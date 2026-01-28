import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np

# 路径Hack：获取当前脚本上两级的目录，即根目录
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from pose_solver.core.config import Config
from pose_solver.models.lnn import LiquidGaitObserver
from pose_solver.data.dataset import AthleteLNNDataset

def pad_collate_fn(batch):
    """
    将不同长度的序列填充到相同长度，以便进行 Batch训练
    """
    # 1. 解包 Batch
    # Batch 是一个列表，包含多个 (imu_input, labels) 元组
    # imu_input shape: (Time, 6)
    # labels shape: (Time, 1)
    inputs = [item[0] for item in batch] # 取出所有样本的输入（IMU数据）
    labels = [item[1] for item in batch] # 取出所有样本的标签（0/1触地）

    # 2. 记录真实长度（可选，用于后续处理）
    lengths = torch.tensor([len(x) for x in inputs])

    # 3. 填充输入
    # batch_first=True -> 输出 (Batch, Max_Time, Feature)
    # padding_value=0 -> 输入用 0 填充
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    # 4. 填充标签
    # padding_value = -1：因为标签是0或1， 我们用-1来标记“这里是填充的，不要加入Loss的计算”
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return inputs_padded, labels_padded, lengths

def calculate_metrics_masked(logits, labels, mask, threshold=0.3):
    """
    只计算真实数据部分的指标，忽略 Padding 部分
    """
    # 1.计算预测值
    probs = torch.sigmoid(logits) # 将模型输出的 Logits 转为 0~1 的概率
    preds = (probs > threshold).float() # 大于 0.5判断为1（触地），否则为 0

    # 2.展平并应用 Mask
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    # 只取出mask 为 True的部分进行计算
    valid_preds = preds_flat[mask_flat]
    valid_labels = labels_flat[mask_flat]

    if len(valid_labels) == 0:
        return {"acc": 0, "prec": 0, "rec": 0, "f1": 0}

    # 3.混淆矩阵
    tp = (valid_preds * valid_labels).sum() # 预测1，真实1
    tn = ((1 - valid_preds) * (1 - valid_labels)).sum() # 预测0，真实0
    fp = (valid_preds * (1 - valid_labels)).sum() # 预测1，真实0（误报）
    fn = ((1 - valid_preds) * valid_labels).sum() # 预测0，真实1（漏报）

    epsilon = 1e-7 # 防止分母为0的微小数值

    # 4. 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon) # 准确率
    precision = tp / (tp + fp + epsilon) # 查准率
    recall = tp / (tp + fn + epsilon) # 查全率
    f1 = 2 * (precision * recall) / (precision + recall + epsilon) # 综合指标 F1-score

    return {
        "acc": accuracy.item(),
        "prec": precision.item(),
        "rec": recall.item(),
        "f1": f1.item()
    }

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
    count = 0

    with torch.no_grad(): # 不计算梯度，节省显存并加速
        for imu_input, labels, _ in val_loader:
            imu_input = imu_input.to(device)
            labels = labels.to(device)

            # 生成 Mask: 标签不等于 -1 的地方是有效数据
            mask = (labels != -1).float()

            logits = model(imu_input, Config.DT)

            # 计算 Masked Loss
            # reduction='none' 使得 criterion 返回的是每个点的loss矩阵
            loss_raw = criterion(logits, labels.float())
            # 只保留 mask 区域的 loss，并求平均
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-7)

            total_loss += loss.item()

            # 计算指标并累加
            batch_metrics = calculate_metrics_masked(logits, labels, mask.bool())
            for k, v in batch_metrics.items():
                metrics_sum[k] += v

            count += 1

    # 计算整个验证集的平均值
    avg_loss = total_loss / max(count, 1)
    avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def train():
    print(f"--- 启动 LNN 训练 ---")

    # 检查 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数: {gpu_count}")

    # 根据显存大小调整 Batch Size
    BATCH_SIZE = 32 * gpu_count
    NUM_WORKERS = 8  # cpu核数

    print(f"[提示] 加载训练集...")
    train_ds = AthleteLNNDataset(Config.ATHLETE_TRAIN_DIR, augment=True)
    # 使用自定义的 collate_fn
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn
    )

    print(f"[提示] 加载验证集...")
    val_ds = AthleteLNNDataset(Config.ATHLETE_VAL_DIR, augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate_fn
    )

    model = LiquidGaitObserver(Config.LNN_INPUT_DIM, Config.LNN_HIDDEN_DIM)

    # 如果有多个 GPU，使用 DataParallel
    if gpu_count > 1:
        print(f"[Info] 启用 DataParallel 在 {gpu_count} 张 GPU 上训练")
        model = nn.DataParallel(model) # 模型被包裹在 module层级下面

    model = model.to(Config.DEVICE)

    # 损失函数: reduction='none'
    # 目的是为了告诉 PyTorch 不要直接算平均 Loss，而是把每个样本的 Loss都返回给我
    # 这样我们才能手动乘以 Mask来剔除无效区域
    pos_weight = torch.tensor([10.0]).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 学习率衰减：每10个 Epoch 学习率变为原来的0.5倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        # 初始化统计变量
        train_loss = 0
        train_metrics = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for imu_input, labels, lengths in pbar:
            # 1. 数据上GPU
            imu_input = imu_input.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # 2. 生成 Mask（标签非-1处为1）
            mask = (labels != -1).float()

            # >>>>>> 新增调试探针 >>>>>>
            # 统计当前 Batch 中有多少个 1 (触地)
            valid_mask = (labels != -1)
            num_pos = (labels == 1).sum().item()
            num_total = valid_mask.sum().item()

            if num_pos == 0:
                # 只有当全是 0 的时候才打印警告，防止刷屏
                if train_count % 50 == 0:  # 每50个batch提醒一次
                    print(f"\n[严重警告] Batch {train_count}: 标签全是 0！模型根本学不到东西！")
            else:
                ratio = num_pos / num_total
                if train_count % 50 == 0:
                    print(f"\n[数据正常] Batch {train_count}: 触地占比 {ratio:.2%} ({num_pos}/{num_total})")
            # <<<<<< 新增调试探针 <<<<<<

            optimizer.zero_grad() # 清空梯度
            logits = model(imu_input, Config.DT) # 前向传播

            # 3. Masked Loss 计算
            loss_raw = criterion(logits, labels.float())
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-7)

            # 4. 反向传播与优化
            loss.backward()
            # 梯度裁剪：防止梯度爆炸（LNN/RNN 常有）
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            batch_metrics = calculate_metrics_masked(logits, labels, mask.bool())
            for k, v in batch_metrics.items():
                train_metrics[k] += v
            train_count += 1

            curr_f1 = train_metrics["f1"] / train_count
            pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'F1': f"{curr_f1:.3f}"})

        scheduler.step()

        # 验证阶段
        print(f"Epoch {epoch + 1} >> 正在验证...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE)

        # 打印报表
        print("-" * 60)
        print(f"Dataset | Loss  | Acc   | Prec  | Rec   | F1-Score")
        print("-" * 60)

        t_acc = train_metrics['acc'] / train_count
        t_f1 = train_metrics['f1'] / train_count
        print(
            f"TRAIN   | {train_loss / train_count:.4f}| {t_acc:.3f} | {train_metrics['prec'] / train_count:.3f} | {train_metrics['rec'] / train_count:.3f} | {t_f1:.3f}")
        print(
            f"VALID   | {val_loss:.4f}| {val_metrics['acc']:.3f} | {val_metrics['prec']:.3f} | {val_metrics['rec']:.3f} | {val_metrics['f1']:.3f}")
        print("-" * 60)

        # 保存模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Config.LNN_WEIGHTS

            # DataParallel 会在模型外包一层 .module
            # 保存 model.module.state_dict()，这样单卡也能加载
            model_to_save = model.module if gpu_count > 1 else model
            torch.save(model_to_save.state_dict(), save_path)

            print(f"发现新纪录 (Best F1: {best_f1:.3f})! 模型已保存。")

        print("\n")

if __name__ == "__main__":
    train()