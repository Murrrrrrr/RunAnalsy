import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np

# 路径 Hack
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from pose_solver.core.config import Config
from pose_solver.models.lnn import LiquidGaitObserver
from pose_solver.data.dataset import AthleteLNNDataset


def pad_collate_fn(batch):
    """
    将不同长度的序列填充到相同长度，以便进行 Batch 训练
    """
    # batch 是一个列表，包含多个 (imu_input, labels) 元组
    # imu_input shape: (Time, 6)
    # labels shape: (Time, 1)
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 记录每个样本的真实长度，后面可以用 pack_padded_sequence (可选) 或 mask
    lengths = torch.tensor([len(x) for x in inputs])

    # 执行填充 (Padding)
    # batch_first=True -> 输出 (Batch, Max_Time, Feature)
    # padding_value=0 -> 输入用 0 填充
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    # 标签用 -1 填充，作为忽略标记 (Ignore Index)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return inputs_padded, labels_padded, lengths


# --- 2. 修改：带 Mask 的指标计算 ---
def calculate_metrics_masked(logits, labels, mask, threshold=0.5):
    """
    只计算真实数据部分的指标，忽略 Padding 部分
    """
    # Sigmoid
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # 展平
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    # 只保留有效区域
    valid_preds = preds_flat[mask_flat]
    valid_labels = labels_flat[mask_flat]

    if len(valid_labels) == 0:
        return {"acc": 0, "prec": 0, "rec": 0, "f1": 0}

    # TP, TN, FP, FN
    tp = (valid_preds * valid_labels).sum()
    tn = ((1 - valid_preds) * (1 - valid_labels)).sum()
    fp = (valid_preds * (1 - valid_labels)).sum()
    fn = ((1 - valid_preds) * valid_labels).sum()

    epsilon = 1e-7

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return {
        "acc": accuracy.item(),
        "prec": precision.item(),
        "rec": recall.item(),
        "f1": f1.item()
    }


# --- 3. 修改：验证循环 (支持 Batch) ---
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
    count = 0

    with torch.no_grad():
        for imu_input, labels, _ in val_loader:  # 注意这里多了一个 lengths 返回值
            imu_input = imu_input.to(device)
            labels = labels.to(device)

            # 生成 Mask: 标签不等于 -1 的地方是有效数据
            mask = (labels != -1).float()

            logits = model(imu_input, Config.DT)

            # 计算 Masked Loss
            # reduction='none' 使得 loss 保持维度 (Batch, Time, 1)
            loss_raw = criterion(logits, labels.float())
            # 乘以 mask 把填充部分的 loss 变为 0
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-7)

            total_loss += loss.item()

            # 计算指标
            batch_metrics = calculate_metrics_masked(logits, labels, mask.bool())
            for k, v in batch_metrics.items():
                metrics_sum[k] += v

            count += 1

    avg_loss = total_loss / max(count, 1)
    avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def train():
    print(f"--- 启动 LNN 训练 (多卡高性能版) ---")

    # 检查 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # 根据显存大小调整 Batch Size
    # 4 张卡的话，如果是 24G 显存，可以尝试 64 或 128
    BATCH_SIZE = 32 * gpu_count
    NUM_WORKERS = 12  # 利用 16 核 CPU，留 4 个给系统和主进程

    print(f"[Info] 加载训练集...")
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

    print(f"[Info] 加载验证集...")
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
        model = nn.DataParallel(model)

    model = model.to(Config.DEVICE)

    # 损失函数: reduction='none' 是为了手动处理 Padding Mask
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    best_f1 = 0.0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        train_metrics = {"acc": 0, "prec": 0, "rec": 0, "f1": 0}
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        # 注意：loader 现在返回 3 个值
        for imu_input, labels, lengths in pbar:
            imu_input = imu_input.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # 生成 Mask
            mask = (labels != -1).float()

            optimizer.zero_grad()
            logits = model(imu_input, Config.DT)

            # Masked Loss 计算
            loss_raw = criterion(logits, labels.float())
            # 只计算 mask 为 1 的部分的平均 loss
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-7)

            loss.backward()
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

        # --- 验证阶段 ---
        print(f"Epoch {epoch + 1} >> 正在验证...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, Config.DEVICE)

        # --- 打印报表 ---
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

        # --- 保存模型 ---
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = Config.LNN_WEIGHTS

            # DataParallel 会在模型外包一层 .module
            # 保存时最好保存 model.module.state_dict()，这样单卡也能加载
            model_to_save = model.module if gpu_count > 1 else model
            torch.save(model_to_save.state_dict(), save_path)

            print(f"发现新纪录 (Best F1: {best_f1:.3f})! 模型已保存。")

        print("\n")

if __name__ == "__main__":
    train()