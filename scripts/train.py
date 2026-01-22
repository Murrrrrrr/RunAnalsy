import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# 导入模块
from pose_solver.core.config import Config
from pose_solver.models.lnn import LiquidGaitObserver
from pose_solver.data.bridge import DataBridge
from pose_solver.modules.physics import PhysicsIMUSimulator


def train():
    print(f"Starting LNN Training on AthletePose dataset...")
    print(f"Data Root: {Config.ATHLETE_ROOT}")

    device = Config.DEVICE

    # 1. 初始化模型
    model = LiquidGaitObserver(Config.LNN_INPUT_DIM, Config.LNN_HIDDEN_DIM).to(device)
    imu_sim = PhysicsIMUSimulator(device)

    # 2. 优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 3. 扫描训练数据
    # [修改] 使用递归搜索 "**", 并且匹配所有 ".npy" 文件
    # 这样能找到 E:\...\data\train_set\S3\Running_0_cam_1.npy
    search_pattern = "**/*.npy"
    train_files = list(Config.ATHLETE_ROOT.glob(search_pattern))

    if not train_files:
        print(f"Error: No .npy files found in {Config.ATHLETE_ROOT}")
        print(f"Search pattern was: {search_pattern}")
        return

    print(f"Found {len(train_files)} training sequences.")

    # 4. 训练循环
    model.train()
    epochs = 30

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        np.random.shuffle(train_files)
        # 每次只取 100 个片段训练，避免一轮太久
        subset_files = train_files[:100]

        pbar = tqdm(subset_files)
        for npy_file in pbar:
            # 加载数据
            pair = DataBridge.get_training_pair(npy_file)
            if pair is None: continue

            gt_pos_world, labels = pair
            if len(gt_pos_world) < 30: continue

            # Sim-to-Sim 在线生成
            with torch.no_grad():
                noise = torch.randn_like(gt_pos_world) * 0.02
                noisy_pos = gt_pos_world + noise
                acc_sim, gyro_sim = imu_sim.simulate_from_position_only(noisy_pos)

                acc_norm = acc_sim / 9.81
                gyro_norm = gyro_sim / 3.14
                imu_input = torch.cat([acc_norm, gyro_norm], dim=-1).unsqueeze(0)

            optimizer.zero_grad()
            logits = model(imu_input, Config.DT)

            loss = criterion(logits, labels.view_as(logits))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / (count + 1e-5):.4f}")

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), Config.LNN_WEIGHTS)
            print(f"Checkpoint saved to {Config.LNN_WEIGHTS}")

    print("Training Complete.")


if __name__ == "__main__":
    train()