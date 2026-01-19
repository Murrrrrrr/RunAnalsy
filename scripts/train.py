import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pose_solver.core.config import Config
from pose_solver.models.lnn import LiquidGaitObserver
from pose_solver.data.bridge import DataBridge

def train():
    print("Starting LNN Training...")
    device = Config.DEVICE
    model = LiquidGaitObserver(Config.LNN_INPUT_DIM, Config.LNN_HIDDEN_DIM).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 扫描训练数据
    train_files = list(Config.PW3D_SEQ_DIR.glob("train/*.pkl"))
    if not train_files:
        print("No training data found.")
        return

    model.train()
    for epoch in range(50):
        total_loss = 0
        count = 0

        pbar = tqdm(train_files[:50])  # 每个 epoch 随机取 50 个序列训练
        for pkl_file in pbar:
            pair = DataBridge.get_training_pair(pkl_file)
            if pair is None: continue

            imu_seq, labels = pair
            imu_seq, labels = imu_seq.to(device).unsqueeze(0), labels.to(device).unsqueeze(0)

            optimizer.zero_grad()
            logits = model(imu_seq, Config.DT)

            loss = criterion(logits, labels)
            loss.backward()

            # 梯度裁剪防爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1
            pbar.set_description(f"Epoch {epoch} Loss: {total_loss / count:.4f}")

    # 保存
    torch.save(model.state_dict(), Config.LNN_WEIGHTS)
    print(f"Model saved to {Config.LNN_WEIGHTS}")


if __name__ == "__main__":
    train()