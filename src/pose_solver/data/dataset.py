import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from ..core.config import Config
from .bridge import DataBridge
from ..modules.physics import PhysicsIMUSimulator

class AthleteLNNDataset(Dataset):
    def __init__(self, data_root, search_pattern="**/*_h36m.npy", device = None, augment=True):
        """
        :param augment: 是否开启数据增强 (训练集True, 验证集False)
        """
        self.data_root = Path(data_root)

        # 搜索所有的 .npy文件
        self.files = list(self.data_root.glob(search_pattern))

        # 【关键参数】决定是否在训练时给数据加噪
        self.augment = augment

        # 初始化物理仿真器，强制使用CPU
        # 如果使用GPU的话，容易导致CUDA初始化报错或者显存碎片化
        self.imu_sim = PhysicsIMUSimulator('cpu')

        print(f"目标帧率：{Config.ATHLETE_FPS}")

        if len(self.files) > 0:
            print(f"[Dataset] Loaded {len(self.files)} files from {self.data_root} (Augment={self.augment})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 优化：使用 while 循环替代递归，防止 RecursionError
        attempts = 0
        max_attempts = 100  # 防止无限死循环

        current_idx = idx

        while attempts < max_attempts:
            file_path = self.files[current_idx]

            # Bridge 现在返回的是 CPU 上的 Tensor
            pair = DataBridge.load_athlete_data(file_path, target_fps=Config.ATHLETE_FPS)

            # 校验
            if pair is not None:
                gt_pos_world, labels = pair
                if len(gt_pos_world) >= 30:
                    # --- 数据有效，开始处理 ---

                    # 1. 噪声注入 (CPU)
                    if self.augment:
                        noise = torch.randn_like(gt_pos_world) * 0.005
                        pos_input = gt_pos_world + noise
                    else:
                        pos_input = gt_pos_world

                    # 2. 物理仿真 (CPU)
                    # 由于 pos_input 已经在 CPU 上，这里没有数据传输开销
                    acc_sim, gyro_sim = self.imu_sim.simulate_from_position_only(pos_input)

                    # 3. 归一化
                    acc_norm = acc_sim / 9.81
                    gyro_norm = gyro_sim / 3.14

                    # 4. 拼接
                    imu_input = torch.cat([acc_norm, gyro_norm], dim=-1)

                    # 返回纯 CPU 数据，PyTorch DataLoader 会自动处理批次堆叠
                    return imu_input, labels.view(-1, 1)

            # --- 数据无效，选择新索引重试 ---
            current_idx = np.random.randint(0, len(self.files))
            attempts += 1

        # 如果尝试了100次都失败，抛出异常或返回零数据（视策略而定）
        raise RuntimeError(f"Failed to load any valid data after {max_attempts} attempts.")

    def _retry(self):
        """
        重试机制：
        如果当前的文件坏了，就随机找另一个文件代替
        避免因为一个坏文件导致整个训练 epoch 崩溃
        """
        new_idx = np.random.randint(0, len(self))
        return self.__getitem__(new_idx)