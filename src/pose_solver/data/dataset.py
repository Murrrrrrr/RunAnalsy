import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from ..core.config import Config
from .bridge import DataBridge
from ..modules.physics import PhysicsIMUSimulator

class AthleteLNNDataset(Dataset):
    def __init__(self, data_root, search_pattern="**/*.npy", device=None, augment=True):
        """
        :param augment: 是否开启数据增强 (训练集True, 验证集False)
        """
        self.data_root = Path(data_root)

        # 搜索所有的 .npy文件
        self.files = list(self.data_root.glob(search_pattern))

        # 设备配置，默认使用Config中定义的
        self.device = device if device else Config.DEVICE

        # 【关键参数】决定是否在训练时给数据加噪
        self.augment = augment

        # 初始化物理仿真器，强制使用CPU
        # 如果使用GPU的话，容易导致CUDA初始化报错或者显存碎片化
        self.imu_sim = PhysicsIMUSimulator('cpu')

        if len(self.files) > 0:
            print(f"[Dataset] Loaded {len(self.files)} files from {self.data_root} (Augment={self.augment})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # 【调用Bridge 加载干净的数据】
        # 返回：干净的质心轨迹(gt_pos_world) 和 触地标签(labels)
        pair = DataBridge.get_training_pair(file_path)

        # 【检验数据有效性】
        # 如果文件损坏（None）或者 帧数太短（<30帧，没法训练序列模型），重试
        if pair is None: return self._retry()
        gt_pos_world, labels = pair
        if len(gt_pos_world) < 30: return self._retry()

        # 【数据增强与物理仿真】
        with torch.no_grad():
            if self.augment:
                # 训练模式：
                # 模拟视觉误差，在真实的轨迹上加入高斯噪声（std=0.02m，即2厘米的抖动）
                noise = torch.randn_like(gt_pos_world) * 0.02
                pos_input = gt_pos_world + noise
            else:
                # 验证模式：
                # 使用干净且完美的轨迹，测试模型的理想性能
                pos_input = gt_pos_world

            # 数据从GPU转移到CPU上
            pos_input = pos_input.cpu()
            labels = labels.cpu()

            # 【物理仿真生成 IMU】
            # 输入的是pos_input
            # 如果是训练模式，不仅位置是噪杂的，算出来的加速度和角速度也会非常抖动
            # 这正是模型需要学习的：从“抖动”的输入中，预测出“干净”的触地标签
            acc_sim, gyro_sim = self.imu_sim.simulate_from_position_only(pos_input)

            # 【归一化】
            # 加速度除以重力加速度G（9.81），角速度除以PI（3.14）
            acc_norm = acc_sim / 9.81
            gyro_norm = gyro_sim / 3.14

            # 拼接特征
            # 输入维度变为（6）：[ax, ay, az, gx, gy, gz]
            imu_input = torch.cat([acc_norm, gyro_norm], dim=-1)

        # 返回
        # imu_input：模拟的传感器数据（带噪声）
        # labels：真实的触地状态（干净的）
        return imu_input, labels.view(-1, 1).cpu()

    def _retry(self):
        """
        重试机制：
        如果当前的文件坏了，就随机找另一个文件代替
        避免因为一个坏文件导致整个训练 epoch 崩溃
        """
        new_idx = np.random.randint(0, len(self))
        return self.__getitem__(new_idx)