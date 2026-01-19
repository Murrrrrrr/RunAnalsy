import torch
import numpy as np
from ..core.config import Config
from ..utils.geometry import batch_rodrigues


class PhysicsIMUSimulator:
    """
    物理惯性仿真器
    作用：当数据集缺失真实 IMU 时，根据位置(Trans)和旋转(Rot)逆向生成虚拟 IMU 数据
    """

    def __init__(self, device):
        self.device = device
        self.g = torch.tensor([0, 0, -Config.GRAVITY], device=device).view(1, 3)  # 3DPW 重力通常在 Z 轴或 Y 轴，视坐标系而定

    def simulate_measurement(self, pose_aa, trans):
        """
        输入:
            pose_aa: (T, 3) 轴角形式的全局旋转 (通常是 SMPL 的前3位)
            trans:   (T, 3) 全局位移
        输出:
            dict {'accel': (T, 3), 'gyro': (T, 3)}
        """
        # 确保数据在设备上
        if pose_aa.device != self.device: pose_aa = pose_aa.to(self.device)
        if trans.device != self.device: trans = trans.to(self.device)

        T = trans.shape[0]
        dt = Config.DT

        # 计算线速度和加速度 (世界坐标系)
        # v = dx / dt
        vel = torch.zeros_like(trans)
        vel[1:] = (trans[1:] - trans[:-1]) / dt

        # a = dv / dt
        acc_world = torch.zeros_like(vel)
        acc_world[1:] = (vel[1:] - vel[:-1]) / dt

        # 计算角速度 (Gyro)
        # 将轴角转换为旋转矩阵 R (T, 3, 3)
        R = batch_rodrigues(pose_aa)

        # R_diff = R_t^T * R_{t+1} (计算相邻帧的旋转差)
        R_curr = R[:-1]
        R_next = R[1:]
        R_diff = torch.bmm(R_curr.transpose(1, 2), R_next)

        from ..utils.geometry import so3_log_map
        w_body = torch.zeros((T, 3), device=self.device)
        # log_map 将旋转差变回角速度向量
        w_body[:-1] = so3_log_map(R_diff) / dt

        # 合成加速度计读数 (Acc)
        # IMU 测量的不是纯加速度，而是 (a - g)，并投影到身体坐标系
        # Acc_body = R^T * (a_world - g_world)
        acc_specific = acc_world - self.g  # 减去重力
        acc_body = torch.bmm(R.transpose(1, 2), acc_specific.unsqueeze(-1)).squeeze(-1)

        # 添加噪声 (模拟真实传感器的不完美)
        # 高斯白噪声
        acc_noise = torch.randn_like(acc_body) * 0.2  # m/s^2
        gyro_noise = torch.randn_like(w_body) * 0.01  # rad/s

        return {
            'accel': acc_body + acc_noise,
            'gyro': w_body + gyro_noise
        }