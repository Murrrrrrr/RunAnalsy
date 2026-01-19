import torch
import pickle
import numpy as np
from ..core.config import Config


class DataBridge:
    @staticmethod
    def load_3dpw_metadata(pkl_path):
        """加载 3DPW pkl 文件 """
        if not pkl_path.exists():
            return None

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"读取错误: {e}")
            return None

        p_idx = 0

        #基础数据提取
        try:
            # 3DPW的poses是 (N, 72)，前3位是全局旋转
            # trans是(N, 3)
            poses_raw = data['poses'][p_idx]
            trans_raw = data['trans'][p_idx]

            # 转 ensor
            gt_trans = torch.tensor(trans_raw, dtype=torch.float32)
            gt_root_pose = torch.tensor(poses_raw[:, :3], dtype=torch.float32)  # 只取根节点旋转

            #是否有真实 IMU？
            if 'imu_accel' in data:
                # 情况A: 有真实数据 (Full Dataset)
                acc = torch.tensor(data['imu_accel'][p_idx], dtype=torch.float32)
                gyro = torch.tensor(data['imu_ang_vel'][p_idx], dtype=torch.float32)
                print(f"[Bridge] 使用真实 IMU 数据")
            else:
                # 情况B:没有数据 (Standard Dataset) -> 启动物理仿真
                print(f"[Bridge] 缺失 IMU 数据，正在进行物理仿真生成...")

                # 动态导入防止循环引用
                from ..modules.physics import PhysicsIMUSimulator

                # 我们需要在 CPU 上计算防止显存打架
                sim = PhysicsIMUSimulator('cpu')

                # 针对 3DPW 数据特性，我们模拟根节点(Root)或脚踝的IMU
                # 这里为了演示，我们模拟根节点 (SMPL Root)
                sim_out = sim.simulate_measurement(gt_root_pose, gt_trans)

                sim_acc = sim_out['accel']
                sim_gyro = sim_out['gyro']

                # 关键：为了骗过 Solver，我们需要构造一个 (T, 17, 3) 的大矩阵
                # 并在 Solver 预期的索引 (比如 8-右脚踝) 填入数据
                T = gt_trans.shape[0]
                acc = torch.zeros(T, 17, 3)
                gyro = torch.zeros(T, 17, 3)

                # 将生成的虚拟数据填入所有关节 (或者只填入脚踝 index 8)
                # 这里我们简单粗暴地填入所有位置，保证 Solver取哪儿都有数
                for i in range(17):
                    acc[:, i, :] = sim_acc
                    gyro[:, i, :] = sim_gyro

                print(f"[Bridge] 虚拟 IMU 数据生成完毕 (Frame: {T})")

            return {
                'acc': acc,
                'gyro': gyro,
                'gt_trans': gt_trans,
                'num_frames': len(gt_trans)
            }

        except Exception as e:
            print(f"数据解析逻辑报错: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def get_training_pair(pkl_path, sensor_idx=8):
        """
        生成 LNN 训练数据: (IMU, Label)
        """
        data = DataBridge.load_3dpw_metadata(pkl_path)
        if data is None: return None

        imu_acc = data['acc'][:, sensor_idx, :]
        imu_gyro = data['gyro'][:, sensor_idx, :]
        trans = data['gt_trans']

        # 自动生成标签: 速度 < 0.08m/s 为支撑相
        vel = torch.norm(trans[1:] - trans[:-1], dim=1)
        vel = torch.cat([vel, vel[-1:]])  # 补齐
        labels = (vel < 0.08).float().view(-1, 1, 1)  # [cite: 16]

        return torch.cat([imu_acc, imu_gyro], dim=-1), labels