import torch
import pickle
import numpy as np
from pathlib import Path
from ..core.config import Config

class DataBridge:
    @staticmethod
    def load_3dpw_metadata(pkl_path):
        """加载 3DPW pkl 文件 """
        if not pkl_path.exists(): return None
       # 若pickle文件是用旧版本python生成的，则抛出错误
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"读取错误: {e}, 请确保.pkl是由python3生成的")
            return None

        try:
            p_idx = 0 #硬编码第一个人
            poses_raw = data['poses'][p_idx]
            trans_raw = data['trans'][p_idx]
            gt_trans = torch.tensor(trans_raw, dtype=torch.float32)
            # [:, :3]是只取了SMPL模型中的前三个数值，及“骨盆/根节点”的三个朝向
            gt_root_pose = torch.tensor(poses_raw[:, :3], dtype=torch.float32)

            if 'imu_accel' in data:
                #情况A：有真实的IMU数据
                acc = torch.tensor(data['imu_accel'][p_idx], dtype=torch.float32)
                gyro = torch.tensor(data['imu_ang_vel'][p_idx], dtype=torch.float32)
                print(f"[Bridge] 使用真实 IMU 数据")
            else:
                #情况B：没有真实的IMU数据 -> 启动物理仿真
                print(f"[Bridge] 缺失 IMU 数据，正在进行物理仿真生成...")
                from ..modules.physics import PhysicsIMUSimulator
                sim = PhysicsIMUSimulator('cpu') # 防止显存冲突，用CPU计算
                #根据视觉捕捉到的动作，反推“如果这里有个传感器，它会检测到什么”
                sim_out = sim.simulate_measurement(gt_root_pose, gt_trans)
                sim_acc = sim_out['accel']
                sim_gyro = sim_out['gyro']

                #数据填充，构造一个（T, 17, 3）的大矩阵
                #Solver求解器期望的输入格式是（Time, Joints, 3），即认为全身17个关键点都应该有数据
                #但上面只用了一个核心部位的数据，于是全部复制粘贴
                T = gt_trans.shape[0]
                acc = torch.zeros(T, 17, 3)
                gyro = torch.zeros(T, 17, 3)
                for i in range(17):
                    acc[:, i, :] = sim_acc
                    gyro[:, i, :] = sim_gyro
                print(f"[Bridge] 虚拟 IMU 数据生成完毕")

            return {
                'acc': acc, 'gyro': gyro, 'gt_trans': gt_trans,
                'num_frames': len(gt_trans)
            }
        except Exception as e:
            print(f"数据解析逻辑报错: {e}")
            return None

    @staticmethod
    def load_athlete_data(data_path):
        """
        [智能版] 加载 AthletePose 数据集并自动修正重力方向
        """
        data_path = Path(data_path)
        print(f"[Bridge] 正在加载 AthletePose 数据: {data_path.name}")

        try:
            # 1. 加载骨架
            kpts_3d_np = np.load(data_path, allow_pickle=True)
            kpts_3d = torch.tensor(kpts_3d_np, dtype=torch.float32)

            # 提取根节点 (假设索引0)
            gt_trans = kpts_3d[:, 0, :]
            T = gt_trans.shape[0]

            # ==========================================
            # 2. [核心修复] 自动检测垂直轴 (Up-Axis)
            # ==========================================
            # 逻辑：跑步时，人是站着的。平均坐标值最大的轴通常是高度轴。
            # (X, Z 通常在原点附近波动，Y 或 Z 会有 ~0.9m 的平均高度)
            mean_vals = torch.mean(gt_trans, dim=0)  # [mean_x, mean_y, mean_z]
            up_axis_idx = torch.argmax(torch.abs(mean_vals)).item()

            axis_names = ['X', 'Y', 'Z']
            print(
                f"[Bridge] 自动检测坐标系: {axis_names[up_axis_idx]}轴 似乎是垂直向上的 (Mean={mean_vals[up_axis_idx]:.2f}m)")

            # ==========================================
            # 3. 物理仿真 (Applied correct gravity)
            # ==========================================
            dt = 1.0 / Config.ATHLETE_FPS

            # v = dx/dt
            vel = torch.zeros_like(gt_trans)
            vel[1:] = (gt_trans[1:] - gt_trans[:-1]) / dt

            # a = dv/dt
            acc_world = torch.zeros_like(vel)
            acc_world[1:] = (vel[1:] - vel[:-1]) / dt

            # 构造重力向量
            g_vec = torch.zeros(3)
            # 假设重力是负方向 (-9.8)，如果此时 up_axis 为正，则 g = [0, -9.8, 0]
            # 传感器读数 = a_world - g
            # 静止时: a=0, 读数 = 0 - (-9.8) = +9.8 (指向上方)
            g_vec[up_axis_idx] = -Config.GRAVITY

            print(f"[Bridge] 应用重力修正: g={g_vec.tolist()}")

            acc_sim = acc_world - g_vec

            # 4. 构造输出
            acc_out = torch.zeros(T, 17, 3)
            gyro_out = torch.zeros(T, 17, 3)

            for i in range(17):
                acc_out[:, i, :] = acc_sim

            return {
                'acc': acc_out,
                'gyro': gyro_out,
                'gt_trans': gt_trans,
                'num_frames': T,
                'fps': Config.ATHLETE_FPS
            }

        except Exception as e:
            print(f"[Error] AthletePose 文件加载失败: {e}")
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
        labels = (vel < 0.08).float().view(-1, 1, 1)

        return torch.cat([imu_acc, imu_gyro], dim=-1), labels