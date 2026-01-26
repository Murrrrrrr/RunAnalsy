import torch
import numpy as np
from pathlib import Path
from ..core.config import Config
import pickle

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
            poses_raw = data['poses'][p_idx] #poses:(T, 72) SMPL姿态参数
            trans_raw = data['trans'][p_idx] #trans:(T, 3) 根节点（骨盆）在世界坐标系的位移
            #转化为Tensor
            gt_trans = torch.tensor(trans_raw, dtype=torch.float32)
            # [:, :3]是只取了SMPL模型中的前三个数值，即“骨盆/根节点”的三个朝向
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
                #输入：根节点旋转 + 根节点位移
                #输出：虚拟的加速度(accel)和 角速度(gyro)
                sim_out = sim.simulate_measurement(gt_root_pose, gt_trans)
                sim_acc = sim_out['accel']
                sim_gyro = sim_out['gyro']

                #数据填充，构造一个（T, 17, 3）的大矩阵
                #Solver求解器期望的输入格式是（Time, Joints, 3），即认为全身17个关键点都应该有数据
                #但上面只用了一个核心部位的数据，于是全部关节的点都复制粘贴
                T = gt_trans.shape[0]
                acc = torch.zeros(T, 17, 3)
                gyro = torch.zeros(T, 17, 3)
                for i in range(17):
                    acc[:, i, :] = sim_acc
                    gyro[:, i, :] = sim_gyro
                print(f"[Bridge] 虚拟 IMU 数据生成完毕")

            #打包好的字典，包含加速度、角速度、真实位移以及帧数
            return {
                'acc': acc,
                'gyro': gyro,
                'gt_trans': gt_trans, # 真实的位移（作为 Ground Truth 用来训练）
                'num_frames': len(gt_trans)
            }
        except Exception as e:
            print(f"数据解析逻辑报错: {e}")
            return None

    @staticmethod
    def load_athlete_data(npy_path, target_fps=60.0):
        """
        加载 AthletePose (.npy) 数据并转换为训练格式
        参数：
            npy_path：文件路径
            target_fps：目标帧率（训练通常用60，原始数据可能是120）
        """
        try:
            # 加载 npy
            raw_pos = np.load(str(npy_path))

            # 情况1：这是一个2D的骨骼数据（x,y）
            # shape 可能是（Time, Joints, 2）
            # 我们需要的是3D数据（x, y, z），所以这种数据直接丢弃
            if raw_pos.ndim == 3 and raw_pos.shape[-1] == 2:
                # 这是一个安静的跳过，不打印错误，防止刷屏
                return None

            # 情况2：数据被压扁了
            # 比如（Time, Joints*3）的二维矩阵
            if raw_pos.ndim == 2:
                # 如果总列数不能被 3 整除，说明数据损坏或者不是XYZ格式，丢弃
                if raw_pos.shape[1] % 3 != 0:
                    return None

                # 尝试还原形状：（T，J*3）->（T, J, 3）
                T = raw_pos.shape[0]
                raw_pos = raw_pos.reshape(T, -1, 3)

            # 情况3：最终确认，如果还不是 3 维或者最后一维不是 3(XYZ)，那就是垃圾数据
            if raw_pos.ndim != 3 or raw_pos.shape[-1] != 3:
                return None

            #获取处理后的维度：帧数，关节点数
            T_raw, num_joints, _ = raw_pos.shape

            # 【下采样处理】
            # 原始数据的帧率（Config.ATHLETE_RAW_FPS）可能是 120
            # 目标帧率（target_fps）是 60
            # step = 120 / 60 =2，意味着每隔一个帧取一个数据
            step = int(max(1, round(Config.ATHLETE_RAW_FPS / target_fps)))
            # 转为PyTorch Tensor，并移动到配置的设备上
            pos = torch.tensor(raw_pos[::step], dtype=torch.float32, device=Config.DEVICE)

            # 【坐标系旋转】
            # 变换: (x, y, z) -> (x, z, -y)
            # 即绕着X轴旋转了-90度
            R_fix = torch.tensor([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ], dtype=torch.float32, device=Config.DEVICE)

            pos_world = torch.matmul(pos, R_fix.T)

            # 【地面高度对齐】
            # 找到整个序列中 Z 的最小值，然后所有人减去这个值
            # 保证人的脚底板最低点刚好踩在 z=0 的地面上
            min_z = torch.min(pos_world[:, :, 2])
            pos_world[:, :, 2] -= min_z

            # 【自动生成触地标签】
            # 因为没有压力板数据，我们需要用算法“猜”什么时候脚踩地了
            idx_l_ankle = 6
            idx_r_ankle = 3

            # 动态调整：如果关节数很少（比如简易骨架），就取最后两个点作为脚踝
            if num_joints <= max(idx_l_ankle, idx_r_ankle):
                idx_l_ankle = num_joints - 2
                idx_r_ankle = num_joints - 1

            # 提取左右脚踝的 Z 坐标（高度）
            z_l = pos_world[:, idx_l_ankle, 2]
            z_r = pos_world[:, idx_r_ankle, 2]

            # 计算垂直速度
            # v = （当前帧高度 - 上一帧高度）/ 时间间隔
            dt = Config.DT
            vel_l = torch.zeros_like(z_l)
            vel_r = torch.zeros_like(z_r)
            vel_l[1:] = (z_l[1:] - z_l[:-1]) / dt
            vel_r[1:] = (z_r[1:] - z_r[:-1]) / dt

            # 核心判据：什么叫“触地”？
            # 1. 高度足够低（z < 8cm）
            # 2. 速度足够慢（v < 1.0m/s，意味着没有剧烈上下移动）
            is_contact_l = (z_l < 0.08) & (vel_l.abs() < 1.0)
            is_contact_r = (z_r < 0.08) & (vel_r.abs() < 1.0)

            # 只要有一只脚触地，就认为人处于“支撑相”（Stance Phase）
            # labels 形状整理为（1, T, 1, 1）方便后续广播计算 loss
            labels = (is_contact_l | is_contact_r).float().view(1, -1, 1, 1)

            # 【质心提取】
            # 如果关键点足够多，取第0个点（通常是骨盆）作为质心
            # 如果关键点太少，取所有点的平均值作为质心
            if num_joints > 4:
                center_mass = pos_world[:, 0, :]
            else:
                center_mass = pos_world.mean(dim=1)

            # 返回处理好的【质心轨迹】和【触地标签】
            return center_mass, labels

        except Exception as e:
            # 只有真正的错误才打印，跳过文件的"错误"不再打印
            if "shapes cannot be multiplied" not in str(e):
                print(f"[Bridge Error] 加载 {npy_path.name} 失败: {e}")
            return None

    @staticmethod
    def get_training_pair(file_path):
        # 如果是 3DPW 的 .pkl文件，目前暂时返回 None
        if str(file_path).endswith('.pkl'):
            return None
        # 如果是 AthletePose 的 .npy文件，调用上面的 load_athlete_data
        elif str(file_path).endswith('.npy'):
            return DataBridge.load_athlete_data(file_path)
        #其他文件不处理
        return None