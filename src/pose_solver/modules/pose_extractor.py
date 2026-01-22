import cv2
import mediapipe as mp
import numpy as np
import torch
from ..core.config import Config


class MediaPipeExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # 使用高精度模型 model_complexity = 2
        self.pose_net = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def extract_from_video(self, video_path):
        """
        输入视频路径，输出：
            raw_pos: (T, 33, 3) 世界坐标系下的关键点（米），已转换为 Z-Up 坐标系
            img_size: (w, h, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        world_landmarks = []

        print(f"[Extractor] 正在处理视频：{video_path} (FPS = {fps:.2f})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_net.process(frame_rgb)

            if results.pose_world_landmarks:
                frame_lm = []
                # MediaPipe 输出的是 normalized landmark list
                for lm in results.pose_world_landmarks.landmark:
                    frame_lm.append([lm.x, lm.y, lm.z])
                world_landmarks.append(frame_lm)
            else:
                # 缺失帧处理：简单复制上一帧或者填0
                if len(world_landmarks) > 0:
                    world_landmarks.append(world_landmarks[-1])
                else:
                    # 33 个关键点
                    world_landmarks.append([[0, 0, 0]] * 33)

        cap.release()

        if len(world_landmarks) == 0:
            print("[Error] 未检测到任何骨骼数据！")
            return torch.zeros(1, 33, 3).to(Config.DEVICE), (w, h, fps)

        # 转换为 Tensor: (T, 33, 3)
        pos_tensor = torch.tensor(np.array(world_landmarks), dtype=torch.float32, device=Config.DEVICE)

        # =========================================================
        # [Critical Fix] 坐标系旋转: MediaPipe (Y-Down) -> Physics (Z-Up)
        # =========================================================
        # MediaPipe 原生: X(右), Y(下), Z(深/Camera)
        # 目标 (Physics): X(右), Y(前/深), Z(上)
        # 变换逻辑: Rotate X -90 deg => (x, y, z) -> (x, z, -y)

        # 旋转矩阵 (3, 3)
        R_x = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32, device=Config.DEVICE)

        # 应用旋转: pos @ R.T
        # pos shape: (T, 33, 3), R shape: (3, 3)
        pos_tensor_rotated = torch.matmul(pos_tensor, R_x.T)

        # 高度对齐 (Ground Alignment)
        # 找到全序列中最低点的 Z 值，将其设为 0 (假设有一刻脚踩在地面)
        # 注意：这里取所有帧、所有点的最小值作为地面参考
        min_z = torch.min(pos_tensor_rotated[:, :, 2])
        pos_tensor_rotated[:, :, 2] -= min_z

        print(f"[Extractor] 完成提取。帧数: {len(pos_tensor_rotated)}, 地面高度修正: {-min_z:.4f}m")

        return pos_tensor_rotated, (w, h, fps)