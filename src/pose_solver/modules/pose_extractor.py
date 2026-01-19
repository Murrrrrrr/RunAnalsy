from gc import enable

import cv2
import mediapipe as mp
import numpy as np
import torch
from statsmodels.graphics.tukeyplot import results

from ..core.config import Config

class MediaPipeExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        #使用高精度模型 model_complexity = 2
        self.pose_net = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 2,
            smooth_landmarks = True,
            enable_segmentation = False,
            min_detection_confidence = 0.6,
            min_tracking_confidence = 0.6
        )

    def extract_from_video(self, video_path):
        """
        输入视频路径，输出：
            raw_pos: (T, 33, 3) 世界坐标系下的关键点（米）
            img_size: (w, h)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FPS))
        h = int(cap.get(cv2.CAP_PROP_FPS))

        world_landmarks = []

        print(f"[Extractor] 正在处理视频：{video_path}(FPS = {fps})")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_net.process(frame_rgb)

            if results.pose_world_landmarks:
                #MediaPipe 的 world_landmarks 单位大约是米
                #原点在臀部中心
                frame_lm = []
                for lm in results.pose_world_landmarks:
                    frame_lm.append([lm.x, lm.y, lm.z])
                world_landmarks.append(frame_lm)
            else:
                #缺失帧处理：简单复制上一帧或者填0
                if len(world_landmarks) > 0:
                    world_landmarks.append(world_landmarks[-1])
                else:
                    world_landmarks.append([[0,0,0]]*33)

                cap.release()

            #转换为Tensor
            #(T, 33, 3)
            pos_tensor = torch.tensor(np.array(world_landmarks), dtype=torch.float32, device=Config.DEVICE)
            #坐标系修正：MediaPipe 的 Y 轴是向下的（图像坐标系），Z轴对应深度
            #Physic模块可能假设Z轴向上（3DPW习惯）。需根据实际的重力方向调整
            #这里不做旋转，后续在 Simulator 里统一处理重力方向
            return pos_tensor, (w, h, fps)