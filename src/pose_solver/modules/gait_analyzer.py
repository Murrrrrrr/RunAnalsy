import numpy as np
import torch
from tensorflow.python.ops.signal.shape_ops import frame

from ..core.config import Config

class GaitEventDetector:
    def __init__(self):
        pass

    def detect_events_from_probs(self, contact_probs, fps=30.0):
        """
        基于LNN输出的触地概率提取关键帧
        :param contact_probs: (T, ) numpy array, 0~1 之间的额值（1=触地，0=腾空）
        :return: event_dict
        """

        #阈值处理
        threshold = 0.5
        is_stance = contact_probs > threshold

        #寻找状态跳变点
        # diff = 1 (0->1): 触地瞬间（Initial Contact, IC）
        # diff = -1 (1->0): 离地瞬间 (Toe Off, TO)
        diff = np.diff(is_stance.astype(int))

        ic_frames = np.where(diff == 1)[0]
        to_frames = np.where(diff == -1)[0]

        #配对步态周期(IC-> TO-> 下一个IC)
        strides = []
        for ic in ic_frames:
            # 找这个IC之后的第一个IO
            next_to = to_frames[to_frames > ic]
            if len(next_to) > 0:
                to = next_to[0]
                next_next_ic = ic_frames[ic_frames > to]
                # 找T0之后的下一个IC
                if len(next_next_ic) > 0:
                    end_ic = next_next_ic[0]
                    strides.append({
                        'start_frame': ic,
                        'to_frame': to,
                        'end_frame': end_ic,
                        'duration': (end_ic - ic) / fps
                    })

        #计算统计指标
        cadence = 0
        if len(strides) > 0:
            avg_duration = np.mean([s['duration'] for s in strides])
            cadence = 60.0 / avg_duration * 2 # 单足->双足步频转换（SPM）

        return {
            'ic_indices': ic_frames,
            'to_indices': to_frames,
            'strides': strides,
            'candence': cadence
        }

    def analyze_pose_metrics(self, optimized_pos, events):
        """
        在关键帧处计算生物力学参数（角度、步幅）
        :param optimized_pos: (T, J, 3)优化后的骨架
        """
        from ..utils.geometry import get_angle_batch

        metrics = []

        pos_np = optimized_pos.cpu().numpy()

        for stride in events['strides']:
            frame = stride['start_frame'] #触地帧

            hip = pos_np[frame, 23]
            knee = pos_np[frame, 25]
            ankle = pos_np[frame, 27]

            #计算向量
            v1 = hip - knee
            v2 = ankle - knee

            # 计算角度
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

            metrics.append({
                'frame': frame,
                'knee_angle_at_contact': angle
            })

        return metrics