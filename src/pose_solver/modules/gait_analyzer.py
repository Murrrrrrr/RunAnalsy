import numpy as np
import torch
from ..core.config import Config
from ..utils.geometry import get_angle_batch


class GaitEventDetector:
    def __init__(self):
        pass

    def detect_events_from_probs(self, contact_probs, fps=30.0):
        """
        基于 LNN 输出的触地概率提取关键帧 (增强版)
        """
        # 0. 安全检查：如果概率全是 0 或全是 1 (没动或一直站着)
        if len(contact_probs) == 0:
            return {'cadence': 0, 'strides': [], 'ic_indices': [], 'to_indices': []}

        max_prob = np.max(contact_probs)
        min_prob = np.min(contact_probs)

        # 打印调试信息，帮你确认 LNN 是否工作正常
        # 如果 Max < 0.5，说明模型觉得脚一直没落地，这是上游数据(物理/Bridge)的问题
        if max_prob < 0.5:
            print(f"[Warning] 触地概率过低 (Max={max_prob:.2f})，无法检测步态。")
            return {'cadence': 0, 'strides': [], 'ic_indices': [], 'to_indices': []}

        # 1. 阈值处理
        threshold = 0.5
        is_stance = contact_probs > threshold

        # 2. 寻找状态跳变点
        diff = np.diff(is_stance.astype(int), prepend=is_stance[0])
        ic_frames = np.where(diff == 1)[0]  # 0->1 触地
        to_frames = np.where(diff == -1)[0]  # 1->0 离地

        # 3. 标准步态周期配对 (IC -> TO -> IC)
        strides = []
        for ic in ic_frames:
            next_to = to_frames[to_frames > ic]
            if len(next_to) > 0:
                to = next_to[0]
                next_next_ic = ic_frames[ic_frames > to]
                if len(next_next_ic) > 0:
                    end_ic = next_next_ic[0]
                    duration = (end_ic - ic) / fps
                    if duration > 0.2:
                        strides.append({
                            'start_frame': ic,
                            'to_frame': to,
                            'end_frame': end_ic,
                            'duration': duration
                        })

        # 4. 步频计算 (四重兜底策略)
        cadence = 0.0

        # [策略 A] 标准计算: 找到了完整的步态周期
        if len(strides) > 0:
            avg_dur = np.mean([s['duration'] for s in strides])
            cadence = (60.0 / avg_dur) * 2

        # [策略 B] 只有 IC (触地) 信号: 比如找到了 2 个 IC，不管中间有没有 TO
        elif cadence == 0 and len(ic_frames) >= 2:
            # 直接计算两个落地点的间隔
            duration = (ic_frames[-1] - ic_frames[0]) / (len(ic_frames) - 1) / fps
            cadence = (60.0 / duration) * 2
            print(f"[Info] 估算: 基于多次触地间隔 ({duration:.2f}s)")

        # [策略 C] 只有 TO (离地) 信号: 同上
        elif cadence == 0 and len(to_frames) >= 2:
            duration = (to_frames[-1] - to_frames[0]) / (len(to_frames) - 1) / fps
            cadence = (60.0 / duration) * 2
            print(f"[Info] 估算: 基于多次离地间隔 ({duration:.2f}s)")

        # [策略 D] 短视频混合信号: 只有一个 IC 和一个 TO (顺序不限)
        elif cadence == 0 and len(ic_frames) > 0 and len(to_frames) > 0:
            # 情况 1: IC 在前 (IC -> TO) -> 这是触地时间 (Stance Time)
            if ic_frames[0] < to_frames[0]:
                stance_time = (to_frames[0] - ic_frames[0]) / fps
                # 跑步经验公式: 触地时间约占 40%
                estimated_stride = stance_time / 0.4
                cadence = (60.0 / estimated_stride) * 2
                print(f"[Info] 估算: 基于触地时间 ({stance_time:.2f}s, IC->TO)")

            # 情况 2: TO 在前 (TO -> IC) -> 这是腾空时间 (Swing Time)
            # 你的报错大概率是因为没处理这个情况！
            else:
                swing_time = (ic_frames[0] - to_frames[0]) / fps
                # 跑步经验公式: 腾空时间约占 60%
                estimated_stride = swing_time / 0.6
                cadence = (60.0 / estimated_stride) * 2
                print(f"[Info] 估算: 基于腾空时间 ({swing_time:.2f}s, TO->IC)")

        return {
            'ic_indices': ic_frames,
            'to_indices': to_frames,
            'strides': strides,
            'cadence': cadence
        }

    def analyze_pose_metrics(self, optimized_pos, events):
        """计算关键帧角度"""
        metrics = []

        # 格式兼容处理
        if isinstance(optimized_pos, np.ndarray):
            pos_tensor = torch.tensor(optimized_pos)
        else:
            pos_tensor = optimized_pos.detach().cpu()

        T = pos_tensor.shape[0]

        # 如果没有完整步态，就分析所有的触地帧(IC)
        target_frames = [s['start_frame'] for s in events['strides']]
        if not target_frames:
            target_frames = events['ic_indices']

        for frame in target_frames:
            if frame >= T: continue

            # 提取 Hip(23), Knee(25), Ankle(27)
            # 注意维度: (1, 3)
            hip = pos_tensor[frame, 23, :].unsqueeze(0)
            knee = pos_tensor[frame, 25, :].unsqueeze(0)
            ankle = pos_tensor[frame, 27, :].unsqueeze(0)

            try:
                angle = get_angle_batch(hip, knee, ankle).item()
                metrics.append({
                    'frame': int(frame),
                    'knee_angle_at_contact': angle
                })
            except Exception as e:
                print(f"[Warning] 角度计算失败: {e}")

        return metrics