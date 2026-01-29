import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

from ..utils.geometry import get_angle_batch

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class GaitAnalysisResult:
    cadence: float = 0.0
    stride_count: int = 0
    ic_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    to_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    strides: List[Dict] = field(default_factory=list)


class GaitEventDetector:
    def __init__(self, threshold: float = 0.5, min_stride_duration: float = 0.2):
        self.threshold = threshold
        self.min_stride_duration = min_stride_duration

    def detect_events_from_probs(self, contact_probs: np.ndarray, fps: float = 30.0) -> GaitAnalysisResult:
        """
        基于 LNN 输出的触地概率提取关键帧
        """
        if len(contact_probs) == 0:
            return GaitAnalysisResult()

        # 转换为 numpy 确保通用性
        if isinstance(contact_probs, torch.Tensor):
            contact_probs = contact_probs.cpu().numpy()

        max_prob = np.max(contact_probs)
        if max_prob < self.threshold:
            logger.warning(f"最大触地概率 ({max_prob:.2f}) 低于阈值，未能检测到有效步态。")
            return GaitAnalysisResult()

        # 1. 状态二值化与边缘检测
        is_stance = contact_probs > self.threshold
        # padding 确保边界处理
        diff = np.diff(is_stance.astype(int), prepend=is_stance[0])

        ic_frames = np.where(diff == 1)[0]  # Initial Contact (触地)
        to_frames = np.where(diff == -1)[0]  # Toe Off (离地)

        # 2. 匹配步态周期 (IC -> TO -> IC)
        strides = []
        for ic in ic_frames:
            # 找到当前 IC 之后的第一个 TO
            valid_to = to_frames[to_frames > ic]
            if len(valid_to) == 0: continue
            to = valid_to[0]

            # 找到该 TO 之后的下一个 IC
            valid_next_ic = ic_frames[ic_frames > to]
            if len(valid_next_ic) == 0: continue
            end_ic = valid_next_ic[0]

            duration = (end_ic - ic) / fps
            if duration > self.min_stride_duration:
                strides.append({
                    'start_frame': int(ic),
                    'to_frame': int(to),
                    'end_frame': int(end_ic),
                    'duration': duration
                })

        # 3. 计算步频 (Cadence)
        cadence = self._calculate_cadence(strides, ic_frames, to_frames, fps)

        return GaitAnalysisResult(
            cadence=cadence,
            stride_count=len(strides),
            ic_indices=ic_frames,
            to_indices=to_frames,
            strides=strides
        )

    def _calculate_cadence(self, strides, ic_frames, to_frames, fps) -> float:
        """封装步频计算策略"""
        # 策略 A: 完美周期
        if strides:
            avg_dur = np.mean([s['duration'] for s in strides])
            return (60.0 / avg_dur) * 2

        # 策略 B: 仅有多点 IC
        if len(ic_frames) >= 2:
            duration = (ic_frames[-1] - ic_frames[0]) / (len(ic_frames) - 1) / fps
            logger.info(f"估算步频: 基于多次触地间隔 ({duration:.2f}s)")
            return (60.0 / duration) * 2

        # 策略 C: 仅有 IC 和 TO (单步片段)
        if len(ic_frames) > 0 and len(to_frames) > 0:
            if ic_frames[0] < to_frames[0]:  # Stance phase
                stance_time = (to_frames[0] - ic_frames[0]) / fps
                return (60.0 / (stance_time / 0.4)) * 2  # 假设触地占 40%
            else:  # Swing phase
                swing_time = (ic_frames[0] - to_frames[0]) / fps
                return (60.0 / (swing_time / 0.6)) * 2

        return 0.0

    def analyze_pose_metrics(self, optimized_pos: torch.Tensor, event_result: GaitAnalysisResult):
        # ... (原有逻辑保持不变，注意适配 GaitAnalysisResult 对象) ...
        pass