import sys
import argparse
import logging
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 动态添加路径 (保持原有 Hack，但更安全) ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- 模块导入 ---
from pose_solver.core.config import Config
from pose_solver.modules.pose_extractor import MediaPipeExtractor
from pose_solver.modules.physics import PhysicsIMUSimulator
from pose_solver.modules.solver import ManifoldSolver
from pose_solver.modules.gait_analyzer import GaitEventDetector

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Runner")


class GaitAnalysisPipeline:
    def __init__(self, output_dir: str):
        self.device = Config.DEVICE
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化管线，运行设备: {self.device}")

        # 初始化子模块
        self.extractor = MediaPipeExtractor()
        self.imu_sim = PhysicsIMUSimulator(self.device)
        self.solver = ManifoldSolver(self.device)
        self.analyzer = GaitEventDetector()

    def run(self, video_path: str):
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return

        logger.info(f"开始处理视频: {video_path.name}")

        # 1. 视觉提取
        raw_pos_tensor, meta = self._step_vision_extraction(video_path)
        fps = meta['fps']

        # 2. 物理仿真
        center_mass, fake_acc, fake_gyro = self._step_physics_sim(raw_pos_tensor)

        # 3. 求解优化 (LNN)
        refined_pos, contact_probs = self._step_solver(center_mass, fake_acc, fake_gyro)

        # 4. 步态分析
        analysis_result = self.analyzer.detect_events_from_probs(contact_probs, fps)

        self._log_results(analysis_result)

        # 5. 可视化 & 输出
        self._visualize_charts(center_mass, refined_pos, contact_probs, analysis_result, video_path.stem)

        # 6. 生成骨架视频 (可选)
        # self._render_output_video(video_path, analysis_result)

    def _step_vision_extraction(self, video_path: Path):
        logger.info(">>> [阶段 1/4] MediaPipe 姿态提取...")
        try:
            # raw_pos shape: (T, 33, 3)
            raw_pos_tensor, (w, h, fps) = self.extractor.extract_from_video(str(video_path))
            return raw_pos_tensor, {'width': w, 'height': h, 'fps': fps}
        except Exception as e:
            logger.error(f"姿态提取失败: {e}")
            sys.exit(1)

    def _step_physics_sim(self, raw_pos_tensor: torch.Tensor):
        logger.info(">>> [阶段 2/4] 物理仿真生成虚拟传感器...")
        # 选取关键点：取两髋中点 (MediaPipe 23: left_hip, 24: right_hip)
        left_hip = raw_pos_tensor[:, 23, :]
        right_hip = raw_pos_tensor[:, 24, :]
        center_mass = (left_hip + right_hip) / 2.0

        # 生成虚拟加速度
        fake_acc, fake_gyro = self.imu_sim.simulate_from_position_only(center_mass)
        return center_mass, fake_acc, fake_gyro

    def _step_solver(self, center_mass, acc, gyro):
        logger.info(">>> [阶段 3/4] LNN 动力学优化...")
        refined_pos, contact_probs = self.solver.solve_liquid_pakr(center_mass, acc, gyro)

        # 转换为 numpy 方便绘图，如果已经是 Tensor 则保留
        if isinstance(contact_probs, torch.Tensor):
            contact_probs = contact_probs.detach().cpu().numpy()

        return refined_pos, contact_probs

    def _log_results(self, res):
        logger.info("-" * 30)
        logger.info(f"分析完成:")
        logger.info(f"  - 完整步数: {res.stride_count}")
        logger.info(f"  - 估算步频: {res.cadence:.1f} SPM")
        logger.info(f"  - 触地帧数: {len(res.ic_indices)}")
        logger.info("-" * 30)

    def _visualize_charts(self, raw_pos, refined_pos, probs, res, filename_stem):
        logger.info(">>> [阶段 4/4] 生成分析图表...")

        plt.style.use('bmh')  # 使用更好看的样式
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        t = np.arange(len(raw_pos))

        # 子图 1: 轨迹对比
        raw_y = raw_pos[:, 1].detach().cpu().numpy()
        ref_y = refined_pos[:, 1].detach().cpu().numpy()

        ax1.plot(t, raw_y, label='Raw Hip Height (Y)', color='gray', alpha=0.5, linestyle='--')
        ax1.plot(t, ref_y, label='LNN Refined Height', color='#1f77b4', linewidth=2)

        # 标记触地事件
        if len(res.ic_indices) > 0:
            ax1.scatter(res.ic_indices, ref_y[res.ic_indices], color='red', marker='x', s=100, zorder=5,
                        label='Initial Contact')

        ax1.set_ylabel("Height (Norm)")
        ax1.set_title(f"Gait Trajectory Analysis: {filename_stem}")
        ax1.legend()

        # 子图 2: 触地概率
        ax2.plot(t, probs, color='#ff7f0e', label='Contact Probability')
        ax2.axhline(0.5, color='black', linestyle=':', alpha=0.5)
        ax2.fill_between(t, 0, probs, alpha=0.3, color='#ff7f0e')

        # 标记步态周期区间
        for stride in res.strides:
            ax2.axvspan(stride['start_frame'], stride['end_frame'], color='green', alpha=0.1)

        ax2.set_ylabel("Probability")
        ax2.set_xlabel("Frame Index")
        ax2.set_title("Stance Phase Detection")

        plt.tight_layout()

        save_path = self.output_dir / f"{filename_stem}_analysis.png"
        plt.savefig(save_path, dpi=150)
        logger.info(f"图表已保存: {save_path}")
        # plt.show() # 服务器端运行时通常关闭 show


def parse_args():
    parser = argparse.ArgumentParser(description="Run Video Gait Analysis Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--output", "-o", type=str, default="output_results", help="结果输出文件夹")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 更新 Config (如果 Config 类支持动态更新)
    Config.DEVICE = args.device

    pipeline = GaitAnalysisPipeline(output_dir=args.output)
    pipeline.run(args.input)