import sys
from pathlib import Path

import cv2
import torch
import matplotlib.pyplot as plt

# 确保能导入 src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from pose_solver.core.config import Config
from pose_solver.modules.pose_extractor import MediaPipeExtractor
from pose_solver.modules.physics import PhysicsIMUSimulator
from pose_solver.modules.solver import ManifoldSolver
from pose_solver.modules.gait_analyzer import GaitEventDetector
from pose_solver.data.video_io import VideoReader, VideoWriter


def main():
    video_path = r"E:\googleDownload\AthletePose3D_data_set\data\train_set\S3\Running_0_cam_1.mp4" # 替换为你的视频路径

    # 1. 初始化模块
    extractor = MediaPipeExtractor()
    imu_sim = PhysicsIMUSimulator(Config.DEVICE)
    solver = ManifoldSolver(Config.DEVICE)
    analyzer = GaitEventDetector()

    print(">>> 正在保存结果视频...")
    reader = VideoReader(video_path)
    output_file = r"D:\PythonProject\RunningAnalsy\video_outputs\result_skeleton.mp4"

    with VideoWriter(output_file, reader.width, reader.height, reader.fps) as writer:
        for i , frame in enumerate(reader):
            cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(frame)

    # 2. 视觉提取 (Visual Front-end)
    print(">>> 阶段1: MediaPipe 姿态提取...")
    # raw_pos shape: (T, 33, 3)
    raw_pos_tensor, (w, h, fps) = extractor.extract_from_video(video_path)

    # 选取关键点：取两髋中点作为躯干中心 (MediaPipe 23, 24)
    left_hip = raw_pos_tensor[:, 23, :]
    right_hip = raw_pos_tensor[:, 24, :]
    center_mass = (left_hip + right_hip) / 2.0

    # 3. 物理仿真 (Physics Simulation)
    print(">>> 阶段2: 逆向动力学仿真生成虚拟传感器数据...")
    # 生成虚拟加速度 (用于输入 LNN 判断触地)
    fake_acc, fake_gyro = imu_sim.simulate_from_position_only(center_mass)

    # 4. 概率优化 (LNN Inference)
    print(">>> 阶段3: LNN 步态相位预测与物理优化...")
    # 注意：solve_liquid_pakr 返回优化后的轨迹和触地概率
    # 这里我们将 center_mass 作为待优化轨迹输入
    refined_pos, contact_probs = solver.solve_liquid_pakr(center_mass, fake_acc, fake_gyro)

    # 5. 关键帧提取 (Keyframe Extraction)
    print(">>> 阶段4: 关键帧事件检测...")
    events = analyzer.detect_events_from_probs(contact_probs, fps)

    print(f"分析结果:")
    print(f"  - 检测到步数: {len(events['strides'])}")
    print(f"  - 估算步频: {events['cadence']:.1f} SPM")
    print(f"  - 触地帧索引: {events['ic_indices']}")

    # 6. 可视化结果
    plt.figure(figsize=(12, 6))

    # 绘制 Z 轴高度变化
    t = range(len(center_mass))
    plt.subplot(2, 1, 1)
    plt.plot(t, center_mass[:, 1].cpu().numpy(), label='Raw Hip Height (Y)', alpha=0.6)
    plt.plot(t, refined_pos[:, 1].cpu().numpy(), label='Refined Hip Height', linewidth=2)
    # 标记关键帧
    plt.scatter(events['ic_indices'], refined_pos[events['ic_indices'], 1].cpu(), c='r', marker='x',
                label='IC (Strike)')
    plt.legend()
    plt.title("Trajectory & Keyframes")

    # 绘制触地概率
    plt.subplot(2, 1, 2)
    plt.plot(t, contact_probs, color='orange', label='LNN Contact Probability')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.fill_between(t, 0, contact_probs, alpha=0.3, color='orange')
    plt.title("Stance Phase Probability")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()