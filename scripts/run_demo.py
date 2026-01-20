import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pose_solver.core.config import Config
from pose_solver.modules.solver import ManifoldSolver
from pose_solver.data.bridge import DataBridge


def main():
    print("[Demo] 正在启动演示程序...")
    solver = ManifoldSolver(Config.DEVICE)

    # 1. 扫描测试文件 (修改了这里)
    # ==========================================

    # 模式选择：你想跑哪个数据集？
    # 修改这个变量: '3DPW' 或 'ATHLETE'
    DATASET_MODE = 'ATHLETE'

    if DATASET_MODE == '3DPW':
        # 原有逻辑
        search_dir = Config.PW3D_SEQ_DIR / "test"
        search_pattern = "*.pkl"
    else:
        # 新增逻辑: 扫描 E 盘的 AthletePose 数据
        search_dir = Config.ATHLETE_ROOT
        # 假设你的数据直接放在这个目录下，或者是子文件夹里
        # 如果是子文件夹，可以用 "**/*.npy" (递归搜索)
        search_pattern = "**/*_h36m.npy"

    print(f"[System] 正在 {search_dir} 下扫描 {search_pattern} ...")
    test_files = list(search_dir.glob(search_pattern))

    if not test_files:
        print(f"错误: 在 {search_dir} 下没找到任何 {search_pattern} 文件！")
        print("请检查: 1. config.py 里的路径是否写对; 2. 文件夹里是否有对应的 .pkl/.npy 文件")
        return

    # 默认取第一个文件跑演示
    target_file = test_files[0]
    print(f"[Target] 目标文件: {target_file.name}")

    # 2. 智能加载数据
    if target_file.suffix == '.pkl':
        # 3DPW 数据
        print("[IO] 检测到 .pkl 文件，使用 3DPW 加载器...")
        data = DataBridge.load_3dpw_metadata(target_file)
        fps = 30.0  # 3DPW 默认帧率
    elif target_file.suffix == '.npy':
        # AthletePose 数据
        print("[IO] 检测到 .npy 文件，使用 AthletePose 加载器...")
        data = DataBridge.load_athlete_data(target_file)
        # 优先使用加载器返回的 FPS，如果没有则默认 120
        fps = data.get('fps', Config.ATHLETE_FPS)
    else:
        print(f"不支持的文件格式: {target_file.suffix}")
        return

    # 3. 提取数据
    try:
        # 根据 bridge.py 返回的字典键名提取
        # 如果你是 AthletePose，可能需要检查键名是不是 'acc'
        idx = 8
        if 'acc' not in data:
            print(f"字典里缺少 'acc' 键。现有键: {list(data.keys())}")
            return

        # 注意维度检查
        if data['acc'].dim() == 3:  # (Frames, 17, 3) - 3DPW 格式
            acc = data['acc'][:, idx, :].to(Config.DEVICE)
            gyro = data['gyro'][:, idx, :].to(Config.DEVICE)
        else:  # (Frames, 3) - 单传感器格式
            acc = data['acc'].to(Config.DEVICE)
            gyro = data['gyro'].to(Config.DEVICE)

        gt_pos = data['gt_trans'].to(Config.DEVICE)

        # 模拟视觉输入
        print("[Solver] 开始运行流形优化...")
        vis_pos = gt_pos + torch.randn_like(gt_pos) * 0.05

        # 4. 运行优化
        refined_pos, probs = solver.solve_liquid_pakr(vis_pos, acc, gyro)
        print("[Solver] 优化完成！")

        # 5. 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(gt_pos[:, 2].cpu(), label='GT (Z-axis)')
        plt.plot(vis_pos[:, 2].cpu(), label='Vision Input', alpha=0.5)
        plt.plot(refined_pos[:, 2].cpu(), label='Refined (LNN)', linewidth=2)
        plt.legend()
        plt.title(f"Optimization Result: {target_file.name}")

        save_path = Config.OUTPUT_DIR / "demo_result.png"
        plt.savefig(save_path)
        print(f"[Output] 结果已保存至: {save_path}")

    except Exception as e:
        print(f"[Runtime Error] 运行时报错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()