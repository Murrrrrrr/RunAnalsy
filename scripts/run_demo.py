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

    # 1. 扫描测试文件
    # 注意：这里需要根据你实际想跑的数据集修改后缀 (.pkl 或 .npz)
    test_files = list(Config.PW3D_SEQ_DIR.glob("test/*.pkl"))

    if not test_files:
        print(f"错误: 在 {Config.PW3D_SEQ_DIR}/test 下没找到测试文件！")
        return

    target_file = test_files[0]
    print(f"[Target] 目标文件: {target_file.name}")

    # 2. 尝试加载数据 (这里加了详细调试)
    print("[IO] 正在调用 DataBridge 加载数据...")

    # --- 分支判断：你是跑 3DPW 还是 Athlete？---
    if target_file.suffix == '.pkl':
        data = DataBridge.load_3dpw_metadata(target_file)
    else:
        # 假设你有这个适配 Athlete 的函数
        data = DataBridge.load_athlete_data(target_file)

    print(f"[IO] 数据加载成功！帧数: {data.get('num_frames', '未知')}")

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