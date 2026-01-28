import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # 把 src 加入路径

from src.pose_solver.data.bridge import DataBridge

# 替换成你自己的一个真实文件路径
test_file = Path(r"E:\googleDownload\AthletePose3D_data_set\data\train_set\S2\Axel_1_cam_1_h36m.npy")

if test_file.exists():
    center, labels = DataBridge.load_athlete_data(test_file)
    print(f"数据长度: {len(center)}")
    print(f"触地帧数: {labels.sum().item()}")
    print(f"触地占比: {labels.sum().item() / len(center):.2%}")

    if labels.sum() == 0:
        print("警告：生成的标签全是 0！请检查数据单位是否为毫米，或坐标轴是否旋转错误（Z轴不是高度）。")
    else:
        print("数据正常，可以开始训练。")
else:
    print("文件不存在，请修改路径测试。")