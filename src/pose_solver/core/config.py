import torch
from pathlib import Path


class Config:
    # --- 路径配置 ---
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "video_outputs"

    # 3DPW 路径 (保留备用)
    PW3D_ROOT = Path("D:/Dataset/3DPW")
    PW3D_SEQ_DIR = PW3D_ROOT / "sequenceFiles"

    # [修改] 这里填您的数据集"最顶层"文件夹
    # 只要填到 AthletePose3D_data_set 这一层即可，程序会自动往里找
    ATHLETE_ROOT = Path(r"E:\googleDownload\AthletePose3D_data_set")

    # 模型权重保存路径
    LNN_WEIGHTS = PROJECT_ROOT / "data" / "models" / "lnn_gait_observer.pth"

    # --- 物理与训练参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 仿真参数
    GRAVITY = 9.81
    DT = 1.0 / 60.0

    # [保留修复] 原始数据的帧率
    ATHLETE_RAW_FPS = 120.0

    # LNN 网络参数
    LNN_INPUT_DIM = 6
    LNN_HIDDEN_DIM = 64
    LNN_ODE_STEPS = 6

    # 损失函数权重
    WEIGHT_DATA_TERM = 1.0
    WEIGHT_ACC_SMOOTH = 0.1
    WEIGHT_ZUPT_LNN = 2.0
    WEIGHT_GEODESIC = 1.0
    WEIGHT_TWIST_SMOOTH = 10.0

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    LNN_WEIGHTS.parent.mkdir(exist_ok=True, parents=True)