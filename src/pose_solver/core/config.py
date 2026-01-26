import torch
from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "video_outputs"
    LNN_WEIGHTS = PROJECT_ROOT / "checkpoints" / "lnn_weights_athlete.pth"

    # 3DPW 路径 (保留备用)
    PW3D_ROOT = Path("D:/Dataset/3DPW")
    PW3D_SEQ_DIR = PW3D_ROOT / "sequenceFiles"

    # --- 路径配置 ---
    # 请确保这里的路径和你的文件夹结构一致
    ATHLETE_ROOT = Path(r"E:\googleDownload\AthletePose3D_data_set\data")
    ATHLETE_TRAIN_DIR = ATHLETE_ROOT / "train_set"
    ATHLETE_VAL_DIR = ATHLETE_ROOT / "valid_set" # [新增] 验证集目录
    ATHLETE_TEST_DIR = ATHLETE_ROOT / "test_set"

    # --- 设备配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 模型与物理参数 ---
    DT = 1.0 / 30.0
    GRAVITY = 9.81
    LNN_INPUT_DIM = 6
    LNN_HIDDEN_DIM = 64
    LNN_ODE_STEPS = 3

    # --- 训练超参 ---
    WEIGHT_DATA_TERM = 1.0
    WEIGHT_ACC_SMOOTH = 0.1
    WEIGHT_ZUPT_LNN = 10.0
    WEIGHT_GEODESIC = 1.0
    WEIGHT_TWIST_SMOOTH = 10.0

    ATHLETE_RAW_FPS = 120.0
    ATHLETE_FPS = 30.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LNN_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)