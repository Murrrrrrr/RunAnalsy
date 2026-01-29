import torch
from pathlib import Path


class Config:
    #---路径管理---
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_ROOT = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "video_outputs"
    LNN_WEIGHTS = PROJECT_ROOT / "checkpoints" / "lnn_weights_athlete.pth"
    VIDEO_FRAMES_DIR = PROJECT_ROOT/"video_frames"

    # 视频文件路径
    VIDEO_PATH = r"E:\googleDownload\AthletePose3D_data_set\data\train_set\S3\Running_0_cam_1.mp4"

    # 3DPW 路径
    PW3D_ROOT = Path("D:/Dataset/3DPW")
    PW3D_SEQ_DIR = PW3D_ROOT / "sequenceFiles"

    # AthletePose3D 路径
    ATHLETE_ROOT = Path(r"E:\googleDownload\AthletePose3D_data_set\data")
    ATHLETE_TRAIN_DIR = ATHLETE_ROOT / "train_set"
    ATHLETE_VAL_DIR = ATHLETE_ROOT / "valid_set"
    ATHLETE_TEST_DIR = ATHLETE_ROOT / "test_set"
    TRAIN_SUBSETS = ['S3','S4']
    VAL_SUBSETS = ['S2']
    TEST_SUBSETS = ['S2']

    # --- 设备配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_CORES = 8 #cpu核数
    GPU_MENMORY = 6

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
    VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True )