import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path


class Config:
    #计算设备配置
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if DEVICE.startswith('cuda'):
        cudnn.benchmark = True
        cudnn.deterministic = False
        print(f"[System] High-Performance GPU Mode: {torch.cuda.get_device_name(0)}")

    #路径管理

    # 自动定位项目根目录
    # 原理: config.py 在 src/pose_solver/core/ 下，往上跳 3 级就是项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    #将输出目录指向项目根目录下的 "outputs"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    # 自动创建这个文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 数据路径 (保持指向 D 盘)
    PW3D_ROOT = Path(r"D:\PythonProject\RunningAnalsy\data\3DPW")
    PW3D_SEQ_DIR = PW3D_ROOT / "sequenceFiles"
    PW3D_IMG_DIR = PW3D_ROOT / "imageFiles"

    # 模型权重路径 (保持在数据盘或移到项目内均可)
    DATA_ROOT = PW3D_ROOT.parent
    MODEL_DIR = DATA_ROOT / "models"
    LNN_WEIGHTS = MODEL_DIR / "lnn_gait_observer.pth"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    #物理常数
    FPS = 30.0
    DT = 1.0 / FPS
    GRAVITY = 9.80665

    #预训练模型路径
    PRETRAINED_DIR = DATA_ROOT / "pretrained_models"
    SMPL_MODEL_DIR = PRETRAINED_DIR / "smpl"
    J_REGRESSOR_PATH = PRETRAINED_DIR / "J_regressor_h36m.npy"
    HYBRIK_CHECKPOINT = PRETRAINED_DIR / "hybrik_hrnet48_w3dpw.pth"

    # LNN 超参数
    LNN_INPUT_DIM = 6
    LNN_HIDDEN_DIM = 64
    LNN_ODE_STEPS = 2

    # 优化权重
    WEIGHT_GEODESIC = 1.0
    WEIGHT_TWIST_SMOOTH = 15.0
    WEIGHT_DATA_TERM = 0.1
    WEIGHT_ACC_SMOOTH = 1.0
    WEIGHT_ZUPT_LNN = 5000.0