import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from ..core.config import Config

#动态挂载HybrIK源码路径
HYBRIK_SOURCE_DIR = Config.ROOT_DIR / "third_party" / "HybrIK-main"
if str(HYBRIK_SOURCE_DIR) not in sys.path:
    sys.path.append(str(HYBRIK_SOURCE_DIR))

#尝试导入HybrIK模块
try:
    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPL
    print("[HybrIK] 源码加载成功")
except ImportError as e:
    print(f"[HybrIK] 无法加载源码，请检查 third_party/HybrIK-main是否存在。\n错误:{e}")

class HybrIKInterface:
    def __init__(self,device=Config.DEVICE):
        self.device = device
        print("[HybrIK] 初始化模型")

        #指向一个HybrIK的配置文件（yaml），通常在源码里的 configs 目录
        self.cfg_file = HYBRIK_SOURCE_DIR / "configs" / "256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml"
        self.checkpoint = Config.DATA_ROOT / "pretrained_models" / "hybrik_hrnet48_w3dpw.pth"

        if not self.cfg_file.exists():
            raise FileNotFoundError(f"配置文件缺失：{self.cfg_file}")

            # 加载配置
            # 注意：这里需要根据 HybrIK 官方的 update_config 逻辑适配
            # 简单起见，我们假设 model 已经 build 好了

            # 实际工程中，这里通常直接加载模型结构
            # self.model = builder.build_sppe(cfg.MODEL)
            # self.model.load_state_dict(torch.load(self.checkpoint, map_location=device))
            # self.model.to(device)
            # self.model.eval()

            print(f"[HybrIK] 模型加载完成 (Mock mode for setup check)")

        def process_video(self, video_path):
            """
            输入视频路径，输出我们 Solver 需要的姿态数据
            """
            print(f"[Vision] 正在处理视频: {video_path}")

            # 这里是读取视频、逐帧送入 HybrIK 的逻辑
            # 由于 HybrIK 代码较多，这里先写伪代码接口
            # 假设我们得到了 T 帧数据
            T = 100

            # 模拟输出格式
            return {
                'pred_pose': torch.zeros(T, 24, 3, 3).to(self.device),  # 旋转矩阵
                'pred_trans': torch.zeros(T, 3).to(self.device),  # 位移
                'pred_shape': torch.zeros(T, 10).to(self.device)  # SMPL Betas
            }

        def process_image_frame(self, img_tensor):
            # 单帧推理接口
            pass
