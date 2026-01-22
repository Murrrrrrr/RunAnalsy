import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from ..core.config import Config

# 1. 动态挂载 HybrIK 源码路径
HYBRIK_SOURCE_DIR = Config.ROOT_DIR / "third_party" / "HybrIK-main"
if str(HYBRIK_SOURCE_DIR) not in sys.path:
    sys.path.append(str(HYBRIK_SOURCE_DIR))

# 2. 导入 HybrIK 模块
try:
    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam
    from hybrik.utils.vis import get_max_iou_box, get_one_box

    print("[HybrIK] 源码加载成功")
except ImportError as e:
    print(f"[HybrIK] 无法加载源码，请检查 third_party/HybrIK-main 是否存在。\n错误:{e}")
    # 为了防止 IDE 报错，定义空的占位符（可选）
    builder = None


class HybrIKInterface:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        print(f"[HybrIK] 正在初始化 (Device: {self.device})...")

        # A. 路径配置
        self.cfg_file = HYBRIK_SOURCE_DIR / "configs" / "256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml"
        self.checkpoint = Config.DATA_ROOT / "pretrained_models" / "hybrik_hrnet48_w3dpw.pth"

        if not self.cfg_file.exists():
            raise FileNotFoundError(f"配置文件缺失：{self.cfg_file}")
        if not self.checkpoint.exists():
            print(f"[Warning] 预训练模型缺失：{self.checkpoint} (请确保已下载)")

        # B. 加载 HybrIK 配置与模型
        # update_config 通常接受字符串路径
        self.cfg = update_config(str(self.cfg_file))

        # 构建 HybrIK 模型
        self.hybrik_model = builder.build_sppe(self.cfg.MODEL)

        # 加载权重
        if self.checkpoint.exists():
            print(f"[HybrIK] 加载权重: {self.checkpoint.name}")
            save_dict = torch.load(self.checkpoint, map_location='cpu')
            if type(save_dict) == dict and 'model' in save_dict:
                self.hybrik_model.load_state_dict(save_dict['model'])
            else:
                self.hybrik_model.load_state_dict(save_dict)

        self.hybrik_model.to(self.device)
        self.hybrik_model.eval()

        # C. 加载检测模型 (Faster R-CNN)
        # HybrIK 需要先检测出人，切图后再输入
        print("[HybrIK] 加载检测器 (Faster R-CNN)...")
        self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.det_model.to(self.device)
        self.det_model.eval()
        self.det_transform = T.Compose([T.ToTensor()])

        # D. 初始化预处理类 (Transformation)
        # 参考 demo_video.py 的逻辑构造 dummy_set
        bbox_3d_shape = getattr(self.cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

        dummy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })

        self.transformation = SimpleTransform3DSMPLCam(
            dummy_set,
            scale_factor=self.cfg.DATASET.SCALE_FACTOR,
            color_factor=self.cfg.DATASET.COLOR_FACTOR,
            occlusion=self.cfg.DATASET.OCCLUSION,
            input_size=self.cfg.MODEL.IMAGE_SIZE,
            output_size=self.cfg.MODEL.HEATMAP_SIZE,
            depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=self.cfg.DATASET.ROT_FACTOR,
            sigma=self.cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=self.cfg.LOSS['TYPE']
        )

        print(f"[HybrIK] 初始化完成")

    def process_video(self, video_path):
        """
        输入视频路径，执行：
        1. 读取视频
        2. Faster R-CNN 人体检测
        3. HybrIK 姿态估计
        4. 格式化输出
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[Vision] 正在处理视频: {Path(video_path).name}")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 结果容器
        results = {
            'pred_pose': [],  # 旋转矩阵 (24, 3, 3)
            'pred_trans': [],  # 全局位移
            'pred_shape': [],  # Betas
            'bboxes': []  # 检测框 (方便后续可视化debug)
        }

        prev_box = None

        # 使用 tqdm 显示进度条
        pbar = tqdm(total=total_frames, desc="HybrIK Inference")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 图像预处理 (BGR -> RGB)
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                # 2. 人体检测 (Detection)
                det_input = self.det_transform(input_image).to(self.device)
                det_output = self.det_model([det_input])[0]

                # 3. 边界框平滑 (Tracking Logic)
                # 如果是第一帧，找置信度最高的框；
                # 如果是后续帧，找和上一帧 IoU 最大的框（防止跳变到其他人身上）
                if prev_box is None:
                    tight_bbox = get_one_box(det_output)  # xyxy
                else:
                    tight_bbox = get_max_iou_box(det_output, prev_box)

                # 如果完全没检测到人，使用上一帧的框，或者跳过
                if tight_bbox is None:
                    # 简单兜底：如果这帧没检测到，就复用上一帧，或者给个空数据
                    if prev_box is not None:
                        tight_bbox = prev_box
                    else:
                        # 极端情况：视频开头就没检测到人
                        pbar.update(1)
                        continue

                prev_box = tight_bbox
                results['bboxes'].append(tight_bbox)

                # 4. HybrIK 姿态估计
                # 裁剪并缩放图像到 256x192
                pose_input, bbox, img_center = self.transformation.test_transform(
                    input_image, tight_bbox)

                pose_input = pose_input.to(self.device)[None, :, :, :]  # (1, 3, 256, 192)

                # 推理
                pose_output = self.hybrik_model(
                    pose_input,
                    flip_test=True,
                    bboxes=torch.from_numpy(np.array(bbox)).to(self.device).unsqueeze(0).float(),
                    img_center=torch.from_numpy(img_center).to(self.device).unsqueeze(0).float()
                )

                # 5. 提取数据
                # pred_theta_mats: (1, 24, 3, 3) - 24个关节的旋转矩阵
                # pred_shape: (1, 10) - SMPL betas
                # transl: (1, 3) - 根节点位移

                rot_mats = pose_output.pred_theta_mats[0].cpu()  # 移回 CPU 存起来节省显存
                betas = pose_output.pred_shape[0].cpu()
                trans = pose_output.transl[0].cpu()

                results['pred_pose'].append(rot_mats)
                results['pred_shape'].append(betas)
                results['pred_trans'].append(trans)

            pbar.update(1)

        cap.release()
        pbar.close()

        # 6. 整合结果
        T_len = len(results['pred_pose'])
        if T_len == 0:
            print("[Error] 未检测到任何姿态数据！")
            return None

        print(f"[HybrIK] 推理结束，共生成 {T_len} 帧数据")

        # Stack 为 Tensor 并转回 Device (根据 Solver 需求，可能在 CPU 或 GPU)
        # 这里统一转为 Tensor
        final_output = {
            'pred_pose': torch.stack(results['pred_pose']).to(self.device),  # (T, 24, 3, 3)
            'pred_trans': torch.stack(results['pred_trans']).to(self.device),  # (T, 3)
            'pred_shape': torch.stack(results['pred_shape']).to(self.device)  # (T, 10)
        }

        return final_output