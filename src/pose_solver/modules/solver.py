import torch
import torch.optim as optim
import numpy as np
from torch.utils.checkpoint import checkpoint

from ..core.config import Config
from ..utils.geometry import so3_exp_map, geodesic_distance, skew_symmetric
from ..models.lnn import LiquidGaitObserver

class ManifoldSolver:
    def __init__(self, device):
        self.device = device
        self.lnn = LiquidGaitObserver(Config.LNN_INPUT_DIM, Config.LNN_HIDDEN_DIM).to(device)

        if Config.LNN_WEIGHTS.exists():
            checkpoint = torch.load(Config.LNN_WEIGHTS, map_location=device)
            self.lnn.load_state_dict(checkpoint)
            self.lnn.eval()
            print(f"[Solver] LNN weights loaded.")
        else:
            print(f"[Solver] No weights found at {Config.LNN_WEIGHTS}. Running random init.")

    def solver_latc(self, visual_rot, imu_orient):
        """
        :param visual_rot:
        :param imu_orient:
        :return:
        阶段一： LATC - 消除绕 Z 轴的视觉扭曲
        """
        print(" [Optim] LATC: Manifold Twist Correction...")
        T = visual_rot.shape[0]
        #优化变量：绕 Z 轴的角度 phi
        phi = torch.zeros()
        optimizer = optim.Adam([phi], lr=0.02)
        z_axis = torch.tensor([[0., 0., 1.]],device=self.device).repeat(T, 1)

        for _ in range(50):
            optimizer.zero_grad()
            # R_new = R_vis * Exp(phi * z)
            correction = so3_exp_map(phi * z_axis)
            R_refined = torch.bmm(visual_rot, correction)

            loss = Config.WEIGHT_GEODESIC * torch.mean(geodesic_distance(R_refined, imu_orient) ** 2) + \
                Config.WEIGHT_TWIST_SMOOTH * torch.mean((phi[1:] - phi[:-1]) ** 2)

            loss.backward()
            optimizer.step()

        return torch.bmm(visual_rot, so3_exp_map(phi.detach() * z_axis))

    def solve_liquid_pakr(self, init_pos, imu_acc, imu_gyro):
        """
        :param init_pos:
        :param imu_acc:
        :param imu_gyro:
        :return:
        阶段2：L-PAKR - LNN 引导的轨迹物理优化
        """

        print(" [Optim] L-PAKR: Bayesian Liquid Refinement...")
        T = init_pos.shape[0]
        dt = Config.DT

        acc_norm = imu_acc / 9.81
        gyro_norm = imu_gyro / 3.14
        imu_in = torch.cat([acc_norm, gyro_norm], dim=-1).unsqueeze(0) #(1, T, 6)

        self.lnn.train() #开启Dropout
        probs_list = []
        with torch.no_grad():
            for _ in range(10): #MC-Dropout 采样 10次
                logits = self.lnn(imu_in, dt)
                probs_list.append(torch.sigmoid(logits))

        probs_stack = torch.stack(probs_list) #(10, 1, T, 1)
        mean_prob = probs_stack.mean(dim=0).squeeze().view(-1)
        uncertainty = probs_stack.var(dim=0).squeeze().view(-1)

        #核心算法：结合不确定性的自适应权重
        #如果LNN很确信是支撑相（Prob高, Var低）， 则ZUPT权重极大
        zupt_weight = mean_prob * torch.clamp(1.0 - uncertainty * 5.0, min=0.0)

        pos_opt = init_pos.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([pos_opt], lr=0.05)

        for _ in range(150):
            optimizer.zero_grad()

            #数值微分
            vel = (pos_opt[1:] - pos_opt[:-1]) / dt
            acc = (vel[1:] - vel[:-1]) / dt

            loss_data = torch.mean(torch.norm(pos_opt - init_pos, dim=1)**2)
            loss_smooth = torch.mean(torch.norm(acc, dim=1)**2)

            #P-ZUPT: 概率加权的零速修正
            loss_zupt = torch.mean(zupt_weight[:-1] * torch.norm(vel, dim=1)**2)

            loss = Config.WEIGHT_DATA_TERM * loss_data + \
                   Config.WEIGHT_ACC_SMOOTH * loss_smooth + \
                   Config.WEIGHT_ZUPT_LNN * loss_zupt

            loss.backward()
            optimizer.step()
        return pos_opt.detach(), mean_prob.cpu().numpy()