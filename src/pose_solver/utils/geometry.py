import torch

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """向量转反对称矩阵（Batch, 3）-> (Batch, 3, 3) """
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2],zero,-v[:, 0],
        -v[:, 0], v[:, 1], zero
    ], dim=1)
    return M.view(-1, 3, 3)
def so3_exp_map(w: torch.Tensor) -> torch.Tensor:
    """李代数 so(3) -> 李群 SO(3) (罗得里格斯公式)"""
    theta = torch.norm(w, p=2, dim=1, keepdim=True)
    K = skew_symmetric(w)
    I = torch.eye(3, device=w.device).unsqueeze(0)

    theta_sq = theta ** 2
    theta_safe = theta.clone()
    theta_safe[theta_safe < 1e-6] = 1.0

    #泰勒展开处理小角度
    A = torch.where(theta < 1e-6, 1.0 - theta_sq/6.0, torch.sin(theta)/theta_safe)
    B = torch.where(theta < 1e-6, 0.5 - theta_sq/24.0, (1-torch.cos(theta))/(theta_safe**2))

    return I + A.unsqueeze(-1) * K + B.unsqueeze(-1)*torch.bmm(K, K)
def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """李群SO(3) -> 李代数so(3)"""
    tr = torch.clamp(R[:, 0, 0]+R[:, 1, 1]+R[:, 2, 2], -1.0+ 1e-7, 3.0- 1e-7)
    theta = torch.acos((tr - 1.0) / 2.0).unsqueeze(1)

    theta_safe = theta.clone()
    theta_safe[theta_safe < 1e-6] = 1.0

    scale = torch.where(theta < 1e-6, 0.5 + theta ** 2/12.0 ,theta / (2 * torch.sin(theta_safe)))

    diff = torch.cat([
        (R[:, 2, 1] - R[:, 1, 2]).unsqueeze(1),
        (R[:, 0, 2] - R[:, 2, 0]).unsqueeze(1),
        (R[:, 1, 0] - R[:,0,1]).unsqueeze(1)
    ], dim=1)
    return scale * diff

def geodesic_distance(R1, R2):
    return torch.norm(so3_log_map(torch.bmm(R1.transpose(1, 2), R2)), dim = 1)

def batch_rodrigues(theta):
    return so3_exp_map(theta)
