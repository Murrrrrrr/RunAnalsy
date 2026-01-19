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

def get_angle_batch(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    计算以 p2 为顶点的夹角 (p1-p2-p3)
    支持批量计算，输入形状通常为 (Batch, 3) 或 (Frames, 3)

    :param p1: 端点 A 坐标
    :param p2: 顶点/中间点坐标 (如膝盖)
    :param p3: 端点 B 坐标
    :return: 角度张量 (Batch,)，单位为度 (Degree)
    """
    # 1. 构建向量：从顶点指向两端
    v1 = p1 - p2
    v2 = p3 - p2

    # 2. 归一化向量 (Normalize)
    # dim=-1 表示沿着 (x,y,z) 维度计算范数
    v1_n = torch.nn.functional.normalize(v1, p=2, dim=-1)
    v2_n = torch.nn.functional.normalize(v2, p=2, dim=-1)

    # 3. 计算点积 (Dot Product)
    # sum(v1_n * v2_n) 等价于 dot(v1_n, v2_n)
    dot_product = torch.sum(v1_n * v2_n, dim=-1)

    # 4. 数值稳定性截断 (Clamping)
    # 防止因为浮点误差导致值略微超过 1.0 或 -1.0，从而使 acos 返回 NaN
    dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    # 5. 反余弦计算弧度 -> 转换为角度
    angle_rad = torch.acos(dot_product)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg
