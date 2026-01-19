import torch
import torch.nn as nn
from ..core.config import Config

class LiquidCell(nn.Module):
    """
    液态时间常数单元（Liquid Time-Constant Cell）
    动力学方程：dy/dt = - (1/tau) * y + f(x)
    """
    def __init__(self, in_features, hidden_features, tau_init=0.5):
        super().__init__()
        self.input_map = nn.Linear(in_features, hidden_features)
        self.recurrent_map = nn.Linear(hidden_features, hidden_features)

        #可学习的时间常数"tau", 限制在（0.01, inf）之间
        self.tau = nn.Parameter(torch.ones(hidden_features) * tau_init)
        self.act = nn.Tanh()

    def forward(self, x, h, dt):
        #当前的输入激励
        numerator = self.act(self.input_map(x) + self.recurrent_map(h))

        decay = torch.exp(-dt / torch.abs(self.tau))
        h_new = h * decay + numerator * (1 - decay)
        return h_new

class LiquidGaitObserver(nn.Module):
    """
    基于LNN的步态观测器
    输入：Acc + Gyro
    输出： 支撑相概率（Stance Probability）
    """
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = LiquidCell(input_dim, hidden_dim)

        #分类头
        self.classifier = nn.Linear(hidden_dim, 1)
        #Dropout用于MC-Dropout不确定性估计
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_seq, dt):
        """
        :param x_seq: (Batch, Time, Features)
        :param dt: 时间步长
        :return:
        """
        batch_size, seq_len, _ = x_seq.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        outputs = []

        #模拟连续时间动力学
        steps = Config.LNN_ODE_STEPS
        steps_dt = dt / steps

        for t in range(seq_len):
            x_t = x_seq[:, t, :]

            for _ in range(steps):
                h = self.cell(x_t, h, steps_dt)
            outputs.append(h)

        rnn_out = torch.stack(outputs, dim=1) #(Batch, Time, Hidden)

        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)
        return logits