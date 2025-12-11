import torch
import torch.nn as nn

class Custom_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, alpha):
        super(Custom_LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.alpha = alpha

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 标准化
        normalized = (x - mean) / (std + 1e-5)
        # 应用权重和偏置，并乘以额外系数
        return self.weight * normalized * (2.95 / self.alpha) + self.bias