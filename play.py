import torch

import torch
import torch.nn as nn


import torch

# 假设 x 是一个三维的 PyTorch tensor，并且 requires_grad=True
x = torch.randn(3, 3, 3, requires_grad=True)  # 创建一个带有梯度追踪的张量

# x_tril = torch.tril(x) * torch.float('inf')
# 使用 torch.zeros_like 来确保类型和 requires_grad 保持一致
# y = torch.where(x <=0, torch.zeros_like(x), torch.log2(x))
y = torch.where(x <=0, torch.tensor(0.0), torch.log2(x))
# 假设你有一个损失函数，这里只是一个简单的平方和损失
loss = y.sum()

# 反向传播计算梯度
loss.backward()

# 查看梯度
print(x.grad)
