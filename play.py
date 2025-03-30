import torch
import torch.nn as nn
import torch.optim as optim


x = torch.tensor([
        [1.5, -0.125, 4.0, -0.25],  # 不同指数和符号的数字
        [-2.0, 0.0625, -3.0, 0.5]
    ], dtype=torch.float16).cuda()
y = torch.floor(torch.log2(torch.abs(x) + 1e-30))
print(x)
print(y)