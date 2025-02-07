import torch

import torch
import torch.nn as nn


import torch

# 创建一个随机 tensor，假设它的维度是 (2048, 4096) 或 (batch_size, 128, height, width)
x = torch.randn(2048, 4096)

# 保存原始形状
original_shape = x.shape

# 将 tensor 转换为 (-1, 128)，注意这里要确保转换后元素个数一致
x_reshaped = x.view(-1, 128)

# 进行一系列处理，例如对 reshaped tensor 进行操作
# 假设这里是一个简单的加法操作
x_reshaped = x_reshaped + 1

# 恢复为原来的形状
x_restored = x_reshaped.view(original_shape)

# 打印检查
print("Original shape:", original_shape)
print("Reshaped shape:", x_reshaped.shape)
print("Restored shape:", x_restored.shape)