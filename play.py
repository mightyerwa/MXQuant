import torch

# 创建一个3维张量 (2, 3, 4)
x = torch.tensor([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]],
                 
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]]])

print(f"原始张量形状: {x.shape}")  # torch.Size([2, 3, 4])

# 在dim=0上求最大值
max_dim0, _ = torch.max(x, dim=0)
print(f"dim=0后形状: {max_dim0.shape}")  # torch.Size([3, 4])

# 在dim=1上求最大值
max_dim1, _ = torch.max(x, dim=1)
print(f"dim=1后形状: {max_dim1.shape}")  # torch.Size([2, 4])

# 在dim=2上求最大值
max_dim2, _ = torch.max(x, dim=2)
print(f"dim=2后形状: {max_dim2.shape}")  # torch.Size([2, 3])
