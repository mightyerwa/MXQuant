import torch

import torch
import torch.nn as nn


import torch
import torch.nn as nn

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding(10, 10)
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # 打印输入数据类型
        print(f"Input dtype: {x.dtype}")

        # 线性层计算
        x = self.linear(x)

        # 打印线性层输出数据类型
        print(f"Linear output dtype: {x.dtype}")

        # 添加一个激活函数（ReLU）
        x = torch.relu(x)

        # 打印激活函数输出数据类型
        print(f"ReLU output dtype: {x.dtype}")

        return x

# 创建模型并转换为 bfloat16
model = MyModel().to(torch.bfloat16).cuda()  # 将模型移动到 GPU 并转换为 bfloat16

# 创建输入数据并转换为 float16
x = torch.randn(1, 10).cuda().to(torch.float16)  # 输入是 float16

# 使用 autocast 进行前向传播
with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 指定 autocast 使用 bfloat16
    output = model(x)

# 打印最终输出数据类型
print(f"Final output dtype: {output.dtype}")