import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 第一层
        self.fc2 = nn.Linear(50, 1)   # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 输入数据和标签
x = torch.randn(32, 10).to("cuda")  # batch_size=32, 特征维度=10
y = torch.randn(32, 1).to("cuda")  # 输出标签

# 初始化模型和优化器
model = SimpleNN().to("cuda")
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义损失函数
loss_fn = nn.MSELoss()

# 分开计算 input_backward 和 weight_backward
def forward_backward_pipeline(model, x, y, optimizer):
    # 正向传播
    y_pred = model(x)
    
    # 计算损失
    loss = loss_fn(y_pred, y)

    # 计算 input backward（反向传播输入的梯度）
    loss.backward(retain_graph=True)  # retain_graph=True 以便后续使用

    # 计算 weight backward（反向传播权重的梯度）
    optimizer.step()

    # 清零梯度
    optimizer.zero_grad()

    return loss.item()

# 训练循环
for epoch in range(100):
    loss = forward_backward_pipeline(model, x, y, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')