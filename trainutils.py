import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer  # Add this import
import math


training_args = {
    # LoRA参数
    "lora_lr": {
        "attention": 5e-5,  # 注意力层
        "ffn": 8e-5,       # FFN层
    },
    # 其他可训练参数
    "other_lr": 1e-4,
    
    # Weight Decay
    "lora_wd": {
        "attention": 0.1,
        "ffn": 0.05,
    },
    "other_wd": 0.01,
}

def get_lora_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_optimizer(model, config):
    # 分组参数
    lora_params_attn = []
    lora_params_ffn = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                lora_params_attn.append(param)
            else:
                lora_params_ffn.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': lora_params_attn, 
         'lr': config.lora_lr['attention'],
         'weight_decay': config.lora_wd['attention']},
        {'params': lora_params_ffn,
         'lr': config.lora_lr['ffn'],
         'weight_decay': config.lora_wd['ffn']},
        {'params': other_params,
         'lr': config.other_lr,
         'weight_decay': config.other_wd}
    ]
    
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
    
class WarmupCosineScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with warmup and cosine annealing
    """
    def __init__(
        self, 
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(torch.tensor(progress) * math.pi)) / 2
                   for base_lr in self.base_lrs]

if __name__ == "__main__":
    # 测试
    class MultiLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.let_layer = nn.Linear(100, 80)
            self.lwc_layer = nn.Linear(80, 60)
            self.lora_layer = nn.Linear(60, 100)
        def forward(self, x):
            x = torch.relu(self.let_layer(x))
            x = torch.relu(self.lwc_layer(x))
            return self.lora_layer(x)
    def plot_multi_lr_schedule(lrs_history, warmup_steps, total_steps, names):
        """绘制多个学习率的变化曲线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制学习率曲线
        for i, (name, lrs) in enumerate(zip(names, lrs_history)):
            plt.plot(lrs, label=f'{name} LR', alpha=0.7)
        
        # 添加warmup分界线 - 调整位置到实际的warmup结束点
        plt.axvline(x=warmup_steps - 1, color='r', linestyle='--', label='Warmup End')
        
        # 设置图表属性
        plt.title('Multiple Learning Rate Schedules')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        
        # 移除对数刻度，使用线性刻度
        plt.yscale('linear')  # 将'log'改为'linear'
        
        # 设置y轴刻度
        all_lrs = [lr for lrs in lrs_history for lr in lrs]  # 展平所有学习率
        max_lr = max(all_lrs)
        min_lr = min(all_lrs)
        
        # 计算合适的刻度间隔（使用10个刻度）
        interval = (max_lr - min_lr) / 10
        ticks = np.arange(min_lr, max_lr + interval, interval)
        plt.yticks(ticks)
        
        # 设置科学计数法格式
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('multi_lr_schedule.png', dpi=300, bbox_inches='tight')
        plt.close()

    
    total_epochs = 100
    warmup_epochs = 10
    let_lr = 0.1
    lwc_lr = 0.1
    lora_lr = 0.05
    model = MultiLayerNet().to("cuda")
    param_groups = [
        {'params': model.let_layer.parameters(), 'lr': let_lr, 'name': 'let'},
        {'params': model.lwc_layer.parameters(), 'lr': lwc_lr, 'name': 'lwc'},
        {'params': model.lora_layer.parameters(), 'lr': lora_lr, 'name': 'lora'}
    ]
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        warmup_start_lr=0,
        eta_min=1e-7
    )
    lrs_history = [[] for _ in range(len(param_groups))]
    names = [group['name'] for group in param_groups]
    
    loss = torch.nn.MSELoss()

    inputs = torch.randn(100, 100).to("cuda")
    labels = torch.randn(100, 100).to("cuda")
    losses = []

    print("Starting learning rate simulation...")
    for epoch in range(total_epochs):
        # 更新学习率
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        
        optimizer.step()
        scheduler.step()
        
        

        losses.append(loss_value.item())
        
        # 记录每组参数的当前学习率
        for i, group in enumerate(optimizer.param_groups):
            # if epoch > 8:
            #     lrs_history[i].append(group['lr'].item())
            # else:
            lrs_history[i].append(group['lr'])
            
        # 打印部分epoch的学习率
        if epoch % 9 == 0:
            print(f"\nEpoch {epoch}:")
            for name, lr in zip(names, [group['lr'] for group in optimizer.param_groups]):
                print(f"{name} LR: {lr:.2e}, loss: {loss_value.item():.2f}")
    
    # 绘制学习率变化曲线
    plot_multi_lr_schedule(lrs_history, warmup_epochs, total_epochs, names)
    print("\nLearning rate visualization saved to multi_lr_schedule.png")