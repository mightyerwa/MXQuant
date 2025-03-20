import torch
import torch.nn as nn




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