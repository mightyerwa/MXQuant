import torch
import torch.nn as nn
import torch.nn.functional as F
from datautils import test_ppl
import math
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn

CLIPMIN = 1e-4


class AlighQuantizer(nn.Module):
    def __init__(self,
                 remain_bit: int = 12,  # 保留的尾数位数
                 group_size: int = 256,  # 分组大小
                 weight = None,
                 ori_dtype = torch.float16):
        super(AlighQuantizer, self).__init__()

        if ori_dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError("Only support float16 and bfloat16")
        
        self.remain_bit = remain_bit
        self.group_size = group_size
        self.ori_dtype = ori_dtype
        
        if weight is not None:
            self.quant_weight = True
            self.quant_activation = False
        else:
            self.quant_weight = False
            self.quant_activation = True

    def get_exponent(self, x):
        """提取浮点数的指数部分，不需要考虑偏移"""
        return torch.floor(torch.log2(torch.abs(x) + 1e-30))

    def align_and_truncate(self, x, max_exp):
        # 保存符号位
        signs = torch.sign(x)
        x_abs = torch.abs(x)
        
        # 获取1.xx格式的尾数 (移除指数)
        mantissa = x_abs / (2.0 ** self.get_exponent(x_abs))
        
        # 计算需要移位的位数
        shift_bits = max_exp.unsqueeze(-1) - self.get_exponent(x_abs)
        assert shift_bits.min() >= 0, "Shift bits should be non-negative"
        
        # 对尾数进行移位操作 (左移为正，右移为负)
        aligned_mantissa = mantissa * (2.0 ** (-shift_bits))
        
        # 计算需要截断的位数
        truncate_bits = self.remain_bit - 2  # -1是为了保留符号位
        
        scale = 2.0 ** truncate_bits
        aligned_mantissa = torch.floor(aligned_mantissa * scale) / scale
        
        # 恢复指数并加回符号位
        result = signs * aligned_mantissa * (2.0 ** (max_exp.unsqueeze(-1)))
        return result

    def forward(self, x: torch.Tensor):
            
        original_shape = x.shape
        
        # 重塑张量以便按group_size分组
        if self.quant_weight:  # 权重矩阵
            num_groups = original_shape[-1] // self.group_size
            x = x.view(*original_shape[:-1], num_groups, self.group_size)
        elif self.quant_activation:  # 激活值
            num_groups = original_shape[-1] // self.group_size
            x = x.view(*original_shape[:-1], num_groups, self.group_size)
        else:
            raise ValueError("Either weight or activation must be provided")   
        # 获取每组的最大指数
        exp = self.get_exponent(x)
        max_exp, _ = torch.max(exp, dim=-1)  # 在group_size维度上取最大值
        
        # 对齐和截断
        x = self.align_and_truncate(x, max_exp)
        
        # 恢复原始形状
        x = x.view(original_shape)
        
        return x

    def change_remain_bit(self, new_remain_bit: int):
        """动态修改保留的尾数位数"""
        self.remain_bit = new_remain_bit

class MXLinear(nn.Module):
    """
    MXLinear Module Perform quantized linear operation
    """
    def __init__(self,
                 org_module: nn.Linear,
                 w_remain_bit: int = 10,
                 a_remain_bit: int = 10,
                 input_type: str = 'float',
                 ):
        super(MXLinear, self).__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        weight_quantizer

        self.weight_quantizer = UniformQuantizer(s_bits = s_bits, e_bits = e_bits_w, group_size = group_size, weight = self.weight)
        self.act_quantizer = UniformQuantizer(s_bits = s_bits, e_bits = e_bits_a, group_size = group_size)
        
        # lora parameters
        self.lora_A = nn.Parameter(torch.empty(self.in_features, l_rank))
        self.lora_B = nn.Parameter(torch.empty(l_rank, self.out_features))
        self.merge_weight = merge_weight

        self.scaling = l_alpha / l_rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.kaiming_normal_(self.lora_A, mode='fan_in', nonlinearity='relu')
        # self.lora_A.data.mul_(0.8)  # 适配SwiGLU特性
        nn.init.zeros_(self.lora_B)

    def forward(self, input):
        pass
        # if self.use_temporary_parameter:
        #     weight = self.temp_weight
        #     bias = self.temp_bias
        # elif self.use_weight_quant:
        #     weight = self.weight_quantizer(self.weight)
        #     bias = self.bias
        # else:
        #     weight = self.weight
        #     bias = self.bias

        # if self.use_act_quant:
        #     input = self.act_quantizer(input)

        # out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # return out

@torch.no_grad()
def evaluate(model, tokenizer, args):
    model.config.use_cache = False
    results = {}

    if args.eval_ppl:
        datasets = ['wikitext2']
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        # import pdb; pdb.set_trace()
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            # num_fewshot=0,
            # task_manager=task_manager,
        )
        print(make_table(results))
        
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        print(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
    return results

def fp_cim_simulate(model):
    layers = model.model.layers
    for block_index in range(len(layers)):
        layer = layers[block_index]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                # Simulate the behavior of the MXLinear class
                # For example, you can set quantization states or modify weights here
                pass
    

def main():
    class Args:
        def __init__(self):
            self.model = '../weight/Llama-2-7b-hf'
            self.eval_ppl = True
            self.eval_batch_size = 16
            self.ppl_seqlen = 2048
            self.eval_tasks = "arc_easy,arc_challenge"
            # self.eval_tasks = "piqa,arc_easy,arc_challenge,hellaswag,winogrande"

    args = Args()
    config = AutoConfig.from_pretrained(args.model, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = False, legacy = False)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map = 'cuda', torch_dtype = torch.bfloat16)

    layers = model.model.layers

    evaluate(model, tokenizer, args)

if __name__ == "__main__":
    main()
