import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformQuantizer

class MXMatMul(nn.Module):
    def __init__(self,
                 s_bits: int = 8,
                 e_bits: int = 8,
                 matmul_func = torch.matmul,
                 group_size: int = None):
        super(MXMatMul, self).__init__()
        # de-activation the quantized forward default
        self.use_act_quant = False
        # initialize quantizer

        self.x1_quantizer = UniformQuantizer(s_bits=s_bits, e_bits=e_bits, group_size=group_size)
        self.x2_quantizer = UniformQuantizer(s_bits=s_bits, e_bits=e_bits, group_size=group_size)
        self.matmul_func = matmul_func

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out
