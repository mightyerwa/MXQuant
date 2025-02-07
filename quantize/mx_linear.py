import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformQuantizer


class MXLinear(nn.Module):
    """
    MXLinear Module Perform quantized linear operation
    """
    def __init__(self,
                 org_module: nn.Linear,
                 s_bits: int = 8,
                 e_bits_w: int = 6,
                 e_bits_a: int = 6,
                 group_size: int = None):
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

        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_temporary_parameter = False

        self.weight_quantizer = UniformQuantizer(s_bits = s_bits, e_bits = e_bits_w, group_size = group_size, weight = self.weight)
        self.act_quantizer = UniformQuantizer(s_bits = s_bits, e_bits = e_bits_a, group_size = group_size)

    def forward(self, input):

        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


