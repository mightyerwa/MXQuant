import torch
import torch.nn as nn

class MXRMSNorm(nn.Module):
    def __init__(self,
                 org_module: nn.Module,
                 eps: float = 1e-6,):
        """
        just for smoothquant
        :param org_module: rms_norm module
        :param eps: eps
        """
        super(MXRMSNorm, self).__init__()
        self.register_buffer('weight', org_module.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        # import pdb; pdb.set_trace()
        return (weight * hidden_states).to(input_dtype) if bias is None else (weight * hidden_states + bias).to(input_dtype)