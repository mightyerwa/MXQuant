import torch
import torch.nn as nn
from collections import OrderedDict
from quantize.smoothquant import *
from quantize.mx_linear import MXLinear
from quantize.mx_matmul import MXMatMul

def load_quantized_model(model_path, wbits, abits, group_size):
    pass

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (MXLinear, MXMatMul)):
            m.set_quant_state(weight_quant, act_quant)

def get_mx_parameters(model, use_shift = True):
    params = []
    template = "smooth_scale"
    for n, p in model.named_parameters():
        if n.find(template) != -1 or n.find("bound_factor") != -1 or n.find("lora") != -1:
            params.append(p)
    return iter(params)

def let_parameters(model):

    params = []
    template = "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) != -1:
            params.append(m)
    return iter(params)

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("bound_factor") != -1:
            params.append(m)
    return iter(params)

def loraa_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("lora_A") != -1:
            params.append(m)
    return iter(params)

def lorab_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("lora_B") != -1:
            params.append(m)
    return iter(params)

def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total


def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, MXLinear):
            module.weight_quantizer.register_scales_and_zeros()

def mx_state_dict(model, destination = None, prefix: str = '', keep_vars = False) -> OrderedDict:

    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') != -1 or name.find('bound_factor') != -1 or name.find('lora') != -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)

def smooth_and_quant_temporary(model, args, isllama):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "smooth_scale" in name:
                param.data = truncate_number(param)

    if isllama:
        # layernorm and qkv matrix
        smooth_ln_fcs_temporary(model.input_layernorm, [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                model.qkv_smooth_scale, model.qkv_smooth_shift)
        # after attention layer, post_attn layernorm and mlp layer
        smooth_ln_fcs_temporary(model.post_attention_layernorm, [model.mlp.up_proj, model.mlp.gate_proj],
                                model.fc1_smooth_scale, model.fc1_smooth_shift)
        # output projection ?????
        smooth_fc_fc_temporary(model.self_attn.v_proj, model.self_attn.o_proj,
                               model.out_smooth_scale, model.out_smooth_shift)

        # smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)
        # last layer of mlp? don't use quantization (detail on paper)
        model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight

    else:
        smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj,model.self_attn.k_proj, model.self_attn.v_proj],
                                model.qkv_smooth_scale, model.qkv_smooth_shift)
        smooth_ln_fcs_temporary(model.final_layer_norm, [model.fc1],
                                model.fc1_smooth_scale, model.fc1_smooth_shift)
        smooth_ln_fcs_temporary(model.self_attn.v_proj, model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        # smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)
        model.fc2.temp_weight = model.fc2.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, MXLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.temp_weight + module.lora_B.t() @ module.lora_A.t() * module.scaling
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                import pdb
                pdb.set_trace()
                print(module)
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter = True

def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, MXLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()
def smooth_and_quant_inplace(model, args, isllama):
    for name, module in model.named_parameters():
        if "smooth_scale" in name:
            module.data = truncate_number(module)
    if isllama:
        smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                model.qkv_smooth_scale,model.qkv_smooth_shift)
        smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                model.fc1_smooth_scale,model.fc1_smooth_shift)
        smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                            model.out_smooth_scale, model.out_smooth_shift)
    else: # opt
        raise NotImplementedError
    # smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, MXLinear):
            module.weight = module.weight + module.lora_B.t() @ module.lora_A.t() * module.scaling
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False
