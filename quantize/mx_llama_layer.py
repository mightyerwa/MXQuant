import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import LlamaConfig

from quantize.mx_linear import MXLinear
from quantize.mx_matmul import MXMatMul

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, LlamaRMSNorm, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig

from quantize.mx_rmsnorm import MXRMSNorm
from collections import OrderedDict

import math

from transformers.activations import ACT2FN
from quantize.smoothquant import *

class MXLlamaMLP(nn.Module):
    def __init__(self,
                 org_module: nn.Module,
                 hidden_act: str,
                 s_bits: int,
                 e_bits: int,
                 group_size: int,):
        super(MXLlamaMLP, self).__init__()
        self.gate_proj = MXLinear(org_module=org_module.gate_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.down_proj = MXLinear(org_module = org_module.down_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.up_proj = MXLinear(org_module=org_module.up_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MXLlamaAttention(nn.Module):
    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 s_bits: int,
                 e_bits: int,
                 group_size: int,):
        super(MXLlamaAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 4096 = 32*128
        self.num_heads = config.num_attention_heads  # 32
        self.head_dim = self.hidden_size // self.num_heads  # 128
        assert self.head_dim * self.num_heads == self.hidden_size, f"Hidden size must be divisible by num_heads, got hidden size {self.hidden_size}, num_heads {self.num_heads}"
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert self.num_key_value_heads * self.num_key_value_groups == self.num_heads, "Number of attention heads must divide num_key_value_heads"
        self.max_position_embeddings = config.max_position_embeddings  # 4096 default

        self.q_proj = MXLinear(org_module=org_module.q_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.k_proj = MXLinear(org_module=org_module.k_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.v_proj = MXLinear(org_module= org_module.v_proj, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.o_proj = MXLinear(org_module = org_module.o_proj,s_bits = s_bits, e_bits = e_bits, group_size = group_size)

        self.qkt_matmul = MXMatMul(s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.pv_matmul = MXMatMul(s_bits = s_bits, e_bits = e_bits, group_size = group_size)

        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # (bsz, token, embed)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.size(-2)
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2] # ?????????????
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)) # why make attn_weights min is min dtype

        # upcast attention to fp 32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant:bool = False, act_quant:bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (MXMatMul, MXLinear)):
                m.set_quant_state(weight_quant, act_quant)

class MXLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 org_layer,
                 s_bits:int,
                 e_bits:int,
                 group_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size  # 4096 128*32
        self.self_attn = MXLlamaAttention(org_module = org_layer.self_attn, config = config, s_bits = s_bits, e_bits = e_bits, group_size = group_size)
        self.mlp = MXLlamaMLP(org_module = org_layer.mlp, hidden_act=config.hidden_act, s_bits = s_bits, e_bits = e_bits, group_size = group_size)

        self.input_layernorm = MXRMSNorm(org_layer.input_layernorm, eps = org_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = MXRMSNorm(org_layer.post_attention_layernorm,eps=org_layer.post_attention_layernorm.variance_epsilon)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position : Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # print(f"cache_position test: {cache_position}")
        # print(f"cache_position.shape: {cache_position.shape}")
        # import sys;
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_embeddings = position_embeddings,
            position_ids = position_ids,
            cache_position = cache_position,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, ) # mean that output is a tuple, if don't has ',', output will be tensor

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for name, m in self.named_modules():
            if isinstance(m, (MXMatMul, MXLinear)):
                m.set_quant_state(weight_quant, act_quant)


    def smooth_and_quant_temporary(self):
        with torch.no_grad():
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)

        smooth_ln_fcs_temporary(self.input_layernorm,
                                [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                self.qkv_smooth_scale, self.qkv_smooth_shift)
        smooth_ln_fcs_temporary(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                                self.fc1_smooth_scale, self.fc1_smooth_shift)
        smooth_fc_fc_temporary(self.self_attn.v_proj, self.self_attn.o_proj,
                               self.out_smooth_scale, self.out_smooth_shift)
        smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                             self.qkt_smooth_scale)
        self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight

        # quant
        for name, module in self.named_modules():
            if isinstance(module, MXLinear):
                if hasattr(module, 'temp_weight'):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, 'temp_bias'):
                    module.temp_bias = module.bias
                module.use_temporary_parameter = True

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, MXLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        for name, module in self.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        smooth_ln_fcs_inplace(self.input_layernorm,
                              [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                              self.qkv_smooth_scale, self.qkv_smooth_shift)
        smooth_ln_fcs_inplace(self.post_attention_layernorm, [self.mlp.up_proj, self.mlp.gate_proj],
                              self.fc1_smooth_scale, self.fc1_smooth_shift)
        smooth_fc_fc_inplace(self.self_attn.v_proj, self.self_attn.o_proj,
                             self.out_smooth_scale, self.out_smooth_shift)
        smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                           self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, MXLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

    # def let_parameters(self, use_shift=True):
    #     params = []
    #     template = "smooth" if use_shift else "smooth_scale"
    #     for n, m in self.named_parameters():
    #         if n.find(template) > -1:
    #             params.append(m)
    #     return iter(params)
    #
    # def lwc_parameters(self):
    #     params = []
    #     for n, m in self.named_parameters():
    #         if n.find('bound_factor') > -1:
    #             params.append(m)
    #     return iter(params)

    def mx_parameters(self, use_shift=False):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, parameter in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(parameter)
        return iter(params)

    def mx_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, MXLinear):
                module.weight_quantizer.register_scales_and_zeros()

