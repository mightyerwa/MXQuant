import torch
import torch.nn as nn

CLIPMIN = 1e-4

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x # TODO

def ceil_ste(x):
    return (x.ceil() - x).detach() + x

def get_shared_exp(x: torch.Tensor, s_bits):
    x = torch.log2(x)
    shared_exp = x - (s_bits - 1)
    shard_exp = ceil_ste(shared_exp)
    return shard_exp


class UniformQuantizer(nn.Module):
    def __init__(self,
                 s_bits: int = 8, # scale bits
                 e_bits: int = 8, # element bits
                 group_size = None,
                 weight = None,
                 disable_zero_point: bool = True,
                 ori_dtype = torch.float16,
                 ):
        super(UniformQuantizer, self).__init__()
        assert 2 <= s_bits <= 8, "scale_bits must be <= 8"
        assert 2 <= e_bits <= 8, "element_bits must be <= 8"

        self.s_bits = s_bits
        self.e_bits = e_bits
        self.group_size = group_size
        self.disable_zero_point = disable_zero_point
        if self.disable_zero_point:
            self.emax = 2 ** (e_bits - 1) - 1
            self.emin = -(2 ** (e_bits - 1))
        else:
            self.emax = 2 ** e_bits - 1
            self.emin = 0

        self.smax = 2 ** (s_bits - 1) - 1
        self.smin = -2 ** (s_bits - 1)

        if weight is not None:
            self.quant_weight = True
            self.quant_activation = False
            dim_out_feature = weight.shape[0]
            if group_size is not None:
                assert weight.shape[1] % group_size == 0, "in_feature of weight must be divisible by group_size"
                dim_out_feature = (weight.shape[1] // group_size) * dim_out_feature
            self.bound_factor = nn.Parameter(torch.ones((dim_out_feature, 1)) * 4)
            self.sigmoid = nn.Sigmoid()
        else:
            self.quant_weight = False
            self.quant_activation = True

        # shard_exp = get_shared_exp(self.s_bits, self.e_bits)

    def change_n_bits(self, s_bits: int, e_bits: int):
        self.s_bits = s_bits
        self.e_bits = e_bits
        if self.disable_zero_point:
            self.emax = 2 ** (e_bits - 1) - 1
            self.emin = -(2 ** (e_bits - 1))
        else:
            self.emax = 2 ** e_bits - 1
            self.emin = 0

        self.smax = 2 ** (s_bits-1) - 1
        self.smin = -2 ** (s_bits-1)

    def forward(self, x: torch.Tensor):
        return self.fake_quant(x)


    def fake_quant(self, x: torch.Tensor):
        dtype = x.dtype
        if self.group_size is not None:
            ori_shape = x.shape
            assert ori_shape[-1] % self.group_size == 0, "hidden_size or in_feature of weight must be divisible by group_size"
            x = x.view(-1, self.group_size)

        if self.quant_weight:
            if self.disable_zero_point:
                xmax = x.amax(dim=-1, keepdim=True)
                xmin = x.amin(dim=-1, keepdim=True)
                abs_max = torch.max(torch.abs(xmax), torch.abs(xmin))
                abs_max = self.sigmoid(self.bound_factor) * abs_max

                self.shard_exp = get_shared_exp(abs_max, self.s_bits)
                self.shard_exp = torch.clamp(self.shard_exp, self.smin, self.smax)

                # x.to(torch.float32)
                # self.shard_exp.to(torch.float32)

                x_int = round_ste(x * (2 ** (-self.shard_exp)))
                x_int = torch.clamp(x_int, self.emin, self.emax)
                x_dequant = x_int * (2 ** self.shard_exp)
                # import pdb; pdb.set_trace()
                # x.to(dtype)
                # self.shard_exp.to(dtype)
            else:
                # x_dequant = x # TODO
                raise NotImplementedError

        elif self.quant_activation:
            if self.disable_zero_point:
                xmax = x.amax(dim=-1, keepdim=True)
                xmin = x.amin(dim=-1, keepdim=True)
                abs_max = torch.max(torch.abs(xmax), torch.abs(xmin))
                self.shard_exp = get_shared_exp(abs_max, self.s_bits)
                self.shard_exp = torch.clamp(self.shard_exp, self.smin, self.smax)

                # x.to(torch.float32)
                # self.shard_exp.to(torch.float32)

                x_int = round_ste(x * (2 ** (-self.shard_exp)))
                x_int = torch.clamp(x_int, self.emin, self.emax)
                x_dequant = x_int * (2 ** self.shard_exp)
                # import pdb; pdb.set_trace()

                # x.to(torch.float32)
                # self.shard_exp.to(torch.float32)
            else:
                # x_dequant = x # TODO
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.group_size is not None:
            x_dequant = x_dequant.view(ori_shape)

        return x_dequant

    def register_scales_and_zeros(self):
        self.register_buffer('shard', self.shard_exp)
        del self.shard_exp







