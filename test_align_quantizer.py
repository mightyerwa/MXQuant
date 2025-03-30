import torch
from fp_cim_eval import AlighQuantizer

def test_alignment():
    # 创建一些测试数据
    x = torch.tensor([
        [1.5, -0.125, 4.0, -0.25],  # 不同指数和符号的数字
        [-2.0, 0.0625, -3.0, 0.5]
    ], dtype=torch.float16).cuda()
    
    # 使用不同的remain_bit测试
    remain_bits = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for remain_bit in remain_bits:
        print(f"\nTesting with remain_bit = {remain_bit}")
        quantizer = AlighQuantizer(remain_bit=remain_bit, group_size=4)
        
        # 打印原始数据
        print("Original:")
        print(x)
        
        # 打印量化后的数据
        quantized = quantizer(x)
        print("Quantized:")
        print(quantized)
        
        # 检查符号是否保持一致
        signs_match = torch.sign(x) == torch.sign(quantized)
        print("Signs preserved:", signs_match.all().item())
        
        # 计算相对误差
        relative_error = torch.abs((x - quantized) / (torch.abs(x) + 1e-30))
        print("Max relative error:", relative_error.max().item())

if __name__ == "__main__":
    test_alignment()