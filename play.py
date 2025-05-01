import torch
import torch.nn as nn
# import torch.optim as optim
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# config = AutoConfig.from_pretrained("../weight/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("../weight/Llama-2-7b-hf", config = config).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("../weight/Llama-2-7b-hf", use_fast = False, legacy = False)

# string = "Once upon a time in a distant galaxy,"

# inputs = tokenizer(string ,return_tensors = "pt").to("cuda")

# output = model.generate(
#     inputs["input_ids"],
#     max_length=200,  # 设置生成文本的最大长度
#     num_return_sequences=1,  # 生成的序列数量
#     temperature=0.7,  # 控制生成的随机性
#     top_k=50,  # 限制最高概率的词汇
#     top_p=0.9,  # nucleus sampling
#     do_sample=True,  # 启用采样
# )

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print("Input Text:", inputs)
# print("Generated Text:", generated_text)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

A = torch.tensor([0.1, 0.7, 1])
B = torch.tensor([0.1, 0.7, 1])

A = F.log_softmax(A)
print(A)
B = F.softmax(B)
print(B)
output = F.kl_div(A, B, reduction = 'batchmean')
print(output)