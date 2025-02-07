import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders, get_wikitext2

try:
    from llava.model import * # required for llava
except ImportError:
    print("if want to quantize llava model, you should manually install llava from ....")

def get_act_scales(model, dataloader, num_samples = 128, alpha = 0.7):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.size(-1)
        tensor = tensor.view(-1, hidden_dim).abs().detach()  # (N, hidden_dim)
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()  # values tuple(values, indices)
        # torch.max(tensor, dim = 0) find max in dim0, so output value dim is (hidden_dim)
        # output tensor has 1 fewer dimension than input tensor
        # import pdb; pdb.set_trace()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name = name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device)) # for each nn.Linear layer find the max for every channel

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples = 128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}
    def stat_tensor(name, tensor):
        hidden_dim = tensor.size(-1)
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 * ((comming_max + comming_min) / 2)
        else:
            act_shifts[name] = (comming_max + comming_min) / 2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name = name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_shifts

def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../weight/Llama-2-7b-hf', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/', help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/', help='where to save the act shifts')
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "pile"], help="Where to extract calibration data from.", )
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataloader, _ = get_wikitext2(
        tokenizer = tokenizer,
        train_size = args.num_samples,
        val_size = 0,
        seed = args.seed,
        seqlen = args.seq_len,
    ) # dataloader output is list of (inp, tar)
    args.net = args.model.split('/')[-1]
    model_family = args.net.split('-')[0].lower()
    if model_family == "llama":
        alpha = 0.85
    else:
        raise Exception(f"{model_family} model family not supported")

    import pdb; pdb.set_trace()
    act_scales = get_act_scales(model, dataloader, num_samples = args.num_samples)
    # model is the class AutoModelForCausalLM
    save_path = os.path.join(args.scales_output_path, f'{args.net}_{args.calib_dataset}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(model, dataloader, num_samples = args.num_samples)
    save_path = os.path.join(args.shifts_output_path, f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)

if __name__ == '__main__':
    main()