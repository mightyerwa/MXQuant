from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from quantize.utils import set_quant_state

config = AutoConfig.from_pretrained("../weight/Llama-2-7b-hf", attn_implementation="eager")
model = AutoModelForCausalLM.from_pretrained("../weight/Llama-2-7b-hf", config = config)
tokenizer = AutoTokenizer.from_pretrained("../weight/Llama-2-7b-hf", use_fast = False, legacy = False)


import os
import torch

# act = torch.load("Llama-2-7b-hf.pt")
# print(act)

from quantize.mx_llama_layer import MXLlamaDecoderLayer

model.config.use_cache = False
config.use_cache = False

layers = model.model.layers
for i in range(len(layers)):
    qlayer = MXLlamaDecoderLayer(config, layers[i], 8, 8)
    set_quant_state(qlayer, weight_quant=False, act_quant=False)
    layers[i] = qlayer

# for module in model.modules():
#     print(module)

import argparse
import random
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='../weight/Llama-2-7b-hf', type=str)

parser.add_argument('--cache_dir', default='./cache', type=str)
parser.add_argument('--output_dir', default = './log/', type=str, help='output log dir')
parser.add_argument('--save_quant_dir', default=None, type=str, help='batch size for eval')

# evaluate parameters
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--ppl_seqlen', default = 2048, type=int, help = 'num of ppl test data samples')
parser.add_argument('--eval_ppl', action='store_true', help='evaluate ppl')
parser.add_argument("--eval_tasks", type=str, default="arc_easy,arc_challenge",
                    help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--max_memory", type=str, default="40GiB", help="The maximum memory of each GPU")
parser.add_argument("--net", type=str, default=None)

# quantize parameters
parser.add_argument("--resume_quant", type=str, default=None, help="quantization model store path")
parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                    choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
                    help="Where to extract calibration data from.")
parser.add_argument("--wbits", type=int, default=4, help="bits of weight")
parser.add_argument("--abits", type=int, default=4, help="bits of activation")

parser.add_argument('--train_size', default=256, type=int, help='num of train data samples')
parser.add_argument('--val_size', default=64, type=int, help='num of val data samples')
parser.add_argument("--training_seqlen", default=2048, type=int, help='seqlen of training data')

parser.add_argument("--epochs", default=10, type=int, help='num of epochs')
parser.add_argument("--batch_size", default=2, type=int, help='batch size')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from main_mx import evaluate
from pathlib import Path
import utils

# init logger
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
if args.cache_dir:
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
if args.save_quant_dir:
    Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
output_dir = Path(args.output_dir)
logger = utils.create_logger(output_dir)
evaluate(model, tokenizer, args, logger)




