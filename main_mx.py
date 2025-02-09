import os
import sys
import random
import numpy as np
import torch
from datautils import test_ppl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from accelerate import infer_auto_device_map, dispatch_model
from pathlib import Path
import utils
import time
from datautils import get_loaders
from quantize.mxquant import mxquant

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    model.config.use_cache = False
    # import pdb; pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
                                       no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        # datasets = ['wikitext2', 'c4']
        datasets = ['wikitext2']
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        # import pdb; pdb.set_trace()
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            # num_fewshot=0,
            # task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
    return results

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../weight/Llama-2-7b-hf', type=str)

    parser.add_argument('--cache_dir', default='./cache', type=str)
    parser.add_argument('--output_dir', default = './log/', type=str, help='output log dir')


    # evaluate parameters
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--ppl_seqlen', default = 2048, type=int, help = 'num of ppl test data samples')
    parser.add_argument('--eval_ppl', default = True, type = bool, help='evaluate ppl')
    parser.add_argument("--eval_tasks", type=str, default="arc_easy,arc_challenge",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="44GiB", help="The maximum memory of each GPU")
    parser.add_argument("--net", type=str, default=None)

    # quantize parameters
    parser.add_argument('--save_quant_dir', default='./quant_model', type=str, help='save quantized model')
    parser.add_argument("--load_quant_model", type=str, default=None, help="quantization model store path")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
                        help="Where to extract calibration data from.")

    parser.add_argument("--s_bits", type=int, default=6, help="bits of scale")
    parser.add_argument("--e_bits_a", type=int, default=6, help="bits of activation")
    parser.add_argument("--e_bits_w", type=int, default=4, help="bits of weight")

    parser.add_argument("--group_size", type = int, default = None, help = "group_size for activation and weight")

    parser.add_argument('--train_size', default=192, type=int, help='num of train data samples')
    parser.add_argument('--val_size', default=64, type=int, help='num of val data samples')
    parser.add_argument("--training_seqlen", default=2048, type=int, help='seqlen of training data')

    parser.add_argument("--epochs", default=30, type=int, help='num of epochs')
    parser.add_argument("--batch_size", default=4, type=int, help='batch size')
    parser.add_argument("--off_load_to_disk", action='store_true', help = "off load train_data to disk")
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--resume", default = None, type = str, help = "resume path")
    parser.add_argument("--alpha", default=0.7, type = float, help = "smoothquant alpha")
    parser.add_argument("--let_lr", type=float, default=1e-4)
    parser.add_argument("--lwc_lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir, args=args)
    logger.info(args)

    if args.net is None:
        args.net = args.model.split('/')[-1]
        args.model_family = args.net.split('-')[0].lower()  # Llama-2-7b-hf
        logger.info(f"net is None, setting as {args.net}")
        logger.info(f"model_family, setting as {args.model_family}")

    if None: # args.resume_quant TODO
        # model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.abits, args.group_size) TODO
        pass
    else:
        # default load fp model
        config = AutoConfig.from_pretrained(args.model, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast = False, legacy = False)
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map = 'cpu', torch_dtype = torch.float16)
    for param in model.parameters():
        param.requires_grad = False

    if args.e_bits_w < 16:
        logger.info("=== start quantization ===")
        tick =time.time()
        # load calibration dataset
        cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
        cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'

        if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
            trainloader = torch.load(cache_trainloader)
            logger.info(f"load trainloader from {cache_trainloader}")
            valloader = torch.load(cache_valloader)
            logger.info(f"load valloader from {cache_valloader}")
        else:
            trainloader, valloader = get_loaders(
                args.calib_dataset,
                tokenizer,
                args.train_size,
                args.val_size,
                seed=args.seed,
                seqlen=args.training_seqlen,
            )
            torch.save(trainloader, cache_trainloader)
            torch.save(valloader, cache_valloader)

        if args.act_scales == None:
            args.act_scales = f"./act_scales/{args.net}.pt"
        if args.act_shifts == None:
            args.act_shifts = f"./act_shifts/{args.net}.pt"

        act_scales = torch.load(args.act_scales)
        act_shifts = torch.load(args.act_shifts)

        mxquant(
            model,
            args,
            trainloader,
            valloader,
            act_scales, # TODO
            logger,
        )

    evaluate(model, tokenizer, args, logger)

    if args.epochs > 0:
        import pdb; pdb.set_trace()
        save_dir = os.path.join(args.save_quant_dir, f'{args.model_family}_sbits{args.s_bits}_ebits{args.e_bits}_epochs{args.epochs}_letlr{args.let_lr}_lwclr{args.lwc_lr}.pt')
        torch.save(model.state_dict(), save_dir)

if __name__ == "__main__":
    print(sys.argv)
    main()
