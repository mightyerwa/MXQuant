import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil

from datautils import BlockTrainDataset
from quantize.mx_llama_layer import MXLlamaDecoderLayer
from quantize.mx_linear import MXLinear
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils

from quantize.utils import set_quant_state, let_parameters, lwc_parameters, get_mx_parameters, \
    smooth_and_quant_temporary,clear_temp_variable, smooth_and_quant_inplace, trainable_parameters_num, \
    register_scales_and_zeros, mx_state_dict




import math
import copy
import time

def update_dataset(layer, dataset, dev, attention_mask, position_ids, position_embeddings):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev) # (batch_size, seqlen, hidden_dim) (2, 2048, 4096)

                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids, position_embeddings = position_embeddings)[0].to('cpu')
                dataset.update_data(index,new_data)

def mxquant(model, args, trainloader, valloader, act_scales, logger):

    logger.info("Starting MXQuant")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    
    model.config.use_cache = False

    # step 1: moving embedding layer and first norm to target device
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # because llama3 put rotary_emb to model layer, it output (cos, sin) tuple
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers = model.model.layers
    layers[0] = layers[0].to(dev)

    dtype = torch.float16

    # step 2: load dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                      model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,
                                      off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,
                                    off_load_to_disk=args.off_load_to_disk)

    # step 3: catch the input of first layer
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super(Catcher, self).__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inps, **kwargs):
            self.dataset.update_data(self.index, inps.squeeze(0).to('cpu')) # for each index (1, batch_size, seq_len, hidden_size)
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            self.position_embeddings = kwargs["position_embeddings"]
            raise ValueError

    # step 3.1 catch the input of training set
    layers[0] = Catcher(layers[0], fp_train_inps)
    iters = len(trainloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0], fp_val_inps)
    iters = len(valloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    position_embeddings = layers[0].position_embeddings
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
        # attention_mask shape is (2， 1， 2048， 2049) batch_size is 2
        # attention_mask_batch shape is (4, 1, 2048, 2049)
        # import pdb; pdb.set_trace()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path) # data total copy
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype,
                                             cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype,
                                           cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype,
                                             cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype,
                                           cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
        for index, data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index, data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)


    if args.resume:
        mx_parameters = torch.load(args.resume)
        mx_parameters_number = 42
    else:
        mx_parameters = {}
        mx_parameters_files = [f for f in os.listdir(args.output_dir) if f.startswith('mx_parameters') and f.endswith('.pth')]
        mx_parameters_number = len(mx_parameters_files) + 1


    # step 6: start training
    loss_func = nn.MSELoss()
    qlayer = [None, None]
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1 replace torch.nn.Linear layer with MXLinear for QAT
        layer = layers[block_index].to(dev)
        qlayer[block_index % 2] = MXLlamaDecoderLayer(config= model.config, org_layer = layer, s_bits = args.s_bits, e_bits_w = args.e_bits_w, e_bits_a = args.e_bits_a, group_size=args.group_size)
        qlayer[block_index % 2] = qlayer[block_index % 2].to(dev)

        # obtain output of full-precision model
        set_quant_state(qlayer[block_index % 2], weight_quant=False, act_quant=False)  # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            update_dataset(qlayer[block_index % 2], fp_train_inps, dev, attention_mask, position_ids, position_embeddings)
            update_dataset(qlayer[block_index % 2], fp_val_inps, dev, attention_mask, position_ids, position_embeddings)
        set_quant_state(qlayer[block_index % 2], weight_quant=False, act_quant=True)  # activate quantization

        is_llama = True
        if is_llama:
            use_shift = False
        qlayer[block_index % 2].register_parameter("qkt_smooth_scale", nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features, device=dev, dtype=dtype)))

        pairs = {
            "qkv": ["q_proj", "k_proj", "v_proj"],
            "out": ["o_proj"],
            "fc1": ["up_proj", "gate_proj"]
        }
        for key, values in pairs.items():
            weight_list = []
            act = None
            for name, module in qlayer[block_index % 2].named_modules():
                if isinstance(module, MXLinear):
                    if any(value in name for value in values):
                        if args.model_family == "llama":
                            layer_name_prefix = "model.layers"
                        else:
                            raise NotImplementedError
                        act = act_scales[f"{layer_name_prefix}.{block_index}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                        weight_list.append(module.weight.abs().max(dim=0)[0].clamp(min=1e-5))
            if weight_list == []:
                import pdb; pdb.set_trace()
            stacked_weights = torch.stack(weight_list, dim = 0)
            weight, _ = torch.max(stacked_weights, dim=0)
            if act is None:
                raise NotImplementedError
            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
            shift = torch.zeros_like(scale) # don't use shift at first
            qlayer[block_index % 2].register_parameter(f"{key}_smooth_scale", torch.nn.Parameter(scale))
            qlayer[block_index % 2].register_parameter(f"{key}_smooth_shift", torch.nn.Parameter(shift))

        # import pdb; pdb.set_trace()

        if args.resume:
            pass # TODO
            # qlayer.load_state_dict(mx_parameters[i], strict = False)

        if args.epochs > 0:
            if qlayer[(block_index + 1) % 2] is not None:
                qlayer[(block_index + 1) % 2].zero_grad()
                for param in get_mx_parameters(qlayer[(block_index + 1) % 2], use_shift=False):
                    param.detach_()
            with torch.no_grad():
                qlayer[block_index % 2].float()  # required for AMP training
                if qlayer[(block_index + 1) % 2] is not None:
                    qlayer[(block_index + 1) % 2].float()
                # create optimizer
                optimizer = torch.optim.AdamW([{"params": let_parameters(qlayer), "lr": args.let_lr},
                                               {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}], weight_decay=args.wd)

                # TODO
                total_training_iteration = args.epochs * args.train_size / args.batch_size
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.let_lr)
                scheduler_let = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.let_lr / 20)
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.lwc_lr)
                scheduler_lwc = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.lwc_lr / 20)

                loss_scaler = utils.NativeScalerWithGradNormCount() #TODO

            trainable_number = trainable_parameters_num(qlayer)
            for param in get_mx_parameters(qlayer, use_shift):
                param.requires_grad = True
            print(f"trainable parameter number: {trainable_number / 1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    with torch.cuda.amp.autocast():
                        smooth_and_quant_temporary(qlayer[block_index % 2], args, is_llama)
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        if block_index != 0:
                            # torch.autograd.set_detect_anomaly(True)
                            # input = input.detach()
                            intermediate_out = qlayer[(block_index + 1) % 2](input, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                            intermediate_out_detach = intermediate_out.detach()
                            quant_out = qlayer[block_index % 2](intermediate_out_detach, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                            # import pdb
                            # pdb.set_trace()
                        else:
                            quant_out = qlayer[block_index % 2](input, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                        loss = reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        import pdb
                        pdb.set_trace()
                    # print(optimizer)
                    scheduler_let.step()
                    scheduler_lwc.step()
                    optimizer.param_groups[0]['lr'] = scheduler_let.get_lr()[0]
                    optimizer.param_groups[1]['lr'] = scheduler_lwc.get_lr()[0]
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_mx_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                # loss_mean = torch.stack(loss_list).mean()
                # norm_mean = torch.stack(norm_list).mean()
                # logger.info(
                #     f"layer {block_index} iter {epoch} train_loss:{loss_mean} train_norm:{norm_mean} ")

                val_loss_list = []
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            if block_index != 0:
                                intermediate_out = qlayer[(block_index + 1) % 2](input, attention_mask=attention_mask,
                                                                                 position_ids=position_ids,
                                                                                 position_embeddings=position_embeddings)[0]
                                quant_out = qlayer[block_index % 2](intermediate_out, attention_mask=attention_mask,
                                                                    position_ids=position_ids,
                                                                    position_embeddings=position_embeddings)[0]

                            else:
                                quant_out = qlayer[block_index % 2](input, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list),
                                     64)  # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num - 1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean:.8f} val_loss:{val_loss_mean:.8f} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024 ** 2:.2f} time {time.time() - start_time:.2f} ")
                # if val_loss_mean < best_val_loss:
                #     best_val_loss = val_loss_mean
                # else:
                #     early_stop_flag += 1
                #     if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                #         break
            optimizer.zero_grad()
            if block_index != 0:
                clear_temp_variable(qlayer[(block_index + 1) % 2])
            del optimizer
        # FIXME

        # real smooth and quantization
        if block_index != 0:
            smooth_and_quant_inplace(qlayer[(block_index + 1) % 2], args, is_llama)

            qlayer[(block_index + 1) % 2].bfloat16()

            if args.epochs > 0:
                update_dataset(qlayer[(block_index + 1) % 2], quant_train_inps, dev, attention_mask, position_ids, position_embeddings)
                update_dataset(qlayer[(block_index + 1) % 2], quant_val_inps, dev, attention_mask, position_ids, position_embeddings)
                register_scales_and_zeros(qlayer[(block_index + 1) % 2])
                layers[block_index - 1] = qlayer[(block_index + 1) % 2].to("cpu")

                mx_parameters[block_index - 1] = mx_state_dict(qlayer[(block_index + 1) % 2])
                torch.save(mx_parameters, os.path.join(args.output_dir, f"mx_parameters_{mx_parameters_number}.pth"))
            else:
                register_scales_and_zeros(qlayer[(block_index + 1) % 2])
                layers[block_index - 1] = qlayer[(block_index + 1) % 2].to("cpu")

        del layer
        torch.cuda.empty_cache()

    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    import gc
    gc.collect()
    model.config.use_cache = use_cache
    return model
