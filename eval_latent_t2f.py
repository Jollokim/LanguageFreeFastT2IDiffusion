# MIT License

# Copyright (c) [2023] [Anima-Lab]

from argparse import ArgumentParser
import os
from collections import OrderedDict
from time import time
from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_fid.fid_score import calculate_fid_given_paths


from models.maskdit import Precond_models
from sample import generate_with_net_t2f
from utils import dist, mprint, get_ckpt_paths, Logger, sample, \
    ddp_sync, init_processes, cleanup, \
    str2bool, parse_str_none, parse_int_list, parse_float_none


# ------------------------------------------------------------
# Training Helper Function

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# ------------------------------------------------------------


def fid_fn(model, config, args, device):
    generate_with_net_t2f(
        args, 
        model, 
        device, 
        feat_path=config.eval.feat_path, 
        feat_dim=config.data.feat_dim, 
        norm_feature=config.data.norm_feature, 
        num_steps=args.num_steps,
        S_churn=args.S_churn, 
        solver=args.solver, 
        discretization=args.discretization,
        schedule=args.schedule, 
        scaling=args.scaling, 
        pretrained_path=args.pretrained_path, 
        ) 
    dist.barrier()
    fid = calculate_fid_given_paths([args.outdir, config.eval.ref_path], config.eval.batchsize, device, 2048, args.num_workers)
    mprint(f'guidance: {args.cfg_scale} FID: {fid}')
    dist.barrier()
    return fid


def eval_loop(args):
    config = OmegaConf.load(args.config)
    mprint('start evaluation...')
    size = args.global_size
    rank = args.global_rank
    print(f'global_rank: {rank}, global_size: {size}, local_rank: {args.local_rank}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    print(f'dist.get_world_size(): {dist.get_world_size()}, dist.get_rank(): {dist.get_rank()}')
    device = torch.device("cuda")

    seed = args.global_rank * args.num_workers + args.global_seed 
    torch.manual_seed(seed)

    # seeds for sampling
    args.seeds = [i for i in range(config.eval.fid_sample_size)]
    mprint(f'Sample size for FID: {len(args.seeds)}')

    # Select batch size per GPU
    num_accumulation_rounds = config.train.grad_accum
    micro_batch = config.train.batchsize
    batch_gpu_total = micro_batch * num_accumulation_rounds
    global_batch_size = batch_gpu_total * size
    mprint(f"Global batchsize: {global_batch_size},  batchsize per GPU: {batch_gpu_total}, micro_batch: {micro_batch}.")

    # Setup an experiment folder
    class_dropout_prob = config.model.class_dropout_prob
    model_name = config.model.model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    data_name = config.data.dataset

    # start a new exp path (and resume from the latest checkpoint if possible)
    if args.experiment_dir is None:
        cond_gen = 'conditional' if class_dropout_prob == 0 else f'clsdrop{class_dropout_prob}'
        exp_name = f'{model_name}-{config.model.precond}-{data_name}-{cond_gen}-m{config.model.mask_ratio}-de{int(config.model.use_decoder)}' \
                    f'-mae{config.model.mae_loss_coef}-bs-{global_batch_size}-lr{config.train.lr}-{config.log.tag}'
        experiment_dir = f"{args.results_dir}/{exp_name}"
    else:
        experiment_dir = args.experiment_dir
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    ckpt_start, ckpt_end = args.ckpt_id_start, args.ckpt_id_end
    ckpt_path_dict = get_ckpt_paths(checkpoint_dir, ckpt_start, ckpt_end)  # Search for all the checkpoints with id falling in [ckpt_start, ckpt_end]
    mprint(f"{len(ckpt_path_dict.keys())} checkpoints found in {checkpoint_dir}.")

    # create evaluation dir
    os.makedirs(f'{experiment_dir}/evaluation', exist_ok=True)

    if rank == 0:
        logger = Logger(file_name=f'{experiment_dir}/log_eval.txt', file_mode="a+", should_flush=True)

    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token,
        use_encoder_feat=config.model.self_cond,
    ).to(device)
    # Note that parameter initialization is done within the model constructor
    model.eval()
    mprint(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mprint(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    # model = torch.compile(model)
    # Load checkpoints

    for ckpt_id, ckpt_path in ckpt_path_dict.items():
        mprint(f'Loading ckpt {ckpt_id}...')
        args.outdir = os.path.join(experiment_dir, 'evaluation_fid', f'edm-steps{args.num_steps}-ckpt{ckpt_id}_cfg{args.cfg_scale}')
        os.makedirs(args.outdir, exist_ok=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['ema'])

        mprint('Sampling...')
        fid = fid_fn(model, config, args, device)
        mprint(f'ID {ckpt_id}, FID: {fid}')
        
        if rank == 0:
            with open(f'{experiment_dir}/evaluation/eval_cfg{args.cfg_scale}.txt', 'a') as f:
                f.write(f'ID: {ckpt_id}, FID: {fid}\n')


    cleanup()
    if rank == 0:
        logger.close()



if __name__ == '__main__':
    parser = ArgumentParser('training parameters')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='localhost', help='address for master')
    parser.add_argument('--master_port', type=str, default='6120', help='address for master')


    # training
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--train_global_batch", type=int, default=1024)
    parser.add_argument('--ckpt_id_start', type=int, default=None, help='Checkpoints with id falling in this interval will be evaluated')
    parser.add_argument('--ckpt_id_end', type=int, default=None, help='Checkpoints with id falling in this interval will be evaluated')

    # sampling
    parser.add_argument('--seeds', type=parse_int_list, default='0-49999', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=100, help='Maximum batch size per GPU during sampling, must be a factor of 50k if torch.compile is used')
    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--ref_path', type=str, default='/diffusion_ws/ViT_Diffusion/assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--fid_batch_size', type=int, default=128, help='Maximum batch size per GPU')


    parser.add_argument('--experiment_dir', type=str, default=None, help='Cusom directory for results')

    args = parser.parse_args()
    args.global_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    
    torch.backends.cudnn.benchmark = True
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            args.global_rank = rank + args.node_rank * args.num_process_per_node
            # print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, args.local_rank, args.global_rank))
            p = mp.Process(target=init_processes, args=(eval_loop, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('Single GPU run')
        assert args.global_size == 1 and args.local_rank == 0
        args.global_rank = 0
        init_processes(eval_loop, args)
