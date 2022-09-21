#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from train.train_utils import train_vanilla,train_vanilla_distributed
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results,validate_results_v2
from termcolor import colored

import torch.distributed as dist
import subprocess
import random
from utils.custom_collate import collate_mil
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_config import build_train_dataloader,build_val_dataloader

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--trBatch', default=None, type=int, help='moe experts number')
parser.add_argument('--valBatch', default=None, type=int, help='moe experts number')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
parser.add_argument("--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
parser.add_argument('--one_by_one',default=False, type=str2bool, help='path to moe pretrained checkpoint')

parser.add_argument('--regu_experts_fromtask',default=False, type=str2bool, help='path to moe pretrained checkpoint')

args = parser.parse_args()
print('os.environ["LOCAL_RANK"]',os.environ["LOCAL_RANK"],args.local_rank)
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    # print(os.environ["LOCAL_RANK"])


def main():
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp, local_rank=args.local_rank)
    # print(os.environ["WORLD_SIZE"])
    if args.trBatch is not None:
        p['trBatch'] = args.trBatch
    if args.valBatch is not None:
        p['valBatch'] = args.valBatch
    distributed = False
    if args.local_rank >=0:
        distributed = True
        print(os.environ["WORLD_SIZE"])
        print('args.local_rank',args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"])
    # if "WORLD_SIZE" in os.environ:
    #     distributed = int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            p['local_rank'] = args.local_rank
        elif args.launcher == "slurm":
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            p['gpus'] = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                # 29500 is torch.distributed default port
                os.environ["MASTER_PORT"] = "29501"
            # use MASTER_ADDR in the environment variable if it already exists
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            p['local_rank'] = int(os.environ["LOCAL_RANK"])

        p['gpus'] = dist.get_world_size()
    else:
        p['local_rank'] = args.local_rank 
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'),local_rank=args.local_rank)
    print(colored(p, 'red'))
    print("Distributed training: {}".format(distributed))
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)


    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    # print('model',model)
    if distributed:
        model = DistributedDataParallel(
            model.cuda(args.local_rank),
            device_ids=[args.local_rank],
            # output_device=args.local_rank,
            # # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    #### old version
    # train_dataloader = get_train_dataloader(p, train_dataset)
    # val_dataloader = get_val_dataloader(p, val_dataset)

    #### new version 1
    # train_sample = DistributedSampler(train_dataset, shuffle=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=p['trBatch'], shuffle=(train_sample is None), \
    #     drop_last=True, num_workers=p['nworkers'], collate_fn=collate_mil)
    # val_sample = DistributedSampler(val_dataset, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=p['valBatch'], shuffle=(val_sample is None), \
    #     drop_last=True, num_workers=p['nworkers'])

    #### new version 2
    train_dataloader = build_train_dataloader(
        train_dataset, p['trBatch'], p['nworkers'], dist=distributed, shuffle=True)
    val_dataloader = build_val_dataloader(
        val_dataset, p['valBatch'], p['nworkers'], dist=distributed)

    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0

        #### don't do it during debug
        save_model_predictions(p, val_dataloader, model)
        if distributed:
            torch.distributed.barrier()
        best_result = eval_all_results(p)
    
    
    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train_vanilla_distributed(args, p, train_dataloader, model, criterion, optimizer, epoch)

        # Evaluate
            # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(p, val_dataloader, model)
            if distributed:
                torch.distributed.barrier()
            curr_result = eval_all_results(p)
            # improves, best_result = validate_results_v2(p, curr_result, best_result)
            improves, best_result = validate_results(p, curr_result, best_result)
            if improves:
                if args.local_rank==0:
                    print('Save new best model')
                    # torch.save(model.state_dict(), p['best_model'])
                    torch.save({'model':model.state_dict()},p['best_model'])

            # Checkpoint
            print('Checkpoint ...')
            if args.local_rank==0:
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                            'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])

    # Evaluate best model at the end
    if args.local_rank==0:
        print(colored('Evaluating best model at the end', 'blue'))
        model.load_state_dict(torch.load(p['best_model'])['model'])
        save_model_predictions(p, val_dataloader, model)
        # if distributed:
        #     torch.distributed.barrier()
        eval_stats = eval_all_results(p)


if __name__ == "__main__":
    main()