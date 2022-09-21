#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
import numpy as np
from collections import Counter
import torch.nn.functional as F
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
import pickle
from os.path import join
from utils.moe_utils import collect_noisy_gating_loss,collect_semregu_loss, collect_regu_subimage_loss
def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES


    if p['model'] == 'mti_net': # Extra losses at multiple scales
        losses = {}
        for scale in range(4):
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    elif p['model'] == 'pad_net': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'padnet_vit': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'papnet_vit': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'jtrl': # Extra losses because of deepsupervision
        losses = {}
        if p['model_kwargs']['tam']:
            for task in tasks:
                losses['tam_%s' %(task)] = AverageMeter('Loss tam %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}

    if 'model_kwargs' in p:
        if 'tam' in p['model_kwargs']:
            if p['model_kwargs']['tam']:
                for task in tasks:
                    # losses['tam_%s' %(task)] = AverageMeter('Loss tam %s' %(task), ':.4e')
                    if 'tam_level0' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level0']:
                            losses['tam_level0_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(0, task), ':.4e')
                    else:
                        losses['tam_level0_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(0, task), ':.4e')
                    
                    if 'tam_level1' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level1']:
                            losses['tam_level1_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(1, task), ':.4e')
                    else:
                        losses['tam_level1_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(1, task), ':.4e')

                    if 'tam_level2' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level2']:
                            losses['tam_level2_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(2, task), ':.4e')
                    else:
                        losses['tam_level2_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(2, task), ':.4e')

    if p['multi_level']:
        for task in tasks:
            for i in range(1,4):
                losses['level%s_%s'%(i,task)] = AverageMeter('At level %s Loss %s' %(i,task), ':.4e')

    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses

def logits_aug(logits, low=0.01, high=9.99):
    batch_size = logits.size(0)
    temp = torch.autograd.Variable(torch.rand(batch_size, 1) * high + low).cuda()
    logits_temp = logits / temp
    return logits_temp

def adjust_epsilon_greedy(p, epoch):
    return 0.5 * max((1 - epoch/(p['epochs'] - p['left'])), 0)

def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def train_mixture_vanilla(p, train_loader, model,prior_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    prior_model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        
        # input_var = Variable(images)
        prior_out, overhead_flop = prior_model(images)
        if p['anneal']:
            prob = adjust_epsilon_greedy(p, epoch)
            print('epsilon greedy prob: {}'.format(prob))
        # output = model(images)
        if p['anneal']:
            # out, masks, costs, flop_percent = model(
            #     images, F.softmax(logits_aug(prior_out) if p['data_aug']
            #                          else prior_out, dim=-1), overhead_flop, prob)
            output = model(
                images, F.softmax(logits_aug(prior_out) if p['data_aug']
                                     else prior_out, dim=-1), overhead_flop, prob)
        else:
            # out, masks, costs, flop_percent = model(
            #     images, F.softmax(logits_aug(prior_out) if p['data_aug'] else
            #                          prior_out, dim=-1), overhead_flop)
            
            output = model(
                images, F.softmax(logits_aug(prior_out) if p['data_aug'] else
                                     prior_out, dim=-1), overhead_flop)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def train_vanilla_distributed(args, p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(args.local_rank, non_blocking=True)
        targets = {task: batch[task].cuda(args.local_rank, non_blocking=True) for task in p.ALL_TASKS.NAMES}
        
        if args.one_by_one:
            optimizer.zero_grad()
            id=0
            for single_task in p.TASKS.NAMES:
                if args.task_one_hot:
                    output = model(images,single_task=single_task, task_id = id)
                else:
                    output = model(images,single_task=single_task)
                id=id+1
                loss_dict = criterion(output, targets, single_task)

                for k, v in loss_dict.items():
                    losses[k].update(v.item())
                performance_meter.update({single_task: get_output(output[single_task], single_task)}, 
                                 {single_task: targets[single_task]})
                
                if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                    loss_dict['total'] += collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                # Backward
                loss_dict['total'].backward()
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                    model.allreduce_params()

            optimizer.step()
                
        else:
            # if (args.regu_sem or args.sem_force or args.regu_subimage) and epoch<args.warmup_epochs:
            #     output = model(images,sem=targets['semseg'])
            # else:
            output = model(images)
            
            
            # Measure loss and performance
            loss_dict = criterion(output, targets)
           
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                loss_dict['total'] += collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                # if args.regu_sem and epoch<args.warmup_epochs:
                #     semregu_loss = collect_semregu_loss(model, args.semregu_loss_weight)
                #     loss_dict['total'] += semregu_loss
                # if args.regu_subimage and epoch<args.warmup_epochs:
                #     regu_subimage_loss = collect_regu_subimage_loss(model, args.subimageregu_weight)
                #     loss_dict['total']+=regu_subimage_loss
            for k, v in loss_dict.items():
                losses[k].update(v.item())
            performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                    {t: targets[t] for t in p.TASKS.NAMES})
            # Backward
            optimizer.zero_grad()
            loss_dict['total'].backward()
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                model.allreduce_params()
            optimizer.step()
            
            
        if i % 25 == 0:
            progress.display(i)
            # for name, param in model.named_parameters():
            #     if 'gamma' in name:
            #         print('gamma',param)
            # if args.regu_sem and epoch<args.warmup_epochs:
            #     print('semregu_loss',semregu_loss)
            # if args.regu_subimage and epoch<args.warmup_epochs:
            #     print('regu_subimage_loss',regu_subimage_loss)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
