#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    
    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, multi_level=False,p=None):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.multi_level = multi_level
        if self.multi_level:
            for key in list(self.loss_weights):
                self.loss_weights[key]=self.loss_weights[key]/4
            print('self.loss_weights',self.loss_weights)
        
        if p is not None: 
            if 'model_kwargs' in p:
                self.tam = p['model_kwargs']['tam']
            else:
                self.tam = False
    
    def forward(self, pred, gt,single_task=None):
        if 'tam_%s'%(self.tasks[0]) in pred: 
            total = 0.
            out = {}
            for task in self.tasks:
                pred_ = pred['tam_%s' %(task)]
                gt_ = gt[task]
                loss_ = self.loss_ft[task](pred_, gt_)
                out['tam_%s' %(task)] = loss_
                total += self.loss_weights[task] * loss_
                
            for task in self.tasks:
                pred_, gt_ = pred[task], gt[task]
                loss_ = self.loss_ft[task](pred_, gt_)
                out[task] = loss_
                total += self.loss_weights[task] * loss_

            out['total'] = total
            return out

        if self.tam:
            total = 0.
            out = {}
            if 'tam_level0_%s'%(self.tasks[0]) in pred: 
                for task in self.tasks:
                    pred_ = pred['tam_level0_%s' %(task)]
                    gt_ = gt[task]
                    loss_ = self.loss_ft[task](pred_, gt_)
                    if torch.any(torch.isnan(loss_)):
                        loss_ = torch.nan_to_num(loss_,nan=0.0)
                    out['tam_level0_%s' %(task)] = loss_
                    total += self.loss_weights[task] * loss_
            if 'tam_level1_%s'%(self.tasks[0]) in pred: 
                for task in self.tasks:
                    pred_ = pred['tam_level1_%s' %(task)]
                    gt_ = gt[task]
                    loss_ = self.loss_ft[task](pred_, gt_)
                    if torch.any(torch.isnan(loss_)):
                        loss_ = torch.nan_to_num(loss_,nan=0.0)
                    out['tam_level1_%s' %(task)] = loss_
                    total += self.loss_weights[task] * loss_

            if 'tam_level2_%s'%(self.tasks[0]) in pred: 
                for task in self.tasks:
                    pred_ = pred['tam_level2_%s' %(task)]
                    gt_ = gt[task]
                    loss_ = self.loss_ft[task](pred_, gt_)
                    if torch.any(torch.isnan(loss_)):
                        loss_ = torch.nan_to_num(loss_,nan=0.0)
                    out['tam_level2_%s' %(task)] = loss_
                    total += self.loss_weights[task] * loss_

            for task in self.tasks:
                pred_, gt_ = pred[task], gt[task]
                loss_ = self.loss_ft[task](pred_, gt_)
                if torch.any(torch.isnan(loss_)):
                    loss_ = torch.nan_to_num(loss_,nan=0.0)
                out[task] = loss_
                total += self.loss_weights[task] * loss_

            out['total'] = total
            return out

        if single_task is None:
            out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
            if 'human_parts' in out:
                if torch.any(torch.isnan(out['human_parts'])):
                    out['human_parts'] = torch.nan_to_num(out['human_parts'],nan=0.0)
            out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        else:
            out = {single_task: self.loss_ft[single_task](pred[single_task], gt[single_task])}
            out['total'] = self.loss_weights[single_task] * out[single_task]
        return out


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                    loss_weights: dict, multi_level=False):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.multi_level = multi_level
        if self.multi_level:
            for key in list(self.loss_weights):
                self.loss_weights[key]=self.loss_weights[key]/4
        print('self.loss_weights',self.loss_weights)
    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_ = F.interpolate(pred['initial_%s' %(task)], img_size, mode='bilinear')
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out['deepsup_%s' %(task)] = loss_
            total += self.loss_weights[task] * loss_
            
        # if self.multi_level:
        #     for task in self.tasks:
        #         for i in range(1,4):
        #             # print(pred['level%s_%s'%(i,task)].shape,task)
        #             pred_ = F.interpolate(pred['level%s_%s'%(i,task)], img_size, mode='bilinear')
        #             gt_ = gt[task]
        #             loss_ = self.loss_ft[task](pred_, gt_)
        #             out['level%s_%s'%(i,task)] = loss_
        #             total += self.loss_weights[task] * loss_
        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out

class JTRLLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                    loss_weights: dict, multi_level=False):
        super(JTRLLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.multi_level = multi_level
        if self.multi_level:
            for key in list(self.loss_weights):
                self.loss_weights[key]=self.loss_weights[key]/4
        print('self.loss_weights',self.loss_weights)
    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        if 'tam_%s'%(self.tasks[0]) in pred: 
            for task in self.tasks:
                pred_ = F.interpolate(pred['tam_%s' %(task)], img_size, mode='bilinear')
                gt_ = gt[task]
                loss_ = self.loss_ft[task](pred_, gt_)
                out['tam_%s' %(task)] = loss_
                total += self.loss_weights[task] * loss_
            
        # if self.multi_level:
        #     for task in self.tasks:
        #         for i in range(1,4):
        #             # print(pred['level%s_%s'%(i,task)].shape,task)
        #             pred_ = F.interpolate(pred['level%s_%s'%(i,task)], img_size, mode='bilinear')
        #             gt_ = gt[task]
        #             loss_ = self.loss_ft[task](pred_, gt_)
        #             out['level%s_%s'%(i,task)] = loss_
        #             total += self.loss_weights[task] * loss_
        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out



class MTINetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict, 
                    loss_weights: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup)
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)]
            pred_scale = {t: F.interpolate(pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            losses_scale = {t: self.loss_ft[t](pred_scale[t], gt[t]) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                out['scale_%d_%s' %(scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out
