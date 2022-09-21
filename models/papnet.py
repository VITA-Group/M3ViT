#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of PAD-Net.
    https://arxiv.org/abs/1805.04409
"""
from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Bottleneck
from models.layers import SEBlock, SABlock
from mmcv.cnn import build_norm_layer
from functools import partial

class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, p, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__() 
        self.tasks = tasks 
        layers = {}
        conv_out = {}
        
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels//4, downsample=downsample)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)


    def forward(self, x):
        out = {}
        
        for task in self.tasks:
            if isinstance(x,dict):
                out['features_%s' %(task)] = self.layers[task](x[task])
            else:
                out['features_%s' %(task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' %(task)])
        
        return out 


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out


class AffinityDiffusionModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, iterations=6, beta=0.05):
        super(AffinityDiffusionModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        print('self.tasks',self.tasks,'self.auxilary_tasks',self.auxilary_tasks)
        # self.self_attention = {}
        # self.alpha = {task:{len(self.auxilary_tasks)} for task in self.tasks}
        # self.alpha = {task:{t:nn.Parameter(torch.Tensor([1]), requires_grad=True) for t in self.auxilary_tasks} for task in self.tasks}
        self.beta=beta
        self.iteration = iterations
        self.alpha = {}
        for task in self.tasks:
            alpha = {}
            for t in self.auxilary_tasks:
                if t==task:
                    alpha[t]=nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
                    # alpha[t]=torch.Tensor([0.8])
                else:
                    alpha[t]=nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
                    # alpha[t]=torch.Tensor([0.2/(len(self.auxilary_tasks)-1)])
            self.alpha[task]=alpha
        print('init alpha', self.alpha)
        # self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        # for t in self.tasks:
        #     other_tasks = [a for a in self.auxilary_tasks if a != t]
        #     self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        # self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        init_shape = x['features_%s' %(self.auxilary_tasks[0])].shape
        device = x['features_%s' %(self.auxilary_tasks[0])].device
        for task in self.tasks:
            self.alpha[task] = {t:torch.exp(self.alpha[task][t]).to(device)/torch.sum(torch.Tensor([torch.exp(self.alpha[task][k]) for k in self.alpha[task].keys()])).to(device) for t in self.alpha[task].keys()}
        # print('during forward',self.alpha)
        PertaskAffinity = {a:F.softmax(torch.matmul(x['features_%s' %(a)].flatten(-2).transpose(2,1), x['features_%s' %(a)].flatten(-2)),dim=2) for a in self.auxilary_tasks}
        # torch.mal()
        Aggregated_PertaskAffinity={}
        for task in self.tasks:
            for t in self.auxilary_tasks:
                if task in Aggregated_PertaskAffinity:
                    # print('have task',task,Aggregated_PertaskAffinity.keys())
                    Aggregated_PertaskAffinity[task]=Aggregated_PertaskAffinity[task]+self.alpha[task][t]*PertaskAffinity[t]
                else:
                    # print('dont have task',task,Aggregated_PertaskAffinity.keys())
                    Aggregated_PertaskAffinity[task]=self.alpha[task][t]*PertaskAffinity[t]
            # Aggregated_PertaskAffinity[task]=torch.Tensor([self.alpha[task][t]*PertaskAffinity[t] for t in self.auxilary_tasks])
        out = {}
        for a in self.tasks:
            x['tranfeatures_%s' %(a)] = x['features_%s' %(a)].flatten(-2).transpose(2,1)
            for i in range(self.iteration):
                x['tranfeatures_%s' %(a)] = torch.matmul(Aggregated_PertaskAffinity[a],x['tranfeatures_%s' %(a)])
            out['aggregated_features_%s' %(a)] = (1-self.beta)*x['features_%s' %(a)].flatten(-2).transpose(2,1) + self.beta*x['tranfeatures_%s' %(a)]
            out['aggregated_features_%s' %(a)] = out['aggregated_features_%s' %(a)].transpose(2,1).reshape(init_shape)
            # x['features_%s' %(a)].flatten(-2).transpose(2,1) + self.bata*torch.matmul(Aggregated_PertaskAffinity[a],x['features_%s' %(a)].flatten(-2).transpose(2,1))
            # out['aggregated_features_%s' %(a)] = (1-self.beta)*x['features_%s' %(a)].flatten(-2).transpose(2,1) + self.bata*torch.matmul(Aggregated_PertaskAffinity[a],x['features_%s' %(a)].flatten(-2).transpose(2,1)) 
        
        
        # adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        # out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}
        return out

class PAPNet_vit(nn.Module):
    def __init__(self, p, backbone, embed_dim=1024, img_size=768, patch_size = 16, norm_layer=partial(nn.LayerNorm, eps=1e-6),\
        align_corners=False,norm_cfg = None):
        super(PAPNet_vit, self).__init__()     
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        # self.channels = backbone_channels
        self.norm_cfg = norm_cfg
        self.channels = embed_dim
        self.norm = norm_layer(embed_dim)
        self.patch_size = patch_size
        self.img_size = img_size 
        self.h = int(self.img_size[0]/self.patch_size)
        self.w = int(self.img_size[1]/self.patch_size)
          
        # Backbone
        self.backbone = backbone
        # self.conv_0 = nn.Conv2d(
        #         embed_dim, 384, kernel_size=3, stride=1, padding=1)
        # # self.conv_1 = nn.Conv2d(
        # #     270, 270, kernel_size=3, stride=1, padding=1)
        # _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 384)
        # # _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 270)

        layers0 = {}
        layers1 = {}
        layers2 = {}
        layers3 = {}
        layers4 = {}
        for task in self.auxilary_tasks:
            conv_0 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, embed_dim)
            layers0[task] = nn.Sequential(conv_0,syncbn_fc_0)

            conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)
            layers1[task] = nn.Sequential(conv_1,syncbn_fc_1)

            conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)
            layers2[task] = nn.Sequential(conv_2,syncbn_fc_2)

            conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)
            layers3[task] = nn.Sequential(conv_3,syncbn_fc_3)

            conv_4 = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], kernel_size=1, stride=1)
            # _, syncbn_fc_4 = build_norm_layer(self.norm_cfg, 256)
            layers4[task] = nn.Sequential(conv_4)


        self.layers0 = nn.ModuleDict(layers0)
        self.layers1 = nn.ModuleDict(layers1)
        self.layers2 = nn.ModuleDict(layers2)
        self.layers3 = nn.ModuleDict(layers3)
        self.layers4 = nn.ModuleDict(layers4)


        self.align_corners = align_corners
        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, 384)
        self.affinity_diffusion = AffinityDiffusionModule(self.tasks, self.auxilary_tasks)
        # Multi-modal distillation
        # self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)

        # Task-specific heads for final prediction
        # heads = {}
        # for task in self.tasks:
        #     bottleneck1 = Bottleneck(256, 256//4, downsample=None)
        #     bottleneck2 = Bottleneck(256, 256//4, downsample=None)
        #     conv_out_ = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
        #     heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        # self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = self.backbone(x) #[8, 270, 120, 160]
        x = x[-1]
        if x.dim() == 3:
            if x.shape[1] % 48 != 0:
                x = x[:, 1:]
            x = self.norm(x)
        if x.dim() == 3:
            n, hw, c = x.shape
            # h = w = int(math.sqrt(hw))
            x = x.transpose(1, 2).reshape(n, c, self.h, self.w)
        
        upscale = {}
        for task in self.auxilary_tasks:
            feature = self.layers0[task](x)
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(x.shape[-2]*2,x.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            upscale[task] = feature
        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(upscale)

        # dict_keys(['features_semseg', 'semseg', 'features_depth', 'depth']) 
        # #torch.Size([8, 256, 120, 160]) torch.Size([8, 40, 120, 160]) torch.Size([8, 256, 120, 160]) torch.Size([8, 1, 120, 160])
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]

        x = self.affinity_diffusion(x)
        
        for task in self.tasks:
            feature = self.layers1[task](x['aggregated_features_%s' %(task)])
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            
            feature = self.layers2[task](feature)
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            
            feature = self.layers3[task](feature)
            feature = F.relu(feature, inplace=True)
            feature = self.layers4[task](feature)
            # feature = F.relu(feature, inplace=True)
            out[task] = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)



        #torch.Size([8, 40, 120, 160]) torch.Size([8, 1, 120, 160]) torch.Size([8, 40, 480, 640]) torch.Size([8, 1, 480, 640])
        return out

