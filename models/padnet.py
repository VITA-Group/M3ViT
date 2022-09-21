#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of PAD-Net.
    https://arxiv.org/abs/1805.04409
"""
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


class PADNet(nn.Module):
    def __init__(self, p, backbone, backbone_channels):
        super(PADNet, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.channels = backbone_channels

        # Backbone
        self.backbone = backbone
        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels)

        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(256, 256//4, downsample=None)
            bottleneck2 = Bottleneck(256, 256//4, downsample=None)
            conv_out_ = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)
    

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = self.backbone(x) #[8, 270, 120, 160]

        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)

        # dict_keys(['features_semseg', 'semseg', 'features_depth', 'depth']) 
        # #torch.Size([8, 256, 120, 160]) torch.Size([8, 40, 120, 160]) torch.Size([8, 256, 120, 160]) torch.Size([8, 1, 120, 160])
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]
 
        # Refine features through multi-modal distillation
        x = self.multi_modal_distillation(x)
        # dict_keys(['semseg', 'depth']) ([8, 256, 120, 160]) torch.Size([8, 256, 120, 160])
        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')

        return out


class PADNet_vit(nn.Module):
    def __init__(self, p, backbone, embed_dim=1024, img_size=768, patch_size = 16, norm_layer=partial(nn.LayerNorm, eps=1e-6),align_corners=False,norm_cfg = None):
        super(PADNet_vit, self).__init__()     
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
        layers1 = {}
        layers2 = {}
        for task in self.auxilary_tasks:
            conv_0 = nn.Conv2d(embed_dim, 270, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, 270)
            layers1[task] = nn.Sequential(conv_0,syncbn_fc_0)
            conv_1 = nn.Conv2d(270, 270, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_1 = build_norm_layer(self.norm_cfg, 270)
            layers2[task] = nn.Sequential(conv_1,syncbn_fc_1)
        self.layers1 = nn.ModuleDict(layers1)
        self.layers2 = nn.ModuleDict(layers2)
            # if embed_dim != intermediate_channels:
            #     downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
            #                                         stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            # else:
            #     downsample = None
            # bottleneck1 = Bottleneck(input_channels, intermediate_channels//4, downsample=downsample)
            # bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            # conv_out_ = nn.Conv2d(intermediate_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            # layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            # conv_out[task] = conv_out_


        # self.conv_0 = nn.Conv2d(
        #         embed_dim, 270, kernel_size=3, stride=1, padding=1)
        # self.conv_1 = nn.Conv2d(
        #     270, 270, kernel_size=3, stride=1, padding=1)
        # _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 270)
        # _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 270)

        self.align_corners = align_corners
        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, 270)

        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(256, 256//4, downsample=None)
            bottleneck2 = Bottleneck(256, 256//4, downsample=None)
            conv_out_ = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)

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
        

        # x = self.conv_0(x)
        # x = self.syncbn_fc_0(x)
        # x = F.relu(x, inplace=True)
        # x = F.interpolate(
        #     x, size=(x.shape[-2]*2,x.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)

        # x = self.conv_1(x)
        # x = self.syncbn_fc_1(x)
        # x = F.relu(x, inplace=True)
        # x = F.interpolate(
        #     x, size=(x.shape[-2]*2,x.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)

        upscale = {}
        for task in self.auxilary_tasks:
            feature = self.layers1[task](x)
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            
            feature = self.layers2[task](feature)
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            upscale[task] = feature
        # print('x',x.shape)
        # Initial predictions for every task including auxilary tasks
        # print(isinstance(upscale,dict))
        x = self.initial_task_prediction_heads(upscale)

        # dict_keys(['features_semseg', 'semseg', 'features_depth', 'depth']) 
        # #torch.Size([8, 256, 120, 160]) torch.Size([8, 40, 120, 160]) torch.Size([8, 256, 120, 160]) torch.Size([8, 1, 120, 160])
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]
 
        # Refine features through multi-modal distillation
        x = self.multi_modal_distillation(x)
        # dict_keys(['semseg', 'depth']) ([8, 256, 120, 160]) torch.Size([8, 256, 120, 160])
        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')
        #torch.Size([8, 40, 120, 160]) torch.Size([8, 1, 120, 160]) torch.Size([8, 40, 480, 640]) torch.Size([8, 1, 480, 640])
        return out
