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
            # print('feature,task',out['features_%s' %(task)].shape,out[task].shape)
        return out 

class TamModule(nn.Module):
    def __init__(self, p, tasks, auxilary_tasks, input_channels, norm_cfg = None):
        super(TamModule, self).__init__() 
        self.tasks = tasks 
        self.auxilary_tasks = auxilary_tasks
        self.norm_cfg = norm_cfg

        # layers = {}
        # conv_out = {}
        conv_0 = nn.Conv2d(len(auxilary_tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers0 = nn.Sequential(conv_0,syncbn_fc_0)
        conv_1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers1 = nn.Sequential(conv_1,syncbn_fc_1)

        conv_2 = nn.Conv2d(len(auxilary_tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_2 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers2 = nn.Sequential(conv_2,syncbn_fc_2)
        # conv_3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        # _, syncbn_fc_3 = build_norm_layer(self.norm_cfg, input_channels)
        # self.layers3 = nn.Sequential(conv_3,syncbn_fc_3)


        # encoders = {}
        encoder_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        _, syncbn_encoder_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.encoder0 = nn.Sequential(encoder_0,syncbn_encoder_0)
        encoder_1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
        _, syncbn_encoder_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.encoder1 = nn.Sequential(encoder_1,syncbn_encoder_1)

        decoder_0 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        _, syncbn_decoder_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.decoder0 = nn.Sequential(decoder_0,syncbn_decoder_0)
        decoder_1 = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1,output_padding=1)
        _, syncbn_decoder_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.decoder1 = nn.Sequential(decoder_1,syncbn_decoder_1)

        layers3 = {}
        layers4 = {}
        for task in self.tasks:
            conv_3 = nn.Conv2d(len(tasks)*input_channels, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)
            layers3[task] = nn.Sequential(conv_3,syncbn_fc_3)
            conv_4 = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], kernel_size=1, stride=1)
            layers4[task] = nn.Sequential(conv_4)

        self.layers3 = nn.ModuleDict(layers3)
        self.layers4 = nn.ModuleDict(layers4)

    def forward(self,deepfeature):
        batch,input_channels,H,W=deepfeature[self.auxilary_tasks[0]].shape
        featurelist = [deepfeature[t] for t in self.auxilary_tasks]
        featureinput = torch.stack(featurelist,dim=1).reshape(batch,len(self.auxilary_tasks)*input_channels,H,W).clone()
        featureinput = self.layers0(featureinput)
        featureinput = F.relu(featureinput, inplace=True)
        B = F.sigmoid(self.layers1(featureinput))
        Fb = torch.cat((deepfeature[self.tasks[0]]*B,deepfeature[self.tasks[1]]*(1-B)),dim=1)

        Fb = self.layers2(Fb)
        Fb = F.relu(Fb, inplace=True)
        # Fb = self.layers3(Fb)
        Fb = self.encoder0(Fb)
        Fb = F.relu(Fb, inplace=True)
        # print('after encoder0',Fb.shape)
        Fb = self.encoder1(Fb)
        Fb = F.relu(Fb, inplace=True)
        # print('after encoder1',Fb.shape)
        Fb = self.decoder0(Fb)
        Fb = F.relu(Fb, inplace=True)
        # print('after decoder0',Fb.shape)
        M = F.sigmoid(self.decoder1(Fb))
        # print('after decoder1',M.shape)
        Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M)),dim=1)
        out = {}
        for task in self.tasks:
            feature = self.layers3[task](Ftam)
            feature = F.relu(feature, inplace=True)
            out[task] = self.layers4[task](feature)
            # out[task] = F.interpolate(
            #     feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
        return out




class JTRL(nn.Module):
    def __init__(self, p, backbone, embed_dim=1024, img_size=768, patch_size = 16, norm_layer=partial(nn.LayerNorm, eps=1e-6),\
        align_corners=False,norm_cfg = None,tam=False):
        super(JTRL, self).__init__()  

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
        if 'multi_level' in p:
            self.multi_level = p['multi_level']
        else:
            self.multi_level = False
        
        self.tam = tam
        print('will use tam for jtrl',self.tam)
        self.inplace=True
        if self.multi_level:
            self.inplace=False
        print('will consider multi level output',self.multi_level)
        if self.tam:
            print('p',p,self.tasks, self.auxilary_tasks)
            self.tam_model = TamModule(p,self.tasks, self.auxilary_tasks,256,norm_cfg=norm_cfg)

        layers0 = {}
        layers1 = {}
        layers2 = {}
        layers3 = {}
        layers4 = {}
        for task in self.auxilary_tasks:
            conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)
            layers0[task] = nn.Sequential(conv_0,syncbn_fc_0)

        # for task in self.tasks:
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
        # self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, 384)
        
    def forward(self, x):
        # print('current in train mode',self.training)
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

        if self.tam:
            deepfeature = {}
        
        for task in self.auxilary_tasks:
            feature = self.layers0[task](x)
            feature = F.relu(feature, inplace=True)
            feature = F.interpolate(
                feature, size=(x.shape[-2]*2,x.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)

            feature = self.layers1[task](feature)
            # if self.multi_level and self.training:
            #     out['level1_%s'%(task)] = self.output_layers1[task](feature)
            feature = F.relu(feature, inplace=self.inplace)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            
            feature = self.layers2[task](feature)
            # if self.multi_level and self.training:
            #     out['level2_%s'%(task)] = self.output_layers2[task](feature)
            feature = F.relu(feature, inplace=self.inplace)
            feature = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)
            
            feature = self.layers3[task](feature)
            # if self.multi_level and self.training:
            #     out['level3_%s'%(task)] = self.output_layers3[task](feature)
            feature = F.relu(feature, inplace=self.inplace)

            if self.tam:
                deepfeature[task]=feature
                
            feature = self.layers4[task](feature)
            out[task] = F.interpolate(
                feature, size=(feature.shape[-2]*2,feature.shape[-1]*2), mode='bilinear', align_corners=self.align_corners)

        if self.tam:
            x = self.tam_model(deepfeature)
            for task in self.tasks:
                out['tam_%s' %(task)] = F.interpolate(x[task], size=img_size, mode='bilinear', align_corners=self.align_corners)
                # out['tam_%s' %(task)] = x[task]
        return out
         