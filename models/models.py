#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
class TamModule(nn.Module):
    def __init__(self, p, tasks, input_channels, norm_cfg = None):
        super(TamModule, self).__init__() 
        self.tasks = tasks 
        self.norm_cfg = norm_cfg

        # layers = {}
        # conv_out = {}
        conv_0 = nn.Conv2d(len(tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_0 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers0 = nn.Sequential(conv_0,syncbn_fc_0)
        conv_1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        _, syncbn_fc_1 = build_norm_layer(self.norm_cfg, input_channels)
        self.layers1 = nn.Sequential(conv_1,syncbn_fc_1)

        conv_2 = nn.Conv2d(len(tasks)*input_channels, input_channels, kernel_size=3, stride=1, padding=1)
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
            conv_4 = nn.Conv2d(256, p.TASKS.NUM_OUTPUT[task], kernel_size=1, stride=1)
            layers4[task] = nn.Sequential(conv_4)

        self.layers3 = nn.ModuleDict(layers3)
        self.layers4 = nn.ModuleDict(layers4)

    def forward(self,deepfeature):
        batch,input_channels,H,W=deepfeature[self.tasks[0]].shape
        featurelist = [deepfeature[t] for t in self.tasks]
        featureinput = torch.stack(featurelist,dim=1).reshape(batch,len(self.tasks)*input_channels,H,W).clone()
        featureinput = self.layers0(featureinput)
        featureinput = F.relu(featureinput, inplace=True)
        B = F.sigmoid(self.layers1(featureinput))
        if len(self.tasks)==2:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B,deepfeature[self.tasks[1]]*(1-B)),dim=1)
        elif len(self.tasks)==3:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B,deepfeature[self.tasks[1]]*(1-B)/2,deepfeature[self.tasks[2]]*(1-B)/2),dim=1)
        elif len(self.tasks)==4:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B/2,deepfeature[self.tasks[1]]*B/2,deepfeature[self.tasks[2]]*(1-B)/2,deepfeature[self.tasks[3]]*(1-B)/2),dim=1)
        elif len(self.tasks)==5:
            Fb = torch.cat((deepfeature[self.tasks[0]]*B/2,deepfeature[self.tasks[1]]*B/2,deepfeature[self.tasks[2]]*(1-B)/3,deepfeature[self.tasks[3]]*(1-B)/3,deepfeature[self.tasks[4]]*(1-B)/3),dim=1)
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
        if len(self.tasks)==2:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M)),dim=1)
        elif len(self.tasks)==3:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M)),dim=1)
        elif len(self.tasks)==4:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M),deepfeature[self.tasks[3]]*(1+M)),dim=1)
        elif len(self.tasks)==5:
            Ftam = torch.cat((deepfeature[self.tasks[0]]*(1+M),deepfeature[self.tasks[1]]*(1+M),deepfeature[self.tasks[2]]*(1+M),deepfeature[self.tasks[3]]*(1+M),deepfeature[self.tasks[4]]*(1+M)),dim=1)
        out = {}
        for task in self.tasks:
            feature = self.layers3[task](Ftam)
            feature = F.relu(feature, inplace=True)
            out[task] = self.layers4[task](feature)
        return out


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list,p=None):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.tasks_id ={}
        id=0
        for task in self.tasks:
            self.tasks_id[task]=id
            id=id+1

        self.tam_level0 = False
        self.tam_level1 = False
        self.tam_level2 = False

        if 'model_kwargs' in p:
            self.tam = p['model_kwargs']['tam']
            self.tam_level0 = True
            self.tam_level1 = True
            self.tam_level2 = True
            if 'tam_level0' in p['model_kwargs']:
                self.tam_level0 = p['model_kwargs']['tam_level0']
            if 'tam_level1' in p['model_kwargs']:
                self.tam_level1 = p['model_kwargs']['tam_level1']
            if 'tam_level2' in p['model_kwargs']:
                self.tam_level2 = p['model_kwargs']['tam_level2']
        else:
            self.tam = False
        
        print('will consider tam in model',self.tam)
        if 'multi_level' in p:
            self.multi_level = p['multi_level']
        else:
            self.multi_level = False
        print('will consider multi level output in model',self.multi_level)

        if 'multi_gate' in p:
            self.multi_gate = p['multi_gate']
        else:
            self.multi_gate = False
        print('will consider multi gate output in model',self.multi_gate)
        if self.tam:
            if self.tam_level0:
                self.tam_model0 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
            if self.tam_level1:
                self.tam_model1 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
            if self.tam_level2:
                self.tam_model2 = TamModule(p,self.tasks, 256,norm_cfg = dict(type='SyncBN', requires_grad=True))
        
    def forward(self, x, single_task=None, task_id = None, sem=None):
        if task_id is not None:
            assert self.tasks_id[single_task]==task_id
        # print('input shape',x.shape)
        out_size = x.size()[2:]
        if not self.multi_gate:
            if task_id is None:
                if sem is None:
                    shared_representation = self.backbone(x)
                else:
                    shared_representation = self.backbone(x, sem=sem)
            else:
                if sem is None:
                    shared_representation = self.backbone(x, task_id=task_id)
                else:
                    shared_representation = self.backbone(x, task_id=task_id, sem=sem)
            # print('shared_representation',shared_representation.shape,out_size)
            if self.tam and self.training:
                if self.tam_level0:
                    deepfeature0 = {}
                if self.tam_level1:
                    deepfeature1 = {}
                if self.tam_level2:
                    deepfeature2 = {}
            out = {}
            if single_task is not None:
                return {single_task: F.interpolate(self.decoders[single_task](shared_representation), out_size, mode='bilinear')}
            
            for task in self.tasks:
                if self.tam and self.training:
                    out[task], feature0, feature1, feature2 = self.decoders[task](shared_representation)
                    if self.tam_level0:
                        deepfeature0[task] = feature0
                    if self.tam_level1:
                        deepfeature1[task] = feature1
                    if self.tam_level2:
                        deepfeature2[task] = feature2
                    out[task] = F.interpolate(out[task], out_size, mode='bilinear')
                else:  
                    out[task] = F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear')
            
            if self.tam and self.training:
                if self.tam_level0:
                    x = self.tam_model0(deepfeature0)
                    for task in self.tasks:
                        out['tam_level0_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level1:
                    x = self.tam_model1(deepfeature1)
                    for task in self.tasks:
                        out['tam_level1_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level2:
                    x = self.tam_model2(deepfeature2)
                    for task in self.tasks:
                        out['tam_level2_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
            return out
        else:
            out = {}
            if self.tam:
                if self.tam_level0:
                    deepfeature0 = {}
                if self.tam_level1:
                    deepfeature1 = {}
                if self.tam_level2:
                    deepfeature2 = {}
            
            for task in self.tasks:
                pertask_representation = self.backbone(x,task_id=self.tasks_id[task])
                if self.tam and self.training:
                    out[task], feature0, feature1, feature2 = self.decoders[task](pertask_representation)
                    if self.tam_level0:
                        deepfeature0[task] = feature0
                    if self.tam_level1:
                        deepfeature1[task] = feature1
                    if self.tam_level2:
                        deepfeature2[task] = feature2
                    
                    out[task] = F.interpolate(out[task], out_size, mode='bilinear')
                else:
                    out[task] = F.interpolate(self.decoders[task](pertask_representation), out_size, mode='bilinear')
                
            if self.tam and self.training:
                if self.tam_level0:
                    x = self.tam_model0(deepfeature0)
                    for task in self.tasks:
                        out['tam_level0_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level1:
                    x = self.tam_model1(deepfeature1)
                    for task in self.tasks:
                        out['tam_level1_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
                if self.tam_level2:
                    x = self.tam_model2(deepfeature2)
                    for task in self.tasks:
                        out['tam_level2_%s' %(task)] = F.interpolate(x[task], out_size, mode='bilinear', align_corners=False)
            return out


                


class MultiTaskModel_Mixture(nn.Module):
    """ Multi-task baseline model with mixture encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel_Mixture, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x, y, overhead=0, prob=0):
        out_size = x.size()[2:]
        shared_representation={task: self.backbone(x,y,task,overhead,prob) for task in self.tasks}
        return {task: F.interpolate(self.decoders[task](shared_representation[task]), out_size, mode='bilinear') for task in self.tasks}
