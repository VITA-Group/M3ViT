#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy
import torch
import torch.nn.functional as F
import math
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil

from .helpers import get_dist_info
from torch.utils.data import DataLoader
from .sampler import (
    DistributedGroupSampler,
    DistributedSampler,
    GroupSampler,
)
from models import vits_gate
from models.vits_gate import VisionTransformerMoCoWithGate
from utils.moe_utils import read_specific_group_experts

"""
    Model getters 
"""
def cvt_state_dict(state_dict, model, p, args, linear_keyword, moe_dir_mode=False, img_h=None, img_w=None):
    # rename moco pre-trained keys
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if not args.pos_emb_from_pretrained and k.startswith('module.pos_embed'):
            del state_dict[k]
        if k.startswith('module.') and (not k.startswith('module.%s' % linear_keyword)) and (not k.startswith('module.norm')):
            if "_aux" in k:
                # print("skip k is {}".format(k))
                continue
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    if args.task_one_hot and (not args.multi_gate) and (not args.regu_experts_fromtask):
        for k in list(state_dict.keys()):
            if 'mlp.gate.w_gate' in k:
                num_experts = state_dict[k].shape[-1]
                if args.gate_task_specific_dim<0:
                    state_dict[k] = torch.cat((state_dict[k],torch.zeros((args.num_tasks,num_experts))),axis=0)
                else:
                    state_dict[k] = torch.cat((state_dict[k],torch.zeros((args.gate_task_specific_dim,num_experts))),axis=0)

    if args.multi_gate:
        for k in list(state_dict.keys()):
            if 'mlp.gate.w_gate' in k:
                state_dict[k[:-len("w_gate")]+'0.'+'w_gate'] = state_dict[k]
                state_dict[k[:-len("w_gate")]+'1.'+'w_gate'] = state_dict[k]
                if args.num_tasks==4:
                    state_dict[k[:-len("w_gate")]+'2.'+'w_gate'] = state_dict[k]
                    state_dict[k[:-len("w_gate")]+'3.'+'w_gate'] = state_dict[k]
                if args.num_tasks==5:
                    state_dict[k[:-len("w_gate")]+'2.'+'w_gate'] = state_dict[k]
                    state_dict[k[:-len("w_gate")]+'3.'+'w_gate'] = state_dict[k]
                    state_dict[k[:-len("w_gate")]+'4.'+'w_gate'] = state_dict[k]
                del state_dict[k]
    # print('curretn state_dict', args.local_rank, state_dict.keys())
    # print('len of moes', args.local_rank,len(state_dict['blocks.1.mlp.experts.htoh4.weight']),args.moe_experts)
    if args.pos_emb_from_pretrained and p['backbone_kwargs']['pos_embed_interp']:
        n, c, hw = state_dict['pos_embed'].transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = state_dict['pos_embed'][:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)
        print(pos_embed_weight.shape)
        if img_w is None:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                img_h), mode='bilinear', align_corners=p['backbone_kwargs']['align_corners'])
        else:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=(img_h, img_w), mode='bilinear', align_corners=p['backbone_kwargs']['align_corners'])
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        cls_token_weight = state_dict['pos_embed'][:, 0].unsqueeze(1)

        state_dict['pos_embed'] = torch.cat(
            (cls_token_weight, pos_embed_weight), dim=1)
        print('=========pos emb is loaded from pretrained weights')

    if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed) and (not moe_dir_mode):
        print('=========will use read_specific_group_experts================')
        state_dict = read_specific_group_experts(state_dict, args.rank, args.moe_experts)
    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    print('=================unmatched keys:================',msg)

    return model


def cvt_state_dict_moe_gate(state_dict, model, p, args, linear_keyword):
    # rename moco pre-trained keys
    from collections import OrderedDict
    feature_state_dict = OrderedDict()
    gate_state_dict = OrderedDict()
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            feature_state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        if k.startswith('module.gate_model'):
            gate_state_dict[k[len("module.gate_model."):]] = state_dict[k]

    feature_state_dict = read_specific_group_experts(feature_state_dict, args.rank, args.moe_experts)

    args.start_epoch = 0
    model.vit_feature.load_state_dict(feature_state_dict, strict=True)
    model.vit_gate.load_state_dict(gate_state_dict, strict=True)

    return model

def get_backbone(p, args=None):
    """ Return the backbone """

    if p['backbone'] == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18(p['backbone_kwargs']['pretrained'])
        backbone_channels = 512
    
    elif p['backbone'] == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(p['backbone_kwargs']['pretrained'])
        backbone_channels = 2048

    elif p['backbone'] == 'hrnet_w18':
        from models.seg_hrnet import hrnet_w18
        backbone = hrnet_w18(p['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]
    
    elif p['backbone'] == 'mixture_inner_resnet_50':
        from models.resnet import mixture_inner_resnet_50
        backbone = mixture_inner_resnet_50(pretrained='',tasks=p.TASKS.NAMES,\
        input_dim=1000,init="uniform,-0.5,1",scale_type='relu',num_classes=1000,\
        dataset="imagenet",drop_ratio=0,drop_input=0,)
        print('backbone',backbone)
        backbone_channels = 2048
    
    elif p['backbone'] == 'VisionTransformer':
        from models.vit import VisionTransformer
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        bn_args = p['backbone_kwargs']
        backbone = VisionTransformer(model_name=bn_args['model_name'],\
            img_size=bn_args['img_size'], patch_size=bn_args['patch_size'], in_chans=bn_args['in_chans'], embed_dim=bn_args['embed_dim'], depth=bn_args['depth'],\
                num_heads=bn_args['num_heads'], num_classes=bn_args['num_classes'], mlp_ratio=bn_args['mlp_ratio'], qkv_bias=bn_args['qkv_bias'], qk_scale=None, drop_rate=bn_args['drop_rate'], attn_drop_rate=bn_args['attn_drop_rate'],\
                    drop_path_rate=bn_args['drop_path_rate'], hybrid_backbone=None,  norm_cfg=norm_cfg,\
                        pos_embed_interp=bn_args['pos_embed_interp'], random_init=bn_args['random_init'], align_corners=bn_args['align_corners'])
        backbone_channels = 2048
        linear_keyword = 'head'
    
    elif p['backbone'] == 'VisionTransformer_moe':
        from models.vision_transformer_moe import VisionTransformerMoE
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        bn_args = p['backbone_kwargs']
        if args.moe_data_distributed:
            moe_world_size = 1
        else:
            moe_world_size = torch.distributed.get_world_size()
            if args.moe_experts % moe_world_size != 0:
                print("experts number of {} is not divisible by world size of {}".format(args.moe_experts, moe_world_size))
            args.moe_experts = args.moe_experts // moe_world_size
        if args.moe_use_gate:
            gate_model = vits_gate.__dict__[args.moe_gate_arch](num_classes=0)
            backbone = VisionTransformerMoE(model_name=bn_args['model_name'],\
                img_size=bn_args['img_size'], patch_size=bn_args['patch_size'], in_chans=bn_args['in_chans'], embed_dim=bn_args['embed_dim'], depth=bn_args['depth'],\
                    num_heads=bn_args['num_heads'], num_classes=bn_args['num_classes'], mlp_ratio=bn_args['mlp_ratio'], qkv_bias=bn_args['qkv_bias'], qk_scale=None, representation_size=None, distilled=bn_args['distilled'],\
                        drop_rate=bn_args['drop_rate'], attn_drop_rate=bn_args['attn_drop_rate'], drop_path_rate=bn_args['drop_path_rate'], hybrid_backbone=None, norm_cfg=norm_cfg,\
                            pos_embed_interp=bn_args['pos_embed_interp'], random_init=bn_args['random_init'], align_corners=bn_args['align_corners'],\
                                act_layer=None, weight_init='', moe_mlp_ratio=bn_args['moe_mlp_ratio'], moe_experts=args.moe_experts, moe_top_k=bn_args['moe_top_k'], world_size=moe_world_size,\
                                    gate_return_decoupled_activation=bn_args['gate_return_decoupled_activation'],gate_dim=gate_model.num_features,)
            backbone = VisionTransformerMoCoWithGate(backbone, gate_model)
        else:
            
            backbone = VisionTransformerMoE(model_name=bn_args['model_name'],\
                img_size=bn_args['img_size'], patch_size=bn_args['patch_size'], in_chans=bn_args['in_chans'], embed_dim=bn_args['embed_dim'], depth=bn_args['depth'],\
                    num_heads=bn_args['num_heads'], num_classes=bn_args['num_classes'], mlp_ratio=bn_args['mlp_ratio'], qkv_bias=bn_args['qkv_bias'], qk_scale=None, representation_size=None, distilled=bn_args['distilled'],\
                        drop_rate=bn_args['drop_rate'], attn_drop_rate=bn_args['attn_drop_rate'], drop_path_rate=bn_args['drop_path_rate'], hybrid_backbone=None, norm_cfg=norm_cfg,\
                            pos_embed_interp=bn_args['pos_embed_interp'], random_init=bn_args['random_init'], align_corners=bn_args['align_corners'],\
                                act_layer=None, weight_init='', moe_mlp_ratio=bn_args['moe_mlp_ratio'], moe_experts=args.moe_experts, moe_top_k=bn_args['moe_top_k'], world_size=moe_world_size, gate_dim=bn_args['gate_dim'],\
                                    gate_return_decoupled_activation=bn_args['gate_return_decoupled_activation'],moe_gate_type=args.moe_gate_type, vmoe_noisy_std=args.vmoe_noisy_std,\
                                        gate_task_specific_dim=args.gate_task_specific_dim, multi_gate=args.multi_gate,
                                        regu_experts_fromtask = args.regu_experts_fromtask, num_experts_pertask = args.num_experts_pertask, num_tasks = args.num_tasks,
                                        gate_input_ahead = args.gate_input_ahead,regu_sem=args.regu_sem,sem_force=args.sem_force,regu_subimage=args.regu_subimage,expert_prune = args.expert_prune)
        linear_keyword = 'head'
        backbone_channels = 2048
    
    else:
        raise NotImplementedError
    print('backbone', backbone)
    if args is not None:
        if args.pretrained:
            if os.path.isfile(args.pretrained) or os.path.isdir(args.pretrained):
                print("=> loading checkpoint '{}'".format(args.pretrained))
                if os.path.isfile(args.pretrained):
                    checkpoint = torch.load(args.pretrained, map_location="cpu")
                    moe_dir_read = False
                elif os.path.isdir(args.pretrained):
                    checkpoint = torch.load(os.path.join(args.pretrained, "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
                    # len_save = min(len([f for f in os.listdir(args.pretrained) if "pth" in f]),int(args.moe_experts * torch.distributed.get_world_size()))
                    len_save = len([f for f in os.listdir(args.pretrained) if "pth" in f])
                    print('===========number of moe loaded from pretrain======',len_save)
                    assert len_save % torch.distributed.get_world_size() == 0
                    response_cnt = [i for i in range(torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                                                    (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
                    # merge all ckpts
                    for cnt, cnt_model in enumerate(response_cnt):
                        if cnt_model != 0:
                            checkpoint_specific = torch.load(os.path.join(args.pretrained, "{}.pth".format(cnt_model)), map_location="cpu")
                            if cnt != 0:
                                for key, item in checkpoint_specific["state_dict"].items():
                                    checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item], dim=0)
                            else:
                                checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                        moe_dir_read = True
                else:
                    raise ValueError("Model {} do not exist".format(args.pretrained))

                if "mae" in args.pretrained and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                    args.start_epoch = 0
                    msg = backbone.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                elif args.moe_use_gate:
                    state_dict = checkpoint['state_dict']
                    backbone = cvt_state_dict_moe_gate(state_dict, backbone, p, args, linear_keyword,backbone.h, backbone.w)
                else:
                    state_dict = checkpoint['state_dict']
                    backbone = cvt_state_dict(state_dict, backbone, p, args, linear_keyword, moe_dir_read, backbone.h, backbone.w)

                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.pretrained))

    if p['backbone_kwargs']['dilated']: # Add dilated convolutions
        assert(p['backbone'] in ['resnet18', 'resnet50'])
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    if 'fuse_hrnet' in p['backbone_kwargs'] and p['backbone_kwargs']['fuse_hrnet']: # Fuse the multi-scale HRNet features
        from models.seg_hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels


def get_priormodel(p):
    if p['prior_arch']=='shallow_embedding_imagenet':
        from models.resnet import shallow_embedding_imagenet
        prior_model = shallow_embedding_imagenet(pretrained='', num_classes=1000)
    else:
        raise NotImplementedError

def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    elif p['head'] == 'hrnet':
        from models.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    
    elif p['head'] == 'VisionTransformerUpHead':
        from models.vit_up_head import VisionTransformerUpHead
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        hd_args = p['head_kwargs']
        return VisionTransformerUpHead(img_size=hd_args['img_size'], patch_size=hd_args['patch_size'],embed_dim=hd_args['embed_dim'],\
            norm_cfg = norm_cfg, \
                num_conv=hd_args['num_conv'], upsampling_method=hd_args['upsampling_method'],\
                num_upsampe_layer=hd_args['num_upsampe_layer'], conv3x3_conv1x1=hd_args['conv3x3_conv1x1'],p=p,\
                    in_channels=hd_args['in_channels'],channels=hd_args['channels'],in_index=hd_args['in_index'],num_classes=p.TASKS.NUM_OUTPUT[task],\
                        align_corners=hd_args['align_corners'],)

    else:
        raise NotImplementedError


def get_model(p,args=None):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p,args=args)
    
    if p['setup'] == 'single_task':
        from models.models import SingleTaskModel
        task = p.TASKS.NAMES[0]
        head = get_head(p, backbone_channels, task)
        model = SingleTaskModel(backbone, head, task)


    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline':
            from models.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskModel(backbone, heads, p.TASKS.NAMES,p=p)
        
        elif p['model'] == 'mixture_baseline':
            from models.models import MultiTaskModel_Mixture
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskModel_Mixture(backbone, heads, p.TASKS.NAMES)


        elif p['model'] == 'cross_stitch':
            from models.models import SingleTaskModel
            from models.cross_stitch import CrossStitchNetwork
            
            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task, 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder
            
            # Stitch the single-task models together
            model = CrossStitchNetwork(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), 
                                        **p['model_kwargs']['cross_stitch_kwargs'])


        elif p['model'] == 'nddr_cnn':
            from models.models import SingleTaskModel
            from models.nddr_cnn import NDDRCNN
            
            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                model.load_state_dict(torch.load(os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task, 'best_model.pth.tar')))
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder
            
            # Stitch the single-task models together
            model = NDDRCNN(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict), 
                                        **p['model_kwargs']['nddr_cnn_kwargs'])


        elif p['model'] == 'mtan':
            from models.mtan import MTAN
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTAN(p, backbone, heads, **p['model_kwargs']['mtan_kwargs'])


        elif p['model'] == 'pad_net':
            from models.padnet import PADNet
            model = PADNet(p, backbone, backbone_channels)
        

        elif p['model'] == 'mti_net':
            from models.mti_net import MTINet
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTINet(p, backbone, backbone_channels, heads)

        elif p['model'] == 'padnet_vit':
            from models.padnet import PADNet_vit
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            hd_args = p['head_kwargs']
            model = PADNet_vit(p, backbone, embed_dim=hd_args['embed_dim'],img_size=hd_args['img_size'], patch_size=hd_args['patch_size'], align_corners=hd_args['align_corners'],norm_cfg = norm_cfg,)

        elif p['model'] == 'papnet_vit':
            from models.papnet import PAPNet_vit
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            hd_args = p['head_kwargs']
            model = PAPNet_vit(p, backbone, embed_dim=hd_args['embed_dim'],img_size=hd_args['img_size'], patch_size=hd_args['patch_size'], align_corners=hd_args['align_corners'],norm_cfg = norm_cfg,)

        elif p['model'] == 'jtrl':
            from models.Jtrl import JTRL
            norm_cfg = dict(type='SyncBN', requires_grad=True)
            hd_args = p['head_kwargs']
            tam = p['model_kwargs']['tam']
            model = JTRL(p, backbone, embed_dim=hd_args['embed_dim'],img_size=hd_args['img_size'], patch_size=hd_args['patch_size'], align_corners=hd_args['align_corners'],norm_cfg = norm_cfg,tam=tam)


        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))


    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))
    

    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms as tr

    # Training transformations
    if p['train_db_name'] == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    elif p['train_db_name'] == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    
    elif p['train_db_name'] == 'CityScapes':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))


    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    
    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(p, transforms):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                          do_edge='edge' in p.ALL_TASKS.NAMES,
                                          do_normals='normals' in p.ALL_TASKS.NAMES,
                                          do_sal='sal' in p.ALL_TASKS.NAMES,
                                          do_human_parts='human_parts' in p.ALL_TASKS.NAMES,
                                          overfit=p['overfit'])

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(split='train', transform=transforms,download=False, do_edge='edge' in p.ALL_TASKS.NAMES, 
                                    do_semseg='semseg' in p.ALL_TASKS.NAMES, 
                                    do_normals='normals' in p.ALL_TASKS.NAMES, 
                                    do_depth='depth' in p.ALL_TASKS.NAMES, overfit=p['overfit'])
    
    elif db_name == 'CityScapes':
        from data.cityscapes import CityScapes
        database = CityScapes(split='train', do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                        do_depth='depth' in p.ALL_TASKS.NAMES,)

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_train_dataloader(p, dataset):
    """ Return the train dataloader """

    trainloader = DataLoader(dataset, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate_mil)
    return trainloader


def get_val_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=p['overfit'])
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(split='val', transform=transforms, download=False, do_edge='edge' in p.TASKS.NAMES, 
                                do_semseg='semseg' in p.TASKS.NAMES, 
                                do_normals='normals' in p.TASKS.NAMES, 
                                do_depth='depth' in p.TASKS.NAMES, overfit=p['overfit'])

    elif db_name == 'CityScapes':
        from data.cityscapes import CityScapes
        database = CityScapes(split='val', do_semseg='semseg' in p.ALL_TASKS.NAMES,
                                        do_depth='depth' in p.ALL_TASKS.NAMES,)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(p, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'])
    return testloader

def build_train_dataloader(
    dataset, batch_size, workers_per_gpu, num_gpus=1, dist=True, **kwargs
):
    shuffle = kwargs.get("shuffle", False)
    if dist:
        rank, world_size = get_dist_info()
        # if shuffle:
        #     sampler = DistributedGroupSampler(dataset, batch_size, world_size, rank)
        # else:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
        num_workers = workers_per_gpu
    else:
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                             num_workers=workers_per_gpu, collate_fn=collate_mil)
        return trainloader

    # TODO change pin_memory
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_mil,
        # pin_memory=True,
        pin_memory=False,
    )

    return data_loader

def build_val_dataloader(
    dataset, batch_size, workers_per_gpu, num_gpus=1, dist=True, **kwargs
):
    shuffle = kwargs.get("shuffle", False)
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, batch_size, world_size, rank)
        else:
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
        num_workers = workers_per_gpu
    else:
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=workers_per_gpu)
        return testloader

    # TODO change pin_memory
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return data_loader

""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True, pos_weight=p['edge_w'])

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()

    elif task == 'normals':
        from losses.loss_functions import NormalsLoss
        criterion = NormalsLoss(normalize=True, size_average=True, norm=p['normloss'])

    elif task == 'sal':
        from losses.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        from losses.loss_functions import DepthLoss
        criterion = DepthLoss(p['depthloss'])

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    if p['setup'] == 'single_task':
        from losses.loss_schemes import SingleTaskLoss
        task = p.TASKS.NAMES[0]
        loss_ft = get_loss(p, task)
        return SingleTaskLoss(loss_ft, task)

    
    elif p['setup'] == 'multi_task':
        if p['loss_kwargs']['loss_scheme'] == 'baseline': # Fixed weights
            from losses.loss_schemes import MultiTaskLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MultiTaskLoss(p.TASKS.NAMES, loss_ft, loss_weights,p['multi_level'],p)


        elif p['loss_kwargs']['loss_scheme'] == 'pad_net': # Fixed weights but w/ deep supervision
            from losses.loss_schemes import PADNetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.ALL_TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return PADNetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights, p['multi_level'])

        elif p['loss_kwargs']['loss_scheme'] == 'jtrl': # Fixed weights but w/ deep supervision
            from losses.loss_schemes import JTRLLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.ALL_TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return JTRLLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights, p['multi_level'])
 

        elif p['loss_kwargs']['loss_scheme'] == 'mti_net': # Fixed weights but at multiple scales
            from losses.loss_schemes import MTINetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in set(p.ALL_TASKS.NAMES)})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MTINetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)

        
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model, args=None):
    """ Return optimizer for a given model and setup """

    if p['model'] == 'cross_stitch': # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    elif p['model'] == 'nddr_cnn': # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for nddr-cnns 
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    else: # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()
    
        if p['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        
        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))
    print('optimizer',optimizer)
    return optimizer
   

def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate_epoch(args, optimizer, epoch):
    if args.dataset == 'imagenet':
        blr = args.blr * (args.step_ratio ** (epoch // 30))
        glr = args.glr * (args.step_ratio ** (epoch // 30))
    else:
        if args.epoch_small <= epoch < args.epoch_big:
            glr = args.glr * (args.step_ratio ** 1)
            blr = args.blr * (args.step_ratio ** 1)
        elif epoch >= args.epoch_big:
            glr = args.glr * (args.step_ratio ** 2)
            blr = args.blr * (args.step_ratio ** 1)
        else:
            glr = args.glr
            blr = args.blr

    logging.info('Epoch [{}] Gate Learning rate: {}, Base Learning rate: {}'.
        format(epoch, glr, blr))
    for param_group in optimizer.param_groups:
        try:
            if param_group['name'] == 'gate':
                param_group['lr'] = glr
            else:
                param_group['lr'] = blr
        except:
            param_group['lr'] = blr

def get_mix_optimizer(p, model, prior_model):
    """ Return optimizer for a given model and setup """
    base_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]

    if p['fine_tune']:
        for p in gate_params:
            p.requires_grad = False
        for p in prior_model.parameters():
            p.requires_grad = False

    if not p['train_base']:
        opt_config = [
            {'params': base_params, 'lr': p['blr'], 'name': 'base'},
            {'params': prior_model.parameters(), 'lr': p['plr'], 'name': 'prior'},
            {'params': gate_params, 'name': 'gate'}
        ]

    if p['model'] == 'cross_stitch': # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    elif p['model'] == 'nddr_cnn': # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for nddr-cnns 
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    else: # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()
    
        if p['optimizer'] == 'sgd':
            # optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])
            optimizer = torch.optim.SGD(base_params if p['fine_tune'] or p['train_base'] else opt_config,
            p['blr'] if p['fine_tune'] or p['train_base'] else p['glr'],
            momentum=p['momentum'],
            weight_decay=p['weight_decay'])

        elif p['optimizer'] == 'adam':
            # optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
            optimizer = torch.optim.Adam(base_params if p['fine_tune'] or p['train_base'] else opt_config,
            lr = p['optimizer_kwargs']['lr'],
            weight_decay = p['optimizer_kwargs']['weight_decay'])
        
        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer