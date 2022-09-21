# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International
from collections import OrderedDict
import torch
import cv2
import numpy as np


import torch.nn.functional as F
import math
import logging
import warnings
import errno
import os
import sys
import re
import zipfile
from urllib.parse import urlparse  # noqa: F401
import torch.distributed as dist
def tens2image(tens):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    if im.ndim == 3:
        im = im.transpose((1, 2, 0))

    return im


def pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def fixed_resize(sample, resolution, flagval=None):
    """
    Fixed resize to
    resolution (tuple): resize image to size specified by tuple eg. (512, 512).
    resolution (int): bring smaller side to resolution eg. image of shape 321 x 481 -> 512 x 767
    """
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[int(np.argmax(sample.shape[:2]))] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def im_normalize(im, max_value=1):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value * (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def ind2sub(array_shape, inds):
    rows, cols = [], []
    for k in range(len(inds)):
        if inds[k] == 0:
            continue
        cols.append((inds[k].astype('int') // array_shape[1]))
        rows.append((inds[k].astype('int') % array_shape[1]))
    return rows, cols


HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_logger = logging.getLogger(__name__)


def load_state_dict_from_url(url, model_dir=None, file_name=None, check_hash=False, progress=True, map_location=None):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn(
            'TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # Directory already exists, ignore.
                pass
            else:
                # Unexpected OSError, re-raise.
                raise
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(
            filename).group(1) if check_hash else None
        torch.hub.download_url_to_file(
            url, cached_file, hash_prefix, progress=progress)
    if zipfile.is_zipfile(cached_file):
        state_dict = torch.load(
            cached_file, map_location=map_location)['model']
    else:
        state_dict = torch.load(cached_file, map_location=map_location)
    return state_dict

def load_pretrained_pos_emb(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, pos_embed_interp=False, num_patches=576, align_corners=False, img_h=None, img_w=None):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        state_dict = torch.load(cfg['pretrained_finetune'])
        print('load pre-trained weight from ' + cfg['pretrained_finetune'])
    else:
        state_dict = load_state_dict_from_url(
            cfg['url'], progress=False, map_location='cpu')
        # print('load pre-trained weight from imagenet21k')

    # if filter_fn is not None:
    #     state_dict = filter_fn(state_dict)
    pos_emb_state_dict = OrderedDict()
    for key, item in state_dict.items():
        if "pos_embed" in key:
            pos_emb_state_dict[key] = state_dict[key]
    
    if pos_embed_interp:
        # print('loaded pos_embed shape',pos_emb_state_dict['pos_embed'].shape)
        n, c, hw = pos_emb_state_dict['pos_embed'].transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_emb_state_dict['pos_embed'][:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)
        # print(pos_embed_weight.shape)
        if img_h is None:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                math.sqrt(num_patches)), mode='bilinear', align_corners=align_corners)
        else:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=(img_h, img_w), mode='bilinear', align_corners=align_corners)
        # print('after interpolation', pos_embed_weight.shape)
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        cls_token_weight = pos_emb_state_dict['pos_embed'][:, 0].unsqueeze(1)
        # print('cls_token_weight', cls_token_weight.shape)
        pos_emb_state_dict['pos_embed'] = torch.cat(
            (cls_token_weight, pos_embed_weight), dim=1)
    strict = False
    msg = model.load_state_dict(pos_emb_state_dict, strict=strict)
    print('=========pos emb is loaded from ================',cfg['url'])

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, pos_embed_interp=False, num_patches=576, align_corners=False, img_h=None, img_w=None):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        state_dict = torch.load(cfg['pretrained_finetune'])
        print('load pre-trained weight from ' + cfg['pretrained_finetune'])
    else:
        state_dict = load_state_dict_from_url(
            cfg['url'], progress=False, map_location='cpu')
        print('load pre-trained weight from imagenet21k')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info(
            'Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I == 3:
            _logger.warning(
                'Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            _logger.info(
                'Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[
                :, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if pos_embed_interp:
        print('loaded pos_embed shape',state_dict['pos_embed'].shape)
        n, c, hw = state_dict['pos_embed'].transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = state_dict['pos_embed'][:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)
        print(pos_embed_weight.shape)
        if img_h is None:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                math.sqrt(num_patches)), mode='bilinear', align_corners=align_corners)
        else:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=(img_h, img_w), mode='bilinear', align_corners=align_corners)
        print('after interpolation', pos_embed_weight.shape)
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        cls_token_weight = state_dict['pos_embed'][:, 0].unsqueeze(1)
        print('cls_token_weight', cls_token_weight.shape)
        state_dict['pos_embed'] = torch.cat(
            (cls_token_weight, pos_embed_weight), dim=1)
    check = False
    if check:
        for i in [1,3,5,7,9,11]:
            state_dict['blocks.%s.mlp.experts.htoh4.weight'%(str(i))] = torch.unsqueeze(state_dict['blocks.%s.mlp.fc1.weight'%(str(i))], 0)
            state_dict['blocks.%s.mlp.experts.htoh4.bias'%(str(i))] = torch.unsqueeze(state_dict['blocks.%s.mlp.fc1.bias'%(str(i))], 0)
            state_dict['blocks.%s.mlp.experts.h4toh.weight'%(str(i))]=torch.unsqueeze(state_dict['blocks.%s.mlp.fc2.weight'%(str(i))], 0)
            state_dict['blocks.%s.mlp.experts.h4toh.bias'%(str(i))]=torch.unsqueeze(state_dict['blocks.%s.mlp.fc2.bias'%(str(i))], 0)

            # state_dict['blocks.1.mlp.experts.htoh4.weight'] = state_dict['blocks.1.mlp.fc1.weight']
            # state_dict['blocks.1.mlp.experts.htoh4.bias'] = state_dict['blocks.1.mlp.fc1.bias']
            # state_dict['blocks.1.mlp.experts.h4toh.weight']=state_dict['blocks.1.mlp.fc2.weight']
            # state_dict['blocks.1.mlp.experts.h4toh.bias']=state_dict['blocks.1.mlp.fc2.bias']

    msg = model.load_state_dict(state_dict, strict=strict)
    print('============load model weights from============',cfg['url'], msg)

def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size