from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
from utils.mypath import MyPath
import cv2
class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_

class RandomScaleCropCityScapes(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        # print('before reshape',img.shape,label.shape,depth.shape,height, width,img[i:i + h, j:j + w].shape)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        # print('after reshape',img_.shape,label_.shape,depth_.shape)
        return img_, label_, depth_ / sc

class RandomScaleCropCityScapes_np(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth):
        height, width = img.shape[:2]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        # img_ = np.squeeze(cv2.resize(img[None, :, i:i + h, j:j + w], (height, width), interpolation=cv2.INTER_LINEAR), axis=0)
        # label_ = np.squeeze(np.squeeze(cv2.resize(label[None, None, i:i + h, j:j + w], (height, width), interpolation=cv2.INTER_NEAREST),axis=0),axis=0)
        # depth_ = np.squeeze(cv2.resize(depth[None, :, i:i + h, j:j + w],(height, width), interpolation=cv2.INTER_NEAREST),axis=0)
        # print('before reshape',img.shape,label.shape,depth.shape,height, width,img[i:i + h, j:j + w].shape)
        img_ = cv2.resize(img[i:i + h, j:j + w], img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        label_ = cv2.resize(label[i:i + h, j:j + w], img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        depth_ = cv2.resize(depth[i:i + h, j:j + w],img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        # print('after reshape',img_.shape,label_.shape,depth_.shape)
        # img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        # label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        # depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        return img_, label_, depth_ / sc

class CityScapes(Dataset):
    def __init__(self, 
                 root=MyPath.db_root_dir('cityscapes'),
                 split='val',
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 unsqueeze=True,
                 squeeze=False):
                #  train=True, augmentation=False):
        # self.train = train
        self.root = os.path.expanduser(root)
        # self.augmentation = augmentation
        self.split = split
        # read the data file
        self.data_path = root + '/'+ self.split
        # if train:
        #     self.data_path = root + '/train'
        # else:
        #     self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

        if self.split == 'train':
            self.augmentation = True
        else:
            self.augmentation = False
        print('will do augmentation', self.augmentation)
        self.do_edge = do_edge
        self.do_semseg = do_semseg
        self.do_normals = do_normals
        self.do_depth = do_depth
        self.unsqueeze = unsqueeze
        self.squeeze = squeeze

    def __getitem__(self, index):
        # print(index)
        sample = {}
        # # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label_7/{:d}.npy'.format(index)))
        semantic[semantic==-1]=255
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        if not self.squeeze:
            depth[depth == 0] = 255.
        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth = RandomScaleCropCityScapes()(image, semantic, depth)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
        
        sample['image'] = image.float()
        if self.do_semseg:
            sample['semseg']=semantic.float()
            if self.unsqueeze:
                sample['semseg']=torch.unsqueeze(semantic.float(),0)
        if self.do_depth:
            sample['depth']=depth.float()
        
        sample['meta'] = {'image': str(index),'im_size': (image.shape[1], image.shape[2])}
        if not self.unsqueeze:
            sample['image'] = sample['image'].numpy()
            if self.do_semseg:
                sample['semseg'] = sample['semseg'].numpy()
        if self.squeeze:
            sample['image'] = sample['image'].numpy()
            if self.do_depth:
                sample['depth'] = torch.squeeze(sample['depth'],0).numpy()
        return sample

    def __len__(self):
        return self.data_len

# class CityScapes(Dataset):
#     def __init__(self, 
#                  root=MyPath.db_root_dir('cityscapes'),
#                  split='val',
#                  do_edge=False,
#                  do_semseg=False,
#                  do_normals=False,
#                  do_depth=False,):
#                 #  train=True, augmentation=False):
#         # self.train = train
#         self.root = os.path.expanduser(root)
#         # self.augmentation = augmentation
#         self.split = split
#         # read the data file
#         self.data_path = root + '/'+ self.split
#         # if train:
#         #     self.data_path = root + '/train'
#         # else:
#         #     self.data_path = root + '/val'

#         # calculate data length
#         self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

#         if self.split == 'train':
#             self.augmentation = True
#         else:
#             self.augmentation = False
#         print('will do augmentation', self.augmentation)
#         self.do_edge = do_edge
#         self.do_semseg = do_semseg
#         self.do_normals = do_normals
#         self.do_depth = do_depth

#     def __getitem__(self, index):
#         # # load data from the pre-processed npy files
#         # image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
#         # semantic = torch.from_numpy(np.load(self.data_path + '/label_7/{:d}.npy'.format(index)))
#         # depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))

#         # # apply data augmentation if required
#         # if self.augmentation:
#         #     image, semantic, depth = RandomScaleCropCityScapes()(image, semantic, depth)
#         #     if torch.rand(1) < 0.5:
#         #         image = torch.flip(image, dims=[2])
#         #         semantic = torch.flip(semantic, dims=[1])
#         #         depth = torch.flip(depth, dims=[2])

#         # return image.float(), semantic.float(), depth.float()

#         sample = {}
#         # _img = np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0).astype(np.float32)
#         _img = np.load(self.data_path + '/image/{:d}.npy'.format(index)).astype(np.float32)
#         sample['image'] = _img
#         # print('image shape is ',_img.shape)

#         if self.do_semseg:
#             _semseg = np.load(self.data_path + '/label_7/{:d}.npy'.format(index)).astype(np.float32)
#             # print('_semantic shape is ', _semseg.shape)
#             # if _semseg.shape != _img.shape[:2]:
#             #     print('RESHAPE SEMSEG')
#             #     _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
#             sample['semseg'] = _semseg

#         if self.do_depth:
#             # _depth = np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)
            
#             # if _depth.shape[:2] != _img.shape[:2]:
#             #     print('RESHAPE DEPTH')
#             #     _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            
#             _depth = np.load(self.data_path + '/depth/{:d}.npy'.format(index))
#             sample['depth'] = _depth
        
#         if self.augmentation:
#             sample['image'], sample['semseg'], sample['depth'] = \
#                 RandomScaleCropCityScapes_np()(sample['image'], sample['semseg'], sample['depth'])
#             if torch.rand(1) < 0.5:
#                 sample['image'] = np.flip(sample['image'], 1)
#                 sample['semseg'] = np.flip(sample['semseg'], 1)
#                 sample['depth'] = np.flip(sample['depth'], 1)

#         sample['image'] = torch.from_numpy(np.moveaxis(sample['image'],-1,0).copy())
#         if self.do_depth:
#             sample['depth'] = np.expand_dims(sample['depth'],-1)
#             sample['depth'] = torch.from_numpy(np.moveaxis(sample['depth'],-1,0).copy())
#         if self.do_semseg:
#             sample['semseg'] = np.expand_dims(sample['semseg'],-1)
#             sample['semseg'] = torch.from_numpy(np.moveaxis(sample['semseg'],-1,0).copy())
#         # print(sample['image'].shape,sample['depth'].shape,sample['semseg'].shape)
#         return sample


#     def __len__(self):
#         return self.data_len
