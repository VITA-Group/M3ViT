#
# Code referenced from  Torchvision

import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from models.model_utils import cal_flops
import math
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        print('ResNet replace_stride_with_dilation',replace_stride_with_dilation)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('x00',x.shape) [2, 3, 480, 640]
        x = self.conv1(x)
        # print('x01',x.shape) [2, 64, 240, 320]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x1',x.shape) [2, 64, 120, 160]
        x = self.layer1(x)
        # print('x2',x.shape) [2, 256, 120, 160]
        x = self.layer2(x)
        # print('x3',x.shape) [2, 512, 60, 80]
        x = self.layer3(x)
        # print('x4',x.shape) [2, 1024, 30, 40]
        x = self.layer4(x)
        # print('x5',x.shape) ([2, 2048, 15, 20]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class MixtureBottleBlock(nn.Module):
    """Keep the input channels and prune all the output channels"""

    expansion = 4

    def __init__(self, inplanes, planes, tasks, expand=1, input_dim=10,
                 stride=1, downsample=None, init='1',  inner=True,
                 scale_type='relu', drop_ratio=0.5, drop_input=0.25,
                 groups=-1,): #num_task=1,
        super(MixtureBottleBlock, self).__init__()
        out_dim = int(planes * expand) if inner else planes
        self.conv1 = nn.Conv2d(inplanes, out_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Conv2d(out_dim, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inner = inner

        self.inplanes = inplanes
        self.planes = planes
        self.input_dim = input_dim
        self.expand = expand
        self.tasks = tasks
        self.num_task = len(tasks)
        # heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})

        # if self.num_task==1:
        #     self.gate1 = nn.Linear(input_dim, out_dim, bias=False)
        #     self.gate2 = nn.Linear(input_dim, out_dim, bias=False)
        #     self.gate3 = nn.Linear(input_dim, planes*4, bias=False)
        # else:
        #     for i in range(self.num_task):
        #         for g in range(2):
        #             setattr(self, 'gate{}_{}'.format(i,g),nn.Linear(input_dim, out_dim, bias=False))
        #         setattr(self, 'gate{}_{}'.format(i,2),nn.Linear(input_dim, planes*4, bias=False))
        for i in range(self.num_task):
            for g in range(2):
                setattr(self, 'gate{}_{}'.format(str(self.tasks[i]),g),nn.Linear(input_dim, out_dim, bias=False))
            setattr(self, 'gate{}_{}'.format(str(self.tasks[i]),2),nn.Linear(input_dim, planes*4, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == '1':
                    m.weight.data.fill_(1)
                else:
                    low = float(init.split(',')[-2])
                    high = float(init.split(',')[-1])
                    nn.init.uniform(m.weight, low, high)

    def forward(self, x, y, task, prob=0):  # use y as input for the gates
        residual = x
        masks, costs = [], []

        # conv1 1x1
        max_cin = x.size(1)
        cin = (x.sum(dim=-1).sum(dim=-1) != 0).float().sum(dim=1).data
        out = self.conv1(x)
        fs = out.size(2)
        out = self.bn1(out)

        # mask
        gprob = getattr(self, 'gate{}_{}'.format(str(task), 0))(y)
        mask = self.relu(gprob)
        # if self.num_task==1:
        #     gprob = self.gate1(y)
        #     mask = self.relu(gprob)
        # else:
        #     mask = []
        #     for i in range(self.num_task):
        #         gprob = getattr(self, 'gate{}_{}'.format(i, 0))(y)
        #         mask.append(self.relu(gprob))
        #     mask = torch.stack(mask)
        #     mask = torch.sum(mask,dim=0)

        costs.append(mask.abs())

        # calculate flops of a batch of inputs
        cout = (mask != 0).float().sum(1).data
        flops = cal_flops(fs, fs, 1, cin, cout)
        flops = flops.mean()
        total_flops = cal_flops(fs, fs, 1, max_cin, out.size(1))

        mask = mask.view(mask.size(0), mask.size(1), 1, 1)
        out = torch.mul(out, mask)
        out = self.relu(out)
        masks.append(mask)

        # conv2 3x3
        max_cin = out.size(1)
        out = self.conv2(out)
        fs = out.size(2)
        out = self.bn2(out)

        gprob = getattr(self, 'gate{}_{}'.format(str(task), 1))(y)
        mask = self.relu(gprob)
        # if self.num_task==1:
        #     gprob = self.gate2(y)
        #     mask = self.relu(gprob)
        # else:
        #     mask = []
        #     for i in range(self.num_task):
        #         gprob = getattr(self, 'gate{}_{}'.format(i, 1))(y)
        #         mask.append(self.relu(gprob))
        #     mask = torch.stack(mask)
        #     mask = torch.sum(mask,dim=0)
        costs.append(mask.abs())

        # calculate flops of a batch of inputs
        cout2 = (mask != 0).float().sum(1).data
        flops += cal_flops(fs, fs, 3, cout, cout2).mean()
        total_flops += cal_flops(fs, fs, 3, max_cin, out.size(1))

        mask = mask.view(mask.size(0), mask.size(1), 1, 1)
        out = torch.mul(out, mask)
        out = self.relu(out)
        masks.append(mask)

        # conv3 1x1
        max_cin = out.size(1)
        out = self.conv3(out)
        fs = out.size(2)
        out = self.bn3(out)

        if not self.inner:
            # gprob = self.gate3(y)
            # mask = self.relu(gprob)
            gprob = getattr(self, 'gate{}_{}'.format(str(task), 2))(y)
            mask = self.relu(gprob)
            # if self.num_task==1:
            #     gprob = self.gate3(y)
            #     mask = self.relu(gprob)
            # else:
            #     mask = []
            #     for i in range(self.num_task):
            #         gprob = getattr(self, 'gate{}_{}'.format(i, 2))(y)
            #         mask.append(self.relu(gprob))
            #     mask = torch.stack(mask)
            #     mask = torch.sum(mask,dim=0)

            costs.append(mask.abs())

            # calculate flops of a batch of inputs
            cout3 = (mask != 0).float().sum(1).data
            flops += cal_flops(fs, fs, 1, cout2, cout3).mean()
            total_flops += cal_flops(fs, fs, 1, max_cin, out.size(1))

            mask = mask.view(mask.size(0), mask.size(1), 1, 1)
            out = torch.mul(out, mask)
            masks.append(mask)
        else:
            flops += cal_flops(fs, fs, 1, cout2, out.size(1)).mean()
            total_flops += cal_flops(fs, fs, 1, max_cin, out.size(1))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, masks, costs, flops, total_flops

########################################
# DeepMoE ResNet with channel routing  #
########################################


class MixtureResNet(nn.Module):
    """ ResNet with routing modules """
    def __init__(self, block, layers,tasks, input_dim=10, num_classes=10, init='1',
                 scale_type='relu', dataset='cifar10', kk=1, groups=-1,
                 drop_ratio=0.5, drop_input=0.25, inner=False, **kwargs):
        super(MixtureResNet, self).__init__()

        self.dataset = dataset
        self.layers = layers
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.init = init
        self.scale_type = scale_type
        self.kk = kk
        self.drop_ratio = drop_ratio
        self.drop_input = drop_input
        self.inner = inner
        self.groups = groups
        self.tasks = tasks

        if dataset == 'imagenet':
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self._make_layer_v2(block, 64, layers[0], 1)
            self._make_layer_v2(block, 128, layers[1], 2, stride=2)
            self._make_layer_v2(block, 256, layers[2], 3, stride=2)
            self._make_layer_v2(block, 512, layers[3], 4, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * block.expansion if self.inner
                                else int(512 * kk * block.expansion), num_classes)

        else:
            self.inplanes = 16
            self.conv1 = conv3x3(3, 16)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)

            self._make_layer_v2(block, 16, layers[0], 1)
            self._make_layer_v2(block, 32, layers[1], 2, stride=2)
            self._make_layer_v2(block, 64, layers[2], 3, stride=2)
            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.fc = nn.Linear(64 * block.expansion if self.inner
                                else int(64 * kk * block.expansion), num_classes)
        
        from functools import partial
        for i in range(self.layers[2]):
            getattr(self, 'layer{}_{}'.format(3, i)).apply(
                partial(self._nostride_dilate, dilate=2))
        for i in range(self.layers[3]):
            getattr(self, 'layer{}_{}'.format(4, i)).apply(
                partial(self._nostride_dilate, dilate=4))
        #         x, mask, cost, _flop, _tflop = getattr(
        #             self, 'layer{}_{}'.format(g, i))(
        #             x, y, prob=prob)
        # getattr(orig_resnet, 'layer{}_{}'.format(3, 1)).apply(
        #         partial(self._nostride_dilate, dilate=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_v2(self, block, planes, blocks, group_id, stride=1):
        downsample = None

        if not self.inner:
            planes *= self.kk
            planes = int(planes)

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        setattr(self, 'layer{}_0'.format(group_id),
                block(self.inplanes, planes, self.tasks, self.kk, self.input_dim,
                      stride, downsample, init=self.init, inner=self.inner,
                      scale_type=self.scale_type,
                      drop_ratio=self.drop_ratio,
                      drop_input=self.drop_input,
                      groups=self.groups,
                      ))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            setattr(self, 'layer{}_{}'.format(group_id, i),
                    block(self.inplanes, planes, self.tasks, self.kk, self.input_dim,
                          init=self.init, inner=self.inner,
                          scale_type=self.scale_type,
                          drop_ratio=self.drop_ratio,
                          drop_input=self.drop_input,
                          groups=self.groups))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, y, task, overhead=0, prob=0):
        # print('input',x.shape)
        masks, costs = [], []

        max_in = x.size(1)
        x = self.conv1(x)
        fs = x.size(2)
        max_out = x.size(1)
        if self.dataset == 'imagenet':
            conv_flop = cal_flops(fs, fs, 7, max_in, max_out)
        else:
            conv_flop = cal_flops(fs, fs, 3, max_in, max_out)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dataset == 'imagenet':
            x = self.maxpool(x)

        flops, total_flops = 0, 0
        for g in range(1, 4+int(self.dataset == 'imagenet')):
            for i in range(self.layers[g-1]):
                x, mask, cost, _flop, _tflop = getattr(
                    self, 'layer{}_{}'.format(g, i))(
                    x, y, task, prob=prob)

                costs.extend(cost)
                masks.extend(mask)
                flops += _flop
                total_flops += _tflop
        # print('x',x.shape)
        return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        in_dim = x.size(1)
        x = self.fc(x)
        linear_flop = in_dim * x.size(-1)

        flops += (conv_flop + linear_flop + overhead)
        total_flops += (conv_flop + linear_flop)

        # print('total flops: ', total_flops)
        # print('flops: ', flops)  # checked ResNet110, 2.5x10^8, matched
        return x, masks, costs, flops/total_flops*100


# ResNet-50
def mixture_inner_resnet_50(pretrained="", tasks=[], **kwargs):
    print('kwargs',kwargs)
    model = MixtureResNet(MixtureBottleBlock, [3, 4, 6, 3],tasks=tasks, kk=2, inner=True,
                          **kwargs)
    if pretrained:
        model = load_pretrained_v2(model, pretrained)
    return model

    

########################################
# Shallow Embedding Network            #
########################################


class ShallowEmbeddingImageNet(nn.Module):
    """ Shallow embedding network for ImageNet """
    def __init__(self, num_classes=1000, **kwargs):
        super(ShallowEmbeddingImageNet, self).__init__()

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = conv3x3(128, 256, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = conv3x3(256, 512, stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = conv3x3(512, 512, stride=2)
        self.bn6 = nn.BatchNorm2d(512)

        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # print('prior input shape',x.shape)
        c_in = x.size(1)
        x = self.conv1(x)
        fs = x.size(2)
        x = self.bn1(x)
        x = self.relu(x)
        flop = cal_flops(fs, fs, 3, c_in, x.size(1))
        x = self.maxpool(x)
        
        c_in = x.size(1)
        x = self.conv2(x)
        fs = x.size(2)
        x = self.bn2(x)
        x = self.relu(x)
        flop += cal_flops(fs, fs, 3, c_in, x.size(1))

        c_in = x.size(1)
        x = self.conv3(x)
        fs = x.size(2)
        x = self.bn3(x)
        x = self.relu(x)
        flop += cal_flops(fs, fs, 3, c_in, x.size(1))

        c_in = x.size(1)
        x = self.conv4(x)
        fs = x.size(2)
        x = self.bn4(x)
        x = self.relu(x)
        flop += cal_flops(fs, fs, 3, c_in, x.size(1))

        c_in = x.size(1)
        x = self.conv5(x)
        fs = x.size(2)
        x = self.bn5(x)
        x = self.relu(x)
        flop += cal_flops(fs, fs, 3, c_in, x.size(1))

        c_in = x.size(1)
        x = self.conv6(x)
        fs = x.size(2)
        x = self.bn6(x)
        x = self.relu(x)
        flop += cal_flops(fs, fs, 3, c_in, x.size(1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        c_in = x.size(1)
        x = self.fc(x)

        flop += c_in * x.size(1)

        # print('overhead: ', flop) # checked, roughly 2% over ResNet-110
        return x, torch.autograd.Variable((torch.ones(1) * flop).cuda()).view(1, 1)


# For ImageNet
def shallow_embedding_imagenet(pretrained="", **kwargs):
    """for imagenet"""
    model = ShallowEmbeddingImageNet(**kwargs)
    if pretrained:
        model = load_pretrained_v2(model, pretrained)
    return model


