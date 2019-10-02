import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .modules.se import SEBlock
from .modules.checkpoint import CheckpointModule
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.mixup import MixUp
from scipy.stats import ortho_group
from torch.autograd import Variable
import numpy as np
import random

__all__ = ['resnet', 'resnet_se']


def create_max_span_matrix(n, m):
    mat = torch.zeros((n, m))
    for i in range(n):
        row = torch.ones((1, m)).normal_(0., 0.1)
        row /= row.sum()
        mat[i][:] = row
    return mat

class OrthNormalProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(OrthNormalProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        mat = torch.from_numpy(ortho_group.rvs(dim=input_size)).float()[:output_size]
        # mat = torch.cat((mat, -mat))
        # pos = torch.eye(input_size)
        # neg = -torch.eye(input_size)
        mat = torch.cat((mat[:output_size // 2], -mat[:output_size // 2]))
        # mat = create_max_span_matrix(output_size, input_size)
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat)

        init_scale = 1. / math.sqrt(self.output_size)

        if fixed_scale is not None:
            self.scale = Variable(torch.Tensor(
                [fixed_scale]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.Tensor([init_scale]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                output_size).uniform_(-init_scale, init_scale))
        else:
            self.register_parameter('bias', None)

        self.eps = 1e-8

    def forward(self, x):
        if not isinstance(self.scale, nn.Parameter):
            self.scale = self.scale.type_as(x)
        # x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.scale * nn.functional.linear(x, w)
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


def init_model(model, init_fc=True):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)
    if init_fc:
        model.fc.weight.data.normal_(0, 0.01)
        model.fc.bias.data.zero_()


def weight_decay_config(value=1e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(BasicBlock, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, dropout=0.):
        super(Bottleneck, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None, mixup=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, dropout=dropout))
        if mixup:
            layers.append(MixUp())
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        features = self.features(x)

        if hasattr(self, 'orth') and not hasattr(self, 'fc'):
            if (self.orth.proj.data.shape[-1] > features.data.shape[-1]):
                features = torch.cat((features, torch.ones((features.size(0), 1)).to(device=features.device)),
                                     dim=1)
            x = features
        else:
            if (self.fc.weight.data.shape[-1] > features.data.shape[-1]):
                features = torch.cat((features, torch.ones((features.size(0), 1)).to(device=features.device)),
                                     dim=1)
            x = self.fc(features)
        if hasattr(self, 'orth'):
            y = self.orth(x)
        else:
            y = x
        mz_loss = None
        if 'mz_mean' in self.regime_name:
            mz_loss = 0.1 * torch.mean((torch.norm(features, dim=1) - 10)**2)
        elif 'mz_all_mean' in self.regime_name:
            mz_loss = (torch.mean(torch.norm(features, dim=1)) - 10)**2
        elif 'mz_relu' in self.regime_name:
            mz_loss = torch.mean(nn.ReLU()(10 - torch.norm(features, dim=1)))

        # if 'mz_eye' in self.regime_name:
        #     orth_proj = self.orth.proj.matmul(self.orth.proj.t())
        #     if mz_loss is not None:
        #         mz_loss += nn.MSELoss()(orth_proj, torch.eye(orth_proj.shape[-1])).to(device=orth_proj.device)
        #     else:
        #         mz_loss = nn.MSELoss()(orth_proj, torch.eye(orth_proj.shape[-1]).to(device=orth_proj.device))
        if 'mz_orth_eye' in self.regime_name and hasattr(self, 'fc'):
            orth_proj = self.fc.weight.matmul(self.fc.weight.t())
            # orth_proj = self.orth.proj.matmul(self.orth.proj.t())
            if mz_loss is not None:
                mz_loss += nn.MSELoss()(orth_proj, torch.eye(orth_proj.shape[-1])).to(device=orth_proj.device)
            else:
                mz_loss = nn.MSELoss()(orth_proj, torch.eye(orth_proj.shape[-1]).to(device=orth_proj.device))

        return y, mz_loss


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1, checkpoint_segments=0, mixup=False):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.regime_name = regime
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i],
                                     mixup=mixup)
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        init_fc = True
        if 'BMz' in regime:
            self.fc = nn.Linear(width[-1] * expansion + 1, width[-1] * expansion, bias=False)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.weight.data[:, width[-1]].zero_()
            self.orth = OrthNormalProj(width[-1] * expansion, self.num_classes, bias=False)
            init_fc = False
        elif 'mz_orth' in regime:
            self.orth = OrthNormalProj(width[-1] * expansion + 1, self.num_classes, fixed_weights=True, bias=False)
            init_fc = False
        else:
            self.fc = nn.Linear(width[-1] * expansion, num_classes)

        init_model(self, init_fc)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'fast':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
                {'epoch': 4,  'lr': 4 * scale_lr * 1e-1},
                {'epoch': 18, 'lr': scale_lr * 1e-1},
                {'epoch': 21, 'lr': scale_lr * 1e-2},
                {'epoch': 35, 'lr': scale_lr * 1e-3},
                {'epoch': 43, 'lr': scale_lr * 1e-4},
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 18, 'input_size': 224, 'batch_size': 64},
                {'epoch': 41, 'input_size': 288, 'batch_size': 32},
            ]
        elif 'small' in regime:
            if regime == 'small_half':
                bs_factor = 2
            else:
                bs_factor = 1
            scale_lr *= 4 * bs_factor
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(1e-4),
                 'momentum': 0.9, 'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': bs_factor * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256 * bs_factor},
                {'epoch': 80, 'input_size': 224, 'batch_size': 64 * bs_factor},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 224, 'batch_size': 512 * bs_factor},
            ]


class ResNet_cifar(ResNet):

    def __init__(self, num_classes=10, inplanes=16,
                 block=BasicBlock, depth=18, width=[16, 32, 64],
                 groups=[1, 1, 1], residual_block=None, regime='normal', dropout=None, mixup=False):
        super(ResNet_cifar, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.regime_name = regime
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x

        self.layer1 = self._make_layer(block, width[0], n, groups=groups[
                                       0], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer2 = self._make_layer(
            block, width[1], n, stride=2, groups=groups[1], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer3 = self._make_layer(
            block, width[2], n, stride=2, groups=groups[2], residual_block=residual_block, dropout=dropout, mixup=mixup)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        init_fc = True
        if 'BMz' in regime:
            self.fc = nn.Linear(width[-1]+1, width[-1], bias=False)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.weight.data[:, width[-1]].zero_()
            self.orth = OrthNormalProj(width[-1], self.num_classes, bias=False)
            init_fc = False
        elif 'QAz' in regime:
            self.fc = nn.Linear(width[-1]+1, self.num_classes, bias=False)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.weight.data[:, width[-1]].zero_()
            self.orth = OrthNormalProj(self.num_classes, self.num_classes, bias=False)
            init_fc = False
        elif 'mz_orth' in regime:
            if 'eye' in regime:
                self.fc = nn.Linear(width[-1] + 1, width[-1], bias=False)
                self.fc.weight.data = torch.from_numpy(ortho_group.rvs(dim=width[-1] + 1)).float()[:width[-1]]
                # self.fc.weight.data.normal_(0, 0.01)
                # self.fc.weight.data[:, width[-1]].zero_()
                self.orth = OrthNormalProj(width[-1], self.num_classes, fixed_weights=True, bias=False)
            else:
                self.orth = OrthNormalProj(width[-1] + 1, self.num_classes, fixed_weights=True, bias=False)
            init_fc = False
        else:
            # self.fc = nn.Linear(width[-1] + 1, self.num_classes, bias=False)
            # pos = torch.eye(width[-1] + 1)
            # neg = -torch.eye(width[-1] + 1)
            # self.fc.weight.data = torch.cat((pos[:self.num_classes//2], neg[:self.num_classes//2]))
            # self.fc.weight.requires_grad = False
            # init_fc = False
            # self.fc = nn.Linear(width[-1], self.num_classes)
            self.fc = nn.Linear(width[-1] + 1, num_classes, bias=False)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.weight.data[:, width[-1]].zero_()
            init_fc = False

        init_model(self, init_fc)

        if 'normal' in regime:
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                 'regularizer': weight_decay_config(1e-4)},
                # {'epoch': 81, 'lr': 1e-2},
                # {'epoch': 122, 'lr': 1e-3},
                # {'epoch': 164, 'lr': 1e-4}
                #{'step': 63260, 'lr': 1e-2},
                #{'step': 95282, 'lr': 1e-3},
                #{'step': 128084, 'lr': 1e-4}
                # {'step': 19525, 'lr': 1e-2},
                 {'step': 24992, 'lr': 1e-2},
                 {'step': 27335, 'lr': 1e-3},
                 {'step': 29678, 'lr': 1e-4}
                # {'step': 14058, 'lr': 1e-2},
                # {'step': 13277, 'lr': 1e-2},
                # {'step': 24211, 'lr': 1e-2},
                # {'step': 15620, 'lr': 1e-3},
                # {'step': 35926, 'lr': 1e-3},
                # {'step': 36707, 'lr': 1e-4}
            ]
        elif 'wide-resnet' in regime:
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                 'regularizer': weight_decay_config(5e-4)},
                # {'epoch': 60, 'lr': 2e-2},
                # {'epoch': 120, 'lr': 4e-3},
                # {'epoch': 160, 'lr': 8e-4}
                # {'step': 46860, 'lr': 2e-2},
                # {'step': 93720, 'lr': 4e-3},
                # {'step': 124960, 'lr': 8e-4}
                # {'step': 22700, 'lr': 2e-2},
                # {'step': 69700, 'lr': 4e-3},
                # {'step': 100960, 'lr': 8e-4}
                # {'step': 23430, 'lr': 4e-3},
                # {'step': 46860, 'lr': 8e-4},
                # {'step': 62480, 'lr': 1.6e-5}
                {'step': 39050, 'lr': 2e-2},
                {'step': 41393, 'lr': 4e-3},
                {'step': 43736, 'lr': 8e-4}
            ]


def resnet(**config):
    dataset = config.pop('dataset', 'imagenet')
    if config.pop('quantize', False):
        from .modules.quantize import QConv2d, QLinear, RangeBN
        torch.nn.Linear = QLinear
        torch.nn.Conv2d = QConv2d
        torch.nn.BatchNorm2d = RangeBN

    bn_norm = config.pop('bn_norm', None)
    if bn_norm is not None:
        from .modules.lp_norm import L1BatchNorm2d, TopkBatchNorm2d
        if bn_norm == 'L1':
            torch.nn.BatchNorm2d = L1BatchNorm2d
        if bn_norm == 'TopK':
            torch.nn.BatchNorm2d = TopkBatchNorm2d

    if 'imagenet' in dataset:
        config.setdefault('num_classes', 1000)
        depth = config.pop('depth', 50)
        if depth == 18:
            config.update(dict(block=BasicBlock,
                               layers=[2, 2, 2, 2],
                               expansion=1))
        if depth == 34:
            config.update(dict(block=BasicBlock,
                               layers=[3, 4, 6, 3],
                               expansion=1))
        if depth == 50:
            config.update(dict(block=Bottleneck, layers=[3, 4, 6, 3]))
        if depth == 101:
            config.update(dict(block=Bottleneck, layers=[3, 4, 23, 3]))
        if depth == 152:
            config.update(dict(block=Bottleneck, layers=[3, 8, 36, 3]))
        if depth == 200:
            config.update(dict(block=Bottleneck, layers=[3, 24, 36, 3]))

        return ResNet_imagenet(**config)

    elif dataset == 'cifar10':
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)


def resnet_se(**config):
    config['residual_block'] = SEBlock
    return resnet(**config)
