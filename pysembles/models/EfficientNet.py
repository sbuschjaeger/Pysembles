# '''
# Taken from https://github.com/kuangliu/pytorch-cifar/blob/master/models/efficientnet.py
#     - I added a Swish module
#     - I removed the class definition to better fit my API 


# EfficientNet in PyTorch.
# Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
# Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from deep_ensembles_v2.Utils import Flatten

# def swish(x):
#     return x * x.sigmoid()

# class Swish(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x * x.sigmoid()


# def drop_connect(x, drop_ratio):
#     keep_ratio = 1.0 - drop_ratio
#     mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
#     mask.bernoulli_(keep_ratio)
#     x.div_(keep_ratio)
#     x.mul_(mask)
#     return x


# class SE(nn.Module):
#     '''Squeeze-and-Excitation block with Swish.'''

#     def __init__(self, in_channels, se_channels):
#         super(SE, self).__init__()
#         self.se1 = nn.Conv2d(in_channels, se_channels,
#                              kernel_size=1, bias=True)
#         self.se2 = nn.Conv2d(se_channels, in_channels,
#                              kernel_size=1, bias=True)

#     def forward(self, x):
#         out = F.adaptive_avg_pool2d(x, (1, 1))
#         out = swish(self.se1(out))
#         out = self.se2(out).sigmoid()
#         out = x * out
#         return out


# class Block(nn.Module):
#     '''expansion + depthwise + pointwise + squeeze-excitation'''

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  expand_ratio=1,
#                  se_ratio=0.,
#                  drop_rate=0.):
#         super(Block, self).__init__()
#         self.stride = stride
#         self.drop_rate = drop_rate
#         self.expand_ratio = expand_ratio

#         # Expansion
#         channels = expand_ratio * in_channels
#         self.conv1 = nn.Conv2d(in_channels,
#                                channels,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(channels)

#         # Depthwise conv
#         self.conv2 = nn.Conv2d(channels,
#                                channels,
#                                kernel_size=kernel_size,
#                                stride=stride,
#                                padding=(1 if kernel_size == 3 else 2),
#                                groups=channels,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)

#         # SE layers
#         se_channels = int(in_channels * se_ratio)
#         self.se = SE(channels, se_channels)

#         # Output
#         self.conv3 = nn.Conv2d(channels,
#                                out_channels,
#                                kernel_size=1,
#                                stride=1,
#                                padding=0,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         # Skip connection if in and out shapes are the same (MV-V2 style)
#         self.has_skip = (stride == 1) and (in_channels == out_channels)

#     def forward(self, x):
#         out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
#         out = swish(self.bn2(self.conv2(out)))
#         out = self.se(out)
#         out = self.bn3(self.conv3(out))
#         if self.has_skip:
#             if self.training and self.drop_rate > 0:
#                 out = drop_connect(out, self.drop_rate)
#             out = out + x
#         return out


# def make_layers(cfg, in_channels):
#     layers = []
#     _cfg = [cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size','stride']]
#     b = 0
#     blocks = sum(cfg['num_blocks'])
#     for expansion, out_channels, num_blocks, kernel_size, stride in zip(*_cfg):
#         strides = [stride] + [1] * (num_blocks - 1)
#         for stride in strides:
#             drop_rate = cfg['drop_connect_rate'] * b / blocks
#             layers.append(
#                 Block(in_channels,
#                         out_channels,
#                         kernel_size,
#                         stride,
#                         expansion,
#                         se_ratio=0.25,
#                         drop_rate=drop_rate))
#             in_channels = out_channels
#     return nn.Sequential(*layers)

# def efficientnet(cfg, num_classes):
#     layers = [
#         nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1,bias=False),
#         nn.BatchNorm2d(32),
#         Swish()
#     ]

#     layers.extend(make_layers(cfg, in_channels=32))
#     layers.extend([
#         nn.AdaptiveAvgPool2d(output_size=1),
#         nn.Dropout(cfg['dropout_rate']),
#         Flatten(),
#         nn.Linear(cfg['out_channels'][-1], num_classes)
#     ])
#     return nn.Sequential(*layers)

# def efficientnetB0(model_type, num_classes = 100): 
#     cfg = {
#         'num_blocks': [1, 2, 2, 3, 3, 4, 1],
#         'expansion': [1, 6, 6, 6, 6, 6, 6],
#         'out_channels': [16, 24, 40, 80, 112, 192, 320],
#         'kernel_size': [3, 3, 5, 3, 5, 5, 3],
#         'stride': [1, 2, 2, 2, 1, 2, 1],
#         'dropout_rate': 0.2,
#         'drop_connect_rate': 0.2,
#     }

#     return efficientnet(cfg, num_classes)

# # def EfficientNetB0():
# #     cfg = {
# #         'num_blocks': [1, 2, 2, 3, 3, 4, 1],
# #         'expansion': [1, 6, 6, 6, 6, 6, 6],
# #         'out_channels': [16, 24, 40, 80, 112, 192, 320],
# #         'kernel_size': [3, 3, 5, 3, 5, 5, 3],
# #         'stride': [1, 2, 2, 2, 1, 2, 1],
# #         'dropout_rate': 0.2,
# #         'drop_connect_rate': 0.2,
# #     }
# #     return EfficientNet(cfg)

'''EfficientNet in PyTorch.
Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride, model_type = "float"):
        super(Block, self).__init__()

        if model_type == "binary":
            ConvLayer = BinaryConv2d
            LinearLayer = BinaryLinear
            Activation = BinaryTanh
        else:
            ConvLayer = nn.Conv2d
            LinearLayer = nn.Linear
            Activation = nn.ReLU

        self.stride = stride
        self.a1 = Activation()
        self.a2 = Activation()
        self.a3 = Activation()

        planes = expansion * in_planes
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3,stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = ConvLayer(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                ConvLayer(in_planes, out_planes, kernel_size=1,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = ConvLayer(out_planes, out_planes//16, kernel_size=1)
        self.fc2 = ConvLayer(out_planes//16, out_planes, kernel_size=1)

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.a2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = self.a3(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out

class EfficientNet(nn.Module):
    def __init__(self, cfg, model_type = "float", num_classes=100):
        super(EfficientNet, self).__init__()
        
        if model_type == "binary":
            ConvLayer = BinaryConv2d
            LinearLayer = BinaryLinear
            Activation = BinaryTanh
        else:
            ConvLayer = nn.Conv2d
            LinearLayer = nn.Linear
            Activation = nn.ReLU

        self.cfg = cfg
        self.a1 = Activation()
        self.conv1 = ConvLayer(3, 32, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, model_type=model_type)
        self.linear = LinearLayer(cfg[-1][1], num_classes)
        self.scale = nn.Sequential()
        
        if model_type == "binary":
            self.scale = nn.Sequential( Scale() )

    def _make_layers(self, in_planes, model_type):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, model_type))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.scale(out)
        return out

def EfficientNetB0(model_type = "float"):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 2),
           (6,  24, 2, 1),
           (6,  40, 2, 2),
           (6,  80, 3, 2),
           (6, 112, 3, 1),
           (6, 192, 4, 2),
           (6, 320, 1, 2)]
    return EfficientNet(cfg, model_type)
