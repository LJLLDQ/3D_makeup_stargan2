
import copy
import math

# from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG as TVGG
from torchvision.models.vgg import load_state_dict_from_url, model_urls, cfgs
# from core.wing import FAN


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        # x = (self._shortcut(x) + self._residual(x)) * 0.2

        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=128, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))

            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, w_hpf=w_hpf, upsample=True))  # stack-like

            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            # Encode
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            # Decode
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None): # x_real:source_image, s_trg：style
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)

        for block in self.decode:
            print(x.shape)
            print(s.shape)
            exit()
            x = block(x, s)
            x = torch.tanh(x)

        return self.to_rgb(x)


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=1, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.Linear = nn.Linear(dim_out, style_dim)


    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        h = self.Linear(h)

        return h





# class VGG(nn.Module):
#     def __init__(self, pool='max'):
#         super(VGG, self).__init__()
#         # vgg modules
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         if pool == 'max':
#             self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
#         elif pool == 'avg':
#             self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

#     def forward(self, x, out_keys):
#         out = {}
#         out['r11'] = F.relu(self.conv1_1(x))
#         out['r12'] = F.relu(self.conv1_2(out['r11']))
#         out['p1'] = self.pool1(out['r12'])
#         out['r21'] = F.relu(self.conv2_1(out['p1']))
#         out['r22'] = F.relu(self.conv2_2(out['r21']))
#         out['p2'] = self.pool2(out['r22'])
#         out['r31'] = F.relu(self.conv3_1(out['p2']))
#         out['r32'] = F.relu(self.conv3_2(out['r31']))
#         out['r33'] = F.relu(self.conv3_3(out['r32']))
#         out['r34'] = F.relu(self.conv3_4(out['r33']))
#         out['p3'] = self.pool3(out['r34'])
#         out['r41'] = F.relu(self.conv4_1(out['p3']))
        
#         out['r42'] = F.relu(self.conv4_2(out['r41']))
#         out['r43'] = F.relu(self.conv4_3(out['r42']))
#         out['r44'] = F.relu(self.conv4_4(out['r43']))
#         out['p4'] = self.pool4(out['r44'])
#         out['r51'] = F.relu(self.conv5_1(out['p4']))
#         out['r52'] = F.relu(self.conv5_2(out['r51']))
#         out['r53'] = F.relu(self.conv5_3(out['r52']))
#         out['r54'] = F.relu(self.conv5_4(out['r53']))
#         out['p5'] = self.pool5(out['r54'])
        
#         return [out[key] for key in out_keys]


# class VGG(TVGG):
#     def forward(self, x):
#         x = self.features(x)
#         return x

# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


# def vgg16(pretrained=False, progress=True, **kwargs):
#     r"""VGG 16-layer model (configuration "D")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
