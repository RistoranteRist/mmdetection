# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import (Conv2d, ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from torch.nn import functional as F

from mmdet.registry import MODELS
from ..layers import CSPLayer


class ConvBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(ConvBlock, self).__init__(init_cfg)
        self.in_channels = in_channels
        expansion = 4
        hidden_channels = out_channels // expansion

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            hidden_channels,
            hidden_channels,
            3,
            stride=stride,
            padding=1,
            groups=groups,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            hidden_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.act3 = build_activation_layer(act_cfg)

        self.residual_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.residual_bn = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        residual = self.residual_conv(residual)
        residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        return x


class Mlp(BaseModule):
    """Implementation of MLP with 1*1 convolutions.

    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 drop_rate=0.,
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.fc1 = Conv2d(in_channels, hidden_channels, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Conv2d(hidden_channels, in_channels, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoding(nn.Module):

    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels = in_channels
        self.num_codes = num_codes
        std = 1. / ((num_codes * in_channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand(
            (b, x.size(1), num_codes, in_channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        reshaped_scale = scale.view((1, 1, num_codes))
        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand(
            (b, x.size(1), num_codes, in_channels))
        assignment_weights = assignment_weights.unsqueeze(3)
        encoded_feat = (assignment_weights *
                        (expanded_x - reshaped_codewords)).sum(1)

        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), dim=2)

        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


class LVCBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_codes,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super(LVCBlock, self).__init__(init_cfg)
        self.out_channels = out_channels
        self.num_codes = num_codes

        self.conv_1 = ConvBlock(
            in_channels=in_channels, out_channels=in_channels, stride=1)

        self.LVC = nn.Sequential(
            Conv2d(in_channels, in_channels, 1, bias=False),
            build_norm_layer(norm_cfg, in_channels)[1],
            build_activation_layer(act_cfg),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            build_norm_layer(dict(type='BN1d'), num_codes)[1],
            build_activation_layer(act_cfg))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        x = self.conv_1(x)
        en = self.LVC(x)
        en = en.mean(1)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = self.act(x + x * y)
        return x


class LightMLPBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 layer_scale_init_value=1e-5,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.out_channels = out_channels

        self.dw = DepthwiseSeparableConvModule(
            in_channels, out_channels, 1, stride=1, act_cfg=dict(type='SiLU'))

        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_channels)[1]

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_dim,
            drop_rate=drop_rate)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. \
            else nn.Identity()

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((out_channels)),
            requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((out_channels)),
            requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.dw(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class EVCBlock(BaseModule):

    def __init__(self, in_channels, out_channels, expansion=2, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            7,
            stride=1,
            padding=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(
            in_channels=in_channels, out_channels=out_channels, num_codes=64)
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(
            in_channels,
            out_channels,
            mlp_ratio=4.,
            drop_rate=0.,
            layer_scale_init_value=1e-5,
            drop_path_rate=0.)
        self.cnv1 = nn.Conv2d(
            out_channels * expansion,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.conv1(x))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x


@MODELS.register_module()
class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 use_evcblock=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # evcblock
        if use_evcblock:
            self.evcblock = EVCBlock(in_channels[-2], in_channels[-2])
        else:
            self.evcblock = nn.Identity()

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)
            if idx == len(self.in_channels) - 1:
                upsample_feat = self.evcblock(upsample_feat)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
