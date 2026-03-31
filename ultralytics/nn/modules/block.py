# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Sequential
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from collections import OrderedDict
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from .lnn import LiquidNeuralModule, LiquidNeuralModuleLite
import math
import numpy as np

from .rep_block import  DiverseBranchBlock, WideDiverseBranchBlock, DeepDiverseBranchBlock,FeaturePyramidAggregationAttention,RecursionDiverseBranchBlock
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1","ELAN","ELAN_H",'MP_1','MP_2','ELAN_t','SPPCSPCSIM','SPPCSPC','A2C2f','YOLOv4_BottleneckCSP','YOLOv4_Bottleneck',
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "CrossAttentionShared","CrossMLCA","TensorSelector","CrossMLCAv2",
    'DiverseBranchBlock', 'WideDiverseBranchBlock', 'DeepDiverseBranchBlock','FeaturePyramidAggregationAttention','RecursionDiverseBranchBlock',
    "C3k2_DeepDBB","C3k2_DBB","C3k2_WDBB",'C2f_DeepDBB','C2f_WDBB','C2f_DBB','C3k_RDBB','C2f_RDBB','C3k2_RDBB',
    'ConvNormLayer', 'BasicBlock', 'BottleNeck', 'Blocks',
    "CrossC2f", "CrossC3k2",
    "CBH","ES_Bottleneck","DWConvblock","ADD",
    'MANet', 'HyperComputeModule', 'MANet_FasterBlock', 'MANet_FasterCGLU', 'MANet_Star',
    "GPT","Add2","Add","CrossTransformerFusion","C2Liquid",
    "C2Liquid_Lite",
    "C2Liquid_Adaptive",
    "LiquidSPPF",
    "LiquidSPPF_Lite",

)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class CrossC2f(nn.Module):
    """Cross-Connected CSP Bottleneck with 2 convolutions and residual connections."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, ratio=0.15):
        super(CrossC2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.ratio = ratio  # residual connection ratio
        self.cv1 = Conv(c1 * 2, 2 * self.c, 1, 1)  # 1x1 conv for information interaction
        self.cv2 = Conv(c1, c2, 1, 1)  # 修改输入通道数为 c1
        self.cv3 = Conv(self.c * (n + 1), c1, 1, 1)  # 修改输入通道数为 self.c * (n + 1)
        self.m1 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m2 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through CrossC2f layer."""
        x1, x2 = x  # unpack input
        # print(x1.shape, x2.shape )
        x_concat = torch.cat([x1, x2], dim=1)  # concatenate along channel dimension
        y = self.cv1(x_concat).split(self.c, dim=1)  # split into two parts after 1x1 conv

        # Cross the outputs
        out_1 = [y[1]]  # Start with the second part of y
        out_2 = [y[0]]  # Start with the first part of y

        # Process out_1 and out_2 through their respective branches
        for m1, m2 in zip(self.m1, self.m2):
            out_1.append(m1(out_1[-1]))
            out_2.append(m2(out_2[-1]))

        # Concatenate the intermediate results of each branch
        out_1 = torch.cat(out_1, dim=1)  # Concatenate all intermediate results of out_1
        out_2 = torch.cat(out_2, dim=1)  # Concatenate all intermediate results of out_2

        # Apply shared convolution to out_1 and out_2
        out_1 = self.cv3(out_1)  # Apply shared conv to out_1
        out_2 = self.cv3(out_2)  # Apply shared conv to out_2

        # Add residual connections
        out_1 = x1 * self.ratio + out_1
        out_2 = x2 * self.ratio + out_2

        # Combine out_1 and out_2 by addition instead of concatenation
        out = out_1 + out_2  # Change from concatenation to addition

        return [out_1, out_2, self.cv2(out)]

class CrossC3k2(CrossC2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True , ratio=0.15):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # print('CrossC3k2:',c1, c2, n, c3k, e, g, shortcut , ratio)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """

    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):


    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y



class CrossAttentionShared(nn.Module):
    """
    Cross-Attention module with weight sharing and additional projection for combined output.
    Both x1 and x2 use the same convolutional layers to generate Query, Key, and Value.
    An additional projection layer combines x1_out and x2_out into a shared output x_out_all.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values of x1 and x2.
        proj_all (Conv): Convolutional layer for projecting the combined output of x1_out and x2_out.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes cross-attention module with shared query, key, and value convolutions and positional encoding."""
        super().__init__()
        # print(dim, num_heads , attn_ratio )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = self.head_dim * num_heads

        # Shared convolutional layer for Query, Key, and Value
        self.qkv = nn.Conv2d(dim, nh_kd * 2 + h, kernel_size=1, bias=False)
        # Shared projection layer for individual outputs
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # Additional projection layer for combined output
        self.proj_all = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        # Shared positional encoding layer
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self,x):
        """
        Forward pass of the Cross-Attention module.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            tuple: A tuple containing the output tensors after cross-attention for x1, x2, and the combined output.
        """
        # print(len(x))
        x1 = x[0]  # 第一个输入张量
        x2 = x[1]  # 第二个输入张量
        B, C, H, W = x1.shape
        N = H * W

        # Compute Query, Key, and Value for x1
        qkv1 = self.qkv(x1).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q1, k1, v1 = qkv1.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # Compute Query, Key, and Value for x2
        qkv2 = self.qkv(x2).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q2, k2, v2 = qkv2.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # Compute attention scores for x1 attending to x2
        attn1 = (q1.transpose(-2, -1) @ k2) * self.scale
        attn1 = attn1.softmax(dim=-1)
        x1_out = (v2 @ attn1.transpose(-2, -1)).view(B, C, H, W)

        # Compute attention scores for x2 attending to x1
        attn2 = (q2.transpose(-2, -1) @ k1) * self.scale
        attn2 = attn2.softmax(dim=-1)
        x2_out = (v1 @ attn2.transpose(-2, -1)).view(B, C, H, W)

        # Add positional encoding
        x1_out = x1_out + self.pe(x1_out)
        x2_out = x2_out + self.pe(x2_out)

        # Project the individual outputs
        x1_out = self.proj(x1_out)
        x2_out = self.proj(x2_out)

        x1_out = x1_out + x1
        x2_out = x2_out + x2
        # Combine x1_out and x2_out and project to a shared output
        x_out_all = self.proj_all(torch.cat([x1_out, x2_out], dim=1))

        # return [x1_out, x2_out, x_out_all]

        return   x_out_all



class TensorSelector(nn.Module):
    """
    A module that selects a specific tensor from a list of tensors based on a fixed index.

    Args:
        index (int): The fixed index of the tensor to be selected.
    """
    def __init__(self, index):
        super(TensorSelector, self).__init__()
        self.index = index

    def forward(self, tensors):
        """
        Forward pass of the TensorSelector module.

        Args:
            tensors (list of torch.Tensor): A list of tensors from which to select.

        Returns:
            torch.Tensor: The selected tensor based on the fixed index.
        """
        if not isinstance(tensors, list) or not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("Input must be a list of torch.Tensor.")
        if self.index < 0 or self.index >= len(tensors):
            raise IndexError("Index out of range.")
        return tensors[self.index]

class CrossMLCA(nn.Module):
    """
    Modified Local Channel Attention (MLCA) module with cross-attention mechanism.
    Global features of x1 interact with local features of x2, and vice versa.
    """
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, local_size=5, gamma=2, b=1):
        super(CrossMLCA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        # Convolutional layer for computing Q, K, V for global features
        self.qkv_global = Conv(dim, h, k=1, act=False)
        self.proj_global = Conv(dim, dim, k=1, act=False)

        # Local average pooling for generating local features (used as positional encoding)
        self.local_avg_pool = nn.AdaptiveAvgPool2d(local_size)

        # ECA-like mechanism for local features
        t = int(abs(math.log(dim, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.combined_conv = Conv(dim * 2, dim, k=1, act=True)
    def forward(self, x):
        x1, x2 = x  # 解包输入张量 x = (x1, x2)

        # Process x1 (global features)
        B, C, H, W = x1.shape
        N = H * W

        # Global features of x1: compute Q, K, V
        qkv_global_x1 = self.qkv_global(x1)  # Shape: (B, C + 2 * nh_kd, H, W)
        q_x1, k_x1, v_x1 = qkv_global_x1.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # Compute attention scores for global features of x1
        attn_x1 = (q_x1.transpose(-2, -1) @ k_x1) * self.scale  # Shape: (B, num_heads, N, N)
        attn_x1 = attn_x1.softmax(dim=-1)

        # Apply attention to V for global features of x1
        attended_v_global_x1 = (v_x1 @ attn_x1.transpose(-2, -1)).view(B, C, H, W)  # Shape: (B, C, H, W)
        global_features_x1 = self.proj_global(attended_v_global_x1)  # Project global features of x1

        # Process x2 (local features)
        local_features_x2 = self.local_avg_pool(x2)  # Shape: (B, C, local_size, local_size)
        B_local, C_local, H_local, W_local = local_features_x2.shape
        N_local = H_local * W_local

        # Flatten and apply ECA-like mechanism to local features of x2
        temp_local_x2 = local_features_x2.view(B_local, C_local, -1).transpose(-1, -2).reshape(B_local, 1, -1)  # Shape: (B, 1, C * local_size^2)
        local_att_x2 = self.conv_local(temp_local_x2)  # Shape: (B, 1, C * local_size^2)
        local_att_x2 = local_att_x2.view(B_local, -1, C_local).transpose(-1, -2).view(B_local, C_local, H_local, W_local)  # Restore shape

        # Upsample local features of x2 to original size
        local_att_x2 = F.interpolate(local_att_x2, size=(H, W), mode='nearest')

        # Combine global features of x1 with local features of x2
        output1 = (global_features_x1 + local_att_x2) * x1 + x1

        #############################################################################

        # Process x2 (global features)
        qkv_global_x2 = self.qkv_global(x2)  # Shape: (B, C + 2 * nh_kd, H, W)
        q_x2, k_x2, v_x2 = qkv_global_x2.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        # Compute attention scores for global features of x2
        attn_x2 = (q_x2.transpose(-2, -1) @ k_x2) * self.scale  # Shape: (B, num_heads, N, N)
        attn_x2 = attn_x2.softmax(dim=-1)

        # Apply attention to V for global features of x2
        attended_v_global_x2 = (v_x2 @ attn_x2.transpose(-2, -1)).view(B, C, H, W)  # Shape: (B, C, H, W)
        global_features_x2 = self.proj_global(attended_v_global_x2)  # Project global features of x2

        # Process x1 (local features)
        local_features_x1 = self.local_avg_pool(x1)  # Shape: (B, C, local_size, local_size)
        temp_local_x1 = local_features_x1.view(B_local, C_local, -1).transpose(-1, -2).reshape(B_local, 1, -1)  # Shape: (B, 1, C * local_size^2)
        local_att_x1 = self.conv_local(temp_local_x1)  # Shape: (B, 1, C * local_size^2)
        local_att_x1 = local_att_x1.view(B_local, -1, C_local).transpose(-1, -2).view(B_local, C_local, H_local, W_local)  # Restore shape
        local_att_x1 = F.interpolate(local_att_x1, size=(H, W), mode='nearest')

        # Combine global features of x2 with local features of x1
        output2 = (global_features_x2 + local_att_x1) * x2 + x2

        # Concatenate output1 and output2
        combined_output = torch.cat([output1, output2], dim=1)  # Shape: (B, 2*C, H, W)

        # Process the combined output through a convolutional layer
        final_output = self.combined_conv(combined_output)  # Shape: (B, C, H, W)
        return [output1, output2,final_output]
        # return  final_output




class ChannelCompressAndExpand(nn.Module):
    def __init__(self, k):
        super(ChannelCompressAndExpand, self).__init__()
        # 1x1卷积层，用于压缩和扩展特征
        out_channels = k * k
        self.k = k

        self.conv1x1 = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        # 全局平均池化层，将通道数压缩到1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((k, k))

    def forward(self, x):
        # x的形状应该是 (batch_size, C, k, k)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 使用全局平均池化将通道数压缩到1
        x_flat_avg = avg_out.view(-1, self.k * self.k)
        x_flat_max = max_out.view(-1, self.k * self.k)
        x_flat = torch.cat([x_flat_avg, x_flat_max], dim=1)
        x_flat = x_flat.unsqueeze(-1)  # 在最后一维扩展一个新的维度
        avg_out_convoluted = self.conv1x1(x_flat)
        # 使用 view 将卷积后的输出调整为 (batch_size, out_channels, k, k)
        output = avg_out_convoluted.view(avg_out_convoluted.size(0), -1, x.size(2), x.size(3))

        return output


class CrossMLCAv2(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(CrossMLCAv2, self).__init__()

        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cae = ChannelCompressAndExpand(local_size)
        # 新增卷积层，用于合并 x1 和 x2，输出通道数降低一半
        self.merge_conv = nn.Conv2d(in_channels=in_size * 2, out_channels=in_size, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2=x
        # 处理 x1 和 x2 的局部和全局信息
        local_arv1 = self.local_arv_pool(x1)
        global_arv1 = self.global_arv_pool(local_arv1)
        local_arv2 = self.local_arv_pool(x2)
        global_arv2 = self.global_arv_pool(local_arv2)

        b, c, m, n = x1.shape
        b_local, c_local, m_local, n_local = local_arv1.shape

        # 共用 conv_cae
        spatial_info_local1 = self.conv_cae(local_arv1)
        spatial_info_local2 = self.conv_cae(local_arv2)

        # 处理局部信息
        temp_local1 = local_arv1.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_local2 = local_arv2.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global1 = global_arv1.view(b, c, -1).transpose(-1, -2)
        temp_global2 = global_arv2.view(b, c, -1).transpose(-1, -2)

        y_local1 = self.conv_local(temp_local1)
        y_global1 = self.conv(temp_global1)
        y_local2 = self.conv_local(temp_local2)
        y_global2 = self.conv(temp_global2)

        # 转换形状
        y_local_transpose1 = y_local1.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose1 = y_global1.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_local_transpose2 = y_local2.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose2 = y_global2.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 应用空间信息
        y_local_transpose1 = spatial_info_local1 * y_local_transpose1
        y_local_transpose2 = spatial_info_local2 * y_local_transpose2

        # 计算注意力
        att_local1 = y_local_transpose1.sigmoid()
        att_global1 = F.adaptive_avg_pool2d(y_global_transpose1.sigmoid(), [self.local_size, self.local_size])
        att_all1 = F.adaptive_avg_pool2d(att_global1 * (1 - self.local_weight) + (att_local1 * self.local_weight), [m, n])

        att_local2 = y_local_transpose2.sigmoid()
        att_global2 = F.adaptive_avg_pool2d(y_global_transpose2.sigmoid(), [self.local_size, self.local_size])
        att_all2 = F.adaptive_avg_pool2d(att_global2 * (1 - self.local_weight) + (att_local2 * self.local_weight), [m, n])

        # 应用注意力
        x1 = x1 * att_all1 +x1
        x2 = x2 * att_all2 +x2

        # 合并 x1 和 x2 并通过卷积降低通道数
        merged = torch.cat([x1, x2], dim=1)  # 合并通道
        output = self.merge_conv(merged)  # 通道数降低一半

        return [x1, x2,output]




######################################## C2f-DDB begin ########################################

class Bottleneck_DBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C3k_DBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DBB(self.c, self.c, shortcut, g) for _ in range(n))

class Bottleneck_WDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = WideDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = WideDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C3k_WDBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_WDBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_WDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WDBB(self.c, self.c, shortcut, g) for _ in range(n))

class Bottleneck_DeepDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeepDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DeepDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

class C3k_DeepDBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DeepDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_DeepDBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DeepDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DeepDBB(self.c, self.c, shortcut, g) for _ in range(n))


class C2f_WDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_WDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class C2f_DeepDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DeepDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class C2f_DBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))



class Bottleneck_RDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # RecursionDiverseBranchBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=False,
        #                             recursion_layer=6)
        self.cv1 = RecursionDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = RecursionDiverseBranchBlock(c_, c2, k[1], 1, groups=g)


class C3k_RDBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_RDBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RDBB(self.c, self.c, shortcut, g) for _ in range(n))



class C2f_RDBB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_RDBB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C2f-DDB end ########################################


# https://blog.csdn.net/weixin_43694096/article/details/131726904
class ELAN(nn.Module):
    def __init__(self, c1, c2, down=False):
        super().__init__()

        c_ = c1 // 2
        if down:
            c_ = c1 // 4

        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c_, 3, 1)
        self.conv4 = Conv(c_, c_, 3, 1)
        self.conv5 = Conv(c_, c_, 3, 1)
        self.conv6 = Conv(c_, c_, 3, 1)
        self.conv7 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.conv2(x)
        y2 = self.conv4(self.conv3(y1))
        y3 = self.conv6(self.conv5(y2))

        return self.conv7(torch.cat((x1, y1, y2, y3), dim=1))


class ELAN_H(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        c_ = c1 // 2
        c__ = c1 // 4

        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(c1, c2, 1, 1)
        self.conv3 = Conv(c2, c__, 3, 1)
        self.conv4 = Conv(c__, c__, 3, 1)
        self.conv5 = Conv(c__, c__, 3, 1)
        self.conv6 = Conv(c__, c__, 3, 1)
        self.conv7 = Conv(c__ * 4 + c_ * 2, c2, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        y = self.conv2(x)
        y1 = self.conv3(y)
        y2 = self.conv4(y1)
        y3 = self.conv5(y2)
        y4 = self.conv6(y3)

        return self.conv7(torch.cat((x1, y, y1, y2, y3, y4), dim=1))


class MP_1(nn.Module):

    def __init__(self, c1, c2, k=2, s=2):
        super(MP_1, self).__init__()

        c_ = c1 // 2
        self.m = nn.MaxPool2d(kernel_size=k, stride=s)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(c_, c_, 3, 2)

    def forward(self, x):
        y1 = self.conv1(self.m(x))
        y2 = self.conv3(self.conv2(x))
        return torch.cat((y1, y2), dim=1)


class MP_2(nn.Module):

    def __init__(self, c1, c2, k=2, s=2):
        super(MP_2, self).__init__()

        self.m = nn.MaxPool2d(kernel_size=k, stride=s)
        self.conv1 = Conv(c1, c1, 1, 1)
        self.conv2 = Conv(c1, c1, 1, 1)
        self.conv3 = Conv(c1, c1, 3, 2)

    def forward(self, x):
        y1 = self.conv1(self.m(x))
        y2 = self.conv3(self.conv2(x))
        return torch.cat((y1, y2), dim=1)


class ELAN_t(nn.Module):
    # Yolov7 ELAN with args(ch_in, ch_out, kernel, stride, padding, groups, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = int(c2 // 2)
        c_out = c_ * 4
        self.cv1 = Conv(c1, c_, k=k, s=s, p=p, g=g, act=act)
        self.cv2 = Conv(c1, c_, k=k, s=s, p=p, g=g, act=act)
        self.cv3 = Conv(c_, c_, k=3, s=s, p=p, g=g, act=act)
        self.cv4 = Conv(c_, c_, k=3, s=s, p=p, g=g, act=act)
        self.cv5 = Conv(c_out, c2, k=k, s=s, p=p, g=g, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = torch.cat((x1, x2, x3, x4), 1)
        return self.cv5(x5)


class SPPCSPCSIM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPCSIM, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = torch.cat([x2] + [m(x2) for m in self.m], 1)
        x4 = self.cv3(x3)
        x5 = torch.cat((x1, x4), 1)
        return self.cv4(x5)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

# -------------------------------------------------YOLOv12------------------------------------------
class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initializes an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention."""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = x + self.pe(v)
        return self.proj(x)

class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        return x + self.mlp(x)

class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )
        # print(c1, c2, n, a2, area)

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

# -------------------------------------------------YOLOv12------------------------------------------


#----------------------YOLOv4---------------------------------

class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()

class ConvBNMish(nn.Module):
    # YOLOv4 conventional convolution module
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super(ConvBNMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=autopad(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class YOLOv4_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, groups=1, expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(YOLOv4_Bottleneck, self).__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = ConvBNMish(c1, c_, 1, 1)
        self.cv2 = ConvBNMish(c_, c2, 3, 1, groups=groups)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class YOLOv4_BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, groups=1, expansion=0.5):
        super(YOLOv4_BottleneckCSP, self).__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = ConvBNMish(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBNMish(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = Mish()
        self.m = nn.Sequential(*[YOLOv4_Bottleneck(c_, c_, shortcut, groups, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


################################### RT-DETR PResnet  来自B站魔鬼面具， 代码 主要用于基本的 rtdetr-r18 ###################################
def get_activation(act: str, inpace: bool = True):
    '''get activation
    '''
    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None,
                 kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1,
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

# PicoDet
class CBH(nn.Module):
    def __init__(self, num_channels, num_filters, filter_size, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x

    def fuseforward(self, x):
        return self.hardswish(self.conv(x))

class ES_SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        out = identity * x
        return out



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ES_Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ES_Bottleneck, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            ES_SEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch3 = nn.Sequential(
            GhostConv(branch_features, branch_features, 3, 1),
            ES_SEModule(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
        )

        self.branch4 = nn.Sequential(
            self.depthwise_conv(oup, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.Hardswish(inplace=True),
        )


    @staticmethod
    def depthwise_conv(i, o, kernel_size=3, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def conv1x1(i, o, kernel_size=1, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x3 = torch.cat((x1, self.branch3(x2)), dim=1)
            out = channel_shuffle(x3, 2)
        elif self.stride == 2:
            x1 = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            out = self.branch4(x1)

        return out



# build DWConvblock
# -------------------------------------------------------------------------
class DWConvblock(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, k, s):
        super(DWConvblock, self).__init__()
        self.p = k // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=s, padding=self.p, groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class ADD(nn.Module):
    # Stortcut a list of tensors along dimension
    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        x1, x2 = x[0], x[1]
        return torch.add(x1, x2, alpha=self.a)

# DWConvblock end
# -------------------------------------------------------------------------


######################################## C2f-Faster begin ########################################

from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3k_Faster(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))


class C3k2_Faster(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_Faster(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block(self.c, self.c) for _ in range(n))


class Bottleneck_PConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Partial_conv3(c1)
        self.cv2 = Partial_conv3(c2)


class C3k_PConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_PConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_PConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_PConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_PConv(self.c, self.c, shortcut, g) for _ in
            range(n))


######################################## C2f-Faster end ########################################

######################################## TransNeXt Convolutional GLU start ########################################

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class Faster_Block_CGLU(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        self.mlp = ConvolutionalGLU(dim)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3k_Faster_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block_CGLU(c_, c_) for _ in range(n)))


class C3k2_Faster_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_Faster_CGLU(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block_CGLU(self.c, self.c) for _ in
            range(n))


######################################## TransNeXt Convolutional GLU end ########################################


######################################## StartNet end ########################################

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



class C3k_Star(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))


class C3k2_Star(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_Star(self.c, self.c, 2, shortcut, g) if c3k else Star_Block(self.c) for _ in range(n))


######################################## StartNet end ########################################


######################################## Hyper-YOLO start ########################################


class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))

class MANet_FasterBlock(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))

class MANet_FasterCGLU(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block_CGLU(self.c, self.c) for _ in range(n))

class MANet_Star(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X

class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x

class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x

######################################## Hyper-YOLO end ########################################



# https://github.com/DocF/multispectral-object-detection   修改版
##################################  CFT   start ############################################
# 多头交叉注意力机制
# Multi-Head Cross Attention
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        # 断言model_dim必须能被num_heads整除
        # Assert that model_dim must be divisible by num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"

        # 可见光特征的查询、键、值线性变换
        # Linear transformations for query, key, value of visual features
        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        # 红外特征的查询、键、值线性变换
        # Linear transformations for query, key, value of infrared features
        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        # 可见光特征的输出线性变换
        # Output linear transformation for visual features
        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        # 红外特征的输出线性变换
        # Output linear transformation for infrared features
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape

        # 可见光特征生成查询、键、值
        # Generate query, key, value for visual features
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        # 红外特征生成查询、键、值
        # Generate query, key, value for infrared features
        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # 为多头注意力重塑张量
        # Reshape tensors for multi-head attention
        Q_vis = Q_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # B, N, C --> B, n_h, N, d_h
        K_vis = K_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_inf = Q_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_inf = K_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_inf = V_inf.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 交叉注意力：可见光查询与红外键，红外查询与可见光键
        # Cross attention: visual query with infrared key, infrared query with visual key
        # Q_vis 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # The shape of Q_vis is (batch_size, num_heads, seq_length, head_dim)
        # K_inf 的形状为 (batch_size, num_heads, head_dim, seq_length)
        # The shape of K_inf is (batch_size, num_heads, head_dim, seq_length)
        # 矩阵乘法后，scores_vis_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # After matrix multiplication, the shape of scores_vis_inf is (batch_size, num_heads, seq_length, seq_length)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # 计算注意力权重
        # Calculate attention weights
        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)

        # 注意力权重与值的矩阵乘法
        # Matrix multiplication of attention weights and values
        # attention_inf 的形状为 (batch_size, num_heads, seq_length, seq_length)
        # The shape of attention_inf is (batch_size, num_heads, seq_length, seq_length)
        # V_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # The shape of V_inf is (batch_size, num_heads, seq_length, head_dim)
        # out_inf 的形状为 (batch_size, num_heads, seq_length, head_dim)
        # The shape of out_inf is (batch_size, num_heads, seq_length, head_dim)
        out_inf = torch.matmul(attention_inf, V_inf)
        out_vis = torch.matmul(attention_vis, V_vis)

        # 将多头结果拼接并投影回原始维度
        # Concatenate multi-head results and project back to original dimension
        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        # 输出线性变换
        # Output linear transformation
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf


# 前向全连接网络
# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 位置编码
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置索引
        # Create position indexes
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算分母项
        # Calculate denominator terms
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))

        pe = torch.zeros(max_len, model_dim)  # 初始化位置编码矩阵 有需要可以采用更多编码，目前只采用了最基础的位置编码
        # Initialize positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用sin函数
        # Even columns use sine function
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用cos函数
        # Odd columns use cosine function

        pe = pe.unsqueeze(0)  # 添加批量维度
        # Add batch dimension
        self.register_buffer('pe', pe)  # 注册为模型缓冲区
        # Register as model buffer

    def forward(self, x):
        # 将位置编码添加到输入中
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 编码器层
# Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        # 交叉注意力机制
        # Cross attention mechanism
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        # 残差连接与归一化
        # Residual connection and normalization
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)

        # 前向全连接网络
        # Feed-forward network
        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)

        # 残差连接与归一化
        # Residual connection and normalization
        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)

        return vis, inf


# Transformer编码器
# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        # 嵌入层
        # Embedding layer
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))

        # 位置编码
        # Positional encoding
        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)

        # 多层编码器
        # Multiple encoder layers
        for layer in self.layers:
            vis, inf = layer(vis, inf)

        return vis, inf


# 交叉注意力
# CrossTransformerFusion
class CrossTransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        # 输入形状为 B, C, H, W
        # Input shape is B, C, H, W
        B, C, H, W = vis.shape

        # 将输入变形为 B, H*W, C
        # Reshape input to B, H*W, C
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)

        # 输入Transformer编码器
        # Input to Transformer encoder
        vis_out, inf_out = self.encoder(vis, inf)

        # 将输出变形为 B, C, H, W
        # Reshape output to B, C, H, W
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        # 在通道维度上进行级联
        # Concatenate on channel dimension
        out = torch.cat((vis_out, inf_out), dim=1)

        return out


##################################  CFT   end ############################################

# https://github.com/DocF/multispectral-object-detection   原始版本
#-------------------------------  GPT  -----------------------------------------------------


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x



class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        # print(rgb_fea.shape,  ir_fea.shape)
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return (rgb_fea_out, ir_fea_out)



class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])

#------------------------------------------- GPT end---------------------------------------


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class SpatialAARMAdapter(nn.Module):
    """
    将AARM概念适配到空间特征图的处理
    """

    def __init__(self, c1, c2, *args):  # 接受额外参数以适配YOLOv5参数传递
        super(SpatialAARMAdapter, self).__init__()

        self.c1 = c1
        self.c2 = c2

        # 特征转换层
        self.query_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.key_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.value_conv = nn.Conv2d(c1, c2, kernel_size=1)

        # 输出转换层
        self.output_conv = nn.Conv2d(c2, c2, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # 生成查询、键和值
        query = self.query_conv(x).view(batch_size, self.c2, -1)  # B x C x HW
        key = self.key_conv(x).view(batch_size, self.c2, -1).permute(0, 2, 1)  # B x HW x C
        value = self.value_conv(x).view(batch_size, self.c2, -1)  # B x C x HW

        # 计算注意力得分和权重
        attention = torch.bmm(query, key)  # B x C x C
        attention = F.softmax(attention, dim=2)

        # 重构特征
        out = torch.bmm(attention, value)  # B x C x HW
        out = out.view(batch_size, self.c2, H, W)
        out = self.output_conv(out)

        # 残差连接
        out = out + x

        return out


class AARM(nn.Module):
    """
    Attention-based Appearance Reconstruction Module (AARM)
    根据提供的架构图重新设计的模块，用于重构外观特征以增强区分能力
    """

    def __init__(self, c1, c2, *args):  # 修改参数列表以适配YOLOv5
        """
        初始化AARM模块

        参数:
            c1: 输入通道数/特征维度
            c2: 输出通道数/特征维度
            args: 额外参数
        """
        super(AARM, self).__init__()

        # 特征映射卷积层（适用于空间特征图）
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)

        # 使用SpatialAARMAdapter处理卷积特征图
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # 应用卷积调整通道数
        x = self.conv(x)

        # 应用空间AARM模块
        out = self.spatial_aarm(x)

        return out


class AARMC3(nn.Module):
    """
    AARM模块与YOLOv5 C3模块的融合
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, *args):
        """
        初始化AARMC3模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
        """
        super(AARMC3, self).__init__()

        c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and c_ % g != 0:
            c_ = math.ceil(c_ / g) * g

        # C3模块的组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 创建Bottleneck序列，确保通道数兼容
        m = []
        for _ in range(n):
            m.append(Bottleneck(c_, c_, shortcut, g, e=1.0))
        self.m = nn.Sequential(*m)

        # AARM模块
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # C3模块的处理
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), 1))

        # AARM模块处理
        out = self.spatial_aarm(out)

        return out


class AARMSPPF(nn.Module):
    """
    AARM模块与SPPF模块的融合
    """

    def __init__(self, c1, c2, k=5, *args):
        """
        初始化AARMSPPF模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: SPPF的kernel size
        """
        super(AARMSPPF, self).__init__()

        c_ = c1 // 2  # 隐藏通道数

        # SPPF模块的组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 确保池化保持空间尺寸不变
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 空间AARM模块
        self.spatial_aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # SPPF模块的处理
        x = self.cv1(x)

        # 保存输入尺寸
        _, _, h, w = x.shape

        # 应用最大池化，并确保输出尺寸与输入相同
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制torch 1.9.0 max_pool2d()警告
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)

            # 确保所有特征具有相同的空间尺寸
            if y1.shape[2:] != x.shape[2:]:
                y1 = F.interpolate(y1, size=(h, w), mode='nearest')
            if y2.shape[2:] != x.shape[2:]:
                y2 = F.interpolate(y2, size=(h, w), mode='nearest')
            if y3.shape[2:] != x.shape[2:]:
                y3 = F.interpolate(y3, size=(h, w), mode='nearest')

            # 连接特征
            out = self.cv2(torch.cat((x, y1, y2, y3), 1))

        # AARM模块处理
        out = self.spatial_aarm(out)

        return out

class C2f_AARM(nn.Module):
    """
    C2f模块与AARM的融合版本
    结合C2f的高效特征融合和AARM的注意力增强机制
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
        """
        super(C2f_AARM, self).__init__()

        self.c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f的核心组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)  # 输入分支卷积
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # 输出融合卷积

        # Bottleneck序列
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # AARM注意力增强模块 - 应用于最终输出
        self.aarm = SpatialAARMAdapter(c2, c2)

        # 可选：在中间特征上也应用轻量级注意力
        self.intermediate_aarm = SpatialAARMAdapter(self.c_, self.c_)
        self.use_intermediate_aarm = True  # 控制是否在中间特征使用AARM

    def forward(self, x):
        """前向传播"""
        # C2f的特征分割和处理
        y = list(self.cv1(x).chunk(2, 1))  # 分割为两个分支

        # 逐步添加Bottleneck处理的特征
        for i, m in enumerate(self.m):
            if self.use_intermediate_aarm and i == len(self.m) // 2:
                # 在中间层应用AARM增强
                enhanced_feature = self.intermediate_aarm(y[-1])
                y.append(m(enhanced_feature))
            else:
                y.append(m(y[-1]))

        # 融合所有特征
        fused_features = self.cv2(torch.cat(y, 1))

        # 应用AARM进行最终的注意力增强
        enhanced_output = self.aarm(fused_features)

        return enhanced_output


class C2f_AARM_Lite(nn.Module):
    """
    C2f_AARM的轻量化版本
    仅在输出层应用AARM，减少计算开销
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM_Lite模块

        参数同C2f_AARM
        """
        super(C2f_AARM_Lite, self).__init__()

        self.c_ = int(c2 * e)  # 隐藏通道数

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f的核心组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # 仅在输出应用AARM
        self.aarm = SpatialAARMAdapter(c2, c2)

    def forward(self, x):
        """前向传播"""
        # 标准C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # 应用AARM增强
        enhanced_output = self.aarm(fused_features)

        return enhanced_output


class C2f_AARM_Adaptive(nn.Module):
    """
    C2f_AARM的自适应版本
    根据特征尺寸动态调整AARM的应用策略
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, *args):
        """
        初始化C2f_AARM_Adaptive模块
        """
        super(C2f_AARM_Adaptive, self).__init__()

        self.c_ = int(c2 * e)

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = math.ceil(self.c_ / g) * g

        # C2f组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n))

        # 多尺度AARM模块
        self.aarm_output = SpatialAARMAdapter(c2, c2)

        # 可学习的权重参数，用于平衡原始特征和增强特征
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """前向传播"""
        # C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # 自适应特征增强
        enhanced_features = self.aarm_output(fused_features)

        # 可学习的特征融合
        output = self.alpha * enhanced_features + (1 - self.alpha) * fused_features

        return output


# ==================== TripleMixer Module ====================

import torch
import warnings
import numpy as np
from torch import autocast
from .wavelet import LiftingScheme2D


class myLayerNorm(nn.LayerNorm):
    """Custom LayerNorm that works with channel-first tensors"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


def build_proj_matrix(indices_non_zeros, occupied_cell, batch_size, num_2d_cells, inflate_ind, channels):
    """Build projection matrix for spatial mixing"""
    num_points = indices_non_zeros.shape[1] // batch_size
    matrix_shape = (batch_size, num_2d_cells, num_points)

    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)
    
    inflate_ind = inflate_ind.unsqueeze(1).expand(-1, channels, -1)
    
    with autocast("cuda", enabled=False):
        num_points_per_cells = torch.bmm(
            inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1))
        )
        
    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate_ind}


class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class ChaMixer(nn.Module):
    """Channel Mixer for TripleMixer"""
    def __init__(self, channels, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.layer_norm = layer_norm
        if layer_norm:
           self.norm = myLayerNorm(channels)
        else:
           self.norm = nn.BatchNorm1d(channels)
        
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  
        
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))


class FreMixer(nn.Module):
    """Frequency Mixer for TripleMixer"""
    def __init__(self, channels, grid_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        self.num_levels = 2
        self.mult = 2
        self.dropout = 0.5
        if layer_norm:
            self.norm = myLayerNorm(channels)
        else:
            self.norm = nn.BatchNorm1d(channels)
            
        self.reduction = nn.Conv2d(channels, int(channels/4), 1)
        
        self.wavelet1 = LiftingScheme2D(in_planes=int(channels/4), share_weights=True)
        self.wavelet2 = LiftingScheme2D(in_planes=int(channels/4), share_weights=True)
        
        self.feedforward2 = nn.Sequential(
                nn.Conv2d(channels, channels,1),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, 1),
                nn.ConvTranspose2d(channels, int(channels/2), 4, stride=2, padding=1),
                nn.BatchNorm2d(int(channels/2))
            )
        
        self.feedforward1 = nn.Sequential(
                nn.Conv2d(channels + int(channels/2), channels,1),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, 1),
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels)
            )   
        
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )   
        
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        with autocast("cuda", enabled=False):
            residual = torch.bmm(
                sp_mat["flatten"], residual.transpose(1, 2).float()
            ).transpose(1, 2)
        # 使用实际的空间尺寸而不是grid_shape
        # 计算实际的空间尺寸
        # 从tokens的总元素数计算空间尺寸
        total_elements = residual.numel()
        spatial_size = int((total_elements / (B * C)) ** 0.5)  # 计算实际的空间尺寸
        residual = residual.reshape(B, C, spatial_size, spatial_size)
        
        residual_re = self.reduction(residual)
        
        # 检查输入尺寸是否适合小波变换
        B, C, H, W = residual_re.shape
        if H >= 8 and W >= 8:  # 确保有足够的尺寸进行小波变换
            # 使用小波变换
            _, _, LL1, LH1, HL1, HH1 = self.wavelet1(residual_re)
            _, _, LL2, LH2, HL2, HH2 = self.wavelet2(LL1)
        else:
            # 如果尺寸太小，使用简化的频域分解
            if H >= 2 and W >= 2:
                # 简单的四分量分解
                LL1 = residual_re[:, :, ::2, ::2]  # Low-Low
                LH1 = residual_re[:, :, ::2, 1::2]  # Low-High
                HL1 = residual_re[:, :, 1::2, ::2]  # High-Low
                HH1 = residual_re[:, :, 1::2, 1::2]  # High-High
                
                # 第二级分解
                if LL1.shape[2] >= 2 and LL1.shape[3] >= 2:
                    LL2 = LL1[:, :, ::2, ::2]
                    LH2 = LL1[:, :, ::2, 1::2]
                    HL2 = LL1[:, :, 1::2, ::2]
                    HH2 = LL1[:, :, 1::2, 1::2]
                else:
                    LL2 = LH2 = HL2 = HH2 = LL1
            else:
                # 如果太小，使用原始输入
                LL1 = LH1 = HL1 = HH1 = residual_re
                LL2 = LH2 = HL2 = HH2 = residual_re
        
        # 确保所有小波分量具有相同的尺寸
        def resize_to_match(tensor, target_shape):
            """将张量调整到目标尺寸"""
            if tensor.shape[2:] == target_shape[2:]:
                return tensor
            else:
                # 使用插值调整尺寸
                return torch.nn.functional.interpolate(tensor, size=target_shape[2:], mode='bilinear', align_corners=False)
        
        # 以LL1的尺寸为基准
        target_shape = LL1.shape
        LL2 = resize_to_match(LL2, target_shape)
        LH2 = resize_to_match(LH2, target_shape)
        HL2 = resize_to_match(HL2, target_shape)
        HH2 = resize_to_match(HH2, target_shape)
        
        # 确保LH1, HL1, HH1与LL1尺寸一致
        LH1 = resize_to_match(LH1, target_shape)
        HL1 = resize_to_match(HL1, target_shape)
        HH1 = resize_to_match(HH1, target_shape)
        
        x2_wavel=torch.cat([LL2,LH2,HL2,HH2],1)
        residual_wavel2 = self.feedforward2(x2_wavel)
        
        x1_wavel=torch.cat([LL1,LH1,HL1,HH1],1) 
        
        # 确保residual_wavel2与x1_wavel的尺寸匹配
        if residual_wavel2.shape[2:] != x1_wavel.shape[2:]:
            residual_wavel2 = torch.nn.functional.interpolate(
                residual_wavel2, 
                size=x1_wavel.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        x1_wavel = torch.cat((x1_wavel,residual_wavel2), 1)
        residual_wavel1 = self.feedforward1(x1_wavel)
        
        residual = residual + residual_wavel1
        
        # FFN
        residual = self.ffn(residual)
        # LayerScale
        # 使用residual的实际尺寸而不是grid_shape
        residual_h, residual_w = residual.shape[2], residual.shape[3]
        # 使用residual的实际通道数，而不是reduction后的C
        residual_channels = residual.shape[1]
        residual = residual.reshape(B, residual_channels, residual_h * residual_w)
        residual = self.scale(residual)
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)


class GeoMixer(nn.Module):
    """Geometry Mixer for TripleMixer"""
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.channels_in, self.channels_out = channels_in, channels_out
        self.norm = nn.BatchNorm1d(channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_out, 1)

        self.fc = nn.Conv2d(2 * channels_in, 2 * channels_in, (1, 1), bias=False)
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(2 * channels_in),
            nn.Conv2d(2 * channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)

    def forward(self, x, neighbors):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""        
        x = self.norm(x)
        point_emb = self.conv1(x)
        
        gather = []
        for ind_nn in range(1, neighbors.shape[1]):  
            temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, x.shape[1], -1)
            gather.append(torch.gather(x, 2, temp).unsqueeze(-1))

        neigh_fea = torch.cat(gather, -1)        
        neigh_emb = neigh_fea - x.unsqueeze(-1)  
         
        neigh_embfea = torch.cat([neigh_fea, neigh_emb], dim=1)
        
        att_activation = self.fc(neigh_embfea)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = neigh_embfea * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)        
        
        finl_emb = self.conv2(f_agg).max(-1)[0]
                
        return self.final(torch.cat((point_emb, finl_emb), dim=1))


class Mixer(nn.Module):
    """Main Mixer combining ChaMixer and FreMixer"""
    def __init__(self, channels, depth, grids_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.channel_mix = nn.ModuleList(
            [ChaMixer(channels, drop_path_prob, layer_norm) for _ in range(depth)]
        )
        self.spatial_mix = nn.ModuleList(
            [
                FreMixer(channels, grids_shape[d % len(grids_shape)], drop_path_prob, layer_norm)
                for d in range(depth)
            ]
        )

    def forward(self, tokens, cell_ind, occupied_cell):
        batch_size, num_points = tokens.shape[0], tokens.shape[-1]
        
        point_ind = (
            torch.arange(num_points, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(1, -1)
        )
        
        batch_ind = (
            torch.arange(batch_size, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, num_points)
            .reshape(1, -1)
        )
        
        non_zeros_ind = []
        for i in range(cell_ind.shape[1]):
            non_zeros_ind.append(
                torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
            )
            
        sp_mat = [
            build_proj_matrix(
                id,
                occupied_cell,
                batch_size,
                np.prod(sh),
                cell_ind[:, i],
                tokens.shape[1],
            )
            for i, (id, sh) in enumerate(zip(non_zeros_ind, self.grids_shape))
        ]
        
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)
        return tokens


class TripleMixer(nn.Module):
    """
    TripleMixer: A unified module combining Channel, Frequency, and Geometry mixers
    
    This module integrates three different mixing strategies:
    1. ChaMixer: Channel-wise mixing for feature refinement
    2. FreMixer: Frequency domain mixing using wavelet transforms
    3. GeoMixer: Geometry-aware mixing for spatial relationships
    
    Args:
        channels (int): Number of input/output channels
        depth (int): Depth of the mixer layers
        grids_shape (list): List of grid shapes for spatial mixing
        drop_path_prob (float): Drop path probability for regularization
        layer_norm (bool): Whether to use layer normalization
        use_geo_mixer (bool): Whether to include geometry mixer
    """
    
    def __init__(self, channels, depth=2, grids_shape=None, drop_path_prob=0.1, 
                 layer_norm=False, use_geo_mixer=True):
        super().__init__()
        
        self.channels = channels
        self.depth = depth
        self.use_geo_mixer = use_geo_mixer
        
        # Default grid shapes if not provided
        if grids_shape is None:
            grids_shape = [(8, 8), (16, 16)]
        
        self.grids_shape = grids_shape
        
        # Initialize the main mixer (ChaMixer + FreMixer)
        self.mixer = Mixer(channels, depth, grids_shape, drop_path_prob, layer_norm)
        
        # Initialize geometry mixer if enabled
        if use_geo_mixer:
            self.geo_mixer = GeoMixer(channels, channels)
        
        # Feature fusion layer
        self.fusion = nn.Conv2d(channels, channels, 1) if use_geo_mixer else nn.Identity()
        
        # Learnable weights for different components
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1)) if use_geo_mixer else None

    def forward(self, x):
        """
        Forward pass of TripleMixer
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape for token-based processing
        tokens = x.view(B, C, H * W)
        
        # Create dummy cell indices and occupied cells for spatial mixing
        # This is a simplified version - in practice, you might want to use actual spatial indices
        cell_ind = torch.zeros(B, 1, H * W, dtype=torch.long, device=x.device)
        occupied_cell = torch.ones(B, H * W, device=x.device)
        
        # Apply main mixer (ChaMixer + FreMixer)
        mixed_tokens = self.mixer(tokens, cell_ind, occupied_cell)
        
        # Reshape back to spatial format
        mixed_features = mixed_tokens.view(B, C, H, W)
        
        if self.use_geo_mixer:
            # Apply geometry mixer
            # Create dummy neighbors for geometry mixing
            # In practice, you would compute actual spatial neighbors
            neighbors = torch.zeros(B, 3, H * W, dtype=torch.long, device=x.device)
            for i in range(H * W):
                neighbors[:, 0, i] = i  # self
                if i > 0:
                    neighbors[:, 1, i] = i - 1  # left neighbor
                if i < H * W - 1:
                    neighbors[:, 2, i] = i + 1  # right neighbor
            
            geo_tokens = mixed_tokens
            geo_features = self.geo_mixer(geo_tokens, neighbors)
            geo_features = geo_features.view(B, C, H, W)
            
            # Fuse features
            fused_features = self.fusion(mixed_features + geo_features)
            
            # Apply learnable weights
            output = self.alpha * mixed_features + self.beta * geo_features
        else:
            output = mixed_features
            
        return output


class TripleMixerBlock(nn.Module):
    """
    TripleMixer Block for YOLOv11 integration
    
    This block can be easily integrated into YOLOv11's architecture
    as a replacement for attention mechanisms or as an additional feature enhancement module.
    """
    
    def __init__(self, c1, c2, depth=2, grids_shape=None, drop_path_prob=0.1, 
                 layer_norm=False, use_geo_mixer=True):
        super().__init__()
        
        # Input projection
        self.input_proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        
        # TripleMixer module
        self.triple_mixer = TripleMixer(
            channels=c2,
            depth=depth,
            grids_shape=grids_shape,
            drop_path_prob=drop_path_prob,
            layer_norm=layer_norm,
            use_geo_mixer=use_geo_mixer
        )
        
        # Output projection
        self.output_proj = Conv(c2, c2, 1)
        
        # Residual connection
        self.residual = c1 == c2

    def forward(self, x):
        """Forward pass through TripleMixer Block"""
        identity = x
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply TripleMixer
        x = self.triple_mixer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Residual connection
        if self.residual:
            x = x + identity
            
        return x


# ==================== Liquid Neural Network Modules ====================

class C2Liquid(nn.Module):
    """
    C2f模块与液态神经网络（LNN）的融合版本
    类似C2AARM结构，在C2f末尾结合LNN模块，不修改bottleneck
    设计目标：提高MAP50, MAP75, MAP50:95，同时降低参数量和FLOPs
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 use_lnn=True, lnn_hidden_dim=None, use_lightweight=True,
                 use_adaptive_fusion=True, *args):
        """
        初始化C2Liquid模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
            use_lnn: 是否使用LNN增强
            lnn_hidden_dim: LNN隐藏维度（None则自动计算）
            use_lightweight: 是否使用轻量级LNN设计
            use_adaptive_fusion: 是否使用自适应融合
        """
        super(C2Liquid, self).__init__()

        # 确保所有参数都是正确的类型
        c1 = max(1, int(c1))
        c2 = max(1, int(c2))
        n = max(1, int(n))
        g = max(1, int(g))
        e = max(0.1, float(e))  # 确保e至少为0.1，避免c_为0

        self.c_ = max(1, int(c2 * e))  # 隐藏通道数，确保至少为1

        # 确保通道数能被分组数整除
        if g > 1 and self.c_ % g != 0:
            self.c_ = max(1, math.ceil(self.c_ / g) * g)  # 确保至少为1

        # C2f的核心组件（不修改bottleneck）
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)  # 输入分支卷积
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)  # 输出融合卷积

        # Bottleneck序列（保持原样）
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        # 液态神经网络模块 - 应用于最终输出（类似C2AARM的AARM位置）
        self.use_lnn = use_lnn
        if use_lnn:
            # 自动计算LNN隐藏维度（使用较小的值以降低参数量）
            if lnn_hidden_dim is None:
                lnn_hidden_dim = max(8, int(c2 * 0.5))  # 使用输出通道数的一半

            self.lnn = LiquidNeuralModule(
                in_channels=c2,
                out_channels=c2,
                hidden_dim=lnn_hidden_dim,
                use_lightweight=use_lightweight,
                use_adaptive_fusion=use_adaptive_fusion
            )

        # 用于存储隐藏状态（用于时序建模，可选）
        self.register_buffer('h_prev', None)
        self.use_temporal = False  # 是否使用时序建模

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            out: 增强后的特征 [B, c2, H, W]
        """
        # C2f的标准处理流程（不修改bottleneck）
        y = list(self.cv1(x).chunk(2, 1))  # 分割为两个分支
        y.extend(m(y[-1]) for m in self.m)  # 通过Bottleneck序列

        # 融合所有特征
        fused_features = self.cv2(torch.cat(y, 1))  # [B, c2, H, W]

        # 应用液态神经网络增强（类似C2AARM的AARM增强）
        if self.use_lnn:
            # 使用隐藏状态（如果启用时序建模）
            h_state = self.h_prev if self.use_temporal else None
            enhanced_output, h_t = self.lnn(fused_features, h_state)

            # 更新隐藏状态
            if self.use_temporal:
                self.h_prev = h_t.detach()  # 分离梯度，避免反向传播到历史状态

            return enhanced_output
        else:
            return fused_features

    def reset_hidden_state(self):
        """重置隐藏状态（用于新的序列开始）"""
        self.h_prev = None


class C2Liquid_Lite(nn.Module):
    """
    C2Liquid的轻量级版本
    进一步降低参数量和FLOPs，适合资源受限场景
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 lnn_hidden_dim_ratio=0.5, *args):
        """
        初始化C2Liquid_Lite模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 隐藏层通道数比例
            lnn_hidden_dim_ratio: LNN隐藏维度比例（降低参数量）
        """
        super(C2Liquid_Lite, self).__init__()

        # 确保所有参数都是正确的类型
        c1 = max(1, int(c1))
        c2 = max(1, int(c2))
        n = max(1, int(n))
        g = max(1, int(g))
        e = max(0.1, float(e))  # 确保e至少为0.1
        lnn_hidden_dim_ratio = max(0.1, float(lnn_hidden_dim_ratio))

        self.c_ = max(1, int(c2 * e))  # 确保至少为1

        if g > 1 and self.c_ % g != 0:
            self.c_ = max(1, math.ceil(self.c_ / g) * g)  # 确保至少为1

        # C2f组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        # 轻量级LNN模块
        self.lnn = LiquidNeuralModuleLite(
            in_channels=c2,
            out_channels=c2,
            hidden_dim_ratio=lnn_hidden_dim_ratio
        )

    def forward(self, x):
        """前向传播"""
        # C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # 轻量级LNN增强
        enhanced_output, h_t = self.lnn(fused_features, None)

        return enhanced_output


class C2Liquid_Adaptive(nn.Module):
    """
    C2Liquid的自适应版本
    根据特征动态调整LNN的应用策略，类似C2AARM_Adaptive
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 use_lightweight=True, *args):
        """
        初始化C2Liquid_Adaptive模块
        """
        super(C2Liquid_Adaptive, self).__init__()

        # 确保所有参数都是正确的类型
        c1 = max(1, int(c1))
        c2 = max(1, int(c2))
        n = max(1, int(n))
        g = max(1, int(g))
        e = max(0.1, float(e))  # 确保e至少为0.1

        self.c_ = max(1, int(c2 * e))  # 确保至少为1

        if g > 1 and self.c_ % g != 0:
            self.c_ = max(1, math.ceil(self.c_ / g) * g)  # 确保至少为1

        # C2f组件
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        # LNN模块
        lnn_hidden_dim = max(8, int(c2 * 0.5))
        self.lnn = LiquidNeuralModule(
            in_channels=c2,
            out_channels=c2,
            hidden_dim=lnn_hidden_dim,
            use_lightweight=use_lightweight,
            use_adaptive_fusion=True  # 强制使用自适应融合
        )

        # 额外的可学习权重参数（类似C2AARM_Adaptive的alpha）
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """前向传播"""
        # C2f处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        fused_features = self.cv2(torch.cat(y, 1))

        # LNN增强
        enhanced_features, h_t = self.lnn(fused_features, None)

        # 自适应融合（结合LNN内部融合和外部可学习权重）
        alpha = torch.sigmoid(self.alpha)  # 限制在[0, 1]
        output = alpha * enhanced_features + (1 - alpha) * fused_features

        return output


class LiquidSPPF(nn.Module):
    """
    SPPF模块与液态神经网络的融合
    类似AARMSPPF结构，在SPPF末尾结合LNN模块
    """

    def __init__(self, c1, c2, k=5, use_lnn=False, lnn_hidden_dim=None,
                 use_lightweight=True, use_adaptive_fusion=True, *args):
        """
        初始化LiquidSPPF模块

        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: SPPF的kernel size
            use_lnn: 是否使用LNN增强
            lnn_hidden_dim: LNN隐藏维度
            use_lightweight: 是否使用轻量级设计
            use_adaptive_fusion: 是否使用自适应融合
        """
        super(LiquidSPPF, self).__init__()

        # 确保所有参数都是正确的类型
        c1 = max(1, int(c1))
        c2 = max(1, int(c2))
        k = max(1, int(k))

        c_ = max(1, c1 // 2)  # 隐藏通道数，确保至少为1

        # SPPF模块的组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 确保池化保持空间尺寸不变
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 液态神经网络模块
        self.use_lnn = use_lnn
        if use_lnn:
            if lnn_hidden_dim is None:
                lnn_hidden_dim = max(8, int(c2 * 0.5))

            self.lnn = LiquidNeuralModule(
                in_channels=c2,
                out_channels=c2,
                hidden_dim=lnn_hidden_dim,
                use_lightweight=use_lightweight,
                use_adaptive_fusion=use_adaptive_fusion
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            out: 增强后的特征 [B, c2, H, W]
        """
        # SPPF模块的处理
        x = self.cv1(x)

        # 保存输入尺寸
        _, _, h, w = x.shape

        # 应用最大池化，并确保输出尺寸与输入相同
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制torch 1.9.0 max_pool2d()警告
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)

            # 确保所有特征具有相同的空间尺寸
            if y1.shape[2:] != x.shape[2:]:
                y1 = F.interpolate(y1, size=(h, w), mode='nearest')
            if y2.shape[2:] != x.shape[2:]:
                y2 = F.interpolate(y2, size=(h, w), mode='nearest')
            if y3.shape[2:] != x.shape[2:]:
                y3 = F.interpolate(y3, size=(h, w), mode='nearest')

            # 连接特征
            out = self.cv2(torch.cat((x, y1, y2, y3), 1))

        # LNN模块处理（类似AARMSPPF的AARM处理）
        if self.use_lnn:
            enhanced_out, h_t = self.lnn(out, None)
            return enhanced_out
        else:
            return out


class LiquidSPPF_Lite(nn.Module):
    """
    LiquidSPPF的轻量级版本
    """

    def __init__(self, c1, c2, k=5, lnn_hidden_dim_ratio=0.25, *args):
        """
        初始化LiquidSPPF_Lite模块
        """
        super(LiquidSPPF_Lite, self).__init__()

        # 确保所有参数都是正确的类型
        c1 = max(1, int(c1))
        c2 = max(1, int(c2))
        k = max(1, int(k))
        lnn_hidden_dim_ratio = max(0.1, float(lnn_hidden_dim_ratio))

        c_ = max(1, c1 // 2)  # 确保至少为1

        # SPPF组件
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 轻量级LNN
        self.lnn = LiquidNeuralModuleLite(
            in_channels=c2,
            out_channels=c2,
            hidden_dim_ratio=lnn_hidden_dim_ratio
        )

    def forward(self, x):
        """前向传播"""
        x = self.cv1(x)
        _, _, h, w = x.shape

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)

            if y1.shape[2:] != x.shape[2:]:
                y1 = F.interpolate(y1, size=(h, w), mode='nearest')
            if y2.shape[2:] != x.shape[2:]:
                y2 = F.interpolate(y2, size=(h, w), mode='nearest')
            if y3.shape[2:] != x.shape[2:]:
                y3 = F.interpolate(y3, size=(h, w), mode='nearest')

            out = self.cv2(torch.cat((x, y1, y2, y3), 1))

        enhanced_out, h_t = self.lnn(out, None)
        return enhanced_out



from timm.layers import DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint


class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
                conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        # if deploy:
        #     print('------------------------------- Note: deploy mode')
        # if self.with_cp:
        #     print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):

        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)

    def switch_to_deploy(self):
        if hasattr(self.dwconv, 'switch_to_deploy'):
            self.dwconv.switch_to_deploy()
        if hasattr(self.norm, 'running_var') and hasattr(self.dwconv, 'lk_origin'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
            self.dwconv.lk_origin.bias.data = self.norm.bias + (
                        self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


class C3_UniRepLKNetBlock(C3):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(UniRepLKNetBlock(c_, k) for _ in range(n)))


class C2f_UniRepLKNetBlock(C2f):
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(UniRepLKNetBlock(self.c, k) for _ in range(n))


class Bottleneck_DRB(Bottleneck):
    """Standard bottleneck with DilatedReparamBlock."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DilatedReparamBlock(c2, 7)


class C3_DRB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DRB(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


class C2f_DRB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DRB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))