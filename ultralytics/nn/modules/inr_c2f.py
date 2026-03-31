# Ultralytics YOLO 🚀, AGPL-3.0 license
"""INR-Enhanced C2f modules for small object detection in drone datasets."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
from .block import Bottleneck


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for INR coordinate representation."""

    def __init__(self, input_dim=2, encoding_dim=64, include_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.include_input = include_input

        # 计算频率（不硬编码dtype，允许后续类型转换）
        self.num_frequencies = encoding_dim // (2 * input_dim)
        frequencies = 2.0 ** torch.linspace(0, self.num_frequencies - 1, self.num_frequencies)
        self.register_buffer('frequencies', frequencies)

        # 输出维度：sin和cos编码 + 原始输入（如果包含）
        self.output_dim = 2 * input_dim * self.num_frequencies + (input_dim if include_input else 0)

    def forward(self, coords):
        """
        Args:
            coords: [B, H, W, 2] 归一化坐标 [-1, 1]
        Returns:
            encoded: [B, H, W, output_dim] 编码后的坐标特征
        """
        # 确保频率与输入坐标类型一致
        frequencies = self.frequencies.to(coords.dtype)

        B, H, W, _ = coords.shape
        coords_expanded = coords.unsqueeze(-1) * frequencies.view(1, 1, 1, 1, -1)  # [B, H, W, 2, num_freq]
        coords_expanded = coords_expanded.reshape(B, H, W, -1)  # [B, H, W, 2*num_freq]

        # 应用sin和cos
        encoded = torch.cat([
            torch.sin(coords_expanded),
            torch.cos(coords_expanded)
        ], dim=-1)  # [B, H, W, encoding_dim]

        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)

        return encoded


class MultiScaleCoordinateEncoder(nn.Module):
    """多尺度坐标编码器，用于捕捉不同尺度的空间信息."""

    def __init__(self, coord_dim=2, hidden_dim=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim

        # 不同尺度的频率编码
        self.scale_encoders = nn.ModuleList([
            PositionalEncoding(coord_dim, hidden_dim // num_scales, include_input=(i == 0))
            for i in range(num_scales)
        ])

        # 融合网络（不硬编码dtype）
        total_dim = sum(enc.output_dim for enc in self.scale_encoders)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, coords):
        """
        Args:
            coords: [B, H, W, 2] 归一化坐标
        Returns:
            encoded: [B, H, W, hidden_dim] 多尺度编码特征
        """
        # 确保融合网络与输入类型一致
        self.fusion[0].weight = nn.Parameter(self.fusion[0].weight.to(coords.dtype))
        self.fusion[0].bias = nn.Parameter(self.fusion[0].bias.to(coords.dtype))
        self.fusion[2].weight = nn.Parameter(self.fusion[2].weight.to(coords.dtype))
        self.fusion[2].bias = nn.Parameter(self.fusion[2].bias.to(coords.dtype))

        encoded_features = []
        for i, encoder in enumerate(self.scale_encoders):
            scale_factor = 2 ** i
            scaled_coords = coords * scale_factor
            encoded = encoder(scaled_coords)
            encoded_features.append(encoded)

        multi_scale_features = torch.cat(encoded_features, dim=-1)
        fused_features = self.fusion(multi_scale_features)

        return fused_features


class INRFeatureEnhancer(nn.Module):
    """INR特征增强器，用于增强小目标特征表示."""

    def __init__(self, feature_dim, coord_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim

        # 坐标编码器
        self.coord_encoder = MultiScaleCoordinateEncoder(coord_dim, hidden_dim)

        # 特征-坐标融合网络（不硬编码dtype）
        input_dim = feature_dim + hidden_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU()
                ])
            elif i == num_layers - 1:
                layers.extend([
                    nn.Linear(hidden_dim, feature_dim),
                    nn.Sigmoid()  # 输出注意力权重
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU()
                ])

        self.fusion_net = nn.Sequential(*layers)

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, features, coords):
        """
        Args:
            features: [B, C, H, W] 输入特征
            coords: [B, H, W, 2] 归一化坐标
        Returns:
            enhanced_features: [B, C, H, W] 增强后的特征
        """
        # 确保融合网络与输入特征类型一致
        for layer in self.fusion_net:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.Parameter(layer.weight.to(features.dtype))
                layer.bias = nn.Parameter(layer.bias.to(features.dtype))

        B, C, H, W = features.shape

        # 坐标编码
        coord_features = self.coord_encoder(coords)  # [B, H, W, hidden_dim]

        # 特征重塑
        features_flat = features.permute(0, 2, 3, 1)  # [B, H, W, C]

        # 特征-坐标融合
        fusion_input = torch.cat([features_flat, coord_features], dim=-1)
        attention_weights = self.fusion_net(fusion_input)  # [B, H, W, C]

        # 应用注意力权重
        enhanced_features = features_flat * attention_weights

        # 残差连接（关键修改：保持Parameter类型）
        residual_weight = self.residual_weight.to(enhanced_features.dtype)  # 先转换为对应类型的张量
        enhanced_features = enhanced_features + residual_weight * features_flat  # 使用转换后的张量计算

        # 重塑回原始形状
        enhanced_features = enhanced_features.permute(0, 3, 1, 2)  # [B, C, H, W]

        return enhanced_features


class SmallObjectAttention(nn.Module):
    """小目标注意力机制，专门针对微小目标进行特征增强."""

    def __init__(self, channels, reduction=16, min_object_size=8):
        super().__init__()
        self.min_object_size = min_object_size

        # 确保channels是正整数
        channels = int(channels)
        reduction = max(1, int(reduction))  # 确保reduction至少为1
        reduced_channels = max(1, channels // reduction)  # 确保reduced_channels至少为1

        # 全局注意力
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

        # 局部细节注意力
        self.local_attn = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

        # 高频细节增强
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            enhanced_x: [B, C, H, W] 增强后的特征
        """
        # 确保卷积层与输入类型一致
        for seq in [self.global_attn, self.local_attn, self.detail_enhance]:
            for layer in seq:
                if isinstance(layer, nn.Conv2d):
                    layer.weight = nn.Parameter(layer.weight.to(x.dtype))
                    layer.bias = nn.Parameter(layer.bias.to(x.dtype))

        # 全局注意力
        global_attn = self.global_attn(x)

        # 局部注意力
        local_attn = self.local_attn(x)

        # 高频细节增强
        detail_attn = self.detail_enhance(x)

        # 多尺度注意力融合
        enhanced_x = x * (global_attn + local_attn + detail_attn) / 3.0

        return enhanced_x


class INREnhancedC2f(nn.Module):
    """INR增强的C2f模块，专门针对无人机小目标检测优化."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 use_inr=True, use_attention=True, coord_encoding_dim=64):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck层数
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 扩展因子
            use_inr: 是否使用INR增强
            use_attention: 是否使用小目标注意力
            coord_encoding_dim: 坐标编码维度
        """
        super().__init__()

        # 确保所有参数都是正确的类型
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = max(1, int(g))  # 确保g至少为1，避免groups=0的错误
        e = float(e)

        self.c = max(1, int(c2 * e))  # hidden channels，确保至少为1
        self.use_inr = use_inr
        self.use_attention = use_attention

        # 原始C2f组件
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

        # INR增强组件
        if use_inr:
            self.inr_enhancer = INRFeatureEnhancer(
                feature_dim=self.c,
                coord_dim=2,
                hidden_dim=coord_encoding_dim,
                num_layers=3
            )

        # 小目标注意力
        if use_attention:
            self.small_object_attn = SmallObjectAttention(self.c)

        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    def create_coordinate_grid(self, B, H, W, device, dtype):
        """创建归一化坐标网格（匹配输入数据类型）"""
        # 创建归一化坐标网格 [-1, 1]，使用指定dtype
        y_coords = torch.linspace(-1, 1, H, dtype=dtype, device=device)
        x_coords = torch.linspace(-1, 1, W, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        return coords

    def forward(self, x):
        """前向传播."""
        B, C, H, W = x.shape

        # 原始C2f前向传播
        y = list(self.cv1(x).chunk(2, 1))

        # 通过Bottleneck层
        for i, m in enumerate(self.m):
            bottleneck_out = m(y[-1])

            # INR增强
            if self.use_inr:
                # 坐标网格使用与输入特征相同的dtype
                coords = self.create_coordinate_grid(
                    B, bottleneck_out.shape[2], bottleneck_out.shape[3],
                    x.device, x.dtype  # 关键：匹配输入数据类型
                )
                enhanced_features = self.inr_enhancer(bottleneck_out, coords)
            else:
                enhanced_features = bottleneck_out

            # 小目标注意力
            if self.use_attention:
                enhanced_features = self.small_object_attn(enhanced_features)

            y.append(enhanced_features)

        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用split()的前向传播."""
        y = list(self.cv1(x).split((self.c, self.c), 1))

        for i, m in enumerate(self.m):
            bottleneck_out = m(y[-1])

            # INR增强
            if self.use_inr:
                B, _, H, W = bottleneck_out.shape
                coords = self.create_coordinate_grid(B, H, W, x.device, x.dtype)
                enhanced_features = self.inr_enhancer(bottleneck_out, coords)
            else:
                enhanced_features = bottleneck_out

            # 小目标注意力
            if self.use_attention:
                enhanced_features = self.small_object_attn(enhanced_features)

            y.append(enhanced_features)

        return self.cv2(torch.cat(y, 1))


# 导出模块
__all__ = [
    'INREnhancedC2f',
    'PositionalEncoding',
    'MultiScaleCoordinateEncoder',
    'INRFeatureEnhancer',
    'SmallObjectAttention'
]