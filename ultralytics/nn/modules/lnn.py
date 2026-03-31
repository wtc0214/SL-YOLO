# Ultralytics YOLO 🚀, AGPL-3.0 license
"""液态神经网络（Liquid Neural Networks, LNN）模块
基于MIT CSAIL的LNN理论，使用可微分的常微分方程（ODE）来动态建模神经元状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class LiquidNeuralUnit(nn.Module):
    """
    液态神经单元（Liquid Neural Unit）
    使用ODE动态建模神经元状态：h_t = h_{t-1} + f_θ(x_t, h_{t-1})
    
    设计为轻量级，降低参数量和FLOPs
    """
    
    def __init__(self, in_channels, hidden_dim=None, use_lightweight=True):
        """
        Args:
            in_channels: 输入通道数
            hidden_dim: 隐藏状态维度（如果None，则使用in_channels）
            use_lightweight: 是否使用轻量级设计（深度可分离卷积）
        """
        super().__init__()
        
        # 确保通道数都是整数类型
        self.in_channels = int(in_channels)
        if hidden_dim is not None:
            self.hidden_dim = int(hidden_dim)
        else:
            self.hidden_dim = int(self.in_channels)
        self.use_lightweight = use_lightweight
        
        # ODE函数 f_θ(x_t, h_{t-1})
        # 使用轻量级设计降低参数量
        # 确保通道数计算为整数
        combined_channels = int(self.in_channels + self.hidden_dim)
        
        if use_lightweight:
            # 深度可分离卷积 + 点卷积
            self.f_theta = nn.Sequential(
                # 深度卷积（空间卷积）
                nn.Conv2d(combined_channels, combined_channels, 
                         kernel_size=3, padding=1, groups=combined_channels, bias=False),
                nn.BatchNorm2d(combined_channels),
                nn.SiLU(inplace=True),
                # 点卷积（通道混合）
                nn.Conv2d(combined_channels, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            # 标准卷积设计
            self.f_theta = nn.Sequential(
                Conv(combined_channels, self.hidden_dim, k=3, s=1),
                Conv(self.hidden_dim, self.hidden_dim, k=1, s=1)
            )
        
        # 输入投影（将x_t投影到隐藏空间）
        self.input_proj = Conv(self.in_channels, self.hidden_dim, k=1, s=1) if self.in_channels != self.hidden_dim else nn.Identity()
        
        # 时间步长（可学习参数，控制ODE积分步长）
        self.dt = nn.Parameter(torch.ones(1) * 0.1)  # 初始步长较小，更稳定
        
        # 初始化隐藏状态（如果第一次调用）
        self.register_buffer('initialized', torch.tensor(False))
    
    def forward(self, x, h_prev=None):
        """
        前向传播
        
        Args:
            x: 当前输入特征 [B, C, H, W]
            h_prev: 前一个时间步的隐藏状态 [B, hidden_dim, H, W]，如果None则初始化为0
        
        Returns:
            h_t: 当前时间步的隐藏状态 [B, hidden_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # 确保输入通道数匹配in_channels
        if C != self.in_channels:
            # 如果通道数不匹配，进行截断或填充
            if C > self.in_channels:
                x = x[:, :self.in_channels]
            else:
                padding = torch.zeros(B, self.in_channels - C, H, W, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
        
        # 初始化隐藏状态（如果未初始化或形状不匹配）
        if h_prev is None or h_prev.shape != (B, self.hidden_dim, H, W):
            h_prev = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        
        # 拼接原始输入和隐藏状态
        # f_theta期望输入是 [B, in_channels + hidden_dim, H, W]
        # 使用原始输入x (in_channels) 和 h_prev (hidden_dim)
        xh = torch.cat([x, h_prev], dim=1)  # [B, in_channels + hidden_dim, H, W]
        
        # 计算ODE函数 f_θ(x_t, h_{t-1})
        f_theta_out = self.f_theta(xh)  # [B, hidden_dim, H, W]
        
        # ODE更新：h_t = h_{t-1} + dt * f_θ(x_t, h_{t-1})
        # 使用可学习的时间步长，并限制在合理范围内
        dt_clamped = torch.clamp(self.dt, min=0.01, max=1.0)
        h_t = h_prev + dt_clamped * f_theta_out
        
        return h_t


class LiquidNeuralModule(nn.Module):
    """
    液态神经网络模块（用于特征增强）
    结合LiquidNeuralUnit和自适应融合机制
    """
    
    def __init__(self, in_channels, out_channels=None, hidden_dim=None, 
                 use_lightweight=True, use_adaptive_fusion=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（如果None，则等于in_channels）
            hidden_dim: 隐藏状态维度
            use_lightweight: 是否使用轻量级设计
            use_adaptive_fusion: 是否使用自适应融合（可学习权重）
        """
        super().__init__()
        
        # 确保通道数都是整数类型
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(self.in_channels)
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # 液态神经单元
        self.liquid_unit = LiquidNeuralUnit(self.in_channels, hidden_dim, use_lightweight)
        
        # 输出投影层（将隐藏状态投影到输出空间）
        if self.liquid_unit.hidden_dim != self.out_channels:
            self.output_proj = Conv(self.liquid_unit.hidden_dim, self.out_channels, k=1, s=1)
        else:
            self.output_proj = nn.Identity()
        
        # 自适应融合权重（平衡原始特征和液态增强特征）
        if use_adaptive_fusion:
            # 使用全局平均池化 + MLP生成自适应权重
            fusion_gate_dim = max(8, self.in_channels // 8)
            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, int(fusion_gate_dim), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(fusion_gate_dim), 2, 1),
                nn.Softmax(dim=1)
            )
        else:
            # 固定可学习权重
            self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x, h_prev=None):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            h_prev: 前一个时间步的隐藏状态（可选，用于时序建模）
        
        Returns:
            out: 增强后的特征 [B, out_channels, H, W]
            h_t: 当前隐藏状态（用于下一时间步）
        """
        # 液态神经单元处理
        h_t = self.liquid_unit(x, h_prev)  # [B, hidden_dim, H, W]
        
        # 投影到输出空间
        enhanced = self.output_proj(h_t)  # [B, out_channels, H, W]
        
        # 自适应融合
        if self.use_adaptive_fusion:
            # 生成自适应权重
            fusion_weights = self.fusion_gate(x)  # [B, 2, 1, 1]
            original_weight = fusion_weights[:, 0:1]  # [B, 1, 1, 1]
            enhanced_weight = fusion_weights[:, 1:2]  # [B, 1, 1, 1]
            
            # 确保x和enhanced通道数匹配
            if x.shape[1] != enhanced.shape[1] or x.shape[2:] != enhanced.shape[2:]:
                # 先调整空间尺寸
                if x.shape[2:] != enhanced.shape[2:]:
                    x_proj = F.interpolate(x, size=enhanced.shape[2:], mode='nearest')
                else:
                    x_proj = x
                # 再调整通道数（使用1x1卷积）
                if x_proj.shape[1] != enhanced.shape[1]:
                    # 使用平均池化或截断/填充来匹配通道
                    if x_proj.shape[1] > enhanced.shape[1]:
                        x_proj = x_proj[:, :enhanced.shape[1]]
                    else:
                        padding = torch.zeros(x_proj.shape[0], enhanced.shape[1] - x_proj.shape[1], 
                                             x_proj.shape[2], x_proj.shape[3], 
                                             device=x_proj.device, dtype=x_proj.dtype)
                        x_proj = torch.cat([x_proj, padding], dim=1)
            else:
                x_proj = x
            
            # 加权融合
            out = original_weight * x_proj + enhanced_weight * enhanced
        else:
            # 固定权重融合
            alpha = torch.sigmoid(self.alpha)  # 限制在[0, 1]
            if x.shape[1] != enhanced.shape[1] or x.shape[2:] != enhanced.shape[2:]:
                # 先调整空间尺寸
                if x.shape[2:] != enhanced.shape[2:]:
                    x_proj = F.interpolate(x, size=enhanced.shape[2:], mode='nearest')
                else:
                    x_proj = x
                # 再调整通道数
                if x_proj.shape[1] != enhanced.shape[1]:
                    if x_proj.shape[1] > enhanced.shape[1]:
                        x_proj = x_proj[:, :enhanced.shape[1]]
                    else:
                        padding = torch.zeros(x_proj.shape[0], enhanced.shape[1] - x_proj.shape[1], 
                                             x_proj.shape[2], x_proj.shape[3], 
                                             device=x_proj.device, dtype=x_proj.dtype)
                        x_proj = torch.cat([x_proj, padding], dim=1)
            else:
                x_proj = x
            out = (1 - alpha) * x_proj + alpha * enhanced
        
        return out, h_t


class LiquidNeuralModuleLite(nn.Module):
    """
    轻量级液态神经网络模块
    进一步降低参数量和FLOPs
    """
    
    def __init__(self, in_channels, out_channels=None, hidden_dim_ratio=0.5):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            hidden_dim_ratio: 隐藏维度相对于输入通道的比例（降低参数量）
        """
        super().__init__()
        
        # 确保通道数都是整数类型
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(self.in_channels)
        hidden_dim = max(8, int(self.in_channels * hidden_dim_ratio))
        
        # 使用更小的隐藏维度
        self.liquid_unit = LiquidNeuralUnit(self.in_channels, hidden_dim, use_lightweight=True)
        
        # 输出投影
        if self.liquid_unit.hidden_dim != self.out_channels:
            self.output_proj = Conv(self.liquid_unit.hidden_dim, self.out_channels, k=1, s=1)
        else:
            self.output_proj = nn.Identity()
        
        # 简单的可学习融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x, h_prev=None):
        """前向传播"""
        h_t = self.liquid_unit(x, h_prev)
        enhanced = self.output_proj(h_t)
        
        # 简单融合
        alpha = torch.sigmoid(self.alpha)
        if x.shape[1] != enhanced.shape[1] or x.shape[2:] != enhanced.shape[2:]:
            # 先调整空间尺寸
            if x.shape[2:] != enhanced.shape[2:]:
                x_proj = F.interpolate(x, size=enhanced.shape[2:], mode='nearest')
            else:
                x_proj = x
            # 再调整通道数
            if x_proj.shape[1] != enhanced.shape[1]:
                if x_proj.shape[1] > enhanced.shape[1]:
                    x_proj = x_proj[:, :enhanced.shape[1]]
                else:
                    padding = torch.zeros(x_proj.shape[0], enhanced.shape[1] - x_proj.shape[1], 
                                         x_proj.shape[2], x_proj.shape[3], 
                                         device=x_proj.device, dtype=x_proj.dtype)
                    x_proj = torch.cat([x_proj, padding], dim=1)
        else:
            x_proj = x
        
        out = (1 - alpha) * x_proj + alpha * enhanced
        return out, h_t


# 导出模块
__all__ = [
    'LiquidNeuralUnit',
    'LiquidNeuralModule',
    'LiquidNeuralModuleLite'
]

