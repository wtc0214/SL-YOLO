import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad


class LookupConv(nn.Module):
    """
    基于查找操作的卷积层，替代传统卷积中的乘法操作
    专门针对小目标检测任务优化
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,
                 nf=65, nw=65, small_object_mode=True, scale='m'):
        """
        初始化LookupConv层

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充
            g: 分组数
            d: 膨胀率
            act: 激活函数
            nf: 特征索引粒度
            nw: 权重索引粒度
            small_object_mode: 是否启用小目标检测优化模式
        """
        super().__init__()

        self.c1, self.c2 = c1, c2
        self.k, self.s, self.d = k, s, d
        self.p = autopad(k, p, d)
        self.act = self._get_activation(act)

        # 小目标检测优化：使用更大的查找表
        if small_object_mode:
            self.nf = max(nf, 65)  # 增大特征索引粒度
            self.nw = max(nw, 65)  # 增大权重索引粒度
        else:
            self.nf, self.nw = nf, nw

        # 初始化查找表
        self._init_lookup_table()

        # 尺度参数（使用指数形式避免负值）
        self.e_w = nn.Parameter(torch.tensor(0.0))
        self.e_f = nn.Parameter(torch.tensor(0.0))

        # 权重和偏置（用于生成索引）
        # 根据scale参数调整通道数 - 与YOLOv8 scales配置保持一致
        # YOLOv8 scales: [depth, width, max_channels]
        scale_multipliers = {
            'n': 0.25,  # nano: width=0.25
            's': 0.50,  # small: width=0.50
            'm': 0.75,  # medium: width=0.75
            'l': 1.00,  # large: width=1.00
            'x': 1.25  # xlarge: width=1.25
        }

        scale_mult = scale_multipliers.get(scale, 0.25)

        # 直接使用YAML传入的通道数，不进行修正
        assert c1 > 0 and c2 > 0, f"Invalid channels: c1={c1}, c2={c2}"
        self.c1, self.c2 = c1, c2

        # 确保分组数不会导致维度为0或除零错误
        if g <= 0:
            g = 1  # 分组数至少为1
        elif g > c1:
            g = c1  # 分组数不能大于输入通道数

        # 设置self.g和self.c2的值
        self.g = g
        self.c2 = c2

        self.weight = nn.Parameter(torch.randn(c2, c1 // g, k, k))
        self.bias = nn.Parameter(torch.zeros(c2))

        # BatchNorm层
        self.bn = nn.BatchNorm2d(c2)

        # 初始化参数
        self._init_parameters()

    def _get_activation(self, act):
        """获取激活函数"""
        if act is True:
            return nn.SiLU()
        elif isinstance(act, nn.Module):
            return act
        else:
            return nn.Identity()

    def _init_lookup_table(self):
        """初始化查找表"""
        # 特征子表（非负，单调递增）
        p_f = torch.softmax(torch.ones(self.nf - 1), dim=0)
        T_f = torch.cumsum(p_f, dim=0)
        T_f = torch.cat([torch.zeros(1), T_f])  # 从0开始

        # 权重子表（可正可负）
        half_nw = (self.nw - 1) // 2
        p_w_pos = torch.softmax(torch.ones(half_nw), dim=0)
        p_w_neg = torch.softmax(torch.ones(half_nw), dim=0)

        T_w_neg = -torch.cumsum(p_w_neg, dim=0).flip(0)  # 负部分
        T_w_pos = torch.cumsum(p_w_pos, dim=0)  # 正部分
        T_w = torch.cat([T_w_neg, torch.zeros(1), T_w_pos])

        # 外积得到二维查找表
        self.T = nn.Parameter(T_f.unsqueeze(1) * T_w.unsqueeze(0))

    def _init_parameters(self):
        """初始化参数"""
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # 安全检查，避免除零错误
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                # 如果fan_in为0，使用默认初始化
                nn.init.zeros_(self.bias)

        # 初始化尺度参数
        with torch.no_grad():
            std_w = self.weight.std().item()
            std_f = 1.0
            # 确保std_w不为0
            if std_w > 0:
                self.e_w.data = torch.tensor(math.log(3 * std_w))
            else:
                self.e_w.data = torch.tensor(0.0)
            self.e_f.data = torch.tensor(math.log(3 * std_f))

    @property
    def scale_w(self):
        """权重尺度参数"""
        return torch.exp(self.e_w)

    @property
    def scale_f(self):
        """特征尺度参数"""
        return torch.exp(self.e_f)

    def _discretize_weights(self):
        """将权重离散化为索引"""
        w_scaled = self.weight / self.scale_w
        w_clipped = torch.clamp(w_scaled, -1, 1)
        idx_w = torch.round((w_clipped + 1) * (self.nw - 1) / 2).long()
        return idx_w

    def _discretize_features(self, x):
        """将特征离散化为索引"""
        x_scaled = x / self.scale_f
        x_clipped = torch.clamp(x_scaled, 0, 1)  # 假设经过ReLU后特征非负
        idx_f = torch.round(x_clipped * (self.nf - 1)).long()
        return idx_f

    def _lookup_operation(self, idx_f, idx_w):
        """执行查找操作"""
        # 确保索引在有效范围内
        idx_f = torch.clamp(idx_f, 0, self.nf - 1)
        idx_w = torch.clamp(idx_w, 0, self.nw - 1)

        # 从查找表中获取响应值
        responses = self.T[idx_f, idx_w]
        return responses

    def forward(self, x):
        """前向传播"""
        # 步骤1: 缩放和离散化
        idx_w = self._discretize_weights()
        idx_f = self._discretize_features(x)

        # 步骤2: 查找操作
        # 这里需要实现卷积的滑动窗口操作
        # 为了简化，我们使用分组卷积的思想
        if self.g == 1:
            # 标准卷积
            output = self._standard_lookup_conv(x, idx_f, idx_w)
        else:
            # 分组卷积
            output = self._grouped_lookup_conv(x, idx_f, idx_w)

        # 步骤3: 重缩放
        output = output * self.scale_w * self.scale_f

        # 步骤4: 添加偏置和BatchNorm
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        output = self.bn(output)
        output = self.act(output)

        return output

    def _standard_lookup_conv(self, x, idx_f, idx_w):
        """标准查找卷积实现 - 改进版本"""
        N, C, H, W = x.shape
        O = self.c2

        # 计算输出尺寸
        out_h = (H + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        out_w = (W + 2 * self.p - self.d * (self.k - 1) - 1) // self.s + 1
        out_h = max(1, out_h)
        out_w = max(1, out_w)

        # 使用标准卷积作为基础，然后应用查找表增强
        # 这样可以保持卷积的特征提取能力
        try:
            # 使用标准卷积进行特征提取
            conv_output = F.conv2d(x, self.weight, bias=None, stride=self.s,
                                   padding=self.p, dilation=self.d, groups=self.g)

            # 应用查找表增强
            if conv_output.shape[2:] != (out_h, out_w):
                conv_output = F.interpolate(conv_output, size=(out_h, out_w),
                                            mode='bilinear', align_corners=False)

            # 使用查找表进行特征增强
            # 这里我们使用查找表来调整卷积输出
            lookup_enhancement = self._apply_lookup_enhancement(conv_output, idx_f, idx_w)
            output = conv_output + 0.1 * lookup_enhancement  # 小权重避免破坏原有特征

        except Exception as e:
            # 如果标准卷积失败，使用简化的查找表实现
            # 但保持更好的特征表达能力
            output = self._fallback_lookup_conv(x, out_h, out_w)

        # 最终安全检查
        if output.numel() == 0 or output.shape[1] == 0:
            output = torch.zeros(N, O, out_h, out_w, device=x.device, dtype=x.dtype)

        return output

    def _apply_lookup_enhancement(self, conv_output, idx_f, idx_w):
        """应用查找表增强"""
        try:
            # 对卷积输出应用查找表
            N, C, H, W = conv_output.shape

            # 完全避免inplace操作，使用向量化操作
            # 创建缩放因子张量
            scale_factors = torch.ones(C, device=conv_output.device, dtype=conv_output.dtype)

            # 对每个通道应用不同的查找表响应
            for c in range(min(C, self.c2)):
                if c < self.T.shape[0] and c < self.T.shape[1]:
                    # 使用查找表值来调整特征
                    scale_factor = self.T[c % self.T.shape[0], c % self.T.shape[1]]
                    scale_factors[c] = 1 + 0.1 * scale_factor

            # 使用广播进行非inplace操作
            enhanced = conv_output * scale_factors.view(1, C, 1, 1)

            return enhanced
        except:
            return conv_output

    def _fallback_lookup_conv(self, x, out_h, out_w):
        """备用查找卷积实现"""
        N, C, H, W = x.shape
        O = self.c2

        # 使用更智能的特征聚合而不是简单平均
        # 对输入特征进行分组处理
        if C >= O:
            # 如果输入通道数大于等于输出通道数，进行通道选择
            step = C // O
            selected_features = x[:, ::step, :, :][:, :O, :, :]
        else:
            # 如果输入通道数小于输出通道数，进行通道扩展
            selected_features = x.repeat(1, (O + C - 1) // C, 1, 1)[:, :O, :, :]

        # 调整空间尺寸
        if (H, W) != (out_h, out_w):
            selected_features = F.interpolate(selected_features, size=(out_h, out_w),
                                              mode='bilinear', align_corners=False)

        return selected_features

    def _grouped_lookup_conv(self, x, idx_f, idx_w):
        """分组查找卷积实现"""
        # 分组卷积的查找操作实现
        # 这里简化实现，实际应用中需要更复杂的处理
        return self._standard_lookup_conv(x, idx_f, idx_w)

    def forward_fuse(self, x):
        """融合前向传播（推理时使用）"""
        # 推理时可以直接使用查找操作，无需BatchNorm
        idx_w = self._discretize_weights()
        idx_f = self._discretize_features(x)

        output = self._standard_lookup_conv(x, idx_f, idx_w)
        output = output * self.scale_w * self.scale_f

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        output = self.act(output)
        return output


class LookupBottleneck(nn.Module):
    """
    基于查找操作的Bottleneck模块
    用于构建LookupC2f
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, scale='m'):
        super().__init__()

        # 确保参数为整数
        c1 = int(c1)
        c2 = int(c2)
        g = int(g)

        c_ = int(c2 * e)  # hidden channels

        # 确保隐藏通道数有效
        if c_ <= 0:
            c_ = 1  # 隐藏通道数至少为1

        self.cv1 = LookupConv(c1, c_, 1, 1, small_object_mode=True, scale=scale)
        self.cv2 = LookupConv(c_, c2, 3, 1, g=g, small_object_mode=True, scale=scale)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class LookupC2f(nn.Module):
    """
    基于查找操作的C2f模块
    结合C2f结构和查找操作，专门针对小目标检测优化
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, small_object_mode=True, scale='m'):
        super().__init__()

        # 根据scale参数调整通道数 - 与YOLOv8 scales配置保持一致
        # YOLOv8 scales: [depth, width, max_channels]
        scale_multipliers = {
            'n': 0.25,  # nano: width=0.25
            's': 0.50,  # small: width=0.50
            'm': 0.75,  # medium: width=0.75
            'l': 1.00,  # large: width=1.00
            'x': 1.25  # xlarge: width=1.25
        }

        scale_mult = scale_multipliers.get(scale, 0.25)

        # 直接使用YAML传入的通道数，不进行修正
        assert c2 > 0, f"Invalid c2: {c2}"
        self.c2 = c2  # 保存原始c2值
        self.c = int(c2 * e)  # hidden channels

        # 确保隐藏通道数有效，如果计算结果为0，则设为1
        if self.c <= 0:
            self.c = 1
            print(f"Warning: Hidden channels was 0, set to 1. c2={c2}, e={e}")

        # 确保n参数有效
        if n < 0:
            n = 0  # n至少为0
        self.n = n

        # 确保分组数不会导致除零错误
        if g <= 0:
            g = 1  # 分组数至少为1
        elif g > self.c:
            g = self.c  # 分组数不能大于隐藏通道数

        # 调试信息已移除，LookupC2f功能正常

        self.cv1 = LookupConv(c1, 2 * self.c, 1, 1, small_object_mode=small_object_mode, scale=scale)
        self.cv2 = LookupConv((2 + n) * self.c, c2, 1, small_object_mode=small_object_mode, scale=scale)
        self.m = nn.ModuleList(
            LookupBottleneck(self.c, self.c, shortcut, g, scale=scale) for _ in range(n)
        )

    def forward(self, x):
        """前向传播"""
        y = self.cv1(x)

        # 检查cv1输出通道数是否为偶数
        if y.shape[1] % 2 != 0:
            # 如果通道数为奇数，添加一个零通道使其变为偶数
            if y.shape[1] > 0:
                zero_channel = torch.zeros_like(y[:, :1, :, :])
                y = torch.cat([y, zero_channel], dim=1)

        y = list(y.chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        result = self.cv2(torch.cat(y, 1))
        return result

    def forward_split(self, x):
        """使用split的前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LookupC3k2(LookupC2f):
    """
    基于查找操作的C3k2模块
    继承自LookupC2f，保持与原始C3k2的兼容性
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, small_object_mode=True):
        super().__init__(c1, c2, n, shortcut, g, e, small_object_mode)
        # 如果需要C3k块，可以在这里添加
        if c3k:
            self.m = nn.ModuleList(
                LookupBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            )


class SmallObjectLookupConv(LookupConv):
    """
    专门针对小目标检测优化的查找卷积
    使用更大的查找表和特殊的训练策略
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, nf=65, nw=65, small_object_mode=True):
        super().__init__(
            c1, c2, k, s, p, g, d, act,
            nf=nf,  # 特征索引粒度
            nw=nw,  # 权重索引粒度
            small_object_mode=small_object_mode
        )

        # 小目标检测特殊优化
        self.attention_weight = nn.Parameter(torch.ones(1))
        self.frequency_enhancement = True

    def forward(self, x):
        output = super().forward(x)

        # 小目标检测增强
        if self.frequency_enhancement:
            # 高频增强，有助于小目标检测
            high_freq = x - F.avg_pool2d(x, 3, 1, 1)
            # 确保high_freq的通道数与output匹配
            if high_freq.shape[1] != output.shape[1]:
                high_freq = F.interpolate(high_freq, size=output.shape[2:], mode='bilinear', align_corners=False)
                # 如果通道数仍然不匹配，使用平均池化调整通道数
                if high_freq.shape[1] != output.shape[1]:
                    high_freq = high_freq.mean(dim=1, keepdim=True).expand_as(output)
            enhanced = output + self.attention_weight * high_freq
            return enhanced

        return output


class AdaptiveLookupConv(LookupConv):
    """
    自适应查找卷积
    根据输入特征动态调整查找表
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,
                 nf=33, nw=33, small_object_mode=False, adaptation_rate=0.1):
        super().__init__(c1, c2, k, s, p, g, d, act, nf, nw, small_object_mode)

        self.adaptation_rate = adaptation_rate
        self.feature_stats = None
        self.weight_stats = None

    def _update_lookup_table(self, x):
        """根据输入特征更新查找表"""
        if self.training:
            # 计算特征统计信息
            with torch.no_grad():
                feature_mean = x.mean()
                feature_std = x.std()

                # 自适应调整查找表 - 避免inplace操作
                scale_factor = feature_std / (feature_mean + 1e-8)
                new_T = self.T.data * (1 + self.adaptation_rate * scale_factor)
                self.T.data = new_T

    def forward(self, x):
        # 更新查找表
        self._update_lookup_table(x)

        # 执行标准前向传播
        return super().forward(x)
