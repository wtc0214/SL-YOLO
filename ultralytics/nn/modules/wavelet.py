

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True

class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))



class LiftingScheme(nn.Module):
    def __init__(self, horizontal, in_planes, modified=True, size=[], splitting=True, k_size=4, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        # 检查输入尺寸是否足够进行小波变换
        B, C, H, W = x.shape
        if H < 4 or W < 4:
            # 如果输入太小，直接返回原始输入的分解
            if H >= 2 and W >= 2:
                c = x[:, :, ::2, ::2]  # 下采样
                d = x[:, :, 1::2, 1::2]  # 下采样
            else:
                # 如果太小，返回原始输入
                c = x
                d = x
            return (c, d)
        
        # 检查padding是否会导致问题
        if H < 8 or W < 8:
            # 对于中等尺寸，使用简化的处理
            if self.splitting:
                (x_even, x_odd) = self.split(x)
            else:
                (x_even, x_odd) = x
            
            # 使用简化的变换，避免复杂的padding
            c = x_even + 0.1 * x_odd  # 简化的update
            d = x_odd - 0.1 * x_even  # 简化的predict
            return (c, d)
        
        # 对于足够大的输入，使用完整的小波变换
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)



class LiftingScheme2D(nn.Module):
    def __init__(self, in_planes, share_weights, modified=True, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingScheme2D, self).__init__()
        self.level1_lf = LiftingScheme(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)
        
        self.share_weights = share_weights
        
        if share_weights:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = self.level2_1_lf  # Double check this
        else:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        # 检查输入尺寸
        B, C, H, W = x.shape
        if H < 4 or W < 4:
            # 如果输入太小，返回简化的分解
            if H >= 2 and W >= 2:
                LL = x[:, :, ::2, ::2]  # Low-Low
                LH = x[:, :, ::2, 1::2]  # Low-High
                HL = x[:, :, 1::2, ::2]  # High-Low
                HH = x[:, :, 1::2, 1::2]  # High-High
                c = x[:, :, ::2, :]  # 中间结果
                d = x[:, :, 1::2, :]  # 中间结果
            else:
                # 如果太小，返回原始输入
                LL = LH = HL = HH = c = d = x
            return (c, d, LL, LH, HL, HH)
        
        # 对于足够大的输入，使用完整的小波变换
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)
        return (c, d, LL, LH, HL, HH)
