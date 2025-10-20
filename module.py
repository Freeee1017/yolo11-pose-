#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11网络模块定义
包含所有基础网络组件：卷积、注意力、特征融合等模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    """
    计算卷积所需的 padding，使输出尺寸与输入保持一致（"same" padding）。
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            # Conv2d 期望 padding 为 int/tuple/str，这里返回 tuple 以满足类型检查
            p = tuple(x // 2 for x in k)
    return p


class Conv(nn.Module):
    """
    标准卷积模块，包含 Conv2d + BN + 激活。
    """
    default_act = nn.SiLU()  # 默认激活函数（SiLU）

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化标准卷积层，支持可配置的卷积核/步幅/分组/扩张与激活。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        前向：Conv2d -> BN -> 激活。
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        前向（融合 BN 后的推理场景）：Conv2d -> 激活（无 BN）。
        """
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    深度可分离卷积中的 Depth-wise 部分（按通道分组卷积）。
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # 输入通道, 输出通道, 卷积核, 步幅, 膨胀, 激活
        """初始化 Depth-wise 卷积参数。"""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)



class Bottleneck(nn.Module):
    """
    标准 Bottleneck 结构：1x1 降维 -> 3x3 升维，可选残差连接。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        初始化 Bottleneck，可选残差（当输入输出通道相等时启用）。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        前向：x -> cv1 -> cv2，若 add=True 则与输入残差相加。
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f：更快的 CSP 变体，通过通道切分与多层 Bottleneck 聚合特征。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化 C2f 模块：1x1 压缩 -> n 个 Bottleneck -> 拼接融合。
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选：使用 FReLU(c2) 作为激活
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        前向：通道切分 -> 级联 n 个 Bottleneck -> 通道拼接 -> 1x1 融合。
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """
        使用 tensor.split() 替代 chunk() 的前向实现，等价但在某些情形下更稳定。
        """
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3K2(nn.Module):
    """C3K2块（YOLO11中的改进CSP）- 支持下采样"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, downsample=False):
        super().__init__()
        self.c = int(c2 * e)
        self.downsample = downsample
        
        # 如果需要下采样，使用stride=2的卷积
        if downsample:
            self.cv1 = Conv(c1, 2 * self.c, 3, 2)  # 下采样
        else:
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.cv3 = Conv((2 + n) * self.c, c2, 1) if c3k else None
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1)) if self.cv3 is not None else self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """
    SPPF（快速空间金字塔池化）：通过多次最大池化堆叠实现多尺度上下文聚合。
    """

    def __init__(self, c1, c2, k=5):
        """
        初始化 SPPF：先 1x1 降维，再进行 3 次相同核大小的最大池化并拼接，最后 1x1 融合。
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        前向：1x1 -> [池化×3 并拼接] -> 1x1。
        """
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """
    多头自注意力（卷积形式的 q/k/v 生成）与位置编码，用于建模长程依赖。
    参数：
        dim: 输入通道数
        num_heads: 注意力头数
        attn_ratio: key 维度相对 head 维度的比例

    属性：
        num_heads: 注意力头数
        head_dim: 每个头的通道维度
        key_dim: key 通道维度
        scale: 注意力缩放系数
        qkv: 生成 q/k/v 的 1x1 卷积
        proj: 输出投影的 1x1 卷积
        pe: 位置编码（深度可分）
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        初始化多头注意力：按通道划分 head，使用 1x1 卷积得到 q/k/v，并加入深度可分的位置编码。
        """
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
        前向：1x1 生成 q/k/v -> 计算注意力 -> 融合 -> 加入位置编码 -> 1x1 投影。
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
    位置敏感注意力块：注意力 + 前馈网络（FFN），可选残差。
    组成：
        attn: 注意力模块
        ffn: 两层 1x1 卷积的前馈网络
        add: 是否启用残差
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        初始化 PSABlock：包含注意力模块与两层 1x1 的 FFN，支持残差。
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        前向：x + Attn(x) -> x + FFN(x)（若启用残差）。
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """
    C2PSA：在 C2 框架内引入 PSA 注意力以加强特征建模。
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        前向：cv1 切分 -> 对后半部分应用 PSA 序列 -> 拼接 -> 1x1 融合。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class PSA(nn.Module):
    """
    PSA：单次注意力与 FFN 的轻量实现（不包含序列堆叠）。
    组成：
        cv1/cv2：通道切分与融合的 1x1 卷积
        attn：注意力模块
        ffn：两层 1x1 的前馈网络
    """

    def __init__(self, c1, c2, e=0.5):
        """
        初始化 PSA：通道切分后对一半通道做注意力和 FFN，再与未处理的一半拼接融合。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        前向：cv1 切分 -> 对后半部分 b 施加注意力与 FFN -> 拼接 -> 1x1 融合。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class DFL(nn.Module):
    """
    分布式焦点回归（DFL）中的积分模块：将离散分布转换为连续偏移。
    """
    
    def __init__(self, c1=16):
        """
        初始化：使用固定权重的 1x1 卷积实现加权求和（0..c1-1 的期望）。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """
        前向：将形状 [B,4*c1,A] 的输入重排后做 softmax，随后进行加权求和得到 [B,4,A] 的连续偏移。
        """
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


if __name__ == "__main__":
    # 测试各个模块
    print("=" * 60)
    print("测试YOLO11网络模块")
    print("=" * 60)
    
    # 测试基础卷积
    print("1. 测试Conv模块:")
    conv = Conv(3, 64, 3, 1)
    x = torch.randn(1, 3, 224, 224)
    out = conv(x)
    print(f"  输入: {x.shape} -> 输出: {out.shape}")
    
    # 测试C3K2模块
    print("2. 测试C3K2模块:")
    c3k2 = C3K2(64, 128, n=2)
    out = c3k2(out)
    print(f"  输入: {conv(x).shape} -> 输出: {out.shape}")
    
    # 测试SPPF模块
    print("3. 测试SPPF模块:")
    sppf = SPPF(128, 256)
    out = sppf(out)
    print(f"  输入: {out.shape} -> 输出: {out.shape}")
    
    # 测试C2PSA模块
    print("4. 测试C2PSA模块:")
    c2psa = C2PSA(256, 256)
    out = c2psa(out)
    print(f"  输入: {out.shape} -> 输出: {out.shape}")
    
    # 测试DFL模块
    print("5. 测试DFL模块:")
    dfl = DFL(16)
    x_dfl = torch.randn(2, 64, 100)  # (batch, 64_channels, anchors)
    out_dfl = dfl(x_dfl)
    print(f"  输入: {x_dfl.shape} -> 输出: {out_dfl.shape}")
    
    print("✓ 所有模块测试通过!")
    print("=" * 60)