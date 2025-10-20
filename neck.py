from module import Conv, C3K2
import torch.nn as nn
import torch
class YOLO11Neck(nn.Module):
    """YOLO11颈部网络 - 严格按照标准架构描述"""
    def __init__(self):
        super().__init__()
        
        # 上采样模块
        self.upsample = nn.Upsample(None, 2, 'nearest')
        
        # 第一阶段融合（节点11-16）
        # 节点13: C3K2 (384x40x40 -> 128x40x40)
        self.layer13 = C3K2(256 + 128, 128, n=1, shortcut=False)  # 384->128
        
        # 节点16: C3K2 (256x80x80 -> 64x80x80)
        self.layer16 = C3K2(128 + 128, 64, n=1, shortcut=False)   # 256->64
        
        # 第二阶段融合（节点17-22）
        # 节点17: CBS (64x80x80 -> 64x40x40)
        self.layer17 = Conv(64, 64, 3, 2)
        
        # 节点19: C3K2 (192x40x40 -> 128x40x40)
        self.layer19 = C3K2(64 + 128, 128, n=1, shortcut=False)   # 192->128
        
        # 节点20: CBS (128x40x40 -> 128x20x20)
        self.layer20 = Conv(128, 128, 3, 2)
        
        # 节点22: C3K2 (384x20x20 -> 256x20x20)
        self.layer22 = C3K2(128 + 256, 256, n=1, shortcut=False)  # 384->256

    def forward(self, x):
        p4, p6, p10 = x  # 节点4(128x80x80), 节点6(128x40x40), 节点10(256x20x20)
        
        # 第一阶段融合
        # 节点11: Upsample (256x20x20 -> 256x40x40)
        p10_up = self.upsample(p10)  # 256x20x20 -> 256x40x40
        
        # 节点12: Concat (256x40x40 + 128x40x40 -> 384x40x40)
        concat_12 = torch.cat([p10_up, p6], 1)  # 384x40x40
        
        # 节点13: C3K2 (384x40x40 -> 128x40x40)
        p13 = self.layer13(concat_12)  # 128x40x40
        
        # 节点14: Upsample (128x40x40 -> 128x80x80)
        p13_up = self.upsample(p13)  # 128x40x40 -> 128x80x80
        
        # 节点15: Concat (128x80x80 + 128x80x80 -> 256x80x80)
        concat_15 = torch.cat([p13_up, p4], 1)  # 256x80x80
        
        # 第二阶段融合
        # 节点16: C3K2 (256x80x80 -> 64x80x80)
        p16 = self.layer16(concat_15)  # 64x80x80
        
        # 节点17: CBS (64x80x80 -> 64x40x40)
        p17 = self.layer17(p16)  # 64x40x40
        
        # 节点18: Concat (64x40x40 + 128x40x40 -> 192x40x40)
        concat_18 = torch.cat([p17, p13], 1)  # 192x40x40
        
        # 节点19: C3K2 (192x40x40 -> 128x40x40)
        p19 = self.layer19(concat_18)  # 128x40x40
        
        # 节点20: CBS (128x40x40 -> 128x20x20)
        p20 = self.layer20(p19)  # 128x20x20
        
        # 节点21: Concat (128x20x20 + 256x20x20 -> 384x20x20)
        concat_21 = torch.cat([p20, p10], 1)  # 384x20x20
        
        # 节点22: C3K2 (384x20x20 -> 256x20x20)
        p22 = self.layer22(concat_21)  # 256x20x20
        
        return p16, p19, p22  # 返回给Head的三个特征图：64x80x80, 128x40x40, 256x20x20