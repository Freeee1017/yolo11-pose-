from module import Conv, C3K2, SPPF, C2PSA
import torch.nn as nn
class YOLO11Backbone(nn.Module):
    """YOLO11骨干网络 - 严格按照标准架构描述"""
    def __init__(self):
        super().__init__()
        
        # CBS模块：共5个（节点0、1、3、5、7）
        self.layer0 = Conv(3, 16, 3, 2)      # 节点0: 3x640x640 -> 16x320x320
        self.layer1 = Conv(16, 32, 3, 2)     # 节点1: 16x320x320 -> 32x160x160
        self.layer3 = Conv(64, 64, 1, 1)     # 节点3: 64x80x80 -> 64x80x80 (保持不变)
        self.layer5 = Conv(128, 128, 3, 2)   # 节点5: 128x80x80 -> 128x40x40
        self.layer7 = Conv(128, 256, 3, 2)   # 节点7: 128x40x40 -> 256x20x20
        
        # C3K2模块：共4个（节点2、4、6、8）
        self.layer2 = C3K2(32, 64, n=2, downsample=True)  # 节点2: 32x160x160 -> 64x80x80
        self.layer4 = C3K2(64, 128, n=2)                  # 节点4: 64x80x80 -> 128x80x80
        self.layer6 = C3K2(128, 128, n=2)                 # 节点6: 128x40x40 -> 128x40x40
        self.layer8 = C3K2(256, 256, n=1)                 # 节点8: 256x20x20 -> 256x20x20
        
        # SPPF模块（节点9）
        self.layer9 = SPPF(256, 256, 5)      # 节点9: 256x20x20 -> 256x20x20
        
        # C2PSA模块（节点10）
        self.layer10 = C2PSA(256, 256, n=1)  # 节点10: 256x20x20 -> 256x20x20

    def forward(self, x):
        # 严格按照标准架构描述的顺序
        x = self.layer0(x)    # 节点0: 16x320x320
        x = self.layer1(x)    # 节点1: 32x160x160
        x = self.layer2(x)    # 节点2: 64x80x80
        x = self.layer3(x)    # 节点3: 64x80x80
        p4 = self.layer4(x)   # 节点4: 128×80×80
        
        x = self.layer5(p4)   # 节点5: 128x40x40
        p6 = self.layer6(x)   # 节点6: 128x40x40
        
        x = self.layer7(p6)   # 节点7: 256x20x20
        x = self.layer8(x)    # 节点8: 256x20x20
        x = self.layer9(x)    # 节点9: 256x20x20
        p10 = self.layer10(x) # 节点10: 256x20x20
        
        return p4, p6, p10    # 128×80×80, 128×40×40, 256×20×20