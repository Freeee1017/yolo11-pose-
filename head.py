import torch
import torch.nn as nn
from utils import make_anchors, dist2bbox
from module import DFL  # 分布式焦点损失模块
from module import Conv, DWConv  # 卷积模块


class SingleTargetPose(nn.Module):
    """单目标YOLO Pose头部：专注于单个目标的检测和21个关键点预测"""

    def __init__(self, nc=1, kpt_shape=(21, 3), ch=()):
        """
        初始化单目标Pose Head
        Args:
            nc: 目标检测类别数（单目标固定为1）
            kpt_shape: 关键点形状，固定为(21, 3)表示21个关键点，每个含(x,y,可见性)
            ch: 输入特征图的通道数列表（如[256, 512, 1024]）
        """
        super().__init__()
        self.nc = nc  # 单目标场景固定为1类
        self.nl = len(ch)  # 检测特征层数量
        self.reg_max = 16  # DFL通道数，用于边界框回归
        self.no = nc + self.reg_max * 4  # 每个锚点的检测输出维度
        self.kpt_shape = kpt_shape  # 固定为21个关键点
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点总维度（21*3=63）
        
        # 可选：Top-K加权融合，用于缓解特征图网格量化误差（默认关闭，保持原行为）
        # 开启后将对得分最高的K个锚点进行权重融合，得到更平滑、更稳健的框与关键点
        self.topk_fuse = False   # 是否启用Top-K融合（建议验证：True/False对比精度）
        self.topk = 5            # 融合的锚点数量K
        self.fuse_temp = 0.5     # 温度系数（越小越“尖锐”，越接近argmax）

        # 1. 边界框回归分支
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = nn.ModuleList([
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        ])

        # 2. 类别预测分支（单目标简化）
        c3 = max(ch[0] // 2, 32)  # 单目标场景减少通道数
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Conv2d(c3, self.nc, 1)  # 简化为两层卷积
            )
            for x in ch
        ])

        # 3. 21关键点预测分支（增强关键点表达能力）
        c4 = max(ch[0] // 2, self.nk)  # 增加关键点分支通道数
        self.cv4 = nn.ModuleList([
            nn.Sequential(Conv(x, c4, 3), 
                          Conv(c4, c4, 3),
                          Conv(c4, c4, 3),  # 增加一层卷积提升精度
                          nn.Conv2d(c4, self.nk, 1))
            for x in ch
        ])

        # DFL模块
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 动态锚点和步长变量
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.shape = None

    def forward(self, x):
        """前向传播：处理多尺度特征图，输出单目标检测和关键点结果
        
        核心流程：
        1. 多尺度特征图分别通过关键点分支和检测分支，生成原始预测
        2. 训练阶段直接返回原始预测（供损失计算）
        3. 推理阶段解码预测结果，筛选出置信度最高的单目标，输出最终结果
        
        Args:
            x: 多尺度特征图列表，形状如 [B, C1, H1, W1], [B, C2, H2, W2], [B, C3, H3, W3]
                （B=批次大小，C=通道数，H/W=特征图高/宽）
        
        Returns:
            训练时：(检测分支原始输出列表, 关键点分支原始输出)
                - 检测分支输出：每个元素形状 [B, no, H, W]（no=检测输出维度）
                - 关键点输出：形状 [B, 63, 总锚点数]（63=21个关键点×3维度）
            推理时：(融合后的单目标结果, (检测分支原始输出, 关键点原始输出))
                - 单目标结果：形状 [B, 68, 1]（4框+1置信度+63关键点）
        """
        bs = x[0].shape[0]  # 获取批次大小（所有尺度特征图批次一致，取第一个即可）

        # 1. 关键点分支计算（21个关键点，每个含x/y/可见性，共21×3=63维）
        kpt = []  # 存储各尺度关键点预测
        for i in range(self.nl):  # self.nl=特征层数量（通常3个：大/中/小尺度）
            # （1）当前尺度特征图通过关键点卷积分支（self.cv4[i]）
            # （2）调整形状：[B, 63, H, W] → [B, 63, H×W]（H×W为当前尺度锚点数）
            kpt_pred = self.cv4[i](x[i]).view(bs, self.nk, -1)  # self.nk=63
            kpt.append(kpt_pred)
        # 拼接所有尺度的关键点预测，维度：[B, 63, 总锚点数]（总锚点数=各尺度H×W之和）
        kpt = torch.cat(kpt, dim=-1)  # dim=-1：沿锚点数量维度拼接,(1,63,8400)

        # 2. 检测分支计算（单目标：边界框回归+类别置信度）
        det_out = []  # 存储各尺度检测预测
        for i in range(self.nl):
            # 边界框回归分支：输出回归参数（用于计算目标框坐标）
            reg_pred = self.cv2[i](x[i])  # 形状 [B, 4×reg_max, H, W]（reg_max=DFL通道数）
            # 类别预测分支：输出单目标置信度（因单目标，nc=1）
            cls_pred = self.cv3[i](x[i])  # 形状 [B, 1, H, W]
            # 拼接回归参数和类别预测，沿通道维度（dim=1）合并
            det_out.append(torch.cat([reg_pred, cls_pred], dim=1))  # 形状 [B, no, H, W]（no=4×reg_max+1）

        # 3. 训练/推理阶段分支处理
        if self.training:
            # 训练阶段：直接返回原始预测（未解码），供后续损失函数计算梯度
            return det_out, kpt
        else:
            # 推理阶段：解码原始预测 → 筛选单目标 → 输出最终结果
            det_final = self._det_decode(det_out)  # 解码检测分支：得到边界框+置信度
            kpt_final = self._kpt_decode(bs, kpt)  # 解码关键点分支：得到图像坐标系下的关键点
            
            # 单目标筛选：使用分类概率 * (关键点可见性均值^0.5) 的组合得分，更稳健。
            # det_final: [B,5,N] -> cls 概率位于 channel=4
            cls_scores = det_final[:, 4, :]  
            vis_probs = kpt_final[:, 2::3, :]  # 关键点可见性位于每3个通道的第3个
            vis_mean = vis_probs.mean(dim=1)   # 计算21个关键点的可见性均值 [B,N]
            # 组合得分：避免分类未充分训练时纯靠其 argmax 产生随机选择
            combined = cls_scores * (vis_mean.clamp_(0, 1) ** 0.5)
            # 判断是否需要 fallback：当组合得分几乎全 0 或方差极低
            comb_max, _ = combined.max(dim=1, keepdim=True)#
            comb_std = combined.std(dim=1, keepdim=True)
            fallback_flag = (comb_max <= 1e-3) | (comb_std <= 1e-6)
            if getattr(self, 'topk_fuse', False):
                # Top-K加权融合（可选）：对Top-K锚点进行权重平均，降低网格量化误差
                bbox_cls, kpts = self._topk_fuse_single_target(det_final, kpt_final, combined, fallback_flag)
            else:
                # 兼容旧逻辑：严格选择单个最大得分锚点
                max_idx = combined.argmax(dim=1, keepdim=True)
                if fallback_flag.any():
                    # fallback：使用可见性均值本身
                    max_idx_vis = vis_mean.argmax(dim=1, keepdim=True)
                    max_idx[fallback_flag] = max_idx_vis[fallback_flag]
                # 根据最高置信度索引，分别筛选检测结果和关键点结果
                bbox_cls = self._select_single_target(det_final, max_idx)  # 筛选后检测结果：[B, 5, 1]
                kpts = self._select_single_target(kpt_final, max_idx)     # 筛选后关键点：[B, 63, 1]
            
            # 融合检测结果和关键点：沿通道维度拼接，形状 [B, 5+63, 1] = [B, 68, 1]
            return torch.cat([bbox_cls, kpts], dim=1), (det_out, kpt)

    def _det_decode(self, det_out):
        """解码检测分支原始输出，将回归参数转换为图像坐标系下的边界框和置信度
        
        核心操作：
        1. 拼接多尺度检测预测，统一处理
        2. 动态生成锚点和步长（适配输入尺寸变化）
        3. DFL解码回归参数 → 计算边界框 → 结合锚点/步长映射到原图
        4. 类别预测转换为置信度（sigmoid激活）
        
        Args:
            det_out: 检测分支原始输出列表，每个元素形状 [B, no, H, W]
        
        Returns:
            解码后的检测结果，形状 [B, 5, 总锚点数]
                - 前4维：边界框（xywh格式，图像尺度）
                - 第5维：目标置信度（0~1）
        """
        # 1. 拼接多尺度检测预测（各尺度H/W不同，需先展平锚点维度）
        bs = det_out[0].shape[0]  # 批次大小
        det_views = []  # 存储各尺度展平后的检测预测
        total_anchors = 0  # 统计所有尺度的总锚点数
        for out in det_out:
            b, c, h, w = out.shape  # 单个尺度检测输出形状
            # 展平特征图维度：[B, no, H, W] → [B, no, H×W]（H×W为当前尺度锚点数）
            det_views.append(out.view(b, c, -1))
            total_anchors += h * w  # 累加当前尺度锚点数
        # 拼接所有尺度：[B, no, 总锚点数]（no=4×reg_max+1）
        det_cat = torch.cat(det_views, dim=-1)

        # 2. 动态生成/更新锚点和步长（仅当特征图形状变化时，避免重复计算）
        shapes = [o.shape for o in det_out]  # 当前各尺度检测输出形状
        if self.shape != shapes:
            # make_anchors：生成各尺度锚点（特征图网格中心）和步长
            # 假设步长固定为 [8,16,32]（对应输入640时特征图尺寸 80×80、40×40、20×20）
            anchors, strides = make_anchors(det_out, [8, 16, 32], 0.5)
            # 转置锚点和步长：适配后续广播计算
            # anchors: (总锚点数, 2) → (2, 总锚点数)；strides: (总锚点数,1) → (1, 总锚点数)
            self.anchors = anchors.transpose(0, 1)
            self.strides = strides.transpose(0, 1)
            self.shape = shapes  # 更新特征图形状记录，避免下次重复计算

        # 3. 解码边界框回归参数
        # 分离回归参数和类别预测：沿通道维度（dim=1）拆分
        reg_params, cls_logits = det_cat.split([self.reg_max * 4, self.nc], dim=1)
        # DFL（分布式焦点损失）解码：将离散回归参数转换为连续坐标偏移量
        bbox = self.dfl(reg_params)  # 形状 [B, 4, 总锚点数]（4=xywh偏移）
        # 结合锚点计算最终边界框（xywh格式），并乘以步长映射到原始图像尺度
        bbox = dist2bbox(bbox, self.anchors.unsqueeze(0), xywh=True) * self.strides

        # 4. 类别预测转换为置信度（sigmoid激活，映射到0~1）
        cls_probs = cls_logits.sigmoid()  # 形状 [B, 1, 总锚点数]（单目标，nc=1）

        # 拼接边界框和置信度：沿通道维度合并，形状 [B, 4+1, 总锚点数] = [B, 5, 总锚点数]
        return torch.cat([bbox, cls_probs], dim=1)

    def _kpt_decode(self, bs, kpt):
        """解码关键点原始预测，将偏移量转换为图像坐标系下的关键点坐标
        
        核心操作：
        1. 处理关键点可见性通道（sigmoid激活，映射到0~1）
        2. 结合锚点和步长，将特征图尺度的偏移量解码为原图坐标
        3. 保持关键点维度结构（21个点×3维度）
        
        Args:
            bs: 批次大小（用于确保解码后形状正确）
            kpt: 关键点原始输出，形状 [B, 63, 总锚点数]
        
        Returns:
            解码后的关键点，形状 [B, 63, 总锚点数]
                - 0/3/6...通道：21个关键点的x坐标（图像尺度）
                - 1/4/7...通道：21个关键点的y坐标（图像尺度）
                - 2/5/8...通道：21个关键点的可见性（0~1，值越高越清晰）
        """
        ndim = self.kpt_shape[1]  # 每个关键点的维度（3：x/y/可见性）
        kpt_decoded = kpt.clone()  # 复制原始预测，避免修改输入数据

        # 1. 处理可见性通道（仅当维度为3时，对可见性通道做sigmoid激活）
        if ndim == 3:
            # 每隔3个通道取1个（第2、5、8...通道，对应可见性），激活后映射到0~1
            kpt_decoded[:, 2::3] = kpt_decoded[:, 2::3].sigmoid()

        # 2. 解码x/y坐标：从特征图尺度映射到原始图像尺度
        # 公式逻辑：
        # x = (原始偏移量×2 + 锚点x - 0.5) × 步长
        # （×2：将偏移量范围从0~1扩展到0~2，适配锚点网格）
        # （锚点x-0.5：将锚点中心坐标转换为网格左上角基准）
        # 处理x坐标：每隔ndim个通道取1个（第0、3、6...通道）
        kpt_decoded[:, 0::ndim] = (kpt_decoded[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        # 处理y坐标：每隔ndim个通道取1个（第1、4、7...通道）
        kpt_decoded[:, 1::ndim] = (kpt_decoded[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides

        return kpt_decoded  # 解码后关键点：[B, 63, 总锚点数]

    def _select_single_target(self, preds, max_idx):
        """从多锚点预测结果中，根据最高置信度索引筛选出单目标结果
        
        核心逻辑：
        1. 调整索引形状，使其与预测结果维度匹配（支持广播）
        2. 使用torch.gather按索引提取目标结果，实现单目标筛选
        
        Args:
            preds: 待筛选的预测结果，形状 [B, C, N]
                （C=特征通道数：检测为5，关键点为63；N=总锚点数）
            max_idx: 最高置信度锚点索引，形状 [B, 1]
        
        Returns:
            筛选后的单目标结果，形状 [B, C, 1]
        """
        # 1. 调整索引形状：适配预测结果的通道维度（C）
        # （1）unsqueeze(1)：在通道维度插入维度，形状从 [B,1] → [B,1,1]
        # （2）repeat(1, preds.shape[1], 1)：沿通道维度重复，形状变为 [B, C, 1]
        # 最终索引形状与preds一致（仅锚点维度为1），支持gather操作
        expanded_idx = max_idx.unsqueeze(1).repeat(1, preds.shape[1], 1)
        
        # 2. 按索引筛选目标：沿锚点维度（dim=-1）提取最高置信度对应的结果
        # torch.gather：在指定维度上，根据索引取值，保持其他维度不变
        return torch.gather(preds, dim=-1, index=expanded_idx)

    def _topk_fuse_single_target(self, det_final, kpt_final, combined, fallback_flag):
        """对Top-K锚点进行加权融合，输出单目标的检测与关键点结果。

        Args:
            det_final: [B, 5, N] 解码后的检测结果（xywh+cls）
            kpt_final: [B, 63, N] 解码后的关键点结果
            combined:  [B, N]    用于排序/加权的组合得分
            fallback_flag: [B,1] 当分布退化时回退到单点选择

        Returns:
            bbox_cls: [B, 5, 1]  融合后的框与置信度
            kpts:     [B, 63, 1] 融合后的关键点
        """
        B, N = combined.shape
        k = min(getattr(self, 'topk', 5), N)
        temp = max(getattr(self, 'fuse_temp', 0.5), 1e-3)

        # 1) 选出Top-K锚点及其得分
        topk_vals, topk_idx = combined.topk(k, dim=1, largest=True)  # [B,k]

        # 2) 归一化权重（softmax with temperature）；回退样本使用one-hot
        weights = torch.softmax(topk_vals / temp, dim=1)  # [B,k]
        if fallback_flag.any():
            # 将退化样本的权重置为one-hot（Top-1）
            ff = fallback_flag.squeeze(1)
            # topk默认按降序排列，Top-1在位置0
            weights = weights.clone()
            weights[ff, :] = 0
            weights[ff, 0] = 1

        # 3) 按Top-K索引收集对应预测
        idx_det = topk_idx.unsqueeze(1).repeat(1, det_final.shape[1], 1)     # [B,5,k]
        idx_kpt = topk_idx.unsqueeze(1).repeat(1, kpt_final.shape[1], 1)     # [B,63,k]
        topk_det = torch.gather(det_final, dim=-1, index=idx_det)            # [B,5,k]
        topk_kpt = torch.gather(kpt_final, dim=-1, index=idx_kpt)            # [B,63,k]

        w = weights.unsqueeze(1)  # [B,1,k]
        eps = 1e-9

        # 4) 框融合：xy线性加权，wh在log域加权更稳健；cls概率线性加权
        xy = (topk_det[:, 0:2, :] * w).sum(dim=-1, keepdim=True)             # [B,2,1]
        wh_log = torch.log(topk_det[:, 2:4, :].clamp_min(eps))               # [B,2,k]
        wh = torch.exp((wh_log * w).sum(dim=-1, keepdim=True))               # [B,2,1]
        cls = (topk_det[:, 4:5, :] * w).sum(dim=-1, keepdim=True)            # [B,1,1]
        bbox_cls = torch.cat([xy, wh, cls], dim=1)                           # [B,5,1]

        # 5) 关键点融合：直接对每个通道做权重平均（可进一步按可见性做点级别权重，后续可拓展）
        kpts = (topk_kpt * w).sum(dim=-1, keepdim=True)                      # [B,63,1]

        return bbox_cls, kpts
