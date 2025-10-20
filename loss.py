
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本模块实现用于姿态估计的损失函数集合：

- DFLoss: Distribution Focal Loss 的实现（用于回归分布）
- BboxLoss: 边界框相关损失，包含 IoU（CIoU）损失与可选的 DFL 损失
- TaskAlignedAssigner: 基于对齐分数的目标分配器，用于匹配 anchors 与 ground-truth
- KeypointLoss: 基于 OKS 思路的关键点定位损失
- PoseLoss: 将上述组件组装用于模型训练的总损失计算

该文件中的实现对训练和验证均适用，并使用中文注释以便项目团队维护。
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import bbox_iou, xywh2xyxy,bbox2dist,dist2bbox,make_anchors


# 统一配置加载
from config_utils import load_config

class DFLoss(nn.Module):
    """用于在训练过程中计算 DFL（分布式焦点损失）的准则类。"""

    def __init__(self, reg_max=16) -> None:
        """初始化 DFL 模块。"""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        计算分布式焦点损失（DFL）的左右分量并返回总和。

        说明：DFL 用于将离散的回归分布（pred_dist）转换为更平滑的回归目标，
        通过对目标在相邻两个整数位置进行线性插值（left/right）并使用交叉熵衡量预测分布与
        插值目标分布的差异来计算损失。

        论文参考：Generalized Focal Loss（包含 Distribution Focal Loss）
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # 左侧整数索引
        tr = tl + 1  # 右侧整数索引
        wl = tr - target  # 左侧权重
        wr = 1 - wl  # 右侧权重
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    

class BboxLoss(nn.Module):
    """用于计算训练中的边界框相关损失的准则类。"""

    def __init__(self, reg_max=16):
        """初始化 BboxLoss，包含最大离散回归段数及是否启用 DFL 的设置。"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        计算边界框相关损失：包括基于 IoU 的回归损失（CIoU）和可选的 DFL 损失。

        参数：
            pred_dist: 网络回归分布输出，形状 [B, A, C]（用于 DFL）
            pred_bboxes: 解码后的预测框，形状 [B, A, 4]（xyxy）
            anchor_points: 锚点坐标，形状 [A, 2]
            target_bboxes: 分配到每个 anchor 的目标框（在 feature map 或 image scale，视调用而定）
            target_scores: 每个 anchor 的目标得分（用于加权），形状与类别 one-hot 相同
            target_scores_sum: 所有目标得分的和（用于归一化）
            fg_mask: 前景/正样本掩码，形状 [B, A]

        返回：
            loss_iou: 基于 CIoU 的回归损失（已按 target_scores 加权并除以 target_scores_sum）
            loss_dfl: DFL 损失（若启用），同样按得分加权并归一化
        """
        # 确保 fg_mask 为布尔类型，便于安全索引
        if not torch.is_tensor(fg_mask):
            fg_mask = torch.tensor(fg_mask, dtype=torch.bool, device=pred_dist.device)
        else:
            fg_mask = fg_mask.to(torch.bool)

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL 损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # 将 target_bboxes 转换为相对于 anchor 的 ltrb 距离分布目标，之后计算 DFL
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
    
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=1, alpha=0.5, beta=6.0):
        """
        TaskAlignedAssigner 构造器。

        参数：
            topk: 每个 GT 选择的 Top-K 候选 anchor，用于候选池构造
            num_classes: 类别数（姿态估计通常为1）
            alpha: 分类分数在对齐得分中的指数权重（s^alpha）
            beta: IoU 在对齐得分中的指数权重（u^beta）

        该分配器基于 Task-Aligned Assigner 思路：对齐分数 = (分类分数^alpha) * (IoU^beta)，
        按每个 GT 选取 topk 候选，然后去重以确保同一 anchor 仅匹配到得分最大的 GT。
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_scores, pred_bboxes, anchors, gt_cls, gt_bboxes, gt_mask=None):
        """
        前向分配：将每个 anchor 与最合适的 GT 匹配（或标记为背景）。

        参数：
            pred_scores: 预测的分类分数，形状 [B, A, C]
            pred_bboxes: 预测的边界框（xyxy），形状 [B, A, 4]
            anchors: 锚点坐标，形状 [A, 2]（此处保留一致性，未直接使用）
            gt_cls: GT 类别索引，形状 [B, G]
            gt_bboxes: GT 边界框（xyxy），形状 [B, G, 4]
            gt_mask: 可选，GT 有效性掩码

        返回：
            target_cls: 每个 anchor 的 one-hot 类别目标，[B, A, C]
            target_bbox: 每个 anchor 的匹配 GT 框，[B, A, 4]
            target_scores: 与 target_cls 一致的 one-hot 分数，[B, A, C]
            target_mask: 正样本掩码，[B, A]
            target_gt_idx: 匹配的 GT 下标，[B, A]（-1 表示背景）
        """

        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        num_gts = gt_bboxes.shape[1]

        # 1. 初始化输出（负样本默认类别为背景，框为0，mask为0）
        target_cls = torch.zeros((batch_size, num_anchors, self.num_classes), device=pred_scores.device)
        target_bbox = torch.zeros((batch_size, num_anchors, 4), device=pred_scores.device)
        target_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=pred_scores.device)
        # target_scores: per-anchor one-hot score (same shape as target_cls)
        target_scores = torch.zeros_like(target_cls)
        # target_gt_idx: matched GT index for each anchor (-1 means background)
        target_gt_idx = torch.full((batch_size, num_anchors), -1, dtype=torch.long, device=pred_scores.device)

        # 遍历每个batch（每张图片）
        for batch_idx in range(batch_size):
            # 当前图片的预测和标注
            scores = pred_scores[batch_idx]  # [num_anchors, num_classes]
            bboxes = pred_bboxes[batch_idx]  # [num_anchors, 4]
            gts = gt_bboxes[batch_idx]       # [num_gts, 4]
            gt_labels = gt_cls[batch_idx]    # [num_gts]

            # 跳过无标注的图片（检查当前图片的有效GT数量）
            valid_gt_mask = (gts.abs().sum(dim=1) > 0)
            if valid_gt_mask.sum() == 0:
                continue
            # 只使用有效的GT条目
            gts_valid = gts[valid_gt_mask]
            gt_labels_valid = gt_labels[valid_gt_mask]
            num_gts_local = gts_valid.shape[0]

            # 2. 计算所有候选框与每个GT的IoU（定位精度）
            # iou.shape: [num_anchors, num_gts]（每个候选框对每个GT的IoU）
            iou = self._bbox_iou(bboxes, gts_valid)

            # 3. 计算每个候选框对每个GT的分类分数（取对应类别的预测概率）
            # 先将GT类别转为one-hot编码：[num_gts, num_classes]
            gt_onehot = torch.zeros((num_gts_local, self.num_classes), device=scores.device)
            # 确保标签为整数类型，便于 scatter/gather 操作
            if not torch.is_floating_point(gt_labels_valid):
                idx_labels = gt_labels_valid.unsqueeze(1).long()
            else:
                idx_labels = gt_labels_valid.unsqueeze(1).long()
            gt_onehot.scatter_(1, idx_labels, 1.0)
            # scores_for_gts.shape: [num_anchors, num_gts_local]
            scores_for_gts = torch.matmul(scores, gt_onehot.T)

            # 4. 计算对齐分数 t = (s^alpha) * (u^beta)
            align_scores = (scores_for_gts ** self.alpha) * (iou ** self.beta)  # [num_anchors, num_gts]

            # 5. 为每个GT选Top-K候选框（按对齐分数降序）
            # topk_inds.shape: [num_gts, topk]（每个GT对应的Top-K候选框索引）
            # topk_scores.shape: [num_gts, topk]（对应的对齐分数）
            topk_scores, topk_inds = torch.topk(align_scores.T, k=self.topk, dim=1, largest=True)

            # 6. 去重：同一候选框若被多个GT选中，保留对齐分数最高的
            # 构建“候选框-分数”映射：key=候选框索引，value=最高对齐分数
            anchor_to_max_score = {}
            anchor_to_gt = {}  # key=候选框索引，value=匹配的GT索引
            for gt_idx in range(num_gts_local):
                for k in range(self.topk):
                    anchor_idx = topk_inds[gt_idx, k]
                    score = topk_scores[gt_idx, k]
                    # 若该候选框未被分配，或当前分数更高，则更新分配
                    if anchor_idx not in anchor_to_max_score or score > anchor_to_max_score[anchor_idx]:
                        anchor_to_max_score[anchor_idx] = score
                        anchor_to_gt[anchor_idx] = gt_idx

            # 7. 标记正样本并赋值目标（类别+框），同时记录匹配的GT索引和得分(one-hot)
            for anchor_idx, gt_idx in anchor_to_gt.items():
                target_mask[batch_idx, anchor_idx] = True  # 标记为正样本
                # 分配目标类别（对应GT的类别，one-hot）
                label_idx = gt_labels_valid[gt_idx]
                if not torch.is_floating_point(label_idx):
                    label_idx = int(label_idx)
                else:
                    label_idx = int(label_idx.long())
                target_cls[batch_idx, anchor_idx, label_idx] = 1.0
                # 目标分数（用于权重计算）与 one-hot 类别一致
                target_scores[batch_idx, anchor_idx] = target_cls[batch_idx, anchor_idx]
                # 分配目标框（对应GT的边界框）
                target_bbox[batch_idx, anchor_idx] = gts_valid[gt_idx]
                # 记录该anchor匹配到的GT索引（相对于该图片的GT索引）
                target_gt_idx[batch_idx, anchor_idx] = gt_idx

        return target_cls, target_bbox, target_scores, target_mask, target_gt_idx

    def _bbox_iou(self, bboxes, gts):
        """辅助函数：计算两组边界框的IoU（交并比）"""
        # 计算每个框的面积
        bbox_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        gt_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        # 计算交集的坐标
        left = torch.max(bboxes[:, None, 0], gts[:, 0])  # [num_anchors, num_gts]
        top = torch.max(bboxes[:, None, 1], gts[:, 1])
        right = torch.min(bboxes[:, None, 2], gts[:, 2])
        bottom = torch.min(bboxes[:, None, 3], gts[:, 3])

        # 计算交集面积（若无交集则为0）
        inter = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)

        # 计算IoU = 交集 / (候选框面积 + GT面积 - 交集)
        iou = inter / (bbox_area[:, None] + gt_area - inter + 1e-6)  # 加1e-6避免除零
        return iou
    
class KeypointLoss(nn.Module):
    """关键点损失计算类，用于计算预测关键点与真实关键点之间的损失
    核心功能：结合欧氏距离和目标尺度归一化，衡量关键点定位精度
    特别适用于人体姿态估计等需要精确定位多个关键点的任务
    """

    def __init__(self, sigmas) -> None:
        """初始化关键点损失计算器
        Args:
            sigmas: 各关键点的标准差数组（与关键点数量对应）
                    用于平衡不同部位关键点的损失权重（如关节点与端点的误差容忍度不同）
        """
        super().__init__()
        self.sigmas = sigmas  # 存储各关键点的标准差参数

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """计算关键点损失（基于改进的欧氏距离损失）
        Args:
            pred_kpts: 预测的关键点坐标，形状为 [N, K, 2] 
                       其中N为样本数，K为关键点数量，2对应(x,y)坐标
            gt_kpts: 真实的关键点坐标，形状同上
            kpt_mask: 关键点可见性掩码，形状为 [N, K]
                      1表示关键点可见（参与损失计算），0表示不可见（忽略）
            area: 目标边界框的面积，形状为 [N, 1]
                  用于根据目标大小归一化损失（大目标允许更大的坐标误差）
        Returns:
            平均关键点损失值
        """
        # 步骤1：计算预测与真实关键点的欧氏距离平方（避免开方运算，提高效率）
        # 分别计算x坐标差的平方和y坐标差的平方，再求和
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)

        # 步骤2：计算关键点损失因子（平衡可见关键点数量不均的问题）
        # 公式：总关键点数量 / 实际可见的关键点数量（+1e-9避免除零）
        # 作用：当某样本可见关键点少时，增大其损失权重，保证学习效果
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)

        # 步骤3：计算归一化误差（基于COCO评估标准的改进公式）
        # 核心逻辑：用目标面积和关键点标准差归一化距离，使损失不受目标大小影响
        # (2*self.sigmas)^2：各关键点的标准差平方（平衡不同部位的重要性）
        # (area + 1e-9)：目标面积（大目标允许更大误差）
        # 整体效果：相同像素误差在大目标上的损失更小，在关键部位（小sigma）上的损失更大
        # 使用 **2 避免静态分析将常量 2 识别为 int 导致的属性访问警告
        e = d / (((2 * self.sigmas) ** 2) * (area + 1e-9) * 2)

        # 步骤4：计算最终损失
        # 1-torch.exp(-e)：将误差转换为损失（类似Smooth L1，对小误差更敏感）
        # *kpt_mask：只计算可见关键点的损失
        # kpt_loss_factor.view(-1, 1)：应用损失因子，平衡样本间可见点数量差异
        # .mean()：求平均损失
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
class PoseLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        device = next(model.parameters()).device
        # 从统一的配置加载工具读取损失系数（允许通过 config.yaml 覆盖）
        cfg = load_config()
    

        # 1. 基础配置（从模型头部提取）
        self.device = device
        self.stride = [8,16,32] # 特征层步长（如[8,16,32]）
        self.nc = 1  # 类别数（姿态估计默认1类：person）
        self.reg_max = 16  # DFL通道数（默认16）
        self.no = self.nc + self.reg_max * 4  # 每个锚点的检测输出维度
        self.kpt_shape = (21,3)  # 关键点形状（21,3）：21个点，每个含(x,y,可见性)
        self.use_dfl = self.reg_max > 1  # 是否启用DFL损失

        # 2. 损失函数初始化
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")  # 类别损失（BCE）
        self.bce_kpt_vis = nn.BCEWithLogitsLoss()  # 关键点可见性损失（BCE）
        self.bbox_loss = BboxLoss(self.reg_max).to(device)  # 边界框损失（含DFL）
        
        # 3. 关键点OKS权重（针对21个关键点的自定义标准差，平衡不同部位损失）
        OKS_SIGMA_21 = np.array([
            0.087, 0.087, 0.087, 0.087, 0.087,  # 0-4: 鼻子、左右眼、左右耳
            0.05, 0.05,                      # 5-6: 左右肩
            0.07, 0.07,                      # 7-8: 左右肘
            0.04, 0.04,                      # 9-10: 左右腕
            0.05, 0.05,                      # 11-12: 左右髋
            0.07, 0.07,                      # 13-14: 左右膝
            0.04, 0.04,                      # 15-16: 左右踝
            0.087, 0.087,                        # 17-18: 左右指尖
            0.087, 0.087                         # 19-20: 左右脚尖
        ])
        sigmas = torch.from_numpy(OKS_SIGMA_21).to(device)
        self.kpt_loss = KeypointLoss(sigmas=sigmas)  # 关键点位置损失

        # 4. 目标分配器（匹配锚点与真实目标，用于损失计算的正负样本划分）
        assign_alpha = cfg.get('assigner_alpha', 0.5)  # 原先固定为0，导致分类退化
        assign_beta = cfg.get('assigner_beta', 6.0)
        self.assigner = TaskAlignedAssigner(
            topk=cfg.get('assigner_topk', 10),  # 每个目标匹配的Top-K锚点数量
            num_classes=self.nc,
            alpha=assign_alpha,
            beta=assign_beta
        )

        # 5. DFL辅助参数（用于分布式焦点损失的坐标解码）
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

        # lambda 系数作为成员变量，优先使用 config.yaml 中的配置
        self.lambda_box = cfg.get('lambda_box', 7.5)
        self.lambda_cls = cfg.get('lambda_cls', 1.5)  # 提高分类权重，鼓励判别性
        self.lambda_kpt = cfg.get('lambda_kpt', 12.0)
        self.lambda_kpt_vis = cfg.get('lambda_kpt_vis', 1.0)
        self.lambda_dfl = cfg.get('lambda_dfl', 1.5)
        # 负样本关键点可见性正则系数（降低 fallback 随机性）
        self.lambda_neg_kpt_vis = cfg.get('lambda_neg_kpt_vis', 0.1)

    def preprocess_targets(self, targets, batch_size, imgsz):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        # 按批次索引分组
        batch_idx = targets[:, 0].long()
        unique_idx, counts = batch_idx.unique(return_counts=True)
        max_obj = counts.max()  # 单张图最大目标数
        processed = torch.zeros(batch_size, max_obj, 5, device=self.device)

        for idx in unique_idx:
            mask = batch_idx == idx
            obj_cnt = mask.sum()
            # 提取类别和边界框，并缩放到图像尺寸
            processed[idx, :obj_cnt, 0] = targets[mask, 1]  # cls
            processed[idx, :obj_cnt, 1:] = targets[mask, 2:] * imgsz[[1, 0, 1, 0]]  # xyxy（适配图像w/h）
        
        return processed

    def decode_bbox(self, anchor_points, pred_dist):
        if self.use_dfl:
            # DFL解码：将分布式预测转换为连续坐标偏移
            B, A, C = pred_dist.shape
            pred_dist = pred_dist.view(B, A, 4, C//4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # 结合锚点计算最终边界框（xyxy格式）
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def decode_kpt(self, anchor_points, pred_kpt):
        decoded = pred_kpt.clone()
        # 坐标偏移解码（适配锚点位置）
        decoded[..., :2] *= 2.0  # 偏移量缩放
        # anchor_points: (A,2) -> split并reshape以便广播到 (B, A, 21)
        ap = anchor_points.to(decoded.device)
        ax = (ap[:, 0] - 0.5).view(1, -1, 1)  # (1, A, 1)
        ay = (ap[:, 1] - 0.5).view(1, -1, 1)  # (1, A, 1)
        # decoded[...,0] shape: (B, A, 21)
        decoded[..., 0] += ax
        decoded[..., 1] += ay
        return decoded

    def calculate_keypoint_loss(self, fg_mask, target_gt_idx, gt_kpt, stride_tensor, target_bbox, pred_kpt):
        if not fg_mask.any():
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        B, A = fg_mask.shape
        fg_mask_bool = fg_mask.to(torch.bool)

        # 构建批次索引网格，并为前景锚点选择匹配的 GT 索引
        batch_idx_grid = torch.arange(B, device=target_gt_idx.device).unsqueeze(1).expand(B, A)
        sel_batch = batch_idx_grid[fg_mask_bool]
        sel_gt = target_gt_idx[fg_mask_bool].long()

        # 过滤无效的 GT 索引
        valid = sel_gt >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        sel_batch = sel_batch[valid]
        sel_gt = sel_gt[valid]

        # 选出的 GT 关键点与预测关键点：（N_valid, K, 3）
        selected_gt_kpt = gt_kpt[sel_batch, sel_gt]  # (N, K, 3)
        selected_pred_kpt = pred_kpt[fg_mask_bool][valid]

        # OKS 归一化所需的面积：target_bbox 为图像尺度的 xyxy
        target_bbox_sel = target_bbox[fg_mask_bool][valid]
        # 将 xyxy 转换为 xywh（如有需要）：此处将其转换为 w、h 以计算面积
        x1 = target_bbox_sel[..., 0]
        y1 = target_bbox_sel[..., 1]
        x2 = target_bbox_sel[..., 2]
        y2 = target_bbox_sel[..., 3]
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        area = (w * h).unsqueeze(-1)

        # 处理 stride：先按批次扩展后再索引；支持 stride 为 1D (A,) 或 2D (B, A)
        if hasattr(stride_tensor, 'dim') and stride_tensor.dim() == 1:
            stride_expand = stride_tensor.unsqueeze(0).expand(B, -1).to(target_gt_idx.device)
        elif hasattr(stride_tensor, 'dim') and stride_tensor.dim() == 2 and stride_tensor.shape[0] == B:
            stride_expand = stride_tensor.to(target_gt_idx.device)
        else:
            # 回退：先展平再扩展（兼容未知形状）
            s = stride_tensor.view(-1)
            stride_expand = s.unsqueeze(0).expand(B, -1).to(target_gt_idx.device)

        stride_sel = stride_expand[fg_mask_bool][valid].float()

        # 将 GT 关键点缩放到特征图坐标系
        selected_gt_kpt = selected_gt_kpt.clone()
        # 将 stride 扩展为 (N,1,1) 以便对 (N,K,2) 的坐标进行广播除法
        selected_gt_kpt[..., :2] /= stride_sel.view(-1, 1, 1)

        # 可见性掩码
        kpt_vis_mask = selected_gt_kpt[..., 2] != 0

        # 计算关键点位置与可见性损失
        kpt_loc_loss = self.kpt_loss(pred_kpts=selected_pred_kpt, gt_kpts=selected_gt_kpt, kpt_mask=kpt_vis_mask, area=area)
        kpt_vis_loss = self.bce_kpt_vis(selected_pred_kpt[..., 2], kpt_vis_mask.float())

        return kpt_loc_loss, kpt_vis_loss

    def forward(self, preds, batch):
        # 1. 解包 preds（兼容训练与推理包装器的输出）
        if isinstance(preds, tuple) and len(preds) == 2:
            first, second = preds
            if isinstance(first, (list, tuple)):
                feats, pred_kpt = first, second
            elif isinstance(second, (list, tuple)):
                feats, pred_kpt = second
            else:
                feats, pred_kpt = preds
        else:
            feats, pred_kpt = preds

        B = feats[0].shape[0]

        # NOTE (head -> loss): 说明 head 输出如何被逐步处理：
        # - `head` 的 forward 在训练时返回 (det_out, kpt_out)：
        #     * det_out: list of detection feature maps per scale, 每个形状 [B, no, H, W]
        #     * kpt_out: keypoint prediction tensor, 形状 [B, nk, total_anchors]
        # - 本函数首先把多尺度的 `det_out` 沿锚点维度展平并拼接：得到
        #     `pred_dist` (回归分支) 和 `pred_cls` (分类分支)，用于目标分配和 bbox 解码。
        # - `pred_kpt` 来自 head 的关键点分支：先调整维度到 [B, A, K, D] 形式
        #     （A=总锚点数, K=关键点数, D=关键点维度(3)），随后通过 `decode_kpt`
        #     把特征图尺度的偏移解码到图像尺度或特征图尺度（取决于实现），再用于关键点损失计算。
        # - 在分配阶段（assigner）使用 `pred_cls`(sigmoid) 与 `pred_bbox`(解码并乘以 stride)
        #   来计算匹配分数，得到正样本掩码 `fg_mask` 和对应的 ground-truth 索引 `target_gt_idx`。
        # - 最终针对正样本锚点：
        #     * 使用 `bbox_loss` 计算回归相关损失；
        #     * 使用 `calculate_keypoint_loss` 计算关键点位置与可见性损失，
        #       其中 `calculate_keypoint_loss` 会挑选每个正样本对应的 GT keypoints，
        #       将它们按 stride 缩放到 feature map 尺度以匹配预测，并计算 OKS 风格的归一化损失。

        # 2. 拼接多尺度检测输出（回归/分类分支）
        pred_dist_list = []
        pred_cls_list = []
        for feat in feats:
            if feat.dim() == 4:
                Bf, Cf, Hf, Wf = feat.shape
            elif feat.dim() == 3:
                Bf, Cf, N = feat.shape
                Hf, Wf = N, 1
            else:
                raise ValueError(f"Unexpected feat shape {tuple(feat.shape)}")
            flat = feat.view(Bf, self.no, -1)
            pred_dist_list.append(flat[:, : self.reg_max * 4, :])
            pred_cls_list.append(flat[:, self.reg_max * 4 :, :])

        pred_dist = torch.cat(pred_dist_list, 2).permute(0, 2, 1).contiguous()
        pred_cls = torch.cat(pred_cls_list, 2).permute(0, 2, 1).contiguous()
        pred_kpt = pred_kpt.permute(0, 2, 1).contiguous().view(B, -1, *self.kpt_shape)

        # 3. 生成锚点与推断图像尺寸
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 4. 组织并对齐 targets（按 batch 分组，转换为图像尺度）
        raw_targets = torch.cat([batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]], dim=1)
        processed = self.preprocess_targets(raw_targets.to(self.device), B, imgsz)
        gt_cls, gt_bbox = processed.split([1, 4], dim=2)
        if gt_cls.dim() == 3 and gt_cls.size(2) == 1:
            gt_cls = gt_cls.squeeze(2)
        gt_mask = gt_bbox.sum(2, keepdim=True).gt(0.0)

        # 5. 分配正负样本（TaskAlignedAssigner）
        pred_bbox = self.decode_bbox(anchor_points, pred_dist)
        _, target_bbox, target_cls_score, fg_mask, target_gt_idx = self.assigner(
            pred_cls.detach().sigmoid(), (pred_bbox.detach() * stride_tensor).type(gt_bbox.dtype), anchor_points * stride_tensor, gt_cls, gt_bbox, gt_mask
        )

        target_cls_sum = target_cls_score.sum()
        if isinstance(target_cls_sum, torch.Tensor):
            target_cls_sum = torch.clamp(target_cls_sum, min=1.0)
        else:
            target_cls_sum = max(target_cls_sum, 1.0)

        # 6. 计算各项损失
        loss = torch.zeros(5, device=self.device)
        loss[3] = self.bce_cls(pred_cls, target_cls_score.to(pred_cls.dtype)).sum() / target_cls_sum

        if fg_mask.any():
            target_bbox_scaled = target_bbox / stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_dist, pred_bbox, anchor_points, target_bbox_scaled, target_cls_score, target_cls_sum, fg_mask)

            # 准备关键点（转换到图像尺度后再按 stride 处理）
            gt_kpt = batch["keypoints"].to(self.device).float()
            gt_kpt[..., 0] *= imgsz[1]
            gt_kpt[..., 1] *= imgsz[0]
            batch_idx_kpt = batch["batch_idx"].long()
            max_obj = torch.unique(batch_idx_kpt, return_counts=True)[1].max()
            batched_gt_kpt = torch.zeros(B, max_obj, self.kpt_shape[0], self.kpt_shape[1], device=self.device)
            for idx in range(B):
                mask = batch_idx_kpt == idx
                batched_gt_kpt[idx, : mask.sum()] = gt_kpt[mask]

            loss[1], loss[2] = self.calculate_keypoint_loss(fg_mask=fg_mask, target_gt_idx=target_gt_idx, gt_kpt=batched_gt_kpt, stride_tensor=stride_tensor, target_bbox=target_bbox, pred_kpt=self.decode_kpt(anchor_points, pred_kpt))

        # 负样本关键点可见性抑制（采样一部分负锚点，鼓励其可见性输出趋近 0）
        if self.lambda_neg_kpt_vis > 0:
            try:
                # pred_kpt shape: [B, A, K, 3] (未decode状态下) ; 我们只关心最后一维 index 2 (vis logits)
                # 这里使用 pred_kpt 原始（未 decode_kpt 偏移）值的第三通道作为 vis logits
                with torch.no_grad():
                    B_all, A_all = pred_dist.shape[0], pred_dist.shape[1]
                vis_logits = pred_kpt[..., 2]  # [B, A, K]
                neg_mask = ~fg_mask  # [B,A]
                if neg_mask.any():
                    # 随机采样固定数量的负锚点（避免全部计算增加开销）
                    neg_indices = neg_mask.nonzero(as_tuple=False)  # [N,2]
                    max_neg = min(neg_indices.shape[0], 2048)
                    if neg_indices.shape[0] > max_neg:
                        rand_perm = torch.randperm(neg_indices.shape[0], device=neg_indices.device)[:max_neg]
                        neg_indices = neg_indices[rand_perm]
                    nb = neg_indices[:, 0]
                    na = neg_indices[:, 1]
                    # 取这些负样本的所有关键点可见性 logits，目标为 0
                    neg_vis_logits = vis_logits[nb, na]  # [N, K]
                    neg_targets = torch.zeros_like(neg_vis_logits)
                    neg_vis_loss = F.binary_cross_entropy_with_logits(neg_vis_logits, neg_targets, reduction='mean')
                    # 累加到 loss[2] (可见性) 分量：保持结构兼容
                    loss[2] += self.lambda_neg_kpt_vis * neg_vis_loss
            except Exception:
                pass

    # 按权重系数组合各项损失
        loss[0] *= self.lambda_box
        loss[1] *= self.lambda_kpt
        loss[2] *= self.lambda_kpt_vis
        loss[3] *= self.lambda_cls
        loss[4] *= self.lambda_dfl

        # `loss` 已在内部按样本/正样本等做了归一化（例如除以 target_scores_sum），
        # 因此不应再乘以批次大小 B，否则会放大损失值。改为直接求和得到总损失。
        total_loss = loss.sum()
        loss_components = loss.detach()
        return total_loss, loss_components
    


