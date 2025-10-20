#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练工具函数
"""

import torch
import numpy as np
import cv2
import math
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""

    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    else:
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [center_x, center_y, w, h]"""
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y
    else:
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Adjust image size and pad to fit new_shape (like YOLO letterbox).

    Returns:
        resized_img, (r, r), (dw, dh)
    where r is scale, dw/dh are padding offsets (may be floats).
    """
    # 取得原图的高和宽（H, W）
    shape = img.shape[:2]  # height, width
    # 允许传入单个 int（方形目标尺寸），统一转换为 (h, w) 形式
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 r，使长宽按相同比例缩放后不超过目标尺寸（保留纵横比）
    # r = min(target_h / src_h, target_w / src_w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 按比例 r 计算缩放后的图像尺寸（未加边框的尺寸），并四舍五入到整数像素
    # 注意顺序：cv2.resize 需要 (W, H)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # 计算与目标尺寸的差值（需要补的总 padding 像素数，左右/上下合计）
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # 将需要补的像素平均分配到两侧（左/右各 dw/2，上/下各 dh/2）
    dw /= 2  # divide padding into two sides
    dh /= 2

    # 若缩放后的尺寸与原尺寸不同，则进行双线性插值缩放到 new_unpad
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 计算四个方向需要填充的整数像素，-0.1/+0.1 是为了在 round 时规避边界取整误差
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 使用常数颜色填充边框，得到目标尺寸图像
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # 返回：
    # 1) 处理后的图像
    # 2) (r, r): 宽高方向的缩放比例（相同）
    # 3) (dw, dh): 每个方向平均分配前的半边 padding（浮点数，用于精确反变换）
    return img, (r, r), (dw, dh)


# --- Inverse letterbox mapping helpers (common for predicter/validater) ---
def inv_letterbox_xyxy(xyxy, pad, ratio):
    """Map a box from letterbox pixel coords back to original image pixel coords.

    Args:
        xyxy: array-like of 4 numbers [x1,y1,x2,y2] in letterbox pixels
        pad: (dw, dh) padding applied by letterbox
        ratio: (rx, ry) scale ratio returned by letterbox

    Returns:
        list[float]: [x1,y1,x2,y2] in original image pixels, or None if invalid.
    """
    import numpy as _np
    try:
        arr = _np.asarray(xyxy, dtype=float).reshape(-1)
        if arr.size < 4:
            return None
        x1, y1, x2, y2 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        px = float(_np.asarray(pad).reshape(-1)[0])
        py = float(_np.asarray(pad).reshape(-1)[1])
        rx = float(_np.asarray(ratio).reshape(-1)[0])
        ry = float(_np.asarray(ratio).reshape(-1)[1])
        if abs(rx) < 1e-6 or abs(ry) < 1e-6:
            return None
        return [(x1 - px) / rx, (y1 - py) / ry, (x2 - px) / rx, (y2 - py) / ry]
    except Exception:
        return None


def inv_letterbox_kpts(kpts, pad, ratio):
    """Map keypoints from letterbox pixel coords back to original image pixel coords.

    Args:
        kpts: array-like of shape (N,2) or (N,3), in letterbox pixels
        pad: (dw, dh) padding applied by letterbox
        ratio: (rx, ry) scale ratio returned by letterbox

    Returns:
        numpy.ndarray with same shape as input, x/y mapped back to original pixels.
        If invalid input, returns the input array converted to float without mapping.
    """
    import numpy as _np
    try:
        arr = _np.asarray(kpts, dtype=float).copy()
        if arr.ndim < 2 or arr.shape[1] < 2:
            return arr
        px = float(_np.asarray(pad).reshape(-1)[0])
        py = float(_np.asarray(pad).reshape(-1)[1])
        rx = float(_np.asarray(ratio).reshape(-1)[0])
        ry = float(_np.asarray(ratio).reshape(-1)[1])
        if abs(rx) < 1e-6 or abs(ry) < 1e-6:
            return arr
        arr[:, 0] = (arr[:, 0] - px) / rx
        arr[:, 1] = (arr[:, 1] - py) / ry
        return arr
    except Exception:
        return _np.asarray(kpts, dtype=float)


class AverageMeter:
    """Computes and stores the average and current value (utility for training loops)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """生成特征图对应的锚点坐标（用于目标检测的先验框基础坐标）
    
    在YOLO等Anchor-Based模型中，锚点是特征图网格的中心点，后续会结合步长(stride)映射到原图坐标
    
    Args:
        feats: 特征图列表或张量，每个元素形状为 [B, C, H, W]，用于确定锚点网格尺寸
        strides: 每个特征图对应的步长列表，如 [8, 16, 32]，表示特征图1像素对应原图的像素数
        grid_cell_offset: 网格偏移量，0.5表示锚点在网格中心（0~1之间）
        
    Returns:
        anchor_points: 所有特征图的锚点坐标，形状为 [N, 2]，N为总锚点数，坐标为(x, y)
        stride_tensor: 每个锚点对应的步长，形状为 [N, 1]
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None, "特征图不能为空"
    # 获取数据类型和设备（保持与特征图一致）
    dtype, device = feats[0].dtype, feats[0].device
    
    for i, stride in enumerate(strides):
        # 获取当前特征图的高和宽（处理列表或单张量两种情况）
        if isinstance(feats, list):
            f = feats[i]
            if f.dim() == 4:
                h, w = f.shape[2], f.shape[3]  # [B, C, H, W] -> H, W
            elif f.dim() == 3:
                # flattened spatial dims: [B, C, N] -> treat as H=N, W=1
                h, w = f.shape[2], 1
            else:
                raise ValueError(f"Unsupported feature tensor dims {f.shape}")
        else:
            h, w = int(feats[i][0]), int(feats[i][1])  # 若为张量直接取尺寸
        
        # 生成x轴和y轴的网格坐标（加上偏移量使锚点位于网格中心）
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # x坐标：[0.5, 1.5, ..., w-0.5]
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # y坐标：[0.5, 1.5, ..., h-0.5]
        
        # 生成网格坐标矩阵（兼容不同PyTorch版本的meshgrid参数）
        try:
            # 新版本PyTorch需要指定indexing="ij"以生成正确的网格
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        except TypeError:
            # 旧版本PyTorch默认索引方式
            sy, sx = torch.meshgrid(sy, sx)
        
        # 将网格坐标展平为[N, 2]的锚点列表（每个网格对应一个锚点）
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        # 生成每个锚点对应的步长张量（与锚点数相同）
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    # 拼接所有特征图的锚点和步长
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=1):
    """将预测的距离偏移量转换为边界框坐标
    
    模型预测的是锚点到目标边界的距离（左、上、右、下），通过此函数转换为实际边界框
    支持两种输出格式：xywh（中心坐标+宽高）或xyxy（左上角+右下角）
    
    Args:
        distance: 预测的距离偏移量，形状为 [B, 4, N] 或 [B, N, 4]，4个值分别为左、上、右、下距离
        anchor_points: 锚点坐标，形状为 [N, 2]，即锚点在特征图上的(x, y)
        xywh: 是否返回xywh格式，True为(中心x,中心y,宽,高)，False为(左上x,左上y,右下x,右下y)
        dim: distance中存储4个距离值的维度
        
    Returns:
        转换后的边界框，形状与distance一致（除dim维度变为4）
    """
    # 目标：在内部统一 distance 的布局为 (B, 4, N)，便于后续计算
    # 记录原始布局信息以便返回时恢复
    orig_dim = dim
    orig_shape = distance.shape

    # 处理无 batch 维度的情况: (4, N) 或 (N, 4)
    if distance.dim() == 2:
        if distance.shape[0] == 4:
            distance = distance.unsqueeze(0)  # -> (1,4,N)
        elif distance.shape[1] == 4:
            distance = distance.unsqueeze(0).permute(0, 2, 1)  # -> (1,4,N)
        else:
            raise ValueError(f"Unsupported 2D distance shape {distance.shape}")

    # 现在处理 3D 情况，支持 (B,4,N) 或 (B,N,4)
    if distance.dim() == 3:
        if distance.shape[1] == 4:
            d = distance  # already (B,4,N)
        elif distance.shape[2] == 4:
            # (B,N,4) -> (B,4,N)
            d = distance.permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported 3D distance shape {distance.shape}, expected 4 in dim=1 or dim=2")
    else:
        raise ValueError(f"Unsupported distance dims {distance.dim()}, expected 2 or 3 dims")

    # 现在 d 的形状为 (B, 4, N)，按通道拆分为 left/top 和 right/bottom
    lt = d[:, :2, :].contiguous()  # (B,2,N)
    rb = d[:, 2:, :].contiguous()  # (B,2,N)

    # 处理 anchor_points：支持 (N,2), (2,N), (1,2,N), (B,2,N)
    ap = anchor_points
    if not isinstance(ap, torch.Tensor):
        raise ValueError(f"anchor_points must be a torch Tensor, got {type(ap)}")

    if ap.dim() == 2:
        # (N,2) 或 (2,N)
        if ap.shape[1] == 2:
            # (N,2) -> (1,2,N)
            ap_norm = ap.transpose(0, 1).unsqueeze(0)
        elif ap.shape[0] == 2:
            # (2,N) -> (1,2,N)
            ap_norm = ap.unsqueeze(0)
        else:
            raise ValueError(f"anchor_points has unsupported 2D shape {ap.shape}, expected (N,2) or (2,N)")
    elif ap.dim() == 3:
        # (1,2,N) or (B,2,N)
        if ap.shape[1] == 2:
            ap_norm = ap
        else:
            raise ValueError(f"anchor_points must have shape (1,2,N) or (B,2,N), got {ap.shape}")
    else:
        raise ValueError(f"anchor_points must have 2 or 3 dims, got {ap.dim()} dims")

    # 若 ap_norm 是 (1,2,N) 且 batch>1，则在批次维度上广播
    if ap_norm.shape[0] == 1 and lt.shape[0] > 1:
        ap_norm = ap_norm.expand(lt.shape[0], -1, -1)

    # 计算边界框的左上角和右下角坐标
    x1y1 = ap_norm - lt  # (B,2,N)
    x2y2 = ap_norm + rb  # (B,2,N)

    # 内部统一返回格式为 (B,4,N)，但在返回前恢复到原始输入的布局
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        out = torch.cat((c_xy, wh), dim=1)  # (B,4,N)
    else:
        out = torch.cat((x1y1, x2y2), dim=1)  # (B,4,N)

    # 如果原始 distance 是 (B,N,4) 或 (N,4)，将结果变回 (B,N,4)
    if orig_shape and len(orig_shape) >= 3:
        if orig_shape[-1] == 4 and (len(orig_shape) == 3 and orig_shape[1] != 4):
            # 原始是 (B, N, 4)
            out = out.permute(0, 2, 1).contiguous()  # -> (B, N, 4)
    elif len(orig_shape) == 2:
        # 原始为 2D：(4,N) 或 (N,4)，去掉 batch 维度并还原顺序
        if orig_shape[0] == 4:
            out = out.squeeze(0)  # (4,N)
        elif orig_shape[1] == 4:
            out = out.squeeze(0).permute(1, 0).contiguous()  # (N,4)

    return out


def bbox2dist(anchor_points, bbox, reg_max):
    """将边界框坐标转换为锚点到边界的距离偏移量（与dist2bbox互为逆操作）
    
    用于训练时将真实边界框转换为模型需要学习的距离目标，配合DFL损失使用
    
    Args:
        anchor_points: 锚点坐标，形状为 [N, 2]
        bbox: 真实边界框，xyxy格式，形状为 [B, N, 4] 或 [B, 4, N]
        reg_max: 距离偏移量的最大值（用于截断，避免预测值过大）
        
    Returns:
        转换后的距离偏移量，形状与bbox一致，4个值分别为左、上、右、下距离
    """
    # 目标：支持多种输入布局（B,4,N）、(B,N,4)、(4,N)、(N,4) 并返回与 bbox 语义一致的距离
    orig_shape = bbox.shape

    # 规范化 bbox 到 (B,4,N)
    if bbox.dim() == 2:
        # (4,N) or (N,4)
        if bbox.shape[0] == 4:
            b = bbox.unsqueeze(0)  # (1,4,N)
        elif bbox.shape[1] == 4:
            b = bbox.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1,4,N)
        else:
            raise ValueError(f"Unsupported 2D bbox shape {bbox.shape}")
    elif bbox.dim() == 3:
        # (B,4,N) or (B,N,4)
        if bbox.shape[1] == 4:
            b = bbox  # already (B,4,N)
        elif bbox.shape[2] == 4:
            b = bbox.permute(0, 2, 1).contiguous()  # (B,4,N)
        else:
            raise ValueError(f"Unsupported 3D bbox shape {bbox.shape}")
    else:
        raise ValueError(f"Unsupported bbox dims {bbox.dim()}, expected 2 or 3 dims")

    B = b.shape[0]

    # 拆分为左上角和右下角 (B,2,N)
    x1y1 = b[:, :2, :].contiguous()
    x2y2 = b[:, 2:, :].contiguous()

    # 处理 anchor_points：支持 (N,2), (2,N), (1,2,N), (B,2,N)
    ap = anchor_points
    if not isinstance(ap, torch.Tensor):
        raise ValueError(f"anchor_points must be a torch Tensor, got {type(ap)}")

    if ap.dim() == 2:
        if ap.shape[1] == 2:
            ap_norm = ap.transpose(0, 1).unsqueeze(0)  # (1,2,N)
        elif ap.shape[0] == 2:
            ap_norm = ap.unsqueeze(0)  # (1,2,N)
        else:
            raise ValueError(f"anchor_points has unsupported 2D shape {ap.shape}, expected (N,2) or (2,N)")
    elif ap.dim() == 3:
        if ap.shape[1] == 2:
            ap_norm = ap  # (B,2,N) or (1,2,N)
        else:
            raise ValueError(f"anchor_points must have shape (1,2,N) or (B,2,N), got {ap.shape}")
    else:
        raise ValueError(f"anchor_points must have 2 or 3 dims, got {ap.dim()} dims")

    # 如果 ap_norm 为 (1,2,N) 而 batch 大于1，则广播到 batch
    if ap_norm.shape[0] == 1 and B > 1:
        ap_norm = ap_norm.expand(B, -1, -1)

    # 计算距离：锚点到边界的左/上 和 右/下
    lt = ap_norm - x1y1  # (B,2,N)
    rb = x2y2 - ap_norm  # (B,2,N)

    out = torch.cat((lt, rb), 1)  # (B,4,N)
    out = out.clamp_(0, reg_max - 0.01)

    # 恢复原始布局：若原始 bbox 是 (B,N,4) 或 (N,4)，则返回 (B,N,4) 或 (N,4)
    if len(orig_shape) == 3 and orig_shape[2] == 4 and orig_shape[1] != 4:
        out = out.permute(0, 2, 1).contiguous()  # (B,N,4)
    elif len(orig_shape) == 2:
        # (4,N) 或 (N,4) -> 去掉 batch 维度并可能转置
        out = out.squeeze(0)
        if orig_shape[1] == 4:
            out = out.permute(1, 0).contiguous()  # (N,4)
    return out


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """计算两个边界框集合的交并比（IoU）及衍生指标（GIoU/DIoU/CIoU）
    
    IoU是目标检测中衡量边界框匹配度的核心指标，值越大表示匹配越好（0~1之间）
    支持多种边界框格式和IoU变体，适应不同的评估和损失计算需求
    
    Args:
        box1: 第一个边界框集合，最后一维为4，形状如[4]、[N,4]、[B,N,4]等
        box2: 第二个边界框集合，格式与box1一致
        xywh: 输入格式是否为xywh（中心x,中心y,宽,高），False则为xyxy（左上x,左上y,右下x,右下y）
        GIoU: 是否计算广义IoU（考虑最小外接矩形）
        DIoU: 是否计算距离IoU（考虑中心距离）
        CIoU: 是否计算完全IoU（考虑中心距离和宽高比）
        eps: 防止除零的小值
        
    Returns:
        计算得到的IoU/GIoU/DIoU/CIoU值，形状为box1和box2的广播后形状（去除最后一维）
    """
    # 1. 转换边界框格式为xyxy（统一计算方式）
    if xywh:
        # 从xywh转换为xyxy：x1 = x - w/2, x2 = x + w/2（y同理）
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        # 直接从xyxy拆分坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        # 计算宽高（加eps避免为0）
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # 2. 计算交并比（IoU）
    # 交集区域：x方向重叠 * y方向重叠（clamp(0)确保无重叠时为0）
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    
    # 并集区域：A + B - 交集
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # 基础IoU
    iou = inter / union

    # 3. 计算衍生IoU指标（若需要）
    if CIoU or DIoU or GIoU:
        # 最小外接矩形的宽和高（包含两个边界框的最小矩形）
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 外接矩形宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 外接矩形高度
        
        if CIoU or DIoU:
            # DIoU/CIoU需要的参数：外接矩形对角线平方、中心距离平方
            c2 = cw.pow(2) + ch.pow(2) + eps  # 外接矩形对角线的平方
            # 两个边界框中心的距离平方（(x1_center - x2_center)^2 + (y1_center - y2_center)^2）
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + 
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            
            if CIoU:
                # CIoU额外考虑宽高比一致性
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    # 避免梯度不稳定的权重因子
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # 完全IoU
            return iou - rho2 / c2  # 距离IoU
        
        # GIoU：考虑外接矩形中未被并集覆盖的部分
        c_area = cw * ch + eps  # 外接矩形面积
        return iou - (c_area - union) / c_area  # 广义IoU
    
    return iou  # 基础IoU



