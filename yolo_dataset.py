#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO关键点检测数据集和数据增强
专门为21个关键点检测任务设计
"""

import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from augment import BabyPoseAugmentor
from config_utils import load_config
from typing import Optional
from utils import letterbox as utils_letterbox
import logging

logger = logging.getLogger(__name__)
class BabyPoseDataset(Dataset):
    """
    婴儿关键点检测数据集类
    支持YOLO格式标签（68个字段：1类别 + 4边界框 + 63关键点）
    """
    
    def __init__(self, img_dir, label_dir=None, img_size=640, is_train=True, 
                 augmentation: Optional[BabyPoseAugmentor] = None, classes=None):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录路径
            label_dir: 标签目录路径
            img_size: 目标图像尺寸
            is_train: 是否为训练模式
            augmentation: 数据增强器
            classes: 类别列表
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = img_size
        self.is_train = is_train
        # 增强器由外部注入：数据集不再内部创建，仅在训练模式下保留
        self.augmentation = augmentation if self.is_train else None
        self.classes = classes or ['baby']
        self.num_classes = len(self.classes)
        
        # 获取图片文件 - 使用不区分大小写的搜索避免重复
        self.img_files = []
        
        # 遍历目录下的所有文件，检查扩展名
        for file_path in self.img_dir.iterdir():
            if file_path.is_file():
                # 获取小写的扩展名
                ext_lower = file_path.suffix.lower()
                if ext_lower in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.img_files.append(file_path)
        
        self.img_files.sort()  # 确保一致的顺序
        
        if not self.img_files:
            raise RuntimeError(f"未在 {img_dir} 中找到图像文件")

        logger.info("加载数据集: %d 张图片 (来自 %s)", len(self.img_files), img_dir)
    
    def __len__(self):
        return len(self.img_files)
    
    def load_yolo_label(self, img_path):
        """加载YOLO格式标签"""
        if not self.label_dir or not self.label_dir.exists():
            return None, None
            
        label_path = self.label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            return None, None
            
        try:
            # 读取所有非空行，适配包含注释或多行的标签文件
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

            if not lines:
                return None, None

            parts = lines[0].split()
            if len(parts) < 68:  # 1+4+63=68
                return None, None
                
            # 解析类别和边界框
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]  # cx, cy, w, h (归一化)
            
            # 解析关键点
            keypoints_data = [float(x) for x in parts[5:68]]  # 63个值
            keypoints = np.array(keypoints_data).reshape(21, 3)  # (21, 3)
            # 将可见性值为 2 的标记为 1（部分数据集使用 0/1/2 表示可见性，训练中我们使用 0/1）
            try:
                vis = keypoints[:, 2]
                # 使用等于 2 的位置替换为 1，保留其他值不变
                keypoints[:, 2] = np.where(np.isclose(vis, 2.0), 1.0, vis)
            except Exception:
                # 如果出现任何问题，忽略并保留原始值
                pass
            
            return keypoints, (class_id, bbox)
            
        except Exception:
            logger.exception("标签加载错误: %s", label_path)
            return None, None
    
    def transform_labels(self, keypoints, bbox_info, ratio, pad, orig_w, orig_h):
        """将像素坐标的 keypoints 和 bbox 映射到 letterbox 输出坐标。

        注意：此函数现在假定输入为像素坐标（绝对像素），
        keypoints 形状为 (N,3) 且 x,y 为像素值；
        bbox_info 为 None 或 (class_id, [cx_px, cy_px, w_px, h_px])（均为像素）。
        """
        if keypoints is None:
            return None, None

        kpts = keypoints.copy()

        # keypoints 已为像素坐标，先应用 letterbox 缩放与偏移得到 letterbox 像素坐标
        kpts[:, 0] = kpts[:, 0] * ratio[0] + pad[0]
        kpts[:, 1] = kpts[:, 1] * ratio[1] + pad[1]

        # 处理边界框（bbox_info 中的值应为像素）并计算 letterbox 下的 [x1,y1,x2,y2]
        bbox_tensor = None
        if bbox_info is not None:
            class_id, bbox = bbox_info
            cx_px, cy_px, w_px, h_px = bbox

            x1 = (cx_px - w_px / 2.0) * ratio[0] + pad[0]
            y1 = (cy_px - h_px / 2.0) * ratio[1] + pad[1]
            x2 = (cx_px + w_px / 2.0) * ratio[0] + pad[0]
            y2 = (cy_px + h_px / 2.0) * ratio[1] + pad[1]

            # 将 bbox 转换为归一化中心格式 (相对于 letterbox 输出尺寸)
            cx_norm = ((x1 + x2) / 2.0) / float(self.img_size)
            cy_norm = ((y1 + y2) / 2.0) / float(self.img_size)
            w_norm = (x2 - x1) / float(self.img_size)
            h_norm = (y2 - y1) / float(self.img_size)
            bbox_tensor = torch.tensor([cx_norm, cy_norm, w_norm, h_norm, class_id]).float()#归一化后

        # 将 keypoints 从 letterbox 像素坐标归一化到 [0,1] 相对于输出尺寸
        kpts[:, 0] = kpts[:, 0] / float(self.img_size)
        kpts[:, 1] = kpts[:, 1] / float(self.img_size)

        return torch.from_numpy(kpts).float(), bbox_tensor
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.img_files[idx]
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("无法读取图像 %s", img_path)
            raise RuntimeError(f"无法读取图像 {img_path}")

        orig_h, orig_w = image.shape[:2] # 原始尺寸,后续需要用到
        
        # 加载标签（YOLO 格式，可能为归一化坐标）
        keypoints, bbox_info = self.load_yolo_label(img_path)

        # 统一将标签从归一化 (0..1) 转换为像素坐标，以便 transform_labels 可以按像素输入工作
        keypoints_px = None
        if keypoints is not None:
            try:
                k = keypoints.copy()
                # 如果标签是归一化的 (<=1)，则转换为像素
                if k.max() <= 1.0:
                    k[:, 0] = k[:, 0] * orig_w
                    k[:, 1] = k[:, 1] * orig_h
                keypoints_px = k
            except Exception:
                keypoints_px = keypoints.copy()

        bbox_info_px = None
        if bbox_info is not None:
            try:
                class_id, bbox = bbox_info
                cx, cy, w_n, h_n = bbox
                # 若是归一化 bbox (<=1.0)，则转换为像素
                if max(cx, cy, w_n, h_n) <= 1.0:
                    cx_abs = cx * orig_w
                    cy_abs = cy * orig_h
                    w_abs = w_n * orig_w
                    h_abs = h_n * orig_h
                    bbox_info_px = (class_id, [cx_abs, cy_abs, w_abs, h_abs])
                else:
                    bbox_info_px = bbox_info
            except Exception:
                bbox_info_px = bbox_info

        # 数据增强（仅训练时）: 将像素坐标传入增强器，增强器返回仍假定为像素坐标
        if self.is_train and self.augmentation is not None and keypoints_px is not None:
            # 准备用于增强器的像素坐标 keypoints 与 bbox (像素)
            kpts_for_aug = keypoints_px.copy()
            bbox_px = None
            if bbox_info_px is not None:
                try:
                    _, bbox_vals = bbox_info_px
                    x1 = bbox_vals[0] - bbox_vals[2] / 2.0
                    y1 = bbox_vals[1] - bbox_vals[3] / 2.0
                    x2 = bbox_vals[0] + bbox_vals[2] / 2.0
                    y2 = bbox_vals[1] + bbox_vals[3] / 2.0
                    bbox_px = (x1, y1, x2, y2) # 转为 (x1,y1,x2,y2) 格式
                except Exception:
                    bbox_px = None

            image, kpts_aug_px, aug_bbox = self.augmentation(image, kpts_for_aug, bbox_px)

            if kpts_aug_px is not None:
                # 使用增强后的像素关键点作为后续处理的输入
                keypoints_px = kpts_aug_px.copy()

            # 解析增强器返回的 bbox（假定为像素坐标 (x1,y1,x2,y2)），并统一为像素中心格式
            if aug_bbox is not None:
                class_id = bbox_info_px[0] if bbox_info_px is not None else 0
                try:
                    if len(aug_bbox) >= 4:
                        x1, y1, x2, y2 = float(aug_bbox[0]), float(aug_bbox[1]), float(aug_bbox[2]), float(aug_bbox[3])
                        cx_px = (x1 + x2) / 2.0
                        cy_px = (y1 + y2) / 2.0
                        w_px = x2 - x1
                        h_px = y2 - y1
                        bbox_info_px = (class_id, [cx_px, cy_px, w_px, h_px])
                except Exception:
                    logger.exception("解析增强器返回的 bbox 时出错: %s", img_path)
                    pass

        # Letterbox变换（直接使用 utils.letterbox）
        image, ratio, pad = utils_letterbox(image, new_shape=self.img_size)

    # 转换为tensor
        image = image.transpose((2, 0, 1))  # HWC to CHW,yolo需要此格式
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0

        # 变换标签坐标：transform_labels 接受像素坐标并输出相对于 letterbox 输出尺寸的归一化值
        keypoints_tensor, bbox_tensor = self.transform_labels(
            keypoints_px, bbox_info_px, ratio, pad, orig_w, orig_h
        )
        #keypoints_tensor, bbox_tensor为归一化后的值
        return {
            'image': image,
            'keypoints': keypoints_tensor,
            'bbox': bbox_tensor,
            'img_path': str(img_path),
            'orig_size': (int(orig_w), int(orig_h)),
            'ratio': (float(ratio[0]), float(ratio[1])),
            'pad': (float(pad[0]), float(pad[1]))
        }


def pad_collate(batch):
    """
    批处理函数，处理不同尺寸的数据
    
    Args:
        batch: 批次数据列表
        
    Returns:
        dict: 包含批处理后的数据
    """
    # 承载图像与元数据（逐样本）
    images = []
    img_paths = []
    orig_sizes = []   # 原图 (w, h) 或 (orig_w, orig_h)
    ratios = []       # letterbox 的缩放比 (rx, ry)
    pads = []         # letterbox 的填充 (dw, dh)

    # object-level accumulators (flattened across batch)
    # 目标级别的聚合容器（将每张图中的单个目标扁平化到一个批级列表中）
    obj_batch_idx = []  # 每个目标所属的图像索引 i
    obj_cls = []        # 每个目标的类别（float，便于后续与张量拼接）
    obj_bboxes_xyxy = []  # 每个目标的边框（xyxy，归一化到 letterbox 输出尺寸）
    obj_keypoints = []    # 每个目标的关键点 (21,3)，归一化到 letterbox 输出尺寸

    for i, item in enumerate(batch):
        # 收集图像张量（已是 CHW、float32、[0,1]）
        images.append(item['image'])
        img_paths.append(item['img_path'])
        # 收集逆 letterbox 所需的元数据（便于后处理把预测还原回原图坐标）
        orig = item.get('orig_size', None)
        rat = item.get('ratio', None)
        pd = item.get('pad', None)
        orig_sizes.append(orig)
        ratios.append(rat)
        pads.append(pd)

        kpts = item.get('keypoints', None)
        bbox = item.get('bbox', None)

        # 若该样本同时具备 bbox 与 keypoints，则将其视为一个有效目标并汇总
        if bbox is not None and kpts is not None:
            # 当前数据集中 bbox 为归一化中心格式 [cx, cy, w, h, class]
            # 为了适配后续损失计算，转换为归一化 xyxy：[x1, y1, x2, y2]
            try:
                cx, cy, w, h, cls = bbox.tolist()
            except Exception:
                # in case bbox is a tensor on CPU/other type
                b = bbox
                cx, cy, w, h, cls = float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            obj_batch_idx.append(i)
            obj_cls.append(float(cls))
            obj_bboxes_xyxy.append([x1, y1, x2, y2])
            obj_keypoints.append(kpts)

    # 将图像在 batch 维度上堆叠：[B, C, H, W]
    batch_images = torch.stack(images, 0)

    # 构建目标级别的张量（可能为空：例如该 batch 内没有任何带标注的样本）
    if len(obj_bboxes_xyxy) > 0:
        batch_idx = torch.tensor(obj_batch_idx, dtype=torch.long)
        cls_tensor = torch.tensor(obj_cls, dtype=torch.float32)
        bboxes_tensor = torch.tensor(obj_bboxes_xyxy, dtype=torch.float32)
        # 关键点列表包含形状 (21,3) 的张量：在第 0 维堆叠为 [N, 21, 3]
        keypoints_tensor = torch.stack([kp for kp in obj_keypoints], 0).float()
    else:
        # 无目标时，返回形状正确但大小为 0 的占位张量，避免下游代码分支判断
        batch_idx = torch.zeros((0,), dtype=torch.long)
        cls_tensor = torch.zeros((0,), dtype=torch.float32)
        bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        keypoints_tensor = torch.zeros((0, 21, 3), dtype=torch.float32)

    batch_data = {
        'images': batch_images,     # [B, C, H, W]
        'batch_idx': batch_idx,     # [N]，每个目标对应的图像索引
        'cls': cls_tensor,          # [N]，每个目标的类别 id（float）
        'bboxes': bboxes_tensor,    # [N, 4]，每个目标的 bbox（xyxy，归一化到 letterbox 输出尺寸）
        'keypoints': keypoints_tensor,  # [N, 21, 3]，关键点（归一化坐标 + 可见性）
        'img_paths': img_paths,     # 长度 B 的图像路径列表
        'batch_size': len(batch),
        'orig_sizes': orig_sizes,   # 长度 B 的原图尺寸，用于逆 letterbox 映射
        'ratios': ratios,           # 长度 B 的缩放比 (rx, ry)
        'pads': pads                # 长度 B 的填充 (dw, dh)
    }

    return batch_data


def build_datasets_and_loaders():
    """
    构建数据集和数据加载器
    
        classes: 类别列表
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    # 读取配置并创建数据增强器，优先使用 cfg 中的设置
    cfg = load_config('config.yaml')
    # use_augmentation: cfg only (fallback True)
    use_aug = cfg.get('use_augmentation', True)
    augmentation = BabyPoseAugmentor(config=cfg) if use_aug else None
    logger.info("%s", "数据增强已启用" if augmentation else "数据增强已禁用")

    # 检查数据目录结构（优先使用 cfg 中的显式路径）
    data_root = cfg.get('data_root', 'dataset')
    train_img_dir = cfg.get('train_img_dir', os.path.join(data_root, 'images', 'train'))
    train_label_dir = cfg.get('train_label_dir', os.path.join(data_root, 'labels', 'train'))
    val_img_dir = cfg.get('val_img_dir', os.path.join(data_root, 'images', 'val'))
    val_label_dir = cfg.get('val_label_dir', os.path.join(data_root, 'labels', 'val'))
    classes = cfg.get('classes', ['baby'])
    # 检查目录是否存在
    if not os.path.exists(train_img_dir):
        raise RuntimeError(f"训练图像目录不存在: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise RuntimeError(f"验证图像目录不存在: {val_img_dir}")
    
    # 创建数据集
    train_dataset = BabyPoseDataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
    img_size=cfg.get('img_size', 640),
        is_train=True,
        augmentation=augmentation,
        classes=classes
    )
    
    val_dataset = BabyPoseDataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
    img_size=cfg.get('img_size', 640),
        is_train=False,
        augmentation=None,  # 验证集不使用数据增强
        classes=classes
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
    batch_size=cfg.get('batch_size', 16),
        shuffle=True,
    num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=pad_collate,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
    batch_size=cfg.get('batch_size', 16),
        shuffle=False,
    num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=pad_collate,
        drop_last=False
    )
    
    logger.info("训练集大小: %d", len(train_dataset))
    logger.info("验证集大小: %d", len(val_dataset))
    logger.info("批次大小: %d", cfg.get('batch_size', 16))
    
    return train_loader, val_loader, train_dataset, val_dataset



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import random
    # 设置matplotlib支持中文显示（仅用于本模块自测可视化）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    # 配置参数（可根据实际情况修改）
    class Args:
        data_root = "dataset"  # 数据根目录
        img_size = 640
        batch_size = 1
        num_workers = 0
        use_augmentation = True
        train_split = 0.9
        seed = 42

    classes = ['baby']
    args = Args()

    train_img_dir = os.path.join(args.data_root, 'images', 'train')
    train_label_dir = os.path.join(args.data_root, 'labels', 'train')

    cfg = load_config('config.yaml')
    augmentor = BabyPoseAugmentor(config=cfg)

    dataset = BabyPoseDataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=args.img_size,
        is_train=True,
        augmentation=augmentor,
        classes=classes
    )

    idx = random.randint(0, len(dataset)-1)
    img_path = dataset.img_files[idx]
    image = cv2.imread(str(img_path))
    orig_h, orig_w = image.shape[:2]
    keypoints, bbox_info = dataset.load_yolo_label(img_path)

    # 原图 letterbox
    image_lb, ratio, pad = utils_letterbox(image, new_shape=args.img_size)
    # 将原图的归一化标签转换为像素坐标后传入 transform_labels（与 __getitem__ 保持一致）
    if keypoints is not None:
        kpts_px_orig = keypoints.copy()
        kpts_px_orig[:, 0] = kpts_px_orig[:, 0] * orig_w
        kpts_px_orig[:, 1] = kpts_px_orig[:, 1] * orig_h
    else:
        kpts_px_orig = None

    if bbox_info is not None:
        class_id, bbox_norm = bbox_info
        cx, cy, w_n, h_n = bbox_norm
        cx_abs = cx * orig_w
        cy_abs = cy * orig_h
        w_abs = w_n * orig_w
        h_abs = h_n * orig_h
        bbox_info_px = (class_id, [cx_abs, cy_abs, w_abs, h_abs])
    else:
        bbox_info_px = None

    kpts_tensor, bbox_tensor = dataset.transform_labels(kpts_px_orig, bbox_info_px, ratio, pad, orig_w, orig_h)

    # 增强图（与 __getitem__ 保持一致的转换）
    image_aug = image.copy()
    if keypoints is not None:
        # 把归一化 keypoints 转为像素坐标传入增强器
        kpts_px = keypoints.copy()
        kpts_px[:, 0] = kpts_px[:, 0] * orig_w
        kpts_px[:, 1] = kpts_px[:, 1] * orig_h
    else:
        kpts_px = None

    # 为保证增强器对目标框进行同步变换，将原始 bbox(像素)传入增强器
    try:
        # 从 bbox_info 构建像素坐标 bbox_px（如果存在）
        bbox_px = None
        if bbox_info is not None:
            _, bbox_norm = bbox_info
            cx, cy, w_n, h_n = bbox_norm
            cx_abs = cx * orig_w
            cy_abs = cy * orig_h
            w_abs = w_n * orig_w
            h_abs = h_n * orig_h
            x1 = cx_abs - w_abs / 2.0
            y1 = cy_abs - h_abs / 2.0
            x2 = cx_abs + w_abs / 2.0
            y2 = cy_abs + h_abs / 2.0
            bbox_px = (x1, y1, x2, y2)
        image_aug, kpts_px, aug_bbox = augmentor(image_aug, kpts_px, bbox_px)
    except Exception:
        logger.exception("增强器报错: %s", img_path)
        image_aug, kpts_px, aug_bbox = image.copy(), kpts_px, None

    if image_aug is None or image_aug.shape != image.shape:
        logger.warning("增强后图像无效，使用原图替代")
        image_aug = image.copy()

    # 将增强后的关键点像素坐标转换回归一化
    # 保持增强后关键点为像素坐标，transform_labels 会处理到 letterbox
    if kpts_px is not None:
        keypoints_aug = kpts_px.copy()
    else:
        keypoints_aug = None

    image_aug_lb, ratio_aug, pad_aug = utils_letterbox(image_aug, new_shape=args.img_size)

    # 解析增强后的 bbox 为像素中心格式 (cx_px, cy_px, w_px, h_px)
    if aug_bbox is not None:
        class_id = bbox_info[0] if bbox_info is not None else 0
        try:
            if len(aug_bbox) >= 4:
                x1, y1, x2, y2 = float(aug_bbox[0]), float(aug_bbox[1]), float(aug_bbox[2]), float(aug_bbox[3])
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                w_px = x2 - x1
                h_px = y2 - y1
                bbox_info_aug = (class_id, [cx_px, cy_px, w_px, h_px])
            else:
                bbox_info_aug = bbox_info
        except Exception:
            bbox_info_aug = bbox_info
    else:
        bbox_info_aug = bbox_info

    kpts_tensor_aug, bbox_tensor_aug = dataset.transform_labels(keypoints_aug, bbox_info_aug, ratio_aug, pad_aug, orig_w, orig_h)

    def plot_img_with_kpts_bbox(ax, img, kpts, bbox, title):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # kpts: normalized 相对于 letterbox 输出尺寸 (N,3)
        if kpts is not None:
            for i in range(kpts.shape[0]):
                x_norm, y_norm, v = kpts[i]
                x = float(x_norm) * img.shape[1]
                y = float(y_norm) * img.shape[0]
                color = 'lime' if v > 0.5 else 'red'
                ax.scatter(x, y, c=color, s=30)
        # bbox: tensor [cx,cy,w,h,class]
        if bbox is not None:
            cx, cy, w, h, cls = bbox.tolist()
            x1 = (cx - w/2.0) * img.shape[1]
            y1 = (cy - h/2.0) * img.shape[0]
            w_px = w * img.shape[1]
            h_px = h * img.shape[0]
            rect = mpatches.Rectangle((x1, y1), w_px, h_px, fill=False, edgecolor='yellow', linewidth=2)
            ax.add_patch(rect)
        ax.set_title(title)
        ax.axis('off')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_img_with_kpts_bbox(axs[0], image_lb, kpts_tensor, bbox_tensor, '原图 (Letterbox)')
    plot_img_with_kpts_bbox(axs[1], image_aug_lb, kpts_tensor_aug, bbox_tensor_aug, '增强后 (Letterbox)')
    plt.tight_layout()
    plt.show()



