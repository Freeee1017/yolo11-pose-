#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试集评估薄封装：复用 Validater 的通用评估与导出能力。"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from config_utils import load_config
from yolo_dataset import BabyPoseDataset, pad_collate
from model import create_model
from my_predict import load_weights
from validater import YOLOValidater


def build_test_loader(img_dir: str, label_dir: str, img_size: int, batch_size: int, num_workers: int) -> DataLoader:
    cfg = load_config('config.yaml')
    dataset = BabyPoseDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        is_train=False,
        augmentation=None,
        classes=cfg.get('classes', ['baby'])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
        drop_last=False
    )


def main():
    cfg = load_config('config.yaml')
    parser = argparse.ArgumentParser(description='在测试集上评估并导出 per-image CSV（复用 Validater）')
    parser.add_argument('--weights', type=str, default=r'runs\train\internet\best.pt', help='权重路径 .pt')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu 或 cuda:0')
    parser.add_argument('--img-size', type=int, default=cfg.get('img_size', 640), help='推理尺寸')
    parser.add_argument('--batch-size', type=int, default=cfg.get('batch_size', 16), help='批大小')
    parser.add_argument('--num-workers', type=int, default=cfg.get('num_workers', 4), help='DataLoader线程数')
    parser.add_argument('--test-img-dir', type=str, default='f:/dataset/images/test', help='测试集图片目录')
    parser.add_argument('--test-label-dir', type=str, default='f:/dataset/labels/test', help='测试集标签目录')
    parser.add_argument('--out-csv', type=str, default='test_results.csv', help='输出CSV文件路径')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 模型与权重
    model = create_model(nc=cfg.get('nc', 1), nk=cfg.get('nk', 21)).to(device)
    try:
        setattr(model, 'config', cfg)
        if hasattr(model, 'apply_config'):
            model.apply_config(cfg)
    except Exception:
        pass
    if args.weights and os.path.exists(args.weights):
        load_weights(model, args.weights, device=device)
    else:
        print(f"⚠ 未找到权重: {args.weights}，将使用随机初始化模型进行评估（仅调试用途）")

    # DataLoader
    test_loader = build_test_loader(
        img_dir=args.test_img_dir,
        label_dir=args.test_label_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 评估并导出
    validator = YOLOValidater(model, criterion=None, device=device, enable_tb=False)
    result = validator.evaluate_and_save(test_loader, args.out_csv, epoch=0, verbose=True)
    print(f"✓ 已保存测试结果到: {args.out_csv}")
    miou = result.get('metrics', {}).get('mean_iou', None)
    if miou is not None:
        print(f"  mean_iou: {float(miou):.4f}")


if __name__ == '__main__':
    main()
