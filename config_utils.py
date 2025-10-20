#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的配置加载与训练配置工具

提供:
- load_config: 读取 config.yaml，返回 dict；文件不存在或解析失败时给出默认值并打印提示
- load_training_config: 在 load_config 基础上合并默认值并支持覆盖（CLI/debug等）
- validate_config: 校验关键路径与核心参数范围
- print_config_summary: 友好的配置摘要打印
- check_environment: 训练环境快速检查（GPU、可选目录创建）
- set_seed: 统一的随机种子设置
- create_experiment_name: 根据配置生成实验名称

尽量将通用逻辑集中，避免 my_train 或其他模块重复实现。
"""
from __future__ import annotations
import os
import yaml
from typing import Dict, Any, Optional
import torch
import numpy as np
import datetime

# 全局默认配置，可按需扩展
DEFAULT_CONFIG: Dict[str, Any] = {
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 0.0005,
    'batch_size': 8,
    'img_size': 640,
    'num_workers': 4,
    'seed': 42,
    # data path specific keys (present in config.yaml)
    'train_img_dir': 'dataset/images/train',
    'train_label_dir': 'dataset/labels/train',
    'val_img_dir': 'dataset/images/val',
    'val_label_dir': 'dataset/labels/val',
    # augmentation probabilities / ranges (from config.yaml)
    'horizontal_flip_prob': 0.5,
    'rotation_prob': 0.3,
    'rotation_range': 15.0,
    'scale_prob': 0.3,
    'scale_range': [0.9, 1.1],
    'translation_prob': 0.3,
    'translation_range': 0.05,
    'color_prob': 0.4,
    'brightness_range': 0.1,
    'contrast_range': 0.1,
    # training schedule / lr strategy (from config.yaml)
    'warmup_epochs': 3,
    # loss related optional keys
    'lambda_box': 7.5,
    'lambda_cls': 0.5,
    'lambda_kpt': 12.0,
    'lambda_kpt_vis': 1.0,
    'lambda_dfl': 1.5,
    'nk': 21,
    'nc': 1,
    'classes': ['baby'],
}

def load_config(path: str = "config.yaml", merge_default: bool = True) -> Dict[str, Any]:
    """加载 YAML 配置。失败时返回默认配置（或空字典）。
    Args:
        path: 配置文件路径
        merge_default: 是否将读取结果与 DEFAULT_CONFIG 合并（优先用户值）
    Returns:
        dict
    """
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                print(f"配置文件格式异常(非字典)，忽略: {path}")
                loaded = {}
            data = loaded
        except Exception as e:
            print(f"读取配置失败: {e} -> 使用默认配置")
    else:
        # 静默：训练初期可能还没写 config.yaml
        pass
    if merge_default:
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    return data

def load_training_config(config_path: str = "config.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """加载训练配置，合并 DEFAULT_CONFIG 与 YAML，并应用可选覆盖。

    Args:
        config_path: 配置文件路径
        overrides: 额外覆盖（例如来自 CLI 的 debug/early_stop 开关）
    Returns:
        dict: 合并后的配置
    """
    cfg = load_config(config_path, merge_default=True)
    if overrides and isinstance(overrides, dict):
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    # 兜底输出路径
    cfg.setdefault('save_dir', 'runs/train')
    cfg.setdefault('log_dir', 'runs/logs')
    return cfg


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置参数的有效性（路径与关键数值范围）。

    - 路径检查与当前 yolo_dataset.build_datasets_and_loaders 约定一致
    - 数值范围：epochs/lr/batch_size > 0
    """
    print("\n验证训练配置...")

    data_root = config.get('data_root', 'dataset')
    if not os.path.exists(data_root):
        print(f"✗ 数据根目录不存在: {data_root}")
        return False

    train_img_dir = config.get('train_img_dir', 'dataset/images/train')
    val_img_dir = config.get('val_img_dir', 'dataset/images/val')
    train_label_dir = config.get('train_label_dir', 'dataset/labels/train')
    val_label_dir = config.get('val_label_dir', 'dataset/labels/val')

    for pth, desc in [
        (train_img_dir, '训练图像目录'),
        (val_img_dir, '验证图像目录'),
        (train_label_dir, '训练标签目录'),
        (val_label_dir, '验证标签目录'),
    ]:
        if not os.path.exists(pth):
            print(f"✗ {desc}不存在: {pth}")
            return False

    if config.get('epochs', 0) <= 0:
        print("✗ epochs必须大于0")
        return False
    if config.get('lr', 0) <= 0:
        print("✗ 学习率必须大于0")
        return False
    if config.get('batch_size', 0) <= 0:
        print("✗ batch_size必须大于0")
        return False

    print("✓ 配置验证通过")
    return True


def print_config_summary(config: Dict[str, Any]) -> None:
    """打印配置摘要。"""
    print("\n" + "="*60)
    print("训练配置摘要")
    print("="*60)
    print("📊 基本参数:")
    print(f"  - 训练轮数: {config.get('epochs', 'N/A')}")
    print(f"  - 批次大小: {config.get('batch_size', 'N/A')}")
    print(f"  - 学习率: {config.get('lr', 'N/A')}")
    print(f"  - 图像尺寸: {config.get('img_size', 'N/A')}")
    print(f"  - 权重衰减: {config.get('weight_decay', 'N/A')}")
    print("\n🎯 模型参数:")
    print(f"  - 类别数: {config.get('nc', 'N/A')}")
    print(f"  - 关键点数: {config.get('nk', 'N/A')}")
    print("\n📁 数据路径:")
    print(f"  - 数据根目录: {config.get('data_root', 'N/A')}")
    print(f"  - 训练图像: {config.get('train_img_dir', 'N/A')}")
    print(f"  - 验证图像: {config.get('val_img_dir', 'N/A')}")
    print("\n💾 输出路径:")
    print(f"  - 模型保存: {config.get('save_dir', 'N/A')}")
    print(f"  - 训练日志: {config.get('log_dir', 'N/A')}")
    print("\n🔧 训练策略:")
    print(f"  - 早停轮数: {config.get('patience', 'N/A')}")
    print(f"  - 验证间隔: {config.get('val_interval', 'N/A')}")
    print(f"  - 保存间隔: {config.get('save_interval', 'N/A')}")
    print(f"  - 早停: {config.get('early_stop', 'N/A')}")
    print("\n🎨 数据增强:")
    # 尝试兼容两套命名（config.yaml 与旧 my_train 默认）
    print(f"  - 水平翻转: {config.get('horizontal_flip_prob', config.get('fliplr', 'N/A'))}")
    print(f"  - 旋转概率: {config.get('rotation_prob', config.get('rotation', 'N/A'))}")
    print(f"  - 缩放概率: {config.get('scale_prob', config.get('scale', 'N/A'))}")
    print(f"  - 亮度调整: {config.get('brightness_range', config.get('brightness', 'N/A'))}")
    print("="*60)


def check_environment(create_dirs: bool = False, save_dir: Optional[str] = None, log_dir: Optional[str] = None) -> bool:
    """检查训练环境。默认仅报告 GPU 情况；目录创建交由 Trainer 兜底。

    Args:
        create_dirs: 是否创建输出目录（通常不需要，Trainer 会创建）
        save_dir: 模型保存目录（当 create_dirs=True 时使用）
        log_dir: 日志目录（当 create_dirs=True 时使用）
    """
    print("检查训练环境...")
    if torch.cuda.is_available():
        try:
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU可用: {device} ({memory:.1f}GB)")
        except Exception:
            print("✓ GPU可用")
    else:
        print("⚠ GPU不可用，将使用CPU训练（速度较慢）")

    if create_dirs and save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"✓ 创建目录: {save_dir}")
        except Exception:
            pass
    if create_dirs and log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"✓ 创建目录: {log_dir}")
        except Exception:
            pass

    print("✓ 环境检查完成")
    return True


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """设置随机种子以提升复现性。"""
    try:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)
        print(f"✓ 已设置随机种子: {seed}")
    except Exception:
        print("⚠ 设置随机种子失败（忽略）")


def create_experiment_name(config: Dict[str, Any], prefix: str = "yolo11_pose") -> str:
    """生成实验名称，如: yolo11_pose_e50_b16_lr0.001_YYYYmmdd_HHMMSS"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs = config.get('epochs', 'unknown')
    batch_size = config.get('batch_size', 'unknown')
    lr = config.get('lr', 'unknown')
    return f"{prefix}_e{epochs}_b{batch_size}_lr{lr}_{timestamp}"


__all__ = [
    "load_config",
    "DEFAULT_CONFIG",
    "load_training_config",
    "validate_config",
    "print_config_summary",
    "check_environment",
    "set_seed",
    "create_experiment_name",
]
