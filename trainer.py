#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11婴儿关键点检测训练器
封装为可重用的训练器类
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import yaml
import json
import logging

# TensorBoard支持
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.getLogger(__name__).warning("TensorBoard 不可用，将不记录训练日志")
    class DummySummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def add_image(self, *args, **kwargs): pass
        def close(self): pass
    SummaryWriter = DummySummaryWriter

logger = logging.getLogger(__name__)

from loss import PoseLoss
from yolo_dataset import build_datasets_and_loaders
from validater import YOLOValidater
from metrics import MetricsTracker  # 精简后只需综合跟踪器
from utils import AverageMeter
from config_utils import load_config


class YOLOTrainer:
    """YOLO关键点检测训练器"""
    
    def __init__(self, model, config=None):
        """
        初始化训练器
        
        Args:
            model: YOLO11KeypointModel实例
            config: 训练配置字典
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.config = config if config is not None else load_config()
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')  # 仍追踪最优验证损失 (兼容旧逻辑)
        # ---- 通用早停指标配置 ----
        self.best_metric_name = self.config.get('early_stop_metric', 'val_loss')
        self.best_metric_mode = str(self.config.get('early_stop_mode', 'min')).lower()
        if self.best_metric_mode not in ('min', 'max'):
            self.best_metric_mode = 'min'
        try:
            self.best_metric_delta = float(self.config.get('early_stop_delta', 0.0))
        except Exception:
            self.best_metric_delta = 0.0
        self.early_stop_enabled = bool(self.config.get('early_stop', True))
        self.best_metric_value = None  # 首次一定视为改进
        self.best_metric_epoch = None
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0
        self.global_iter = 0
        self.base_lrs = None

        # 随机种子
        self._set_seed(self.config.get('seed'))

        # 初始化组件
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_directories()
        self._setup_tensorboard()
        self._setup_validater()
        self._setup_metrics_tracker()  # 添加指标跟踪器初始化
        # 将配置挂到模型上，便于验证器读取 (如 kpt_vis_threshold)
        try:
            setattr(self.model, 'config', self.config)
        except Exception:
            pass
        # 将配置同步到模型的 head（如 Top-K 融合等推理策略）
        try:
            if hasattr(self.model, 'apply_config'):
                self.model.apply_config(self.config)
            else:
                # 回退：直接写到 head 属性上（若存在）
                if hasattr(self.model, 'head') and self.model.head is not None:
                    hk = getattr(self.model.head, 'topk', None)
                    setattr(self.model.head, 'topk_fuse', bool(self.config.get('topk_fuse', getattr(self.model.head, 'topk_fuse', False))))
                    setattr(self.model.head, 'topk', int(self.config.get('topk', hk if hk is not None else 5)))
                    setattr(self.model.head, 'fuse_temp', float(self.config.get('fuse_temp', getattr(self.model.head, 'fuse_temp', 0.5))))
        except Exception:
            pass
        # 可视化相关配置
        self.visual_interval = self.config.get('visual_interval', 5)
        self.max_vis_images = self.config.get('visual_max_images', 2)

        # AMP 支持
        self.amp_enabled = bool(self.config.get('amp') and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        # CSV 路径
        self.csv_path = os.path.join(self.config['save_dir'], 'results.csv')
        # JSON 路径: 用于保存每个 epoch 的训练/验证损失
        self.epoch_losses_path = os.path.join(self.config['save_dir'], 'epoch_losses.json')
    
    def _setup_optimizer(self):
        """设置优化器（参数组区分是否应用 weight_decay）"""
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() == 1 or n.endswith('.bias'):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {'params': decay, 'weight_decay': self.config['weight_decay']},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        # 优先尝试 AdamW, 不可用则回退 SGD
        opt = None
        AdamWCls = getattr(optim, 'AdamW', None)
        if AdamWCls is not None:
            try:
                opt = AdamWCls(param_groups, lr=self.config['lr'])
            except Exception:
                opt = None
        if opt is None:
            SGDCls = getattr(optim, 'SGD', None)
            if SGDCls is None:
                raise RuntimeError('未找到可用优化器 AdamW 或 SGD')
            opt = SGDCls(param_groups, lr=self.config['lr'], momentum=0.9)
        self.optimizer = opt
        self.base_lrs = [g.get('lr', self.config['lr']) for g in self.optimizer.param_groups]
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['lr'] * 0.01
        )
    
    def _setup_criterion(self):
        """设置损失函数 & 动态损失项名称"""
        self.criterion = PoseLoss(self.model)
        # 现在 PoseLoss 返回五个分量：bbox, kpt_loc, kpt_vis, cls, dfl
        self.loss_names = ['bbox', 'kpt_loc', 'kpt_vis', 'cls', 'dfl']
    
    def _setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
    
    def _setup_tensorboard(self):
        """设置TensorBoard"""
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.config['log_dir'])
        else:
            self.writer = SummaryWriter()
    
    def _setup_validater(self):
        """设置验证器并配置可选可视化"""
        vis_enable = bool(self.config.get('val_visualize', True))
        vis_interval = int(self.config.get('val_visual_interval', 1))
        vis_max = int(self.config.get('val_visual_max_images', 2))
        if TENSORBOARD_AVAILABLE and hasattr(self.writer, 'add_scalar'):
            self.validater = YOLOValidater(
                self.model,
                self.criterion,
                self.device,
                writer=self.writer,  # type: ignore[arg-type]
                enable_tb=True,
                save_dir=os.path.join(self.config['save_dir'], 'val_vis'),
                visualize=vis_enable,
                vis_interval=vis_interval,
                vis_max_images=vis_max
            )
        else:
            self.validater = YOLOValidater(
                self.model,
                self.criterion,
                self.device,
                enable_tb=False,
                save_dir=os.path.join(self.config['save_dir'], 'val_vis'),
                visualize=vis_enable,
                vis_interval=vis_interval,
                vis_max_images=vis_max
            )
    
    def _setup_metrics_tracker(self):
        """设置指标跟踪器"""
        vis_thr = float(self.config.get('kpt_vis_threshold', 0.5))
        self.metrics_tracker = MetricsTracker(
            num_classes=self.model.nc,
            num_keypoints=self.model.nk,
            vis_threshold=vis_thr
        )

    def preprocess_batch(self, batch):
        """预处理 dataloader 返回的 batch（来自 `pad_collate`）。

        - 将所有 tensor 移动到设备并规范 dtype
        - 保留非 tensor 项（如 img_paths）原样返回
        返回: images, batch_on_device
        """
        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                try:
                    batch_on_device[k] = v.to(self.device)
                except Exception:
                    batch_on_device[k] = v
            else:
                batch_on_device[k] = v

        # 规范常用 key 的 dtype
        if 'images' in batch_on_device and isinstance(batch_on_device['images'], torch.Tensor):
            images = batch_on_device['images'].float()
            batch_on_device['images'] = images
        else:
            images = None

        if 'cls' in batch_on_device and isinstance(batch_on_device['cls'], torch.Tensor):
            batch_on_device['cls'] = batch_on_device['cls'].float()
        if 'bboxes' in batch_on_device and isinstance(batch_on_device['bboxes'], torch.Tensor):
            batch_on_device['bboxes'] = batch_on_device['bboxes'].float()
        if 'batch_idx' in batch_on_device and isinstance(batch_on_device['batch_idx'], torch.Tensor):
            batch_on_device['batch_idx'] = batch_on_device['batch_idx'].long()
        if 'keypoints' in batch_on_device and isinstance(batch_on_device['keypoints'], torch.Tensor):
            batch_on_device['keypoints'] = batch_on_device['keypoints'].float()

        return images, batch_on_device
    
    def setup_data_loaders(self):
        """设置数据集与加载器"""
        self.train_loader, self.val_loader, train_dataset, val_dataset = build_datasets_and_loaders()
            
        if len(train_dataset) == 0:
            raise ValueError(f"训练数据集为空，请检查数据路径: {self.config['data_root']}")
        
        if len(val_dataset) == 0:
            raise ValueError(f"验证数据集为空，请检查数据路径: {self.config['data_root']}")
        # 挂载到实例，便于外部访问并在此处打印信息（避免模块导入时执行）
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        logger.info("训练集: %d 样本", len(train_dataset))
        logger.info("验证集: %d 样本", len(val_dataset))
            

    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        losses = AverageMeter()
        bbox_losses = AverageMeter()
        kpt_loc_losses = AverageMeter()
        kpt_vis_losses = AverageMeter()
        cls_losses = AverageMeter()
        dfl_losses = AverageMeter()
        
        # 重置epoch指标
        self.metrics_tracker.start_epoch()
        
        # 创建进度条
        # Use dynamic_ncols so the progress bar adapts to terminal width and avoids truncation
        pbar = tqdm(
            self.train_loader,
            desc=f'Train Epoch {epoch+1}/{self.config["epochs"]}',
            dynamic_ncols=True,
            leave=False  # 训练完成后清除进度条
        )
        
        warmup_epochs = max(0, int(self.config.get('warmup_epochs', 0)))
        # 线性 warmup: 直接在 optimizer.param_groups 上修改 lr, 与后续 CosineAnnealingLR.step() 叠加。
        # 当前实现: warmup 期间手动线性提升到 base_lr, 之后 scheduler.cosine 会从当前组 lr 继续衰减。
        # 若需精确的 "warmup + cosine" 组合，可改为使用 SequentialLR 或在 scheduler 前置 warmup 调度器。
        warmup_iters = warmup_epochs * len(self.train_loader) if warmup_epochs > 0 else 0

        for batch_idx, batch in enumerate(pbar):
            # 通过 preprocess_batch 统一移动并格式化 batch
            images, batch_targets = self.preprocess_batch(batch)
            if images is None:
                # 回退到旧逻辑（保证向后兼容）
                images = batch['images'].to(self.device)
                batch_targets = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_targets[k] = v.to(self.device)
                    else:
                        batch_targets[k] = v
            
            # 前向 + 反向 (AMP 支持)
            self.optimizer.zero_grad(set_to_none=True)
            # 兼容不同 PyTorch 版本的 autocast 接口 (优先新API)
            try:  # 新版 (PyTorch >= 2.0) 建议使用 torch.amp.autocast
                from torch.amp import autocast as _autocast_new  # type: ignore
                autocast_ctx = _autocast_new(device_type='cuda', enabled=self.amp_enabled)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.amp_enabled)
            with autocast_ctx:
                predictions = self.model(images)  # train: (det_out_list, kpt_raw)
                total_loss, loss_items = self.criterion(predictions, batch_targets)
                # 分类分布统计（可配置开关）
                if self.config.get('log_cls_stats', True):
                    try:
                        # 从 criterion 内部的 pred_cls 无法直接拿，这里快速再 decode：
                        # predictions 训练态下来自 head: (det_out_list, kpt_raw)
                        if isinstance(predictions, (list, tuple)) and len(predictions) == 2:
                            det_out_list, _kpt_raw = predictions
                            # 聚合分类 logits
                            cls_logits_all = []
                            for feat in det_out_list:
                                # feat: [B, no, H, W]; 分类在最后 1 通道 (reg_max*4 之后)
                                reg_ch = self.criterion.reg_max * 4
                                cls_part = feat[:, reg_ch:, :, :].contiguous().view(feat.size(0), 1, -1)
                                cls_logits_all.append(cls_part)
                            cls_logits = torch.cat(cls_logits_all, dim=2)  # [B,1,N]
                            cls_probs = cls_logits.sigmoid()
                            cls_mean = cls_probs.mean().item()
                            cls_std = cls_probs.std().item()
                            cls_max = cls_probs.max().item()
                            # topk 平均（k=50 或所有）
                            k = min(50, cls_probs.shape[-1])
                            topk_vals, _ = torch.topk(cls_probs.view(cls_probs.size(0), -1), k=k, dim=1)
                            topk_mean = topk_vals.mean().item()
                            if batch_idx % 50 == 0:
                                self.writer.add_scalar('Train/cls_prob_mean', cls_mean, self.global_iter)
                                self.writer.add_scalar('Train/cls_prob_std', cls_std, self.global_iter)
                                self.writer.add_scalar('Train/cls_prob_max', cls_max, self.global_iter)
                                self.writer.add_scalar('Train/cls_prob_topk_mean', topk_mean, self.global_iter)
                    except Exception:
                        pass

            if self.amp_enabled:
                self.scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            # Warmup 线性增大学习率
            if warmup_iters > 0 and self.global_iter < warmup_iters and self.base_lrs:
                ratio = (self.global_iter + 1) / warmup_iters
                for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs or []):
                    try:
                        g['lr'] = base_lr * ratio
                    except Exception:
                        pass
            self.global_iter += 1
            
            # 记录损失
            losses.update(total_loss.item(), images.size(0))
            # 动态损失名称
            loss_names = self.loss_names
            # loss_items contains 5 components (scaled by lambdas in PoseLoss)
            loss_meters = [bbox_losses, kpt_loc_losses, kpt_vis_losses, cls_losses, dfl_losses]
            for meter, loss_val in zip(loss_meters, loss_items[:5]):
                try:
                    meter.update(loss_val.item(), images.size(0))
                except Exception:
                    pass
            
            # 更新指标跟踪器损失
            loss_dict = {
                'total_loss': total_loss.item(),
                **{name: loss_val.item() for name, loss_val in zip(loss_names, loss_items)}
            }
            self.metrics_tracker.update_losses(loss_dict)

            # 更新关键点指标 (从 raw 输出解码，逐图统计)
            try:
                # 仅在训练态下，predictions 为 (det_out_list, kpt_raw)
                if isinstance(predictions, (tuple, list)) and len(predictions) == 2:
                    det_out_list, kpt_raw = predictions
                    # 解码，确保 anchors/strides 初始化
                    try:
                        det_full = self.model.head._det_decode(det_out_list)  # [B,5,N]
                        kpt_dec = self.model.head._kpt_decode(images.size(0), kpt_raw)  # [B,63,N]
                        # 选择单锚点：与 head 推理一致的组合分数
                        cls_scores = det_full[:, 4, :]  # [B,N]
                        vis_probs = kpt_dec[:, 2::3, :]  # [B,21,N]
                        vis_mean = vis_probs.mean(dim=1)  # [B,N]
                        combined = cls_scores * (vis_mean.clamp(0,1) ** 0.5)
                        comb_max, _ = combined.max(dim=1, keepdim=True)
                        comb_std = combined.std(dim=1, keepdim=True)
                        fallback_flag = (comb_max <= 1e-3) | (comb_std <= 1e-6)
                        max_idx = combined.argmax(dim=1, keepdim=True)
                        if fallback_flag.any():
                            max_idx_vis = vis_mean.argmax(dim=1, keepdim=True)
                            max_idx[fallback_flag] = max_idx_vis[fallback_flag]
                        # 按索引选出每图关键点并重塑为 (B,21,3)
                        kpt_sel = self.model.head._select_single_target(kpt_dec, max_idx)  # [B,63,1]
                        kpt_sel = kpt_sel.squeeze(-1).view(kpt_sel.size(0), -1, 3)  # [B,21,3]

                        # 构建 GT 每图关键点 (与 validator 逻辑一致, 归一化到 [0,1] 的 letterbox 相对坐标)
                        B = images.size(0)
                        nk = getattr(self.model, 'nk', 21)
                        gt_kpts_per_image = torch.zeros((B, nk, 3), device=kpt_sel.device)
                        if 'batch_idx' in batch_targets and batch_targets['batch_idx'].numel() > 0 and batch_targets.get('keypoints') is not None:
                            obj_batch_idx = batch_targets['batch_idx']
                            kp_objs = batch_targets['keypoints']
                            for obj_i in range(obj_batch_idx.size(0)):
                                img_i = int(obj_batch_idx[obj_i].item())
                                if 0 <= img_i < B:
                                    gt_kpts_per_image[img_i] = kp_objs[obj_i].to(kpt_sel.device)

                        # 归一化与像素单位同时更新
                        img_h = images.shape[2]; img_w = images.shape[3]
                        kpt_px = kpt_sel.clone()
                        kpt_norm = kpt_px.clone()
                        kpt_norm[:, :, 0] = kpt_norm[:, :, 0] / float(img_w)
                        kpt_norm[:, :, 1] = kpt_norm[:, :, 1] / float(img_h)

                        # 训练指标中不再累计关键点 L1/可见点相关统计
                    except Exception:
                        pass
            except Exception:
                pass
            
            # 更新学习率到指标跟踪器
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics_tracker.update_lr(current_lr)
            
            # 每隔一定步数进行指标评估 (为了节省计算时间)
            # 简化版本: 暂不做中途预测指标统计（单目标回归）
            
            # 实时更新进度条显示（包含分项损失）
            try:
                # 使用更短的键并标注为已乘系数的值（scaled），以便一眼识别这是乘过 lambda 后的损失
                # include cls* and dfl* which are also provided by PoseLoss
                pbar.set_postfix({
                    'L*': f'{losses.avg:.4f}',
                    'BBox*': f'{bbox_losses.avg:.4f}',
                    'KLoc*': f'{kpt_loc_losses.avg:.4f}',
                    'KVis*': f'{kpt_vis_losses.avg:.4f}',
                    'Cls*': f'{cls_losses.avg:.4f}',
                    'DFL*': f'{dfl_losses.avg:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            except Exception:
                # 保险回退到仅显示总损失
                try:
                    pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'LR': f'{current_lr:.6f}'})
                except Exception:
                    pass
            
            # TensorBoard记录
            if batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', total_loss.item(), step)
                self.writer.add_scalar('Train/LearningRate', current_lr, step)
                
                for i, (name, loss_val) in enumerate(zip(loss_names, loss_items)):
                    if hasattr(loss_val, 'item'):
                        self.writer.add_scalar(f'Train/{name}', loss_val.item(), step)
        
        # 关闭进度条
        pbar.close()
        
        # 换行后记录详细的损失信息
        logger.info("Epoch %d Training Results:", epoch+1)
        logger.info("   Total Loss: %.6f", losses.avg)
        logger.info("   BBox Loss:  %.6f", bbox_losses.avg)
        logger.info("   Kpt Loc:    %.6f", kpt_loc_losses.avg)
        logger.info("   Kpt Vis:    %.6f", kpt_vis_losses.avg)
        logger.info("   Cls Loss:   %.6f", cls_losses.avg)
        logger.info("   DFL Loss:   %.6f", dfl_losses.avg)
        logger.info("   LR:         %.8f", current_lr)

        # 训练阶段不再统计/记录关键点辅助指标（按需求删除）

        # 结束epoch指标统计
        self.metrics_tracker.end_epoch()
        train_stats = {
            'train_total_loss': losses.avg,
            'train_bbox_loss': bbox_losses.avg,
            'train_kpt_loc_loss': kpt_loc_losses.avg,
            'train_kpt_vis_loss': kpt_vis_losses.avg,
            'train_cls_loss': cls_losses.avg,
            'train_dfl_loss': dfl_losses.avg,
            'lr': current_lr
        }
        return train_stats

    
    # _extract_predictions_for_metrics 已移除（单目标简化不需要）
    
    def validate(self, epoch):
        """验证 - 调用外部验证器并集成指标跟踪"""
        logger.info("🔍 Validating Epoch %d...", epoch+1)

        val_loss, metrics = self.validater.validate_epoch(
            self.val_loader,
            epoch=epoch,
            verbose=True  # 让验证器显示进度条
        )

        # 换行后记录验证结果
        logger.info("📈 Epoch %d Validation Results:", epoch+1)
        logger.info("   Val Loss:   %.6f", val_loss)
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info("   %s: %.6f", key, value)

        # 不再记录训练阶段关键点 L1 到验证面板

        # TensorBoard记录：Val/Loss 由 Trainer 写入；细分 Val/* 指标由 Validater 统一写入，避免重复
        self.writer.add_scalar('Val/Loss', val_loss, epoch)

        # 更新最佳指标到跟踪器
        self.metrics_tracker.training_metrics.update_best_metrics(metrics)

        # 自动保存 per-image 验证结果为 CSV（默认启用，可在 config 中关闭）
        try:
            if self.config.get('save_val_per_image_csv', True):
                csv_dir = os.path.join(self.config['save_dir'], 'validation')
                os.makedirs(csv_dir, exist_ok=True)
                csv_path = os.path.join(csv_dir, f'val_per_image_epoch{epoch+1}.csv')
                try:
                    self.validater.save_validation_csv(csv_path)
                    logger.info("Saved per-image validation CSV: %s", csv_path)
                except Exception:
                    logger.exception("保存 per-image CSV 失败: %s", csv_path)
        except Exception:
            logger.exception("尝试保存 per-image CSV 时发生错误")

        return val_loss, metrics
    
    def evaluate_model(self):
        """使用验证器进行轻量评估 (返回验证损失与指标)。"""
        if not hasattr(self, 'val_loader'):
            self.setup_data_loaders()
        return self.validater.evaluate_model(self.val_loader, epoch=0)
    
    def save_checkpoint(self, epoch, loss, metrics, is_best=False):
        """保存 last + (可选) best 检查点，包含随机与 AMP 状态"""
        ckpt_dir = os.path.join(self.config['save_dir'], 'weights')
        os.makedirs(ckpt_dir, exist_ok=True)
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': self.best_metric_value,
            'best_metric_mode': self.best_metric_mode,
            'best_metric_delta': self.best_metric_delta,
            'best_metric_epoch': self.best_metric_epoch,
            'amp': self.amp_enabled,
            'scaler': self.scaler.state_dict() if self.amp_enabled else None,
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        last_path = os.path.join(ckpt_dir, 'last.pt')
        try:
            torch.save(state, last_path)
        except Exception:
            logger.exception("保存 last 失败: %s", last_path)
        # 同时可选地保存一个仅包含 model.state_dict() 的轻量级副本，便于直接对比/替换
        try:
            save_weights_only = bool(self.config.get('save_weights_only_copy', True))
        except Exception:
            save_weights_only = True

        if save_weights_only:
            try:
                last_weights_path = os.path.join(ckpt_dir, 'last_weights.pt')
                torch.save(self.model.state_dict(), last_weights_path)
            except Exception:
                logger.exception("保存 last_weights 失败: %s", last_weights_path)

        if is_best:
            best_path = os.path.join(ckpt_dir, 'best.pt')
            try:
                torch.save(state, best_path)
                logger.info("新的最佳模型已保存: %s (Val Loss: %.6f)", best_path, loss)
            except Exception as e:
                logger.exception("保存 best 失败: %s", best_path)
            if save_weights_only:
                try:
                    best_weights_path = os.path.join(ckpt_dir, 'best_weights.pt')
                    torch.save(self.model.state_dict(), best_weights_path)
                except Exception:
                    logger.exception("保存 best_weights 失败: %s", best_weights_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点并恢复训练/随机/AMP状态"""
        if not os.path.isfile(checkpoint_path):
            logger.error("检查点文件不存在: %s", checkpoint_path)
            return False

        logger.info("加载检查点: %s", checkpoint_path)
        try:
            # 首先尝试完整加载
            ckpt = torch.load(checkpoint_path, map_location=self.device)

            # 判断是完整 checkpoint (包含 epoch/optimizer 等) 还是仅 weights/state_dict
            if isinstance(ckpt, dict) and ('epoch' in ckpt or 'optimizer_state_dict' in ckpt or 'best_loss' in ckpt):
                full_ckpt = True
            else:
                full_ckpt = False

            if full_ckpt:
                state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
            else:
                # 这是一个仅包含 state_dict 的文件（或纯权重字典）
                state_dict = ckpt
                logger.info("检测到 weights-only checkpoint，仅加载模型权重")

            if not isinstance(state_dict, dict):
                logger.warning("检查点不包含有效的 state_dict，尝试按纯权重字典解析")
            self.model.load_state_dict(state_dict, strict=False)

            # 优化器等可选
            if full_ckpt:
                opt_state = ckpt.get('optimizer_state_dict')
                if opt_state:
                    try:
                        self.optimizer.load_state_dict(opt_state)
                    except Exception:
                        logger.exception("优化器状态恢复失败(忽略)")
                sch_state = ckpt.get('scheduler_state_dict')
                if sch_state and self.scheduler:
                    try:
                        self.scheduler.load_state_dict(sch_state)
                    except Exception:
                        logger.exception("调度器状态恢复失败(忽略)")
                if self.amp_enabled and ckpt.get('scaler'):
                    try:
                        self.scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        logger.warning("AMP scaler 状态恢复失败(忽略)")

                self.start_epoch = ckpt.get('epoch', -1) + 1
                self.best_loss = ckpt.get('best_loss', self.best_loss)
            else:
                # weights-only 文件: 不能恢复优化器/epoch 等，用户应从 epoch 0 重新训练或手动设置
                self.start_epoch = 0

            # 随机状态恢复（类型检查）
            try:
                rng_state = ckpt.get('rng_state')
                if rng_state is not None:
                    try:
                        torch.set_rng_state(rng_state)
                    except Exception:
                        logger.warning("无法恢复 torch RNG state，类型: %s", type(rng_state))
            except Exception:
                logger.exception("RNG 状态恢复失败(跳过)")

            try:
                np_state = ckpt.get('numpy_rng_state')
                if np_state is not None:
                    np.random.set_state(np_state)
            except Exception:
                logger.warning("numpy 随机状态恢复失败(忽略)")

            try:
                py_state = ckpt.get('python_rng_state')
                if py_state is not None:
                    random.setstate(py_state)
            except Exception:
                logger.warning("python 随机状态恢复失败(忽略)")

            if ckpt.get('cuda_rng_state_all') and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(ckpt['cuda_rng_state_all'])
                except Exception:
                    logger.warning("CUDA RNG 状态恢复失败(忽略)")

            logger.info("成功加载检查点(权重+部分状态)，从 epoch %d 继续", self.start_epoch)
            return True

        except Exception:
            logger.exception("加载检查点失败: %s", checkpoint_path)
            logger.info("回退: 仅尝试加载模型权重 (strict=False)")
            try:
                weights = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(weights, dict):
                    cand = weights.get('model_state_dict') or weights.get('state_dict') or weights
                    if isinstance(cand, dict):
                        try:
                            self.model.load_state_dict(cand, strict=False)
                            logger.info("仅权重加载成功，重新从 epoch 0 训练")
                            self.start_epoch = 0
                            return True
                        except Exception:
                            logger.exception("仅权重加载也失败: %s", checkpoint_path)
            except Exception:
                logger.exception("尝试仅权重加载也失败: %s", checkpoint_path)
            return False
    
    def train(self, resume_from=None):
        """
        开始训练
        
        Args:
            resume_from: 恢复训练的检查点路径
        """
        logger.info("开始训练，设备: %s", self.device)

        # 设置数据加载器 - 如果失败会直接抛出异常
        self.setup_data_loaders()
        
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        start_time = time.time()
        
        try:
            # CSV 头部是否已写
            csv_header_written = os.path.exists(self.csv_path)
            metric_header_keys = None  # 记录验证指标列顺序

            for epoch in range(self.start_epoch, self.config['epochs']):
                epoch_start_time = time.time()

                # 训练
                train_stats = self.train_epoch(epoch)
                train_loss = train_stats['train_total_loss']
                self.train_losses.append(train_loss)

                # 预设验证占位
                val_loss = None
                metrics = {}

                performed_val = False
                if epoch % self.config['val_interval'] == 0:
                    val_loss, metrics = self.validate(epoch)
                    self.val_losses.append(val_loss)
                    performed_val = True
                    # 始终维护 best_loss (兼容)
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss

                    monitor_key = self.best_metric_name or 'val_loss'
                    if monitor_key == 'val_loss':
                        monitor_val = val_loss
                    else:
                        monitor_val = metrics.get(monitor_key)
                        if monitor_val is None:
                            logger.warning("early_stop_metric '%s' 不存在于当前 metrics, 回退使用 val_loss", monitor_key)
                            monitor_key = 'val_loss'
                            monitor_val = val_loss

                    improved = False
                    if self.best_metric_value is None:
                        improved = True
                    else:
                        if self.best_metric_mode == 'min':
                            improved = (monitor_val < self.best_metric_value - self.best_metric_delta)
                        else:
                            improved = (monitor_val > self.best_metric_value + self.best_metric_delta)

                    if improved:
                        self.best_metric_value = float(monitor_val)
                        self.best_metric_epoch = epoch
                        self.patience_counter = 0
                        is_best = True
                    else:
                        self.patience_counter += 1
                        is_best = False

                    if self.early_stop_enabled and self.patience_counter >= self.config['patience']:
                        logger.info(
                            "⚠️  Early stopping on %s (%s mode, Δ<=%.4f) no improvement for %d epochs | best=%.6f @ epoch %s",
                            monitor_key, self.best_metric_mode, self.best_metric_delta, self.config['patience'],
                            self.best_metric_value if self.best_metric_value is not None else float('nan'),
                            (self.best_metric_epoch+1) if self.best_metric_epoch is not None else 'N/A'
                        )
                        self.save_checkpoint(epoch, val_loss, metrics, is_best)
                        if self.config.get('save_csv', True):
                            self._append_results_csv(epoch, train_stats, val_loss, metrics,
                                                     csv_header_written, metric_header_keys)
                        break

                    # 保存改进模型
                    self.save_checkpoint(epoch, val_loss, metrics, is_best)

                # 更新学习率（放在验证后，保持与大多数实践一致）
                self.scheduler.step()

                # 训练期关键点覆盖图功能已移除，避免无效分支与混淆

                # 写入 / 追加 CSV （包含验证列）
                if self.config.get('save_csv', True):
                    # 首次验证时建立指标列顺序
                    if performed_val and metric_header_keys is None:
                        metric_header_keys = [k for k, v in metrics.items() if isinstance(v, (int, float, np.number))]
                    self._append_results_csv(epoch, train_stats, val_loss if performed_val else None, metrics,
                                             csv_header_written, metric_header_keys)
                    csv_header_written = True

                # Persist epoch losses to JSON (one file for easy plotting & inspection)
                try:
                    self._save_epoch_losses_json(epoch+1, train_loss, val_loss if performed_val else None)
                except Exception:
                    logger.exception("写入 epoch_losses.json 失败")

                # 打印总结
                self._print_epoch_summary(epoch, train_loss, epoch_start_time, start_time)
                
        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
        except Exception as e:
            logger.exception("训练过程中发生错误: %s", e)
        finally:
            self.writer.close()
            logger.info('训练结束！')
            
            # 打印最终训练总结
            self._print_final_summary()

    def _append_results_csv(self, epoch, train_stats, val_loss, val_metrics, header_written, metric_keys):
        """将单个 epoch 的训练与验证结果写入 CSV。

    列结构:
    epoch,train_total_loss,train_bbox_loss,train_kpt_loc_loss,train_kpt_vis_loss,train_cls_loss,train_dfl_loss,
    val_total_loss,val_bbox_loss,val_kpt_loc_loss,val_kpt_vis_loss,val_cls_loss,val_dfl_loss,
    lr,<val_metric_1>...
        仅当本 epoch 进行了验证才写入 val_loss 与指标, 否则为空。
        """
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            # 构建字段字典，便于与既有头对齐或生成新头
            fields = {}
            # epoch
            fields['epoch'] = str(epoch+1)
            # 训练损失
            fields['train_total_loss'] = f"{train_stats['train_total_loss']:.6f}"
            fields['train_bbox_loss'] = f"{train_stats.get('train_bbox_loss',0.0):.6f}"
            fields['train_kpt_loc_loss'] = f"{train_stats.get('train_kpt_loc_loss',0.0):.6f}"
            fields['train_kpt_vis_loss'] = f"{train_stats.get('train_kpt_vis_loss',0.0):.6f}"
            fields['train_cls_loss'] = f"{train_stats.get('train_cls_loss',0.0):.6f}"
            fields['train_dfl_loss'] = f"{train_stats.get('train_dfl_loss',0.0):.6f}"
            # 学习率
            fields['lr'] = f"{train_stats['lr']:.8f}"

            # 验证损失
            if val_loss is not None:
                fields['val_total_loss'] = f"{val_loss:.6f}"
                fields['val_loss'] = f"{val_loss:.6f}"  # 兼容旧列名
                if isinstance(val_metrics, dict):
                    def fmt(v):
                        return f"{float(v):.6f}" if isinstance(v,(int,float,np.number)) else ""
                    fields['val_bbox_loss'] = fmt(val_metrics.get('bbox_loss'))
                    fields['val_kpt_loc_loss'] = fmt(val_metrics.get('kpt_loc_loss'))
                    fields['val_kpt_vis_loss'] = fmt(val_metrics.get('kpt_vis_loss'))
                    fields['val_cls_loss'] = fmt(val_metrics.get('cls_loss'))
                    fields['val_dfl_loss'] = fmt(val_metrics.get('dfl_loss'))
                    # 其它验证指标
                    miou = val_metrics.get('mean_iou')
                    if isinstance(miou,(int,float,np.number)):
                        fields['val_mean_iou'] = f"{float(miou):.6f}"
                    # 每关键点 MAE
                    for i in range(getattr(self.model, 'nk', 21)):
                        key = f'kpt_{i}_mae'
                        v = val_metrics.get(key)
                        fields['val_'+key] = f"{float(v):.6f}" if isinstance(v,(int,float,np.number)) else ""
                    # 其它数值型 metrics，避免重复
                    if metric_keys:
                        for k in metric_keys:
                            if k in ('mean_iou','bbox_loss','kpt_loc_loss','kpt_vis_loss','cls_loss','dfl_loss') or k.startswith('kpt_'):
                                continue
                            v = val_metrics.get(k)
                            if isinstance(v,(int,float,np.number)):
                                fields['val_'+k] = f"{float(v):.6f}"

            # 如果已有头，按现有列对齐；否则写新头（新顺序满足你的要求）
            if header_written:
                with open(self.csv_path,'r',encoding='utf-8') as f:
                    existing_header = f.readline().strip()
                existing_cols = existing_header.split(',') if existing_header else []
                line_parts = [fields.get(col, '') for col in existing_cols]
                header_cols = existing_cols
            else:
                header_cols = [
                    'epoch',
                    'train_total_loss','train_bbox_loss','train_kpt_loc_loss','train_kpt_vis_loss','train_cls_loss','train_dfl_loss',
                    'val_total_loss','val_bbox_loss','val_kpt_loc_loss','val_kpt_vis_loss','val_cls_loss','val_dfl_loss',
                    'lr',
                    'val_mean_iou',
                ]
                # 逐个追加 val_kpt_i_mae
                header_cols.extend([f'val_kpt_{i}_mae' for i in range(getattr(self.model, 'nk', 21))])
                # 追加其它 val_* 指标（若 metric_keys 指定）
                if metric_keys:
                    for k in metric_keys:
                        if k in ('mean_iou','bbox_loss','kpt_loc_loss','kpt_vis_loss','cls_loss','dfl_loss') or k.startswith('kpt_'):
                            continue
                        col = 'val_'+k
                        if col not in header_cols:
                            header_cols.append(col)
                line_parts = [fields.get(col, '') for col in header_cols]

            # 写文件
            with open(self.csv_path,'a',encoding='utf-8') as f:
                if not header_written:
                    f.write(','.join(header_cols)+'\n')
                f.write(','.join(line_parts)+'\n')
        except Exception as e:
            logger.exception("写入 CSV 失败: %s", e)

    def _save_epoch_losses_json(self, epoch: int, train_loss: float, val_loss):
        """Append epoch loss record to a JSON file.

        The file stores a list of records, each:
            {"epoch": int, "train_loss": float, "val_loss": float|null, "timestamp": "..."}

        Args:
            epoch: 1-based epoch index
            train_loss: training loss value
            val_loss: validation loss value or None
        """
        try:
            os.makedirs(os.path.dirname(self.epoch_losses_path), exist_ok=True)
            # load existing
            records = []
            if os.path.exists(self.epoch_losses_path):
                try:
                    with open(self.epoch_losses_path, 'r', encoding='utf-8') as f:
                        records = json.load(f) or []
                except Exception:
                    # corrupt or unreadable -> back up and start fresh
                    try:
                        backup = self.epoch_losses_path + ".bak"
                        os.replace(self.epoch_losses_path, backup)
                    except Exception:
                        pass
                    records = []

            import datetime
            rec = {
                'epoch': int(epoch),
                'train_loss': float(train_loss) if train_loss is not None else None,
                'val_loss': float(val_loss) if val_loss is not None else None,
                'timestamp': datetime.datetime.now().isoformat()
            }
            records.append(rec)
            # atomic write
            tmp_path = self.epoch_losses_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.epoch_losses_path)
        except Exception:
            logger.exception("无法写入 epoch losses JSON")
    
    def _print_epoch_summary(self, epoch, train_loss, epoch_start_time, total_start_time):
        """简化的epoch总结"""
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time
        remaining_epochs = self.config['epochs'] - epoch - 1
        remaining_time = elapsed_time * remaining_epochs / (epoch - self.start_epoch + 1) if epoch > self.start_epoch else 0
        
        logger.info("⏱️  Epoch %d/%d Summary:", epoch+1, self.config['epochs'])
        logger.info("   Time: %.1fs | Elapsed: %.1fh | ETA: %.1fh", epoch_time, elapsed_time/3600.0, remaining_time/3600.0)
        
        # 验证损失和早停信息
        if self.val_losses:
            if self.early_stop_enabled:
                metric_str = f"{self.best_metric_name}={self.best_metric_value:.6f}" if self.best_metric_value is not None else "N/A"
                logger.info("   Best Loss: %.6f | Monitored(%s,%s): %s | Patience %d/%d",
                            self.best_loss, self.best_metric_mode, f"Δ>{self.best_metric_delta}" if self.best_metric_delta>0 else 'Δ>0',
                            metric_str, self.patience_counter, self.config['patience'])
            else:
                logger.info("   Best Loss: %.6f | Early Stop: disabled", self.best_loss)
            
        logger.info("%s", "-" * 60)
    
    def _print_final_summary(self):
        """打印最终训练总结"""
        logger.info("%s", '='*80)
        logger.info("训练完成总结")
        logger.info("%s", '='*80)
        
        # 基本统计
        total_epochs = len(self.train_losses)
        logger.info("总共训练: %d epochs", total_epochs)
        logger.info("最佳验证损失: %.4f", self.best_loss)
        
        if self.train_losses:
            logger.info("最终训练损失: %.4f", self.train_losses[-1])
            logger.info("最佳训练损失: %.4f", min(self.train_losses))
        
        if self.val_losses:
            logger.info("最终验证损失: %.4f", self.val_losses[-1])
        if self.best_metric_value is not None:
            logger.info("监控指标 %s (%s) 最佳值: %.6f (epoch %s)",
                        self.best_metric_name, self.best_metric_mode,
                        self.best_metric_value,
                        (self.best_metric_epoch+1) if self.best_metric_epoch is not None else 'N/A')
        
        # 获取完整的训练总结
        try:
            final_metrics = self.metrics_tracker.compute_all_metrics()
            training_summary = final_metrics.get('training_summary', {}) if isinstance(final_metrics, dict) else {}

            if isinstance(training_summary, dict):
                logger.info("训练效率统计:")
                total_time_hours = training_summary.get('total_time_hours', 0.0)
                avg_epoch_time_minutes = training_summary.get('avg_epoch_time_minutes', 0.0)
                logger.info("  总训练时间: %.2f 小时", total_time_hours)
                logger.info("  平均每epoch时间: %.1f 分钟", avg_epoch_time_minutes)

                best_metrics = training_summary.get('best_metrics', {})
                if isinstance(best_metrics, dict) and best_metrics:
                    logger.info("验证过程中的最佳指标:")
                    for key, value in best_metrics.items():
                        try:
                            logger.info("  %s: %.4f", key, float(value))
                        except Exception:
                            logger.info("  %s: %s", key, value)
            
        except Exception:
            logger.exception("获取训练总结时出错")
        
        logger.info("%s", '='*80)
    
    def get_training_summary(self):
        """获取训练总结"""
        return {
            'total_epochs': len(self.train_losses),
            'best_loss': self.best_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    # 设定随机种子
    def _set_seed(self, seed):
        if seed is None:
            return
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            logger.exception("设定随机种子失败")



