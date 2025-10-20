#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, Tuple, cast
import csv
import torch
import os
import numpy as _np
from utils import xywh2xyxy, inv_letterbox_xyxy, inv_letterbox_kpts
from metrics import ValidationMetrics
from torch.utils.data import DataLoader

try:  # TensorBoard 可选
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except Exception:  # 运行期缺失时提供占位
    _TB_AVAILABLE = False
    class SummaryWriter:  # type: ignore
        pass


class YOLOValidater:
    def __init__(self, model, criterion, device=None,
                 writer: Optional[SummaryWriter] = None,
                 enable_tb: bool = True,
                 save_dir: Optional[str] = None,
                 visualize: bool = False,
                 vis_interval: int = 1,
                 vis_max_images: int = 2):
        self.model = model
    # 训练期的损失函数对象保留用于兼容（本验证流程以指标计算为主，仍可选计算损失）
        self.criterion = criterion
        self.device = device or next(model.parameters()).device
        self.writer = writer
        self.enable_tb = enable_tb and writer is not None and _TB_AVAILABLE
        
    # 可视化相关配置
        self.save_dir = save_dir
        self.visualize = visualize
        self.vis_interval = max(1, int(vis_interval))
        self.vis_max_images = max(1, int(vis_max_images))
        # 可见性阈值：从模型配置读取，供可视化使用
        try:
            cfg = getattr(model, 'config', None)
            self.vis_threshold = float(cfg.get('kpt_vis_threshold', 0.5)) if isinstance(cfg, dict) else 0.5
        except Exception:
            self.vis_threshold = 0.5

    # --- 使用 utils 中的反 letterbox 映射工具 ---

    def _maybe_visualize_batch(self, images: torch.Tensor, preds, batch: Dict[str, Any], epoch: int):
        # 对一个 batch 的前若干张图进行预测可视化：
        # - 若可取得 head 的原始输出，则按推理路径解码（与 head 的逻辑一致）
        # - 选择置信度最大的 anchor（与 head 相同的选择规则）
        # - 将 bbox/关键点从 letterbox 尺度映射回原图坐标并保存图片
        if not self.visualize:
            return
        if (epoch + 1) % self.vis_interval != 0:
            return
        try:
            save_dir = self.save_dir or os.path.join('runs', 'val_vis')
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            return

        B = images.size(0)
        max_img = min(self.vis_max_images, B)

    # 尝试从 preds 中获取原始输出（以便按 head 的路径进行解码）
        det_out = None
        kpt_raw = None
        single_tensor = None
        try:
            if isinstance(preds, tuple) and len(preds) >= 2 and isinstance(preds[1], (list, tuple)):
                # preds = (single, (det_out, kpt_raw)) per head.py
                det_out, kpt_raw = preds[1]
            elif isinstance(preds, tuple) and len(preds) >= 1:
                # fallback: only single target tensor
                single_tensor = preds[0]
            elif torch.is_tensor(preds):
                single_tensor = preds
        except Exception:
            single_tensor = preds if torch.is_tensor(preds) else None

    # 若能取到原始输出，则进行解码并按 head 的规则选择单一目标
        pb_xyxy_lb_list = []
        pk_lb_list = []
        pconf_list = []
        try:
            if det_out is not None and kpt_raw is not None and hasattr(self.model, 'head'):
                # ensure on same device/cpu
                with torch.no_grad():
                    det_final = self.model.head._det_decode(det_out)  # [B,5,N]
                    # 在置信度通道上对 N 个候选取 argmax
                    max_idx = det_final[:, 4:5, :].argmax(dim=-1)  # [B,1]
                    kpt_final = self.model.head._kpt_decode(B, kpt_raw)  # [B,63,N]
                    # 根据最大索引收集单一目标
                    expanded_idx = max_idx.unsqueeze(1).repeat(1, det_final.shape[1], 1)  # [B,5,1]
                    picked_det = torch.gather(det_final, dim=-1, index=expanded_idx)  # [B,5,1]
                    # bbox: xywh（letterbox 像素） -> xyxy（letterbox 像素）
                    try:
                        b_xywh = picked_det[:, 0:4, 0].cpu()  # Tensor [B,4]
                        b_xyxy_any = xywh2xyxy(b_xywh.clone())
                        if torch.is_tensor(b_xyxy_any):
                            t_any = cast(torch.Tensor, b_xyxy_any)
                            b_xyxy_np = t_any.cpu().numpy()
                        else:
                            b_xyxy_np = _np.asarray(b_xyxy_any, dtype=float)
                    except Exception:
                        try:
                            b_xyxy_np = picked_det[:, 0:4, 0].cpu().numpy()
                        except Exception:
                            b_xyxy_np = None
                    conf = picked_det[:, 4, 0].cpu().numpy()
                    # 关键点解码并根据最大索引收集
                    expanded_idx_k = max_idx.unsqueeze(1).repeat(1, kpt_final.shape[1], 1)  # [B,63,1]
                    picked_k = torch.gather(kpt_final, dim=-1, index=expanded_idx_k)  # [B,63,1]
                    picked_k = picked_k[:, :, 0].view(B, -1, 3).cpu().numpy()  # [B,21,3]
                    pb_xyxy_lb_list = [b_xyxy_np[i] for i in range(B)] if b_xyxy_np is not None else [None]*B
                    pk_lb_list = [picked_k[i] for i in range(B)]
                    pconf_list = [float(conf[i]) for i in range(B)]
            elif single_tensor is not None and torch.is_tensor(single_tensor):
                st = cast(torch.Tensor, single_tensor)
                s = st.detach().cpu()
                if s.dim() == 3 and s.size(-1) == 1:
                    s = s.squeeze(-1)
                # s: [B,C]
                if s.dim() == 2:
                    for i in range(min(B, s.size(0))):
                        si = s[i].numpy()
                        bbox_vals = si[0:4].astype(float)
                        score = float(si[4]) if si.shape[0] > 4 else 1.0
                        # bbox may be normalized to img_size (letterbox), assume pixels handled in caller
                        if bbox_vals.max() <= 1.0:
                            # assume input size equals images H/W
                            H = images.shape[2]
                            bbox_vals = bbox_vals * float(H)
                        try:
                            bx = xywh2xyxy(_np.array([bbox_vals]))[0]
                        except Exception:
                            bx = bbox_vals
                        pb_xyxy_lb_list.append(bx)
                        # keypoints
                        if si.shape[0] >= 5 + 63:
                            kf = si[5:5+63].astype(float).reshape(21, 3)
                            if kf.max() <= 1.0:
                                H = images.shape[2]
                                kf[:, 0] *= float(H)
                                kf[:, 1] *= float(H)
                            pk_lb_list.append(kf)
                        else:
                            pk_lb_list.append(None)
                        pconf_list.append(score)
        except Exception:
            # decoding failed; skip visualization
            return

    # 绘制并保存前若干张图片
        import cv2
        img_paths = batch.get('img_paths', [None]*B)
        ratios = batch.get('ratios', [None]*B)
        pads = batch.get('pads', [None]*B)
        for i in range(max_img):
            try:
                # 优先加载原图；若不可用则从 letterbox 张量还原 BGR 图
                img = None
                if img_paths and img_paths[i] and os.path.exists(img_paths[i]):
                    img = cv2.imread(img_paths[i])
                if img is None:
                    # fallback to letterbox tensor
                    im = images[i].cpu().numpy()
                    im = (im * 255.0).clip(0, 255).astype(_np.uint8)
                    img = _np.transpose(im, (1, 2, 0))[:, :, ::-1].copy()
                # 将预测从 letterbox 坐标映射回原图坐标
                pb = pb_xyxy_lb_list[i] if i < len(pb_xyxy_lb_list) else None
                pk = pk_lb_list[i] if i < len(pk_lb_list) else None
                pad = pads[i] if pads and i < len(pads) else (0.0, 0.0)
                ratio = ratios[i] if ratios and i < len(ratios) else (1.0, 1.0)
                if pb is not None:
                    pb_orig = inv_letterbox_xyxy(pb, pad, ratio) or pb
                    x1, y1, x2, y2 = [int(round(v)) for v in pb_orig]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img, f"conf:{pconf_list[i]:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                if pk is not None:
                    pk_orig = inv_letterbox_kpts(pk, pad, ratio)
                    thr = getattr(self, 'vis_threshold', 0.5)
                    for j in range(pk_orig.shape[0]):
                        x = int(round(pk_orig[j, 0])); y = int(round(pk_orig[j, 1])); v = float(pk_orig[j, 2])
                        color = (0, 0, 255) if v > thr else (128, 128, 128)
                        cv2.circle(img, (x, y), 2, color, -1)
                # save
                fn = os.path.join(save_dir, f"epoch{epoch+1}_img{i}.jpg")
                cv2.imwrite(fn, img)
            except Exception:
                continue

    def _move_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 使用 pad_collate 产生的“目标级”张量：'batch_idx','cls','bboxes','keypoints'
        # 将其移动到验证器所在设备，并确保 dtype 正确
        out = {}
        if 'keypoints' in batch:
            out['keypoints'] = batch['keypoints'].to(self.device)
        else:
            out['keypoints'] = torch.zeros((0, 21, 3), device=self.device)
        if 'bboxes' in batch:
            out['bboxes'] = batch['bboxes'].to(self.device)
        else:
            out['bboxes'] = torch.zeros((0, 4), device=self.device)
        # batch_idx indicates object->image mapping (object-level)
        if 'batch_idx' in batch:
            out['batch_idx'] = batch['batch_idx'].to(self.device)
        else:
            out['batch_idx'] = torch.zeros((0,), dtype=torch.long, device=self.device)
        # also include cls if present
        if 'cls' in batch:
            out['cls'] = batch['cls'].to(self.device)
        else:
            out['cls'] = torch.zeros((0,), device=self.device)
        return out

    def validate_epoch(self, val_loader: DataLoader, epoch: int = 0, verbose: bool = True) -> Tuple[float, Dict[str, float]]:
    # 验证流程：模型设为 eval，进行前向/解码，聚合指标（并可选计算损失用于监控）。
        self.model.eval()
    # 默认不在每个 epoch 中保存可视化结果；如需，可开启 visualize 或外部调用。

    # 指标累积改由 ValidationMetrics 负责
        val_metrics_accum = ValidationMetrics(num_keypoints=getattr(self.model, 'nk', 21))

    # 验证损失的本地累积变量
        val_loss_sum = 0.0
        val_loss_count = 0

        with torch.no_grad():
            # 按样本累积损失各分量（用于 epoch 级平均）
            comp_sums = [0.0, 0.0, 0.0, 0.0, 0.0]  # bbox, kpt_loc, kpt_vis, cls, dfl
            visualized = False
            for batch in val_loader:
                images = batch['images'].to(self.device)
                bs = images.size(0)
                # 前向推理（模型 eval 路径应返回已解码或可解析的单目标结果）
                preds = self.model(images)
                
                # 可选：仅对首个 batch 进行可视化
                if not visualized:
                    try:
                        self._maybe_visualize_batch(images, preds, batch, epoch)
                        visualized = True
                    except Exception:
                        pass

                # 采用训练同款 criterion 计算验证损失（可选），用于日志与对比
                try:
                    targets = self._move_targets(batch)
                    total_loss, loss_items = self.criterion(preds, targets)
                    # total_loss 可能为张量或标量
                    if isinstance(total_loss, torch.Tensor):
                        loss_val = float(total_loss.item())
                    else:
                        loss_val = float(total_loss)
                    val_loss_sum += loss_val * bs
                    val_loss_count += bs
                    # 按分量累积（缩放至样本数）
                    try:
                        # PoseLoss 通常返回长度>=5 的分项
                        for i in range(min(5, len(loss_items))):
                            v = loss_items[i]
                            if hasattr(v, 'item'):
                                comp_sums[i] += float(v.item()) * bs
                            else:
                                comp_sums[i] += float(v) * bs
                    except Exception:
                        pass
                except Exception:
                    # if loss computation fails for some layout mismatch, skip loss accumulation
                    pass

                # 解析每张图的关键点与检测框/置信度
                pred_kpts_per_image = None
                pred_boxes_per_image = None
                pred_scores_per_image = None
                try:
                    # 处理 tuple 输出：(single_res, (det_out, kpt_raw)) 或直接张量
                    if isinstance(preds, tuple) and len(preds) >= 1:
                        single = preds[0]
                    else:
                        single = preds
                    # single 期望形状为 [B, C, 1] 或 [B, C]
                    if isinstance(single, torch.Tensor):
                        s = single
                        if s.dim() == 3 and s.size(-1) == 1:
                            s = s.squeeze(-1)
                        # 若通道包含 [x1,y1,x2,y2,conf,cls,后接63个关键点] 之类布局
                        if s.size(1) >= 5 + 63:
                            kpt_flat = s[:, 5:5+63]
                        elif s.size(1) == 63:
                            kpt_flat = s
                        else:
                            # 未知布局，跳过关键点解析
                            kpt_flat = None
                        if kpt_flat is not None:
                            # 关键点重塑为 [B,21,3]
                            pred_kpts = kpt_flat.view(bs, 21, 3).detach()
                            # 同时解析预测框（xywh）与分数，假设 [:,0:4] 为框，[:,4] 为置信度
                            if s.size(1) >= 5:
                                pred_bbox_xywh = s[:, :4].detach()
                                pred_score = s[:, 4].detach()
                            else:
                                pred_bbox_xywh = None
                                pred_score = None

                            # 若关键点看起来是归一化坐标（<=1），则转换到像素尺度
                            img_h = images.shape[2]
                            img_w = images.shape[3]
                            pred_kpts_pix = pred_kpts.clone()
                            # If keypoints are normalized (<=1), multiply by image dims
                            if pred_kpts_pix.max() <= 1.0:
                                pred_kpts_pix[..., 0] = pred_kpts_pix[..., 0] * float(img_w)
                                pred_kpts_pix[..., 1] = pred_kpts_pix[..., 1] * float(img_h)

                            # 存储像素尺度的 per-image 预测
                            pred_kpts_per_image = pred_kpts_pix.cpu()
                            if pred_bbox_xywh is not None:
                                # 确保在 CPU 上处理 bbox
                                pb = pred_bbox_xywh.clone().cpu()
                                # 若 bbox 似乎为归一化，映射到像素尺度
                                try:
                                    if pb.max() <= 1.0:
                                        pb = pb.clone()
                                        pb[:, 0] = pb[:, 0] * float(img_w)  # cx
                                        pb[:, 1] = pb[:, 1] * float(img_h)  # cy
                                        pb[:, 2] = pb[:, 2] * float(img_w)  # w
                                        pb[:, 3] = pb[:, 3] * float(img_h)  # h
                                except Exception:
                                    pass
                                # xywh -> xyxy
                                try:
                                    pb_xyxy = xywh2xyxy(pb)
                                except Exception:
                                    pb_xyxy = pb
                                pred_boxes_per_image = pb_xyxy
                            if pred_score is not None:
                                pred_scores_per_image = pred_score.clone().cpu()
                except Exception:
                    pred_kpts_per_image = None

                # 构造每图的 GT 关键点 (B,21,3) 与 GT 框 (B,4)
                gt_kpts_per_image = torch.zeros((bs, 21, 3))
                gt_boxes_per_image = torch.zeros((bs, 4))
                has_gt = torch.zeros((bs,), dtype=torch.bool)

                # 预期 pad_collate 提供的 keypoints 是相对 letterbox 输出尺寸的归一化坐标；
                # bboxes 则为相对 letterbox 输出的归一化 xyxy（0..1）。先还原为 letterbox 像素，
                # 随后使用 ratio/pad 逆映射到原图像素坐标。
                if 'batch_idx' in batch and batch['batch_idx'].numel() > 0:
                    obj_batch_idx = batch['batch_idx']
                    kp_objs = batch['keypoints']
                    bboxes_objs = batch.get('bboxes', None)
                    for obj_i in range(obj_batch_idx.size(0)):
                        img_i = int(obj_batch_idx[obj_i].item())
                        if 0 <= img_i < bs:
                            kpt = kp_objs[obj_i].clone()
                            # convert normalized (0..1) relative to letterbox output to letterbox pixels
                            img_h = images.shape[2]
                            img_w = images.shape[3]
                            kpt_pix = kpt.clone()
                            try:
                                if kpt_pix.max() <= 1.0:
                                    kpt_pix[..., 0] = kpt_pix[..., 0] * float(img_w)
                                    kpt_pix[..., 1] = kpt_pix[..., 1] * float(img_h)
                            except Exception:
                                pass
                            gt_kpts_per_image[img_i] = kpt_pix
                            # bboxes in pad_collate are stored as normalized xyxy relative to letterbox output
                            if bboxes_objs is not None and bboxes_objs.numel() > 0:
                                bb = bboxes_objs[obj_i].clone()
                                bb_pix = bb.clone()
                                try:
                                    if bb_pix.max() <= 1.0:
                                        bb_pix[0] = bb_pix[0] * float(img_w)
                                        bb_pix[1] = bb_pix[1] * float(img_h)
                                        bb_pix[2] = bb_pix[2] * float(img_w)
                                        bb_pix[3] = bb_pix[3] * float(img_h)
                                except Exception:
                                    pass
                                gt_boxes_per_image[img_i] = bb_pix
                            has_gt[img_i] = True

                # 追加到指标聚合器

                for i in range(bs):
                    # 预测：若可获取映射元数据，执行 letterbox->原图 的逆映射
                    orig_sizes = batch.get('orig_sizes', None)
                    ratios = batch.get('ratios', None)
                    pads = batch.get('pads', None)

                    # map preds
                    pred_box_mapped = None
                    pred_kpts_mapped = None
                    pred_score_val = None
                    # predicted box
                    if pred_boxes_per_image is not None:
                        try:
                            pb_np = pred_boxes_per_image[i].numpy()
                            if orig_sizes is not None and ratios is not None and pads is not None:
                                pred_box_mapped = inv_letterbox_xyxy(pb_np, pads[i], ratios[i])
                            else:
                                pred_box_mapped = pb_np.tolist()
                        except Exception:
                            pred_box_mapped = None
                    # predicted score
                    if pred_scores_per_image is not None:
                        try:
                            pred_score_val = float(pred_scores_per_image[i].item())
                        except Exception:
                            pred_score_val = 0.0
                    else:
                        pred_score_val = 0.0
                    # predicted kpts
                    if pred_kpts_per_image is not None:
                        try:
                            pk_np = pred_kpts_per_image[i].numpy()
                            if orig_sizes is not None and ratios is not None and pads is not None:
                                pred_kpts_mapped = inv_letterbox_kpts(pk_np, pads[i], ratios[i])
                            else:
                                pred_kpts_mapped = pk_np
                        except Exception:
                            pred_kpts_mapped = None

                    # map GT
                    gt_box_mapped = None
                    gt_kpts_mapped = None
                    if has_gt[i]:
                        try:
                            gb_np = gt_boxes_per_image[i].numpy()
                            gk_np = gt_kpts_per_image[i].numpy()
                            if orig_sizes is not None and ratios is not None and pads is not None:
                                gt_box_mapped = inv_letterbox_xyxy(gb_np, pads[i], ratios[i])
                                gt_kpts_mapped = inv_letterbox_kpts(gk_np, pads[i], ratios[i])
                            else:
                                gt_box_mapped = gb_np.tolist()
                                gt_kpts_mapped = gk_np
                        except Exception:
                            gt_box_mapped = None
                            gt_kpts_mapped = None

                    # image path
                    try:
                        img_paths = batch.get('img_paths', None)
                        img_path = img_paths[i] if img_paths is not None else None
                    except Exception:
                        img_path = None

                    # add to accumulator
                    val_metrics_accum.add_image(
                        pred_box=pred_box_mapped,
                        gt_box=gt_box_mapped,
                        pred_kpts=pred_kpts_mapped,
                        gt_kpts=gt_kpts_mapped,
                        img_path=img_path,
                        pred_score=pred_score_val,
                    )

                # 为避免每个 epoch 均落盘，可视化在上方仅对首个 batch 进行
                

    # 计算指标（集中在 ValidationMetrics）
        metrics = val_metrics_accum.compute()
        # 生成 per-image 行并暴露出去
        self.last_validation_rows = val_metrics_accum.build_rows()
    # 写入 TensorBoard 日志（可选）
        if self.enable_tb and self.writer is not None:
            try:
                for k, v in metrics.items():
                    if v is not None:
                        self.writer.add_scalar(f'Val/{k}', float(v), epoch)  # type: ignore[attr-defined]
            except Exception:
                pass

        if verbose:
            # 打印平均 IoU 以及各关键点的平均误差（像素）
            mean_iou = metrics.get('mean_iou', 0.0)
            print(f"[Val] Epoch {epoch+1}: mean_iou={mean_iou:.4f}")
            # 收集 keypoint MAE 键并按序打印
            # 按关键点编号的数字顺序打印（避免字典键的字典序将 kpt_10 放到 kpt_2 前面）
            kpt_keys = [k for k in metrics.keys() if k.startswith('kpt_') and k.endswith('_mae')]
            def _kpt_index(key: str):
                try:
                    # key 格式 kpt_{idx}_mae
                    parts = key.split('_')
                    return int(parts[1])
                except Exception:
                    return 999
            kpt_keys = sorted(kpt_keys, key=_kpt_index)
            if kpt_keys:
                print(f"[Val] Per-keypoint MAE (px):")
                for k in kpt_keys:
                    v = metrics.get(k)
                    if v is None:
                        print(f"  {k}: None")
                    else:
                        print(f"  {k}: {v:.3f}")
            else:
                print(f"[Val] Epoch {epoch+1}: no keypoint MAE available")

    # 计算平均验证损失（按样本平均）
        avg_val_loss = float(val_loss_sum / val_loss_count) if val_loss_count > 0 else 0.0
    # 计算各分量平均损失，附加到 metrics（字段名不加 'val_' 前缀；CSV 保存时再加）
        if val_loss_count > 0:
            comp_avgs = [s / val_loss_count for s in comp_sums]
            metrics['bbox_loss'] = float(comp_avgs[0])
            metrics['kpt_loc_loss'] = float(comp_avgs[1])
            metrics['kpt_vis_loss'] = float(comp_avgs[2])
            metrics['cls_loss'] = float(comp_avgs[3])
            metrics['dfl_loss'] = float(comp_avgs[4])

        return avg_val_loss, metrics

    def validate_single_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, Any]]:
    # 单个 batch 的快速验证：返回预测与（可选）指标，不计算损失
        self.model.eval()
        with torch.no_grad():
            images = batch['images'].to(self.device)
            preds = self.model(images)
            return 0.0, {
                'metrics': {},
                'predictions': preds
            }

    def save_validation_csv(self, csv_path: str, rows: Optional[list] = None) -> None:
        """将上一次验证得到的 per-image 行数据保存为 CSV。

        参数:
            csv_path: 输出 CSV 文件路径
            rows: 可选的行列表（字典），默认使用 `self.last_validation_rows`。
        """
        if rows is None:
            rows = getattr(self, 'last_validation_rows', None)
        if not rows:
            raise ValueError('No validation rows to save. Run validate_epoch first or pass rows.')
        # allow optional detailed fields if rows contain them
        fieldnames = ['index', 'img_path', 'pred_score', 'iou', 'kpt_mean_euclid_px']
        # detect detailed keys
        extra_keys = set()
        for r in rows:
            for k in ['pred_box', 'gt_box', 'pred_kpts', 'gt_kpts']:
                if k in r:
                    extra_keys.add(k)
        fieldnames.extend(sorted(list(extra_keys)))
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: (r.get(k) if r.get(k) is not None else '') for k in fieldnames})
        except Exception:
            raise

    def evaluate_model(self, val_loader: DataLoader, epoch: int = 0) -> Dict[str, Any]:
        val_loss, metrics = self.validate_epoch(val_loader, epoch=epoch, verbose=False)
        return {
            'validation_loss': val_loss,
            'metrics': metrics,
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device)
            }
        }

    def evaluate_and_save(self, loader: DataLoader, out_csv: str, epoch: int = 0, verbose: bool = False) -> Dict[str, Any]:
        # 对任意 DataLoader 进行评估并保存 per-image 行至 CSV。
        # 返回包含 validation_loss 与 metrics 的字典，便于外部使用。
        val_loss, metrics = self.validate_epoch(loader, epoch=epoch, verbose=verbose)
        # ensure directory exists and save rows from last run
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        self.save_validation_csv(out_csv)
        return {
            'validation_loss': val_loss,
            'metrics': metrics
        }


def create_validater(model, criterion, device=None, writer: Optional[SummaryWriter] = None, enable_tb: bool = True):
    return YOLOValidater(model, criterion, device, writer=writer, enable_tb=enable_tb)

