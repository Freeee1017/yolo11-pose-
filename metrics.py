"""Minimal training metrics tracker.

Tracked:
 - Loss components (external update)
 - LR history & epoch timing
"""
from typing import Dict, Any, Optional, List, Union
import time
from collections import defaultdict
import torch
from utils import bbox_iou as _bbox_iou
import numpy as np


class TrainingMetrics:
    def __init__(self):
        self.reset()
    def reset(self):
        self.losses = defaultdict(list)
        self.learning_rates = []
        self.epoch_times = []
        self.epoch_count = 0
        self.best_metrics = {}
    def update_loss(self, loss_dict: Dict[str,float]):
        for k,v in loss_dict.items():
            # 仅当 v 具有 item 且不是内置 float 才尝试转换
            if hasattr(v, 'item') and not isinstance(v, (float, int)):
                try:
                    v = v.item()  # type: ignore
                except Exception:
                    pass
            self.losses[k].append(float(v))
    def update_lr(self, lr: float):
        self.learning_rates.append(float(lr))
    def update_epoch_time(self, seconds: float):
        self.epoch_times.append(seconds)
        self.epoch_count += 1
    def update_best_metrics(self, metrics: Dict[str,float]):
        for k,v in metrics.items():
            if isinstance(v,(int,float)) and (k not in self.best_metrics or v > self.best_metrics[k]):
                self.best_metrics[k] = v
    def get_summary(self):
        total_time = sum(self.epoch_times)
        avg_epoch = total_time / len(self.epoch_times) if self.epoch_times else 0
        loss_summary = {}
        for name, vals in self.losses.items():
            if vals:
                loss_summary[f'{name}_final'] = vals[-1]
                loss_summary[f'{name}_best'] = min(vals)
                loss_summary[f'{name}_avg'] = sum(vals)/len(vals)
        return {
            'total_epochs': self.epoch_count,
            'total_time_hours': total_time/3600,
            'avg_epoch_time_minutes': avg_epoch/60,
            'final_lr': self.learning_rates[-1] if self.learning_rates else 0.0,
            'loss_summary': loss_summary,
            'best_metrics': self.best_metrics
        }


class MetricsTracker:
    def __init__(self, num_classes: int = 1, num_keypoints: int = 21, vis_threshold: float = 0.5):
        # store configuration so external callers (Validater/Trainer) can set visibility threshold
        self.num_classes = int(num_classes) if num_classes is not None else 1
        self.num_keypoints = int(num_keypoints) if num_keypoints is not None else 21
        self.vis_threshold = float(vis_threshold) if vis_threshold is not None else 0.5
        # internal trackers
        self.training_metrics = TrainingMetrics()
        self.start_time = time.time()
        self.epoch_start_time = None
    def start_epoch(self):
        self.epoch_start_time = time.time()
    def end_epoch(self):
        if self.epoch_start_time is not None:
            self.training_metrics.update_epoch_time(time.time()-self.epoch_start_time)
    def update_losses(self, loss_dict: Dict[str,float]):
        self.training_metrics.update_loss(loss_dict)
    def update_lr(self, lr: float):
        self.training_metrics.update_lr(lr)
    def compute_all_metrics(self) -> Dict[str,Any]:
        ts = self.training_metrics.get_summary()
        return {
            'training_summary': ts,
            'total_training_time': time.time()-self.start_time
        }
    def print_metrics(self, epoch: int = -1):
        m = self.compute_all_metrics()
        print('\n==== Metrics ====')
        if epoch >= 0:
            print(f'Epoch {epoch}')
        print('=================')


class ValidationMetrics:
    """验证阶段指标聚合器

    负责逐图累积预测与 GT，并在 epoch 结束时统一计算：
    - mean_iou（检测框）
    - 每图关键点平均欧氏距离（仅可见点，用于行级输出）
    - 每个关键点的像素级 MAE（仅可见点）

    说明：
    - 预测/GT 框均按 xyxy 像素坐标传入；允许为 None（缺失时按 0 处理 IoU）。
    - 关键点为 (K,3) 的像素坐标数组/张量，通道 [x,y,v]，v!=0 视为可见。
    - 可通过 build_rows() 生成 per-image 行数据，用于 CSV/调试。
    """
    def __init__(self, num_keypoints: int = 21):
        self.num_keypoints = int(num_keypoints) if num_keypoints is not None else 21
        self._pred_boxes: List[Optional[List[float]]] = []
        self._gt_boxes: List[Optional[List[float]]] = []
        self._pred_kpts: List[Optional[np.ndarray]] = []
        self._gt_kpts: List[Optional[np.ndarray]] = []
        self._img_paths: List[Optional[str]] = []
        self._pred_scores: List[float] = []

    def add_image(
        self,
        pred_box: Optional[Union[torch.Tensor, List[float], np.ndarray]],
        gt_box: Optional[Union[torch.Tensor, List[float], np.ndarray]],
        pred_kpts: Optional[Union[torch.Tensor, np.ndarray]],
        gt_kpts: Optional[Union[torch.Tensor, np.ndarray]],
        img_path: Optional[str] = None,
        pred_score: Optional[float] = None,
    ) -> None:
        # 统一为 list/np 格式便于后续处理
        def to_list4(b):
            if b is None:
                return None
            if isinstance(b, torch.Tensor):
                b = b.detach().cpu().view(-1).tolist()
            elif isinstance(b, np.ndarray):
                b = b.reshape(-1).tolist()
            return [float(x) for x in b[:4]] if b is not None else None

        def to_np_kpts(k):
            if k is None:
                return None
            if isinstance(k, torch.Tensor):
                k = k.detach().cpu().numpy()
            k = np.asarray(k)
            if k.ndim != 2 or k.shape[1] < 2:
                return None
            # ensure shape (K,3) with visibility channel
            if k.shape[1] == 2:
                vis = np.ones((k.shape[0], 1), dtype=k.dtype)
                k = np.concatenate([k, vis], axis=1)
            return k.astype(float)

        self._pred_boxes.append(to_list4(pred_box))
        self._gt_boxes.append(to_list4(gt_box))
        self._pred_kpts.append(to_np_kpts(pred_kpts))
        self._gt_kpts.append(to_np_kpts(gt_kpts))
        self._img_paths.append(img_path)
        self._pred_scores.append(float(pred_score) if pred_score is not None else 0.0)

    def _compute_ious(self) -> List[float]:
        ious: List[float] = []
        for pb, gb in zip(self._pred_boxes, self._gt_boxes):
            if pb is None or gb is None:
                ious.append(0.0)
                continue
            try:
                pb_t = torch.tensor(pb, dtype=torch.float32).view(1, 4)
                gb_t = torch.tensor(gb, dtype=torch.float32).view(1, 4)
                iou = float(_bbox_iou(pb_t, gb_t, xywh=False).item())
            except Exception:
                iou = 0.0
            ious.append(iou)
        return ious

    def _compute_per_image_kpt_mean(self) -> List[Optional[float]]:
        means: List[Optional[float]] = []
        for pk, gk in zip(self._pred_kpts, self._gt_kpts):
            if pk is None or gk is None:
                means.append(None)
                continue
            if pk.shape[0] != self.num_keypoints or gk.shape[0] != self.num_keypoints:
                means.append(None)
                continue
            total = 0.0
            cnt = 0
            for j in range(self.num_keypoints):
                vis = float(gk[j, 2]) != 0.0
                if not vis:
                    continue
                dx = float(pk[j, 0]) - float(gk[j, 0])
                dy = float(pk[j, 1]) - float(gk[j, 1])
                dist = (dx * dx + dy * dy) ** 0.5
                total += dist
                cnt += 1
            means.append(total / cnt if cnt > 0 else None)
        return means

    def _compute_kpt_mae(self) -> Dict[str, Optional[float]]:
        sums = [0.0] * self.num_keypoints
        counts = [0] * self.num_keypoints
        for pk, gk in zip(self._pred_kpts, self._gt_kpts):
            if pk is None or gk is None:
                continue
            if pk.shape[0] != self.num_keypoints or gk.shape[0] != self.num_keypoints:
                continue
            for j in range(self.num_keypoints):
                vis = float(gk[j, 2]) != 0.0
                if not vis:
                    continue
                dx = float(pk[j, 0]) - float(gk[j, 0])
                dy = float(pk[j, 1]) - float(gk[j, 1])
                dist = (dx * dx + dy * dy) ** 0.5
                sums[j] += dist
                counts[j] += 1
        out: Dict[str, Optional[float]] = {}
        for j in range(self.num_keypoints):
            key = f'kpt_{j}_mae'
            out[key] = (sums[j] / counts[j]) if counts[j] > 0 else None
        return out

    def compute(self) -> Dict[str, Any]:
        ious = self._compute_ious()
        metrics: Dict[str, Any] = {}
        metrics['mean_iou'] = float(sum(ious) / len(ious)) if len(ious) > 0 else 0.0
        # 添加每关键点 MAE
        metrics.update(self._compute_kpt_mae())
        return metrics

    def build_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        per_img_kpt_mean = self._compute_per_image_kpt_mean()
        ious = self._compute_ious()
        for idx in range(len(self._pred_scores)):
            row: Dict[str, Any] = {
                'index': idx,
                'img_path': self._img_paths[idx] if idx < len(self._img_paths) else None,
                'pred_score': self._pred_scores[idx],
                'iou': ious[idx] if idx < len(ious) else 0.0,
                'kpt_mean_euclid_px': per_img_kpt_mean[idx] if idx < len(per_img_kpt_mean) else None,
            }
            pb = self._pred_boxes[idx] if idx < len(self._pred_boxes) else None
            gb = self._gt_boxes[idx] if idx < len(self._gt_boxes) else None
            pk = self._pred_kpts[idx] if idx < len(self._pred_kpts) else None
            gk = self._gt_kpts[idx] if idx < len(self._gt_kpts) else None
            if pb is not None:
                row['pred_box'] = pb
            if gb is not None:
                row['gt_box'] = gb
            if pk is not None:
                row['pred_kpts'] = pk.tolist()
            if gk is not None:
                row['gt_kpts'] = gk.tolist()
            rows.append(row)
        return rows


# 便捷函数
def compute_bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """计算两组边界框的IoU（统一调用 utils.bbox_iou）。

    - 输入均为 xyxy 格式
    - 返回 (N, M) 的 IoU 矩阵
    """
    if not isinstance(box1, torch.Tensor):
        box1 = torch.as_tensor(box1)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.as_tensor(box2)
    # 添加维度以便广播：(N,1,4) vs (1,M,4) -> (N,M)
    b1 = box1.unsqueeze(1)
    b2 = box2.unsqueeze(0)
    iou_b = _bbox_iou(b1, b2, xywh=False)
    # iou_b 的形状应为 (N, M, 1) 或 (N, M)，根据内部实现做 squeeze
    return iou_b.squeeze(-1) if iou_b.dim() == 3 and iou_b.size(-1) == 1 else iou_b


def compute_keypoint_distance(kpt1: torch.Tensor, kpt2: torch.Tensor) -> torch.Tensor:
    """
    计算关键点距离
    
    Args:
        kpt1: (N, 2) 或 (N, 3)
        kpt2: (N, 2) 或 (N, 3)
        
    Returns:
        距离 (N,)
    """
    dx = kpt1[:, 0] - kpt2[:, 0]
    dy = kpt1[:, 1] - kpt2[:, 1]
    return torch.sqrt(dx * dx + dy * dy)


if __name__ == "__main__":
    # 简单自检（仅训练损失/学习率/时间）
    mt = MetricsTracker(num_keypoints=21)
    mt.start_epoch()
    mt.update_losses({'total_loss':1.2,'bbox':0.4,'kpt_loc':0.5,'kpt_vis':0.3})
    mt.update_lr(0.001)
    mt.end_epoch()
    mt.print_metrics(1)
