import random
import numpy as np
import cv2
import yaml
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
from config_utils import load_config

class BabyPoseAugmentor:
    """21关键点婴儿姿态增强器 (支持原始边界框同步增强)"""

    # 左右对称关键点索引 (0-based)
    LR_PAIRS: List[Tuple[int, int]] = [
        (1, 2),   # left_eye <-> right_eye
        (3, 4),   # left_ear <-> right_ear
        (5, 6),   # left_shoulder <-> right_shoulder
        (7, 8),   # left_elbow <-> right_elbow
        (9, 10),  # left_wrist <-> right_wrist
        (11, 12), # left_hip <-> right_hip
        (13, 14), # left_knee <-> right_knee
        (15, 16), # left_ankle <-> right_ankle
        (17, 18), # left_foot <-> right_foot
        (19, 20), # left_hand <-> right_hand
    ]

    def __init__(self, config_path: str = "config.yaml", config: Optional[Dict[str, Any]] = None):
        # 支持直接传入 config dict，或传入配置文件路径（使用 load_config 合并默认值）
        if config is None:
            config = load_config(config_path)
        # 直接从根配置读取各个增强参数
        self.horizontal_flip_prob = config.get('horizontal_flip_prob', 0.5)
        self.rotation_prob = config.get('rotation_prob', 0.3)
        self.rotation_range = config.get('rotation_range', 15.0)
        self.scale_prob = config.get('scale_prob', 0.3)
        self.scale_range = tuple(config.get('scale_range', [0.9, 1.1]))
        self.translation_prob = config.get('translation_prob', 0.3)
        self.translation_range = config.get('translation_range', 0.05)
        self.color_prob = config.get('color_prob', 0.4)
        self.brightness_range = config.get('brightness_range', 0.1)
        self.contrast_range = config.get('contrast_range', 0.1)
        self.debug = config.get('debug', False)
        self._set_random_seed(config.get('seed'))
        # 边界框填充颜色，默认与图像边界填充一致
        self.pad_color = tuple(config.get('pad_color', [114, 114, 114]))

    def _set_random_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.debug:
            print(f"[BabyPoseAug] {msg}")

    @staticmethod
    def _clone(image, keypoints, bbox):
        kpts = None if keypoints is None else keypoints.copy()
        bbx = None if bbox is None else list(bbox)
        return image.copy(), kpts, bbx

    @staticmethod
    def _visibility_mask(kpts: np.ndarray) -> np.ndarray:
        return (kpts[:, 2] > 0).astype(bool)

    @staticmethod
    def _clip_and_visibility(kpts: np.ndarray, w: int, h: int):
        vis = BabyPoseAugmentor._visibility_mask(kpts)
        # 超出边界置 v=0
        out = (
            (kpts[:, 0] < 0) | (kpts[:, 0] >= w) |
            (kpts[:, 1] < 0) | (kpts[:, 1] >= h)
        ) & vis
        kpts[out, 2] = 0
        return kpts

    @staticmethod
    def _clip_bbox(bbox: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
        """将边界框裁剪到图像范围内"""
        x1, y1, x2, y2 = bbox
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _recompute_bbox(kpts: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
        if kpts is None:
            return None
        vis = kpts[:, 2] > 0
        if not np.any(vis):
            return None
        xs = kpts[vis, 0]
        ys = kpts[vis, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        # 裁剪到图像内
        return BabyPoseAugmentor._clip_bbox((x1, y1, x2, y2), w, h)

    @staticmethod
    def _bbox_to_corners(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """将边界框(x1,y1,x2,y2)转换为四个角点坐标"""
        x1, y1, x2, y2 = bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    @staticmethod
    def _corners_to_bbox(corners: np.ndarray) -> Tuple[float, float, float, float]:
        """将角点坐标转换为边界框(x1,y1,x2,y2)"""
        x1 = float(np.min(corners[:, 0]))
        y1 = float(np.min(corners[:, 1]))
        x2 = float(np.max(corners[:, 0]))
        y2 = float(np.max(corners[:, 1]))
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # 各种增强算子 (均保证返回 (image, keypoints, bbox))
    # ------------------------------------------------------------------
    def horizontal_flip(self, image, keypoints, bbox):
        if (keypoints is None and bbox is None) or random.random() > self.horizontal_flip_prob:
            return image, keypoints, bbox
            
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 1)
        new_kpts = keypoints.copy() if keypoints is not None else None
        new_bbox = None
        
        # 处理关键点
        if new_kpts is not None:
            vis = new_kpts[:, 2] > 0
            new_kpts[vis, 0] = (w - 1) - new_kpts[vis, 0]
            # 左右配对交换
            for a, b in self.LR_PAIRS:
                if a < len(new_kpts) and b < len(new_kpts):
                    new_kpts[[a, b]] = new_kpts[[b, a]]
        
        # 处理边界框
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # 水平翻转边界框
            new_x1 = (w - 1) - x2
            new_x2 = (w - 1) - x1
            new_bbox = (new_x1, y1, new_x2, y2)
            # 确保x1 < x2
            if new_x1 > new_x2:
                new_x1, new_x2 = new_x2, new_x1
                new_bbox = (new_x1, y1, new_x2, y2)
        
        self._log("apply horizontal flip")
        return flipped, new_kpts, new_bbox

    def random_rotation(self, image, keypoints, bbox):
        if (keypoints is None and bbox is None) or random.random() > self.rotation_prob:
            return image, keypoints, bbox
            
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=self.pad_color)
        
        new_kpts = keypoints.copy() if keypoints is not None else None
        new_bbox = None
        
        # 处理关键点
        if new_kpts is not None:
            vis_mask = new_kpts[:, 2] > 0
            if np.any(vis_mask):
                pts = new_kpts[vis_mask, :2]
                ones = np.ones((pts.shape[0], 1))
                homo = np.hstack([pts, ones])
                trans = (M @ homo.T).T
                new_kpts[vis_mask, :2] = trans
            new_kpts = self._clip_and_visibility(new_kpts, w, h)
        
        # 处理边界框
        if bbox is not None:
            # 将边界框转换为角点并应用旋转
            corners = self._bbox_to_corners(bbox)
            ones = np.ones((corners.shape[0], 1))
            homo_corners = np.hstack([corners, ones])
            trans_corners = (M @ homo_corners.T).T
            # 转换回边界框并裁剪到图像内
            new_bbox = self._corners_to_bbox(trans_corners)
            new_bbox = self._clip_bbox(new_bbox, w, h)
        
        self._log(f"apply rotation angle={angle:.2f}")
        return rotated, new_kpts, new_bbox

    def random_scale(self, image, keypoints, bbox):
        if (keypoints is None and bbox is None) or random.random() > self.scale_prob:
            return image, keypoints, bbox
            
        h, w = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        
        # 避免不必要的缩放操作
        if abs(scale - 1.0) < 1e-6:
            return image, keypoints, bbox
            
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        scaled_img = cv2.resize(image, (new_w, new_h))
        
        new_kpts = keypoints.copy() if keypoints is not None else None
        new_bbox = None
        
        if scale < 1.0:
            # 缩小 -> 放置在大画布中心
            canvas = np.full((h, w, 3), self.pad_color, dtype=np.uint8)
            dx = (w - new_w) // 2
            dy = (h - new_h) // 2
            canvas[dy:dy+new_h, dx:dx+new_w] = scaled_img
            scaled_img = canvas
            
            # 处理关键点
            if new_kpts is not None:
                new_kpts[:, 0] = new_kpts[:, 0] * scale + dx
                new_kpts[:, 1] = new_kpts[:, 1] * scale + dy
                new_kpts = self._clip_and_visibility(new_kpts, w, h)
            
            # 处理边界框
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                new_x1 = x1 * scale + dx
                new_y1 = y1 * scale + dy
                new_x2 = x2 * scale + dx
                new_y2 = y2 * scale + dy
                new_bbox = self._clip_bbox((new_x1, new_y1, new_x2, new_y2), w, h)
                
        else:
            # 放大 -> 中心裁剪回原尺寸
            dx = (new_w - w) // 2
            dy = (new_h - h) // 2
            scaled_img = scaled_img[dy:dy+h, dx:dx+w]
            
            # 处理关键点
            if new_kpts is not None:
                new_kpts[:, 0] = new_kpts[:, 0] * scale - dx
                new_kpts[:, 1] = new_kpts[:, 1] * scale - dy
                new_kpts = self._clip_and_visibility(new_kpts, w, h)
            
            # 处理边界框
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                new_x1 = x1 * scale - dx
                new_y1 = y1 * scale - dy
                new_x2 = x2 * scale - dx
                new_y2 = y2 * scale - dy
                new_bbox = self._clip_bbox((new_x1, new_y1, new_x2, new_y2), w, h)
        
        self._log(f"apply scale factor={scale:.3f}")
        return scaled_img, new_kpts, new_bbox

    def random_translation(self, image, keypoints, bbox):
        if (keypoints is None and bbox is None) or random.random() > self.translation_prob:
            return image, keypoints, bbox
            
        h, w = image.shape[:2]
        max_dx = self.translation_range * w
        max_dy = self.translation_range * h
        dx = int(round(random.uniform(-max_dx, max_dx)))
        dy = int(round(random.uniform(-max_dy, max_dy)))
        
        # 避免不必要的平移操作
        if dx == 0 and dy == 0:
            return image, keypoints, bbox
            
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        translated = cv2.warpAffine(image, M, (w, h), borderValue=self.pad_color)
        
        new_kpts = keypoints.copy() if keypoints is not None else None
        new_bbox = None
        
        # 处理关键点
        if new_kpts is not None:
            new_kpts[:, 0] += dx
            new_kpts[:, 1] += dy
            new_kpts = self._clip_and_visibility(new_kpts, w, h)
        
        # 处理边界框
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = x2 + dx
            new_y2 = y2 + dy
            new_bbox = self._clip_bbox((new_x1, new_y1, new_x2, new_y2), w, h)
        
        self._log(f"apply translation dx={dx} dy={dy}")
        return translated, new_kpts, new_bbox

    def color_augmentation(self, image, keypoints, bbox):
        if random.random() > self.color_prob:
            return image, keypoints, bbox
            
        img = image.copy()
        # HSV 抖动
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_gain = random.uniform(-0.02, 0.02)
        s_gain = random.uniform(-0.4, 0.4)
        v_gain = random.uniform(-0.4, 0.4)
        hsv[..., 0] = (hsv[..., 0] + h_gain * 180) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + s_gain), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + v_gain), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 亮度对比度
        alpha = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
        beta = random.uniform(-self.brightness_range * 255, self.brightness_range * 255)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        self._log(f"apply color (alpha={alpha:.2f}, beta={beta:.1f})")
        # 颜色增强不影响关键点和边界框
        return img, keypoints, bbox

    # ------------------------------------------------------------------
    # 主调用
    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray, keypoints: Optional[np.ndarray], 
                 bbox: Optional[Tuple[float, float, float, float]] = None):
        """
        执行一系列增强, 返回增强后的图像、关键点和边界框
        
        参数:
            image: 输入图像
            keypoints: 关键点数组, 形状为(N, 3), 第三列表示可见性
            bbox: 边界框, 格式为(x1, y1, x2, y2)
            
        返回:
            aug_image: 增强后的图像
            aug_keypoints: 增强后的关键点
            aug_bbox: 增强后的边界框, 优先使用原始边界框增强后的结果, 
                      若原始边界框不存在则基于关键点重新计算
        """
        img, kpts, bbx = self._clone(image, keypoints, bbox)
        
        if kpts is None and bbx is None:
            # 无关键点和边界框时只做颜色增强
            img, _, _ = self.color_augmentation(img, None, None)
            return img, None, None

        # 顺序: flip -> rotate -> scale -> translate -> color
        img, kpts, bbx = self.horizontal_flip(img, kpts, bbx)
        img, kpts, bbx = self.random_rotation(img, kpts, bbx)
        img, kpts, bbx = self.random_scale(img, kpts, bbx)
        img, kpts, bbx = self.random_translation(img, kpts, bbx)
        img, kpts, bbx = self.color_augmentation(img, kpts, bbx)

        h, w = img.shape[:2]
        # 如果原始边界框不存在, 基于关键点重新计算
        if bbx is None:
            bbx = self._recompute_bbox(kpts, w, h)
            
        return img, kpts, bbx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 尝试从本地 dataset 中读取一张训练图像及其标签
    base = Path(__file__).parent
    data_root = base / 'dataset'
    img_dir = data_root / 'images' / 'train'
    label_dir = data_root / 'labels' / 'train'

    # 查找第一张图片
    img_path = None
    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                img_path = p
                break

    if img_path is None:
        print('未找到训练图像，测试结束')
    else:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 读取对应标签（YOLO + 21 keypoints 格式）
        keypoints = None
        bbox = None
        label_path = label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                parts = line.split()
                if len(parts) >= 68:
                    class_id = int(parts[0])
                    bbox_norm = [float(x) for x in parts[1:5]]  # cx,cy,w,h
                    kpt_data = [float(x) for x in parts[5:68]]
                    keypoints = np.array(kpt_data).reshape(21, 3).astype(np.float32)
                    # 标签中的关键点为归一化坐标 (0..1)，转换为像素坐标
                    keypoints[:, 0] = keypoints[:, 0] * w
                    keypoints[:, 1] = keypoints[:, 1] * h
                    cx, cy, ww, hh = bbox_norm
                    cx_abs = cx * w
                    cy_abs = cy * h
                    ww_abs = ww * w
                    hh_abs = hh * h
                    x1 = cx_abs - ww_abs/2
                    y1 = cy_abs - hh_abs/2
                    x2 = cx_abs + ww_abs/2
                    y2 = cy_abs + hh_abs/2
                    bbox = (x1, y1, x2, y2)
            except Exception as e:
                print(f'读取标签失败: {e}')

        augmentor = BabyPoseAugmentor(config_path=str(base / 'config.yaml'))

        aug_img, aug_kpts, aug_bbox = augmentor(img.copy(), None if keypoints is None else keypoints.copy(), bbox)

        # 如果增强器没有返回 bbox，但有关键点，可重新计算
        if aug_bbox is None and aug_kpts is not None:
            aug_bbox = augmentor._recompute_bbox(aug_kpts, aug_img.shape[1], aug_img.shape[0])

        # 绘制关键点与 bbox 的辅助函数（在拷贝图上绘制）
        def draw_on(img_in, kpts, bbx):
            out = img_in.copy()
            # 先绘制 bbox
            if bbx is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbx]
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
            # 再绘制关键点，按可见性着色
            if kpts is not None:
                for kp in kpts:
                    x, y, v = kp
                    try:
                        vx = float(v)
                    except Exception:
                        vx = 1.0
                    color = (0, 255, 0) if vx > 0.5 else (0, 0, 255)
                    cv2.circle(out, (int(round(x)), int(round(y))), 4, color, -1)
            return out

        img_vis = draw_on(img, keypoints, bbox)
        aug_vis = draw_on(aug_img, aug_kpts, aug_bbox)

        # 显示原图与增强后对比
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title('原图 (with kpts/bbox)')
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(aug_vis, cv2.COLOR_BGR2RGB))
        axs[1].set_title('增强后 (with kpts/bbox)')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()
