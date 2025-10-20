"""
YOLO11婴儿关键点检测可视化模块
提供关键点、边界框等可视化功能
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import os


class YOLOVisualizer:
    """YOLO关键点检测可视化器"""
    
    # 关键点名称
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4: 鼻子、左右眼、左右耳
        'left_shoulder', 'right_shoulder',                         # 5-6: 左右肩
        'left_elbow', 'right_elbow',                               # 7-8: 左右肘
        'left_wrist', 'right_wrist',                               # 9-10: 左右腕
        'left_hip', 'right_hip',                                   # 11-12: 左右髋
        'left_knee', 'right_knee',                                 # 13-14: 左右膝
        'left_ankle', 'right_ankle',                               # 15-16: 左右踝
        'left_index_finger', 'right_index_finger',                 # 17-18: 左右指尖
        'left_toe', 'right_toe'                                    # 19-20: 左右脚尖
    ]
    
    # 颜色配置
    COLORS = {
        'box': (0, 255, 0),           # 绿色边界框
        'keypoint': (0, 255, 0),      # 绿色关键点 (统一改为绿色)
        'text': (255, 255, 255),      # 白色文字
        'confidence': (0, 255, 0),    # 绿色置信度
        'invisible': (128, 128, 128)  # 灰色不可见点
    }
    
    def __init__(self, conf_thres: float = 0.25, keypoint_thres: float = 0.1):
        """
        初始化可视化器
        
        Args:
            conf_thres: 检测置信度阈值
            keypoint_thres: 关键点可见性阈值
        """
        self.conf_thres = conf_thres
        self.keypoint_thres = keypoint_thres
    
    def visualize_predictions(self, 
                            image: np.ndarray, 
                            predictions: Dict,
                            save_path: Optional[str] = None,
                            show_confidence: bool = True,
                            show_keypoint_names: bool = True,
                            show_keypoint_indices: bool = False,
                            line_thickness: int = 2,
                            point_radius: int = 3) -> np.ndarray:
        """
        可视化预测结果
        
        Args:
            image: 原始图像 (H, W, 3)
            predictions: 预测结果字典
            save_path: 保存路径
            show_confidence: 是否显示置信度
            show_keypoint_names: 是否显示关键点名称
            line_thickness: 线条粗细
            point_radius: 关键点半径
            
        Returns:
            可视化后的图像
        """
        # 复制图像以避免修改原图
        vis_image = image.copy()
        
        # 获取检测结果
        detections = predictions.get('detections', [])
        
        if not detections:
            return vis_image
        
        # 遍历每个检测结果
        for detection in detections:
            # 绘制边界框
            vis_image = self._draw_bbox(
                vis_image, 
                detection, 
                show_confidence, 
                line_thickness
            )
            
            # 绘制关键点和骨架（可选显示名称与序号）
            vis_image = self._draw_keypoints(
                vis_image, 
                detection, 
                show_keypoint_names,
                show_keypoint_indices,
                point_radius,
                line_thickness
            )
        
        # 添加统计信息
        vis_image = self._add_info_panel(vis_image, predictions)
        
        # 保存图像
        if save_path:
            self._save_image(vis_image, save_path)
        
        return vis_image
    
    def _draw_bbox(self, 
                   image: np.ndarray, 
                   detection: Dict, 
                   show_confidence: bool,
                   thickness: int) -> np.ndarray:
        """绘制边界框"""
        bbox = detection.get('bbox', [])
        confidence = detection.get('confidence', 0)
        
        if len(bbox) != 4:
            return image
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), self.COLORS['box'], thickness)
        
        # 添加置信度标签
        if show_confidence:
            label = f'Baby: {confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # 绘制标签背景
            cv2.rectangle(
                image, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0] + 10, y1), 
                self.COLORS['box'], 
                -1
            )
            
            # 绘制标签文字
            cv2.putText(
                image, 
                label, 
                (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                self.COLORS['text'], 
                1
            )
        
        return image
    
    def _draw_keypoints(self, 
                        image: np.ndarray, 
                        detection: Dict, 
                        show_names: bool,
                        show_indices: bool,
                        point_radius: int,
                        line_thickness: int) -> np.ndarray:
        """绘制关键点(移除骨架连接)"""
        keypoints = detection.get('keypoints', [])

        if keypoints is None:
            return image

        # Normalize keypoints into a Nx3 numpy array where N == len(KEYPOINT_NAMES)
        kpts_arr = None
        try:
            arr = np.asarray(keypoints, dtype=float)
            # flat 63-length vector [x0,y0,v0,...]
            if arr.ndim == 1 and arr.size == len(self.KEYPOINT_NAMES) * 3:
                kpts_arr = arr.reshape(len(self.KEYPOINT_NAMES), 3)
            # already (21,3)
            elif arr.ndim == 2 and arr.shape[0] == len(self.KEYPOINT_NAMES) and arr.shape[1] >= 2:
                # if has only x,y (no v) pad v=1
                if arr.shape[1] == 2:
                    vcol = np.ones((arr.shape[0], 1), dtype=float)
                    kpts_arr = np.hstack([arr, vcol])
                else:
                    kpts_arr = arr[:, :3]
            # list of tuples/lists
            elif isinstance(keypoints, (list, tuple)) and len(keypoints) == len(self.KEYPOINT_NAMES):
                tmp = []
                for item in keypoints:
                    try:
                        it = np.asarray(item, dtype=float)
                        if it.size == 3:
                            tmp.append(it[:3])
                        elif it.size == 2:
                            tmp.append(np.array([it[0], it[1], 1.0], dtype=float))
                        else:
                            # unexpected, fill zeros
                            tmp.append(np.array([0.0, 0.0, 0.0], dtype=float))
                    except Exception:
                        tmp.append(np.array([0.0, 0.0, 0.0], dtype=float))
                kpts_arr = np.vstack(tmp)
        except Exception:
            kpts_arr = None

        if kpts_arr is None or kpts_arr.shape[0] != len(self.KEYPOINT_NAMES):
            return image
        
        # 只绘制关键点，不绘制骨架连接
        for i in range(kpts_arr.shape[0]):
            x, y, v = kpts_arr[i, 0], kpts_arr[i, 1], kpts_arr[i, 2]
            try:
                if (float(x) == 0.0 and float(y) == 0.0 and float(v) == 0.0) or float(v) < float(self.keypoint_thres):
                    continue
            except Exception:
                continue

            color = self.COLORS['keypoint'] if float(v) >= float(self.keypoint_thres) else self.COLORS['invisible']

            cv2.circle(image, (int(round(x)), int(round(y))), point_radius, color, -1)
            cv2.circle(image, (int(round(x)), int(round(y))), point_radius + 1, (0, 0, 0), 1)

            if show_names and i < len(self.KEYPOINT_NAMES):
                name = self.KEYPOINT_NAMES[i]
                try:
                    cv2.putText(
                        image,
                        name,
                        (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        self.COLORS['text'],
                        1
                    )
                except Exception:
                    pass

            if show_indices:
                try:
                    idx_label = str(i)
                    cv2.putText(
                        image,
                        idx_label,
                        (int(round(x)) + 5, int(round(y)) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        self.COLORS['text'],
                        1
                    )
                except Exception:
                    pass
        
        return image
    
    def _add_info_panel(self, image: np.ndarray, predictions: Dict) -> np.ndarray:
        """添加信息面板"""
        h, w = image.shape[:2]
        
        # 准备信息文本
        info_lines = []
        
        # 检测统计
        num_detections = predictions.get('num_detections', 0)
        info_lines.append(f'Detections: {num_detections}')
        
        # 推理时间
        inference_time = predictions.get('inference_time', 0)
        if inference_time > 0:
            info_lines.append(f'Inference: {inference_time:.3f}s')
        
        # 图像尺寸
        image_shape = predictions.get('image_shape', None)
        if image_shape:
            info_lines.append(f'Size: {image_shape[1]}x{image_shape[0]}')
        
        # 绘制信息面板背景
        panel_height = len(info_lines) * 25 + 10
        cv2.rectangle(
            image, 
            (10, 10), 
            (250, 10 + panel_height), 
            (0, 0, 0), 
            -1
        )
        cv2.rectangle(
            image, 
            (10, 10), 
            (250, 10 + panel_height), 
            self.COLORS['box'], 
            2
        )
        
        # 绘制信息文本
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(
                image, 
                line, 
                (20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                self.COLORS['text'], 
                1
            )
        
        return image
    
    def _save_image(self, image: np.ndarray, save_path: str):
        """保存图像"""
        try:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 保存图像
            success = cv2.imwrite(save_path, image)
            if success:
                print(f"可视化结果已保存到: {save_path}")
            else:
                print(f"保存失败: {save_path}")
                
        except Exception as e:
            print(f"保存图像时出错: {e}")
    
    def visualize_batch(self, 
                       images: List[np.ndarray], 
                       predictions_list: List[Dict],
                       save_dir: str = "visualizations",
                       **kwargs) -> List[np.ndarray]:
        """
        批量可视化
        
        Args:
            images: 图像列表
            predictions_list: 预测结果列表
            save_dir: 保存目录
            **kwargs: 其他可视化参数
            
        Returns:
            可视化图像列表
        """
        results = []
        
        # 确保保存目录存在
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        for i, (image, predictions) in enumerate(zip(images, predictions_list)):
            # 生成保存路径
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"result_{i:04d}.jpg")
            
            # 可视化单张图像
            vis_image = self.visualize_predictions(
                image, 
                predictions, 
                save_path=save_path,
                **kwargs
            )
            
            results.append(vis_image)
        
        return results
    
    def create_comparison_grid(self, 
                              original_images: List[np.ndarray],
                              visualized_images: List[np.ndarray],
                              grid_cols: int = 2,
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        创建对比网格图
        
        Args:
            original_images: 原始图像列表
            visualized_images: 可视化图像列表
            grid_cols: 网格列数
            save_path: 保存路径
            
        Returns:
            网格图像
        """
        if len(original_images) != len(visualized_images):
            raise ValueError("原始图像和可视化图像数量不匹配")
        
        n_images = len(original_images)
        if n_images == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 计算网格尺寸
        grid_rows = (n_images + grid_cols - 1) // grid_cols
        
        # 获取单张图像尺寸（假设所有图像尺寸相同）
        h, w = original_images[0].shape[:2]
        
        # 创建网格画布
        grid_h = grid_rows * h
        grid_w = grid_cols * w * 2  # *2因为要并排显示原图和结果
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # 填充网格
        for i in range(n_images):
            row = i // grid_cols
            col = i % grid_cols
            
            # 计算位置
            y_start = row * h
            y_end = y_start + h
            x_start_orig = col * w * 2
            x_end_orig = x_start_orig + w
            x_start_vis = x_end_orig
            x_end_vis = x_start_vis + w
            
            # 放置原始图像
            if i < len(original_images):
                resized_orig = cv2.resize(original_images[i], (w, h))
                grid_image[y_start:y_end, x_start_orig:x_end_orig] = resized_orig
            
            # 放置可视化图像
            if i < len(visualized_images):
                resized_vis = cv2.resize(visualized_images[i], (w, h))
                grid_image[y_start:y_end, x_start_vis:x_end_vis] = resized_vis
        
        # 添加分隔线
        for i in range(1, grid_cols):
            x = i * w * 2
            cv2.line(grid_image, (x, 0), (x, grid_h), (255, 255, 255), 2)
        
        for i in range(1, grid_rows):
            y = i * h
            cv2.line(grid_image, (0, y), (grid_w, y), (255, 255, 255), 2)
        
        # 保存网格图
        if save_path:
            self._save_image(grid_image, save_path)
        
        return grid_image
    
    def set_thresholds(self, conf_thres: Optional[float] = None, keypoint_thres: Optional[float] = None):
        """设置阈值"""
        if conf_thres is not None:
            self.conf_thres = conf_thres
        if keypoint_thres is not None:
            self.keypoint_thres = keypoint_thres


# 便捷函数
def visualize_single_prediction(image: np.ndarray, 
                               predictions: Dict,
                               save_path: Optional[str] = None,
                               **kwargs) -> np.ndarray:
    """
    可视化单个预测结果的便捷函数
    
    Args:
        image: 输入图像
        predictions: 预测结果
        save_path: 保存路径
        **kwargs: 其他可视化参数
        
    Returns:
        可视化后的图像
    """
    visualizer = YOLOVisualizer()
    # ensure keypoint names are shown by default unless explicitly overridden
    if 'show_keypoint_names' not in kwargs:
        kwargs['show_keypoint_names'] = True
    return visualizer.visualize_predictions(image, predictions, save_path, **kwargs)


