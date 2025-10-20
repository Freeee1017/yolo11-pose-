"""
YOLO11 单目标关键点模型封装

这个文件将当前工作区中的 `backbone.py`、`neck.py`、`head.py` 组合成一个可用的模型类，
并提供方便的预测/可视化接口（调用 `predicter` 和 `visualize` 模块）。
"""
from typing import Union, List, Dict, Optional, Any
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# local modules
from backbone import YOLO11Backbone
from neck import YOLO11Neck
from head import SingleTargetPose


class YOLO11KeypointModel(nn.Module):
    """组合 Backbone + Neck + SingleTargetPose 的模型类"""

    def __init__(self, nc: int = 1, nk: int = 21):
        super().__init__()
        self.nc = int(nc)
        self.nk = int(nk)

        # backbone / neck
        self.backbone = YOLO11Backbone()
        self.neck = YOLO11Neck()

        # head：SingleTargetPose 期望传入各尺度通道数 (ch)
        # 根据 neck/backbone 实现，传入 (64,128,256)
        self.head = SingleTargetPose(nc=self.nc, kpt_shape=(self.nk, 3), ch=(64, 128, 256))

        # 延迟创建的工具对象
        self._predicter = None
        self._visualizer = None

    def forward(self, x: torch.Tensor):
        """前向返回 head 的原始输出格式（训练时不解码/推理时解码）"""
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        preds = self.head(neck_feats)
        return preds

    # 将外部配置同步到 Head（例如 Top-K 融合相关超参）
    def apply_config(self, cfg: Optional[Dict[str, Any]]):
        if not isinstance(cfg, dict):
            return
        # Top-K 融合策略（推理时生效）
        try:
            if hasattr(self, 'head') and self.head is not None:
                if 'topk_fuse' in cfg:
                    self.head.topk_fuse = bool(cfg.get('topk_fuse'))
                if 'topk' in cfg:
                    self.head.topk = int(cfg.get('topk', getattr(self.head, 'topk', 5)))
                if 'fuse_temp' in cfg:
                    self.head.fuse_temp = float(cfg.get('fuse_temp', getattr(self.head, 'fuse_temp', 0.5)))
        except Exception:
            # 静默失败，避免训练流程被配置细节阻断
            pass

    @property
    def predicter(self):
        if self._predicter is None:
            from predicter import YOLOPredicter
            self._predicter = YOLOPredicter(model=self, device=next(self.parameters()).device)
        return self._predicter

    @property
    def visualizer(self):
        if self._visualizer is None:
            try:
                from visualize import YOLOVisualizer
                self._visualizer = YOLOVisualizer()
            except Exception:
                self._visualizer = None
        return self._visualizer

    def predict(self, image_input: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
                img_size: int = 640,
                conf_thres: float = 0.25,
                return_original: bool = False,
                visualize: bool = False,
                save_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """便捷预测接口，内部调用 `predicter` 完成推理。"""
        # apply thresholds (single-target: IoU threshold not used)
        self.predicter.set_thresholds(conf_thres)

        if isinstance(image_input, list):
            results = self.predicter.predict_batch(image_input, img_size=img_size)
            if visualize and self.visualizer is not None and save_path:
                # load original images
                originals = []
                for inp in image_input:
                    if isinstance(inp, str) and os.path.exists(inp):
                        img = cv2.imread(inp)
                        if img is not None:
                            originals.append(img)
                    elif isinstance(inp, np.ndarray):
                        originals.append(inp)
                if originals:
                    try:
                        self.visualizer.visualize_batch(originals, results, save_dir=os.path.dirname(save_path))
                    except Exception:
                        pass
            return results

        # single image
        res = self.predicter.predict_single(image_input, img_size=img_size, return_original=return_original)
        if visualize and self.visualizer is not None:
            original_image = None
            if isinstance(image_input, str) and os.path.exists(image_input):
                original_image = cv2.imread(image_input)
            elif isinstance(image_input, np.ndarray):
                original_image = image_input
            elif isinstance(res, dict) and 'original_image' in res:
                original_image = res.get('original_image')
            if original_image is not None:
                try:
                    vis = self.visualizer.visualize_predictions(original_image, res, save_path=save_path)
                    res['visualization'] = vis
                except Exception:
                    pass
        return res

    # convenience wrappers
    def predict_image(self, image_path: str, **kwargs):
        return self.predict(image_path, **kwargs)

    def predict_array(self, image_array: np.ndarray, **kwargs):
        return self.predict(image_array, **kwargs)


def create_model(nc: int = 1, nk: int = 21) -> YOLO11KeypointModel:
    model = YOLO11KeypointModel(nc=nc, nk=nk)
    # ensure params require grad by default
    for p in model.parameters():
        p.requires_grad_(True)
    return model


if __name__ == '__main__':
    # quick smoke test (CPU)
    try:
        m = create_model(nc=1, nk=21)
        m.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = m(x)
        print('forward OK, output type:', type(out))
        # if eval-mode head returns (decoded, (feats,kpt_raw)) tuple
        if isinstance(out, tuple):
            print('tuple output lengths:', len(out))
    except Exception as e:
        print('self-test failed:', e)
