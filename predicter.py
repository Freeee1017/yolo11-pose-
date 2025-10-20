import os
from typing import List, Union, Optional, Dict, Any

import cv2
import numpy as np
import torch

from utils import letterbox, xywh2xyxy, xyxy2xywh, inv_letterbox_xyxy, inv_letterbox_kpts


class YOLOPredicter:
    """Simple predicter for YOLO11 single-target pose model.

    - Preprocesses images with `letterbox` (same as dataset)
    - Runs model in eval mode
    - Parses head output (single-target tensor) and maps detections/keypoints
      from letterbox pixels back to original image pixels.
    """

    def __init__(self, model, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.conf_thres = 0.25
        # 关键点可见性阈值，低于此阈值视为不可见并置为0
        self.kpt_vis_thres = 0.2

    def set_thresholds(self, conf: float):
        self.conf_thres = float(conf)

    def _preprocess(self, img: Union[str, np.ndarray], img_size: int):
        # load image if path
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(img)
            img0 = cv2.imread(img)
            if img0 is None:
                raise RuntimeError(f'failed to read image: {img}')
        else:
            img0 = img.copy()

        orig_h, orig_w = img0.shape[:2]
        img_lb, ratio, pad = letterbox(img0, new_shape=img_size)
        img_in = img_lb.transpose((2, 0, 1))  # HWC->CHW
        img_in = np.ascontiguousarray(img_in)
        img_in = torch.from_numpy(img_in).float() / 255.0
        return img_in, img0, (orig_w, orig_h), ratio, pad

    # use shared utils for inverse mapping
    def _inv_map_xyxy(self, xyxy: Union[List[float], np.ndarray], pad: tuple, ratio: tuple) -> Optional[List[float]]:
        return inv_letterbox_xyxy(xyxy, pad, ratio)

    def _inv_map_kpts(self, kpts: np.ndarray, pad: tuple, ratio: tuple) -> np.ndarray:
        return inv_letterbox_kpts(kpts, pad, ratio)

    def _build_detection(self, s_np: np.ndarray, img_size: int, ratio: tuple, pad: tuple) -> Optional[Dict[str, Any]]:
        """Parse a single prediction vector and map to multiple coord systems.

        Returns a detection dict or None (e.g., filtered by conf_thres or empty).
        """
        if s_np is None or s_np.size == 0:
            return None

        # bbox and score
        bbox_vals = s_np[0:4].astype(float)
        score = float(s_np[4]) if s_np.shape[0] > 4 else 1.0
        if score < self.conf_thres:
            return None

        # Determine whether bbox is normalized (<=1) or in pixels
        if bbox_vals.max() <= 1.0:
            # normalized relative to letterbox output size (img_size x img_size)
            bbox_vals_px = bbox_vals * float(img_size)
        else:
            bbox_vals_px = bbox_vals

        # convert xywh->xyxy (numpy), fallback if already xyxy
        try:
            bbox_xyxy_lb = xywh2xyxy(np.array([bbox_vals_px]))[0]
        except Exception:
            bbox_xyxy_lb = bbox_vals_px

        # keypoints
        kpt_flat = None
        if s_np.shape[0] >= 5 + 63:
            kpt_flat = s_np[5:5+63].astype(float)
        elif s_np.shape[0] == 63:
            kpt_flat = s_np.astype(float)

        kpts_pixels = None
        if kpt_flat is not None:
            kpt_arr = kpt_flat.reshape(21, 3).copy()
            # if normalized to [0,1], scale to letterbox px
            if kpt_arr.max() <= 1.0:
                kpt_arr[:, 0] = kpt_arr[:, 0] * float(img_size)
                kpt_arr[:, 1] = kpt_arr[:, 1] * float(img_size)
            # mask low-visibility keypoints
            try:
                low_vis_mask = (kpt_arr[:, 2] < self.kpt_vis_thres)
                if low_vis_mask.any():
                    kpt_arr[low_vis_mask, :] = 0.0
            except Exception:
                pass
            kpts_pixels = kpt_arr

        # inverse map from letterbox pixels to original image pixels
        try:
            bbox_xyxy_lb_arr = np.asarray(bbox_xyxy_lb, dtype=float).reshape(4,)
        except Exception:
            bbox_xyxy_lb_arr = np.array(list(bbox_xyxy_lb), dtype=float)
        bbox_xyxy_lb_list = bbox_xyxy_lb_arr.tolist()
        bbox_xyxy_orig = self._inv_map_xyxy(bbox_xyxy_lb_arr, pad, ratio)
        if bbox_xyxy_orig is None:
            # fallback to letterbox coords if inverse mapping failed
            bbox_xyxy_orig = bbox_xyxy_lb_arr.tolist()

        kpts_orig = None
        if kpts_pixels is not None:
            kpts_orig = self._inv_map_kpts(kpts_pixels, pad, ratio)

        # normalized bbox relative to letterbox size
        bbox_lb_xywh = None
        bbox_norm_xywh = None
        try:
            bbox_lb_np = np.array(bbox_xyxy_lb_list, dtype=float)
            bbox_lb_xywh = xyxy2xywh(bbox_lb_np.reshape(1, 4))[0].tolist()
            bbox_norm_xywh = [
                bbox_lb_xywh[0] / float(img_size), bbox_lb_xywh[1] / float(img_size),
                bbox_lb_xywh[2] / float(img_size), bbox_lb_xywh[3] / float(img_size)
            ]
        except Exception:
            bbox_lb_xywh = None
            bbox_norm_xywh = None

        # original bbox xywh
        try:
            bbox_xyxy_np = np.array(bbox_xyxy_orig, dtype=float)
            bbox_xywh_orig = xyxy2xywh(bbox_xyxy_np.reshape(1, 4))[0].tolist()
        except Exception:
            bbox_xywh_orig = None

        # keypoints representations
        kpts_lb = None
        kpts_norm = None
        if kpts_pixels is not None:
            kpts_lb = kpts_pixels.tolist()
            kpts_norm = kpts_pixels.copy()
            kpts_norm[:, 0] = kpts_norm[:, 0] / float(img_size)
            kpts_norm[:, 1] = kpts_norm[:, 1] / float(img_size)
            kpts_norm = kpts_norm.tolist()

        det = {
            'bbox': [int(round(x)) for x in bbox_xyxy_orig],                    # original px xyxy
            'bbox_xywh': [float(x) for x in bbox_xywh_orig] if bbox_xywh_orig is not None else None,
            'bbox_lb_xyxy': [float(x) for x in bbox_xyxy_lb_list],              # letterbox px xyxy
            'bbox_lb_xywh': [float(x) for x in bbox_lb_xywh] if bbox_lb_xywh is not None else None,
            'bbox_norm': [float(x) for x in bbox_norm_xywh] if bbox_norm_xywh is not None else None,
            'confidence': float(score),
            'keypoints': kpts_orig.tolist() if kpts_orig is not None else [],   # original px
            'keypoints_lb': kpts_lb,                                             # letterbox px
            'keypoints_norm': kpts_norm                                           # normalized (0..1) relative to letterbox
        }
        return det

    def predict_single(self, image_input: Union[str, np.ndarray], img_size: int = 640, return_original: bool = False) -> Dict[str, Any]:
        img_tensor, orig_img, orig_size, ratio, pad = self._preprocess(image_input, img_size)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            im = img_tensor.unsqueeze(0).to(self.device)
            preds = self.model(im)

        # extract single-target tensor
        if isinstance(preds, tuple) and len(preds) >= 1:
            single = preds[0]
        else:
            single = preds

        if isinstance(single, torch.Tensor):
            s = single.detach().cpu()
            if s.dim() == 3 and s.size(-1) == 1:
                s = s.squeeze(-1)
            s = s.squeeze(0) if s.size(0) == 1 and s.dim() == 2 else s
        else:
            raise RuntimeError('unexpected model output')

        # default empty result with metadata similar to validater
        result = {
            'num_detections': 0,
            'detections': [],
            'original_image': orig_img if return_original else None,
            'orig_size': orig_size,
            'ratio': ratio,
            'pad': pad,
            'img_size': img_size
        }

        # s expected shape (C,) or (1,C)
        if s.numel() == 0:
            return result

        # ensure 1D
        if s.dim() == 2:
            s = s[0]

        det = self._build_detection(s.numpy(), img_size, ratio, pad)
        if det is not None:
            result['num_detections'] = 1
            result['detections'] = [det]
        return result

    def predict_batch(self, inputs: List[Union[str, np.ndarray]], img_size: int = 640) -> List[Dict[str, Any]]:
        # preprocess all images, keep metadata
        imgs = []
        metas = []
        for inp in inputs:
            img_t, orig, orig_size, ratio, pad = self._preprocess(inp, img_size)
            imgs.append(img_t)
            metas.append((orig, orig_size, ratio, pad))

        batch = torch.stack(imgs, 0).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(batch)

        if isinstance(preds, tuple) and len(preds) >= 1:
            single = preds[0]
        else:
            single = preds

        if isinstance(single, torch.Tensor):
            single = single.detach().cpu()
        else:
            try:
                single = torch.tensor(single).detach().cpu()
            except Exception:
                # fallback: try to convert iterable of tensors/numbers into stacked tensor
                try:
                    items = [x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in single]
                    single = torch.stack(items)
                except Exception:
                    raise RuntimeError('unexpected model output type for batch predictions')
        if single.dim() == 3 and single.size(-1) == 1:
            single = single.squeeze(-1)

        results = []
        for i in range(single.size(0)):
            s = single[i]
            result = {
                'num_detections': 0,
                'detections': [],
                'original_image': metas[i][0],
                'orig_size': metas[i][1],
                'ratio': metas[i][2],
                'pad': metas[i][3],
                'img_size': img_size
            }

            det = self._build_detection(s.numpy(), img_size, metas[i][2], metas[i][3])
            if det is not None:
                result['num_detections'] = 1
                result['detections'] = [det]
            results.append(result)

        return results


if __name__ == '__main__':
    # quick smoke: create model if available
    try:
        from model import create_model
        m = create_model()
        p = YOLOPredicter(m)
        print('predicter created')
    except Exception:
        print('predicter module loaded')
