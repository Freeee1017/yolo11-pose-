import os
import argparse
from typing import Optional

import cv2
import torch

from model import create_model
from predicter import YOLOPredicter
from config_utils import load_config


def load_weights(model, weights_path: str, device: Optional[torch.device] = None):
    """Load checkpoint or state_dict into model (best-effort)."""
    device = device or next(model.parameters()).device
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    ckpt = torch.load(weights_path, map_location=device)
    # allow both full checkpoint or plain state_dict
    if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'state_dict' in ckpt):
        state = ckpt.get('model_state_dict') or ckpt.get('state_dict')
        model.load_state_dict(state, strict=False)
    elif isinstance(ckpt, dict):
        # try direct load or nested keys fallback
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception:
            for k in ['model_state_dict', 'state_dict', 'weights']:
                if k in ckpt:
                    try:
                        model.load_state_dict(ckpt[k], strict=False)
                        break
                    except Exception:
                        continue
    else:
        raise RuntimeError('Unsupported checkpoint format')
    print(f'Loaded weights from {weights_path}')


def run(weights: str,
        img: str,
        img_size: int,
        device: str,
        save_dir: str,
        save_original: bool):
    """Thin entry: build model, load weights, run predicter, visualize via YOLOVisualizer."""
    dev = torch.device(device)
    model = create_model()
    model.to(dev)
    # apply inference-related config (e.g., topk_fuse) if available
    cfg = {}
    try:
        cfg = load_config('config.yaml')
        # keep a reference for downstream tools
        setattr(model, 'config', cfg)
        if hasattr(model, 'apply_config'):
            model.apply_config(cfg)
    except Exception:
        # keep cfg as empty dict on failure
        cfg = cfg or {}
    if weights:
        load_weights(model, weights, device=dev)

    pred = YOLOPredicter(model, device=dev)
    # apply keypoint visibility threshold from config if present
    try:
        vis_thr = float(cfg.get('kpt_vis_threshold', getattr(model, 'config', {}).get('kpt_vis_threshold', 0.5) if isinstance(getattr(model, 'config', None), dict) else 0.5))
    except Exception:
        vis_thr = 0.5
    try:
        pred.kpt_vis_thres = vis_thr
    except Exception:
        pass
    res = pred.predict_single(img, img_size=img_size, return_original=True)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(img)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(save_dir, f'{name}_pred.jpg')

    # get original image
    orig = res.get('original_image')
    if orig is None:
        orig = cv2.imread(img)

    # visualize using central visualizer
    try:
        from visualize import YOLOVisualizer
        # use same visibility threshold for visualizer
        try:
            viz = YOLOVisualizer(keypoint_thres=vis_thr)
        except Exception:
            viz = YOLOVisualizer()
        vis = viz.visualize_predictions(orig, res, save_path=None)
    except Exception:
        vis = orig  # fallback to original image if visualizer not available

    cv2.imwrite(out_path, vis)
    print(f'Saved visualization to {out_path}')
    if save_original and orig is not None:
        orig_out = os.path.join(save_dir, f'{name}_orig.jpg')
        cv2.imwrite(orig_out, orig)
        print(f'Saved original to {orig_out}')


def main():
    parser = argparse.ArgumentParser(description='YOLO11 single-image prediction entry')
    # keep original-style CLI options and defaults close to previous behavior
    parser.add_argument('--weights', type=str, default=r'runs\train\internet\last.pt', help='path to weights file')
    parser.add_argument('--img', type=str, default=r'F:\\dataset\\images\\test\\abnormal_012_060.jpg', help='path to input image')
    parser.add_argument('--img-size', type=int, default=640, help='inference image size')
    parser.add_argument('--device', type=str, default='cpu', help='device: cpu or cuda:0')
    parser.add_argument('--save-dir', type=str, default='runs/predict', help='output directory')
    parser.add_argument('--save-original', action='store_true', help='also save original image copy')
    args = parser.parse_args()

    run(weights=args.weights,
        img=args.img,
        img_size=args.img_size,
        device=args.device,
        save_dir=args.save_dir,
        save_original=args.save_original)


if __name__ == '__main__':
    main()
