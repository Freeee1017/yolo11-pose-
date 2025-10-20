#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速统计关键点标签情况

假设标签格式 YOLO 关键点：
class cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... (所有坐标为归一化 0-1)
若你的格式不同，请按需求修改 parse_label_line。
"""
import os
from pathlib import Path
from typing import List, Tuple
import math
import json

LABEL_DIR = Path('dataset/labels/train')  # 可手动修改
NK = 21  # 与 config.yaml 一致


def parse_label_line(line: str, nk: int) -> Tuple[int, List[Tuple[float, float, float]]]:
    parts = line.strip().split()
    if len(parts) < 5 + nk * 3:
        return -1, []
    cls = int(float(parts[0]))
    # 跳过 bbox 5 个 (cls + cx cy w h) -> 后面 nk*3
    kpts = []
    k_raw = parts[5:5 + nk * 3]
    for i in range(nk):
        try:
            kx = float(k_raw[i*3])
            ky = float(k_raw[i*3+1])
            vis = float(k_raw[i*3+2])
            kpts.append((kx, ky, vis))
        except Exception:
            kpts.append((math.nan, math.nan, 0.0))
    return cls, kpts


def scan_labels(label_dir: Path, nk: int):
    files = list(label_dir.glob('*.txt'))
    total_files = len(files)
    cls_count = {}
    vis_sum = [0.0]*nk
    vis_cnt = [0]*nk
    missing_files = 0

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                lines = [l for l in fh.readlines() if l.strip()]
            if not lines:
                continue
            # 本工程假设每张只有一个目标，取第一行
            cls, kpts = parse_label_line(lines[0], nk)
            if cls >= 0:
                cls_count[cls] = cls_count.get(cls, 0) + 1
            for i,(kx,ky,v) in enumerate(kpts):
                if not math.isnan(kx) and not math.isnan(ky):
                    vis_sum[i] += v
                    vis_cnt[i] += 1
        except Exception:
            missing_files += 1

    vis_avg = [ (vis_sum[i]/vis_cnt[i]) if vis_cnt[i]>0 else 0.0 for i in range(nk) ]
    report = {
        'total_label_files': total_files,
        'parsed_files': total_files - missing_files,
        'class_distribution': cls_count,
        'keypoint_visibility_avg': vis_avg,
        'keypoint_covered_counts': vis_cnt
    }
    return report


def main():
    report = scan_labels(LABEL_DIR, NK)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
