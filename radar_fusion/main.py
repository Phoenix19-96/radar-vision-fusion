#!/usr/bin/env python
"""Main entry point for the radar-camera fusion perception pipeline.

Usage:
    cd D:/wzr/PyWorkspace/Fusion
    PYTHONPATH="nuscenes-devkit/python-sdk;radar_fusion" python radar_fusion/main.py

Pipeline steps (per-frame):
  3. Radar detection  (DBSCAN clustering on xy-plane)
  4. Camera detection (YOLOv8n, COCO-pretrained)
  5. Decision-level fusion (Hungarian matching)
  6. Tracking (CA-Kalman + state machine)
  7. Visualization (camera overlay + BEV → .mp4 video)
"""

import sys
import os
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","nuscenes-devkit", "python-sdk"))

from config import CONFIG
from data_loader import FusionDataLoader
from radar_detector import RadarDetector
from camera_detector import CameraDetector
from fusion import FusionModule
from tracker import MultiObjectTracker
from visualize import render_frame, render_bev, CLASS_NAMES
from evaluate import FusionEvaluator


def main():
    print("=" * 60)
    print("Radar-Camera Fusion Perception System")
    print("=" * 60)

    # ============================================================
    # Step 1: 解析数据 (Data Loading)
    # ============================================================
    print("\n[1/7] Loading nuScenes data...")
    loader = FusionDataLoader(version=CONFIG["version"],
                              dataroot=CONFIG["dataroot"])
    scene_idx = CONFIG["scene_idx"]
    scene = loader.nusc.scene[scene_idx]
    print(f"  Scene {scene_idx}: {scene['name']}")
    print(f"  Description: {scene['description']}")

    # ============================================================
    # Step 2: 算法参数初始化 (Initialize Detectors, Fusion, Tracker)
    # ============================================================
    print("\n[2/7] Initializing algorithm modules...")
    radar_det = RadarDetector(CONFIG)
    camera_det = CameraDetector(CONFIG)
    print(f"  Radar:  point-wise detection "
          f"(each pre-clustered radar target → one detection)")

    fusion = FusionModule(CONFIG)
    print(f"  Fusion: Late fusion with Hungarian matching")

    tracker = MultiObjectTracker(CONFIG)
    print(f"  Tracker: CA-Kalman filter, confirm={CONFIG['track_born_confirm']}, "
          f"coast_max={CONFIG['track_coast_max']}")

    # --- Evaluation ---
    eval_enabled = CONFIG.get("eval_enabled", False)
    evaluator = None
    if eval_enabled:
        evaluator = FusionEvaluator(
            dist_threshold=CONFIG.get("eval_dist_threshold", 2.0),
        )
        print(f"  Eval:  enabled, dist_threshold={evaluator.dist_threshold}m")

    # --- Load all frames for the scene ---
    frames = list(loader.iter_scene(scene_idx))
    if not frames:
        print("No frames to render.")
        return

    # --- Precompute canvas size from sample frame ---
    sample_bev = render_bev(peaks=frames[0]["peaks"], figsize=(6, 6))
    sample_img = frames[0]["image"]
    bev_h, bev_w = sample_bev.shape[:2]
    img_h, img_w = sample_img.shape[:2]
    total_w = img_w + bev_w
    total_h = max(img_h, bev_h)

    # --- Create output video ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scene_{scene_idx}_fusion.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 10.0, (total_w, total_h))

    # ============================================================
    # Per-frame pipeline loop
    # ============================================================
    # --- Evaluation state ---
    radar_eval_results = [] if eval_enabled else None
    fusion_eval_results = [] if eval_enabled else None

    print(f"\nProcessing {len(frames)} frames...")
    t_start = time.time()

    for i, data in enumerate(frames):
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames)}")

        # ============================================================
        # Step 3: 雷达算法运行 (Radar Detection — point-wise)
        # ============================================================
        radar_dets = radar_det.detect(data["peaks"],
                                      debug=CONFIG["radar_debug_plot"],
                                      image=data["image"])

        # ============================================================
        # Step 4: 视觉算法运行 (Camera Detection — YOLOv8n)
        # ============================================================
        cam_dets = camera_det.detect(data["image"])

        # ============================================================
        # Step 5: 决策级融合算法 (Decision-level Fusion)
        # ============================================================
        fused = fusion.fuse(radar_dets, cam_dets, data["cs_radar"],
                            data["cs_cam"], data["camera_intrinsic"])

        # --- Evaluate radar-only and fusion outputs ---
        if eval_enabled and evaluator is not None:
            gt = evaluator.load_gt(loader.nusc, data["sample_token"],
                                   data["cs_radar"], data["ego_pose"])
            if len(radar_dets) > 0:
                radar_eval_results.append(
                    evaluator.evaluate_frame(radar_dets, gt))
            if len(fused) > 0:
                fusion_eval_results.append(
                    evaluator.evaluate_frame(fused, gt))

        # ============================================================
        # Step 6: 跟踪及目标管理 (Tracking & Target Management)
        # ============================================================
        tracks = tracker.update(fused, data["timestamp"])

        # ============================================================
        # Step 7: 可视化展示 (Visualization)
        # ============================================================
        # 7a. Camera image with detection/track overlays
        img_overlay = render_frame(data["image"], radar_dets=radar_dets,
                                   camera_dets=cam_dets, fused_objects=fused,
                                   tracks=tracks, cs_radar=data["cs_radar"],
                                   cs_cam=data["cs_cam"],
                                   camera_intrinsic=data["camera_intrinsic"])

        # 7b. Bird's Eye View with radar points, fused objects, track trails
        bev_img = render_bev(peaks=data["peaks"],
                             radar_dets=radar_dets,
                             fused_objects=fused, tracks=tracks)

        # 7c. Compose side-by-side and write to video
        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined[:img_h, :img_w] = img_overlay
        bev_resized = cv2.resize(bev_img, (bev_w, bev_h))
        combined[:bev_h, img_w:img_w + bev_w] = bev_resized
        out.write(combined)

    out.release()

    # --- Print evaluation summary ---
    if eval_enabled and evaluator is not None:
        if radar_eval_results:
            radar_summary = evaluator.summarize(
                radar_eval_results, scene_name=scene["name"] + " [radar-only]")
            evaluator.print_summary(radar_summary)
        if fusion_eval_results:
            fusion_summary = evaluator.summarize(
                fusion_eval_results, scene_name=scene["name"] + " [fusion]")
            evaluator.print_summary(fusion_summary)

    elapsed = time.time() - t_start

    fps = len(frames) / elapsed if elapsed > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Complete: {len(frames)} frames in {elapsed:.1f}s "
          f"({elapsed/len(frames):.2f}s/frame, {fps:.1f} FPS)")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
