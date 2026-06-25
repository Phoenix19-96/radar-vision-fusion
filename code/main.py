#!/usr/bin/env python
"""Main entry point for the radar-camera fusion perception pipeline.

Usage:
    cd D:/wzr/PyWorkspace/Fusion
    PYTHONPATH="nuscenes-devkit/python-sdk;code" python code/main.py

Processes scene 0 of nuScenes v1.0-mini through the full pipeline:
  Radar clustering -> Camera detection -> Fusion -> Tracking -> Visualization.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                 "nuscenes-devkit", "python-sdk"))

from config import CONFIG
from data_loader import FusionDataLoader
from radar_detector import RadarDetector
from camera_detector import CameraDetector
from fusion import FusionModule
from tracker import MultiObjectTracker
from visualize import render_scene_video, CLASS_NAMES


def main():
    print("=" * 60)
    print("Radar-Camera Fusion Perception System")
    print("=" * 60)

    # --- Initialize ---
    print("\n[1/5] Loading nuScenes data...")
    loader = FusionDataLoader(version=CONFIG["version"],
                              dataroot=CONFIG["dataroot"])
    scene_idx = CONFIG["scene_idx"]
    scene = loader.nusc.scene[scene_idx]
    print(f"  Scene {scene_idx}: {scene['name']}")
    print(f"  Description: {scene['description']}")

    print("\n[2/5] Initializing detectors...")
    radar_det = RadarDetector(CONFIG)
    camera_det = CameraDetector(CONFIG)
    print(f"  Radar: DBSCAN eps={CONFIG['radar_eps']}m, "
          f"min_samples={CONFIG['radar_min_samples']}")
    print(f"  Camera: YOLOv8n, conf={CONFIG['camera_confidence']}")

    print("\n[3/5] Initializing fusion module...")
    fusion = FusionModule(CONFIG)
    print("  Late fusion with Hungarian matching")

    print("\n[4/5] Initializing tracker...")
    tracker = MultiObjectTracker(CONFIG)
    print(f"  CA-Kalman filter, confirm={CONFIG['track_born_confirm']}, "
          f"coast_max={CONFIG['track_coast_max']}")

    print("\n[5/5] Running pipeline and rendering video...")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scene_{scene_idx}_fusion.mp4")

    t_start = time.time()
    render_scene_video(loader, radar_det, camera_det, fusion, tracker,
                       scene_idx=scene_idx, out_path=out_path)
    elapsed = time.time() - t_start

    frames = list(loader.iter_scene(scene_idx))
    fps = len(frames) / elapsed if elapsed > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Complete: {len(frames)} frames in {elapsed:.1f}s "
          f"({elapsed/len(frames):.2f}s/frame, {fps:.1f} FPS)")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
