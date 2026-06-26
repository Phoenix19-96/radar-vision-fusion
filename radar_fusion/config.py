"""Global configuration for the radar-camera fusion system."""

import os

CONFIG = {
    # --- Radar Detection ---
    "radar_eps": 2.0,              # DBSCAN eps in meters (x-y plane)
    "radar_min_samples": 3,        # DBSCAN min points per cluster
    "radar_min_rcs": -10.0,        # Min RCS (dB) to filter noise points

    # --- Camera Detection ---
    "camera_confidence": 0.3,      # YOLO confidence threshold
    "camera_nms_iou": 0.45,        # NMS IoU threshold
    # COCO class IDs: 2=car, 5=bus, 7=truck, 0=person, 1=bicycle, 3=motorcycle
    "camera_classes": [2, 5, 7, 0, 1, 3],

    # --- Fusion ---
    "fusion_max_depth_diff": 5.0,   # Max depth discrepancy (meters)
    "fusion_outside_penalty": 50.0, # Penalty for radar point outside camera bbox

    # --- Tracking ---
    "track_born_confirm": 2,       # Consecutive hits to confirm new track
    "track_coast_max": 5,          # Max coasting frames before deletion
    "track_init_variance": 1.0,    # Initial position variance (m^2)
    "track_process_noise": 0.5,    # Process noise std for ax/ay (m/s^2)
    "track_meas_noise_pos": 0.3,   # Position measurement noise std (m)
    "track_meas_noise_vel": 0.1,   # Velocity measurement noise std (m/s)

    # --- Data ---
    "dataroot": "D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes",
    "version": "v1.0-mini",
    "scene_idx": 0,
}
