"""Decision-level fusion of radar and camera detections.

Projects radar cluster centroids to the camera image plane, then uses
Hungarian matching to associate radar and camera object proposals.

Three outcomes per detection:
  - Matched (radar ∩ camera): position from radar, class from camera
  - Radar-only: class = -1 (unknown), velocity available
  - Camera-only: velocity = 0 (unavailable), position estimated from bbox
"""

import numpy as np
from config import CONFIG
from utils.coordinate import radar_to_pixel
from utils.association import build_fusion_cost_matrix, hungarian_match


class FusionModule:
    """Fuse radar and camera detections via geometric projection + association.

    Radar detections provide accurate 3D position + velocity.
    Camera detections provide 2D bounding box + class + confidence.
    """

    def __init__(self, config=None):
        cfg = config if config is not None else CONFIG
        self.max_depth_diff = cfg.get("fusion_max_depth_diff", 5.0)
        self.outside_penalty = cfg.get("fusion_outside_penalty", 50.0)

    def fuse(self, radar_detections, camera_detections, cs_radar, cs_cam, camera_intrinsic):
        """Fuse radar and camera detections for one frame.

        Args:
            radar_detections: (K, 8) from RadarDetector.detect().
            camera_detections: (N, 6) from CameraDetector.detect().
            cs_radar: calibrated_sensor record for radar.
            cs_cam: calibrated_sensor record for camera.
            camera_intrinsic: (3, 3) camera K matrix.

        Returns:
            fused_objects: (M, 8) float64 array:
                [x, y, z, vx, vy, class_id, conf_radar, conf_camera]
        """
        K = radar_detections.shape[0]
        N = camera_detections.shape[0]

        fused = []

        # --- Project radar centroids to image plane ---
        if K > 0:
            radar_xyz = radar_detections[:, :3].T  # (3, K)
            radar_uv, depths = radar_to_pixel(radar_xyz, cs_radar, cs_cam, camera_intrinsic)

            # Build cost matrix and match
            if N > 0 and radar_uv.shape[1] > 0:
                cfg = {
                    "fusion_max_depth_diff": self.max_depth_diff,
                    "fusion_outside_penalty": self.outside_penalty,
                }
                cost = build_fusion_cost_matrix(radar_uv, depths, camera_detections, cfg)
                row_ind, col_ind, unmatched_r, unmatched_c = hungarian_match(cost, max_cost=100.0)

                # --- Matched pairs ---
                for r_idx, c_idx in zip(row_ind, col_ind):
                    obj = np.zeros(8)
                    obj[0:3] = radar_detections[r_idx, 0:3]      # x, y, z from radar
                    obj[3:5] = radar_detections[r_idx, 3:5]      # vx, vy from radar
                    obj[5] = camera_detections[c_idx, 4]          # class from camera
                    obj[6] = 0.8  # conf_radar
                    obj[7] = camera_detections[c_idx, 5]          # conf_camera
                    fused.append(obj)

                # --- Radar-only (camera missed) ---
                for r_idx in unmatched_r:
                    if r_idx < len(depths):
                        obj = np.zeros(8)
                        obj[0:3] = radar_detections[r_idx, 0:3]
                        obj[3:5] = radar_detections[r_idx, 3:5]
                        obj[5] = -1  # unknown class
                        obj[6] = 0.5
                        obj[7] = 0.0
                        fused.append(obj)

                # --- Camera-only (radar missed) ---
                for c_idx in unmatched_c:
                    u1, v1, u2, v2 = camera_detections[c_idx, :4]
                    bbox_height = v2 - v1
                    est_depth = 1000.0 / max(bbox_height, 1)
                    fx = camera_intrinsic[0, 0]
                    cx = camera_intrinsic[0, 2]
                    cx_img = (u1 + u2) / 2
                    obj = np.zeros(8)
                    obj[0] = est_depth
                    obj[1] = (cx_img - cx) * est_depth / fx
                    obj[2] = 0.0
                    obj[3:5] = 0.0
                    obj[5] = camera_detections[c_idx, 4]
                    obj[6] = 0.0
                    obj[7] = camera_detections[c_idx, 5]
                    fused.append(obj)
            else:
                # No camera detections — all radar as radar-only
                for i in range(K):
                    obj = np.zeros(8)
                    obj[0:3] = radar_detections[i, 0:3]
                    obj[3:5] = radar_detections[i, 3:5]
                    obj[5] = -1
                    obj[6] = 0.5
                    obj[7] = 0.0
                    fused.append(obj)
        elif N > 0:
            # No radar detections — all camera as camera-only
            for c_idx in range(N):
                u1, v1, u2, v2 = camera_detections[c_idx, :4]
                est_depth = 1000.0 / max(v2 - v1, 1)
                obj = np.zeros(8)
                obj[0] = est_depth
                obj[1] = 0.0
                obj[2] = 0.0
                obj[3:5] = 0.0
                obj[5] = camera_detections[c_idx, 4]
                obj[6] = 0.0
                obj[7] = camera_detections[c_idx, 5]
                fused.append(obj)

        if not fused:
            return np.empty((0, 8))
        return np.array(fused)
