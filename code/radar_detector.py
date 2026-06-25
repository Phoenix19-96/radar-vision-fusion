"""Radar point cloud clustering via DBSCAN.

Filters raw nuScenes radar points, clusters in the x-y plane using DBSCAN,
and extracts per-cluster features (centroid, velocity, RCS, size, point count).
"""

import numpy as np
from sklearn.cluster import DBSCAN

from config import CONFIG


class RadarDetector:
    """Cluster radar point clouds into object proposals.

    Filtering:
      - invalid_state == 0 (valid points only)
      - ambig_state == 3 (unambiguous)
      - dyn_prop in [0..6] (all dynamic states: moving, stationary, stopped)
      - rcs >= radar_min_rcs (optional noise filter)

    Clustering: DBSCAN on (x, y) plane.
    Feature extraction: centroid x/y/z, mean compensated velocity, RCS, size, count.
    """

    def __init__(self, config=None):
        cfg = config if config is not None else CONFIG
        self.eps = cfg.get("radar_eps", 2.0)
        self.min_samples = cfg.get("radar_min_samples", 3)
        self.min_rcs = cfg.get("radar_min_rcs", -10.0)

    def detect(self, radar_points):
        """Run filtering, clustering, and feature extraction.

        Args:
            radar_points: (18, N) float32 array from nuScenes RadarPointCloud.

        Returns:
            detections: (K, 8) float64 array, columns:
                [x, y, z, vx, vy, rcs_mean, size_xy, n_points]
            Returns (0, 8) empty array if no clusters found.
        """
        # --- Step 1: Filtering ---
        invalid = radar_points[14, :]  # invalid_state
        ambig = radar_points[11, :]    # ambig_state
        dyn_prop = radar_points[3, :]  # dyn_prop
        rcs = radar_points[5, :]       # rcs

        mask = (
            (invalid == 0) &
            (ambig == 3) &
            (dyn_prop >= 0) & (dyn_prop <= 6)
        )
        if self.min_rcs is not None:
            mask = mask & (rcs >= self.min_rcs)

        valid = radar_points[:, mask]
        if valid.shape[1] < self.min_samples:
            return np.empty((0, 8))

        # --- Step 2: DBSCAN Clustering (x, y plane) ---
        xy = valid[:2, :].T  # (M, 2)
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clusterer.fit_predict(xy)

        # --- Step 3: Feature Extraction per cluster ---
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # exclude noise (-1)

        if len(unique_labels) == 0:
            return np.empty((0, 8))

        detections = np.zeros((len(unique_labels), 8))

        for i, label in enumerate(unique_labels):
            cluster_mask = labels == label
            cluster_pts = valid[:, cluster_mask]  # (18, M_c)

            detections[i, 0] = np.mean(cluster_pts[0, :])   # x centroid
            detections[i, 1] = np.mean(cluster_pts[1, :])   # y centroid
            detections[i, 2] = np.mean(cluster_pts[2, :])   # z centroid
            # Use compensated velocity: vx_comp (index 8), vy_comp (index 9)
            detections[i, 3] = np.mean(cluster_pts[8, :])   # vx mean
            detections[i, 4] = np.mean(cluster_pts[9, :])   # vy mean
            detections[i, 5] = np.mean(cluster_pts[5, :])   # rcs mean
            detections[i, 6] = np.std(cluster_pts[0, :]) + np.std(cluster_pts[1, :])  # size_xy
            detections[i, 7] = cluster_pts.shape[1]          # n_points

        return detections
