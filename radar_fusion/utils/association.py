"""Association utilities: cost matrix construction and Hungarian matching.

Used in two places:
  - Fusion (Task 7): match radar detections to camera detections
  - Tracking (Task 9): match predicted tracks to current observations
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_fusion_cost_matrix(radar_uv, radar_depths, camera_dets, config):
    """Build cost matrix for radar-camera fusion association.

    For each radar detection projected to pixel (u, v, depth), computes
    cost against each camera detection bounding box.

    Cost strategy:
      - If (u,v) inside bbox: cost = normalized distance from bbox center
      - If outside: cost = pixel_distance_to_bbox + outside_penalty

    Args:
        radar_uv: (2, K) array of projected pixel coordinates.
        radar_depths: (K,) array of depths in camera frame (meters).
        camera_dets: (N, 6) array [u1, v1, u2, v2, class_id, score].
        config: dict with "fusion_max_depth_diff", "fusion_outside_penalty".

    Returns:
        cost_matrix: (K, N) float64 array. Large values = poor match.
    """
    K = radar_uv.shape[1]
    N = camera_dets.shape[0]
    outside_penalty = config.get("fusion_outside_penalty", 50.0)

    cost = np.full((K, N), 1e9)

    for i in range(K):
        u_r, v_r = radar_uv[0, i], radar_uv[1, i]

        for j in range(N):
            u1, v1, u2, v2 = camera_dets[j, 0], camera_dets[j, 1], camera_dets[j, 2], camera_dets[j, 3]

            # Check if radar projection is inside the camera bbox
            inside = (u1 <= u_r <= u2) and (v1 <= v_r <= v2)

            if inside:
                # Distance from bbox center, normalized by bbox diagonal
                cx, cy = (u1 + u2) / 2, (v1 + v2) / 2
                diag = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2) + 1e-6
                dist = np.sqrt((u_r - cx) ** 2 + (v_r - cy) ** 2) / diag
                cost[i, j] = dist  # 0 = center, grows toward edges
            else:
                # Distance to nearest bbox edge
                dx = max(u1 - u_r, 0, u_r - u2)
                dy = max(v1 - v_r, 0, v_r - v2)
                pixel_dist = np.sqrt(dx ** 2 + dy ** 2)
                cost[i, j] = pixel_dist + outside_penalty

    return cost


def hungarian_match(cost_matrix, max_cost=50.0):
    """Solve optimal assignment using the Hungarian algorithm.

    Args:
        cost_matrix: (K, N) array where cost_matrix[i, j] is cost of
                     matching row i to column j.
        max_cost: Maximum allowed cost for a valid match. Pairs exceeding
                  this are left unmatched.

    Returns:
        row_ind: (M,) int array of matched row indices.
        col_ind: (M,) int array of matched column indices.
        unmatched_rows: list of unmatched row indices.
        unmatched_cols: list of unmatched column indices.
    """
    K, N = cost_matrix.shape

    # Hungarian algorithm — minimizes total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter by max_cost
    valid = cost_matrix[row_ind, col_ind] <= max_cost
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    # Determine unmatched
    matched_rows = set(row_ind)
    matched_cols = set(col_ind)
    unmatched_rows = [r for r in range(K) if r not in matched_rows]
    unmatched_cols = [c for c in range(N) if c not in matched_cols]

    return row_ind, col_ind, unmatched_rows, unmatched_cols
