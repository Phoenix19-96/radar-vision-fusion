"""Lightweight evaluation module for radar-camera fusion.

Matches system detections to nuScenes ground-truth annotations in the radar
sensor frame (BEV center-distance matching via Hungarian algorithm) and
computes per-frame / scene-level metrics.

Metrics:
  - Recall:      matched_gt / total_gt
  - Precision:   matched_gt / total_det
  - mATE (m):    mean L2 distance between matched centers (xy plane)
  - mAVE (m/s):  mean L2 distance between matched velocity vectors (xy plane)
"""

import numpy as np
from pyquaternion import Quaternion

from utils.association import hungarian_match


# nuScenes category name prefixes for target classes
EVAL_CLASS_PREFIXES = [
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.motorcycle",
]


def _matches_eval_classes(category_name, prefixes=None):
    """Check whether a nuScenes category name matches any of the target prefixes."""
    for prefix in (prefixes or EVAL_CLASS_PREFIXES):
        if category_name.startswith(prefix):
            return True
    return False


class FusionEvaluator:
    """Match fusion / radar detections to nuScenes GT and compute metrics."""

    def __init__(self, dist_threshold=2.0, target_prefixes=None):
        """
        Args:
            dist_threshold: max BEV center distance (m) for a valid match.
            target_prefixes: list of nuScenes category name prefixes.
        """
        self.dist_threshold = dist_threshold
        self.target_prefixes = target_prefixes or EVAL_CLASS_PREFIXES

    # ------------------------------------------------------------------
    # Ground-truth loading
    # ------------------------------------------------------------------

    def load_gt(self, nusc, sample_token, cs_radar, ego_pose):
        """Extract GT boxes for one sample, transformed to radar sensor frame.

        Args:
            nusc: NuScenes instance.
            sample_token: current sample token.
            cs_radar: calibrated_sensor record for RADAR_FRONT.
            ego_pose: ego_pose record for the sample.

        Returns:
            dict with keys:
                centers:   (G, 2)  x, y in radar sensor frame
                velocities:(G, 2)  vx, vy in radar sensor frame (may contain np.nan)
                classes:   list[str] of length G
        """
        sample = nusc.get("sample", sample_token)
        centers = []
        velocities = []
        classes = []

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            inst = nusc.get("instance", ann["instance_token"])
            cat = nusc.get("category", inst["category_token"])

            if not _matches_eval_classes(cat["name"], self.target_prefixes):
                continue

            # --- GT box in global frame ---
            box_global = _make_eval_box(ann)
            vel_global = nusc.box_velocity(ann_token)  # [vx, vy, vz] or [nan, nan, nan]

            # --- Transform to radar sensor frame ---
            box_radar = _global_to_sensor(box_global, ego_pose, cs_radar)

            centers.append(box_radar.center[:2])       # (x, y)
            velocities.append(vel_global[:2])           # (vx, vy) — global ≈ radar for velocity of nearby targets
            classes.append(cat["name"])

        if not centers:
            return {"centers": np.empty((0, 2)), "velocities": np.empty((0, 2)),
                    "classes": []}

        return {
            "centers": np.array(centers),         # (G, 2)
            "velocities": np.array(velocities),   # (G, 2)
            "classes": classes,                   # list[str]
        }

    # ------------------------------------------------------------------
    # Per-frame evaluation
    # ------------------------------------------------------------------

    def evaluate_frame(self, detections, gt):
        """Match detections to GT and compute frame-level metrics.

        Args:
            detections: (K, 8+) array, columns [x, y, z, vx, vy, ...].
            gt: dict from load_gt().

        Returns:
            dict with keys: recall, precision, mATE_m, mAVE_ms,
                            match_count, gt_count, det_count.
        """
        det_centers = detections[:, :2]   # (K, 2)
        det_vels = detections[:, 3:5]     # (K, 2)
        gt_centers = gt["centers"]
        gt_vels = gt["velocities"]

        K = len(det_centers)
        G = len(gt_centers)

        if K == 0 or G == 0:
            return {
                "recall": 0.0,
                "precision": 0.0,
                "mATE_m": float("nan"),
                "mAVE_ms": float("nan"),
                "match_count": 0,
                "gt_count": G,
                "det_count": K,
            }

        # Build cost matrix: pairwise BEV center distance
        cost = np.linalg.norm(
            det_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :],
            axis=2,
        )  # (K, G)

        # Hungarian matching
        row_ind, col_ind, _, _ = hungarian_match(cost, max_cost=self.dist_threshold)

        match_count = len(row_ind)

        # --- Compute errors on matched pairs ---
        pos_errors = []
        vel_errors = []

        for k, g in zip(row_ind, col_ind):
            pos_errors.append(cost[k, g])  # center distance already computed

            v_det = det_vels[k]
            v_gt = gt_vels[g]
            if not np.any(np.isnan(v_gt)):
                vel_errors.append(np.linalg.norm(v_det - v_gt))

        return {
            "recall": match_count / G,
            "precision": match_count / K,
            "mATE_m": np.mean(pos_errors) if pos_errors else float("nan"),
            "mAVE_ms": np.mean(vel_errors) if vel_errors else float("nan"),
            "match_count": match_count,
            "gt_count": G,
            "det_count": K,
        }

    # ------------------------------------------------------------------
    # Scene-level summary
    # ------------------------------------------------------------------

    def summarize(self, per_frame_results, scene_name=""):
        """Aggregate per-frame results into scene-level statistics.

        Args:
            per_frame_results: list of dicts from evaluate_frame().
            scene_name: optional scene identifier for print header.

        Returns:
            dict with aggregated metrics.
        """
        if not per_frame_results:
            return {}

        recs = [r["recall"] for r in per_frame_results]
        precs = [r["precision"] for r in per_frame_results]
        ates = [r["mATE_m"] for r in per_frame_results if not np.isnan(r["mATE_m"])]
        aves = [r["mAVE_ms"] for r in per_frame_results if not np.isnan(r["mAVE_ms"])]
        total_gt = sum(r["gt_count"] for r in per_frame_results)
        total_det = sum(r["det_count"] for r in per_frame_results)
        total_matched = sum(r["match_count"] for r in per_frame_results)

        summary = {
            "scene": scene_name,
            "num_frames": len(per_frame_results),
            "total_gt": total_gt,
            "total_det": total_det,
            "total_matched": total_matched,
            "mean_recall": np.mean(recs) if recs else 0,
            "mean_precision": np.mean(precs) if precs else 0,
            "mean_ATE_m": np.mean(ates) if ates else float("nan"),
            "mean_AVE_ms": np.mean(aves) if aves else float("nan"),
        }
        return summary

    @staticmethod
    def print_summary(summary):
        """Pretty-print scene-level evaluation summary."""
        print(f"\n{'=' * 60}")
        print(f"Evaluation Summary — {summary.get('scene', '')}")
        print(f"{'=' * 60}")
        print(f"  Frames:            {summary['num_frames']}")
        print(f"  Total GT objects:  {summary['total_gt']}")
        print(f"  Total detections:  {summary['total_det']}")
        print(f"  Matched pairs:     {summary['total_matched']}")
        print(f"  Mean Recall:       {summary['mean_recall']:.2%}")
        print(f"  Mean Precision:    {summary['mean_precision']:.2%}")
        ate = summary['mean_ATE_m']
        ave = summary['mean_AVE_ms']
        print(f"  mATE (center err): {ate:.3f} m" if not np.isnan(ate) else "  mATE:  n/a")
        print(f"  mAVE (velocity err):{ave:.3f} m/s" if not np.isnan(ave) else "  mAVE: n/a")
        print(f"{'=' * 60}\n")


# ------------------------------------------------------------------
# Internal helpers — coordinate transforms
# ------------------------------------------------------------------

class _SimpleBox:
    """Minimal box wrapper providing .translation, .rotation, .size for transforms."""
    def __init__(self, translation, rotation, size):
        self.translation = translation
        self.rotation = rotation
        self.size = size
        self.center = tuple(translation)


def _make_eval_box(ann):
    """Build a _SimpleBox from a nuScenes sample_annotation dict."""
    return _SimpleBox(
        translation=ann["translation"],
        rotation=ann["rotation"],
        size=ann["size"],
    )


def _global_to_sensor(box, pose_record, cs_record):
    """Transform a box from global frame to sensor frame.

    Chain: global → ego (inverse pose) → sensor (inverse calibration)

    Args:
        box: _SimpleBox with .translation (x,y,z), .rotation (w,x,y,z).
        pose_record: ego_pose dict with "translation" and "rotation".
        cs_record: calibrated_sensor dict with "translation" and "rotation".

    Returns:
        _SimpleBox in sensor frame.
    """
    # Global → Ego
    t_pose = np.array(pose_record["translation"])
    q_pose = Quaternion(pose_record["rotation"])
    t_ego = q_pose.inverse.rotate(np.array(box.translation) - t_pose)
    q_ego = q_pose.inverse * Quaternion(box.rotation)

    # Ego → Sensor
    t_cs = np.array(cs_record["translation"])
    q_cs = Quaternion(cs_record["rotation"])
    t_sensor = q_cs.inverse.rotate(t_ego - t_cs)
    q_sensor = q_cs.inverse * q_ego

    return _SimpleBox(
        translation=t_sensor.tolist(),
        rotation=q_sensor.elements.tolist(),
        size=box.size,
    )
