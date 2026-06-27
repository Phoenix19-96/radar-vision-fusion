"""Radar point cloud detector.

Filters raw nuScenes radar points and treats each surviving point as an
individual detection.  No clustering is applied because the Continental
ARS 408 radar already outputs pre-clustered targets — each point is a
radar-internal cluster.

Debug interface: set radar_debug_plot=True in config, or pass debug=True to
detect(), to render a 2-panel figure: left = radar xy-plane, right = camera image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from config import CONFIG
from peaks_info import PeaksInfo


class RadarDetector:
    """Detect objects from radar point cloud.

    Each filtered radar peak becomes one detection.  Clustering is intentionally
    skipped — the Continental ARS 408 radar firmware already outputs pre-clustered
    targets (the nuScenes "points" are actually clusters).

    Filtering (three validity signals only):
      - invalid_state == 0  (valid point, not an artifact)
      - ambig_state == 3    (Doppler ambiguity successfully resolved)
      - pdh0 <= 2           (false-alarm probability ≤ 50%)
    """

    def __init__(self, config=None):
        cfg = config if config is not None else CONFIG
        self.debug_plot = cfg.get("radar_debug_plot", False)
        self._debug_fig = None   # reused across frames
        self._debug_axes = None  # (ax_radar, ax_cam)

    def detect(self, peaks, debug=None, image=None):
        """Filter radar peaks and convert each to a detection.

        Each surviving point becomes an individual detection — no clustering is
        applied because the ARS 408 radar already outputs pre-clustered targets.

        Args:
            peaks: list[PeaksInfo] — parsed radar peaks from data_loader.
            debug: If True, render debug plot.
                   Falls back to self.debug_plot (from config) when None.
            image: (H, W, 3) uint8 BGR camera image, shown alongside radar in debug plots.

        Returns:
            detections: (K, 8) float64 array, columns:
                [x, y, z, vx, vy, rcs_mean, size_xy, n_points]
            Returns (0, 8) empty array if no points survive filtering.
        """
        show_plot = debug if debug is not None else self.debug_plot

        # Convert list[PeaksInfo] → (18, N) array for vectorised filtering
        radar_points = PeaksInfo.to_array(peaks)

        # --- Step 1: Filtering ---
        # 对输入点云进行预处理，按照以下条件滤除相应的点：
        # 1. 只保留标准有效的点   (invalid_state == 0)
        # 2. 只保留解模糊准确的点  (ambig_state == 3)
        # 3. 滤除高虚警概率的点   (pdh0 <= 2, i.e. ≤50% false-alarm)
        invalid = radar_points[14, :]  # invalid_state
        ambig = radar_points[11, :]    # ambig_state
        pdh0 = radar_points[15, :]     # false-alarm probability

        mask = (
            (invalid == 0) &
            (ambig == 3) &
            (pdh0 <= 2)
        )
        valid = radar_points[:, mask]
        if valid.shape[1] == 0:
            return np.empty((0, 8))

        M = valid.shape[1]  # number of surviving points

        # --- Debug: render filtered points ---
        if show_plot:
            xy = valid[:2, :].T
            vel = valid[[8, 9], :].T
            self._plot_debug(xy, vel=vel, image=image,
                             title=f"Radar detections ({M} points)")

        # --- Step 2: Each filtered point → one detection ---
        detections = np.zeros((M, 8))
        detections[:, 0] = valid[0, :]   # x
        detections[:, 1] = valid[1, :]   # y
        detections[:, 2] = valid[2, :]   # z
        detections[:, 3] = valid[8, :]   # vx (vx_comp)
        detections[:, 4] = valid[9, :]   # vy (vy_comp)
        detections[:, 5] = valid[5, :]   # rcs
        detections[:, 6] = 0.0           # size_xy (N/A for single point),reserved
        detections[:, 7] = 1.0           # n_points (each is a pre-clustered target),reserved

        return detections

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    # palette of distinct colors for clusters
    _PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def _init_debug_figure(self):
        """Create the debug figure once; reused across frames."""
        plt.ion()  # enable interactive mode so plt.pause() refreshes in-place
        self._debug_fig, (ax_radar, ax_cam) = plt.subplots(1, 2, figsize=(16, 7))
        self._debug_axes = (ax_radar, ax_cam)

        # -- Left panel: fixed decoration --
        ax_radar.set_aspect("equal")
        ax_radar.set_xlabel("y (m) — lateral")
        ax_radar.set_ylabel("x (m) — forward")
        ax_radar.set_title("Radar (FOV)")
        ax_radar.grid(True, alpha=0.3)
        ax_radar.set_xlim(50, -50)
        ax_radar.set_ylim(0, 100)

        # -- Right panel: fixed decoration --
        ax_cam.set_title("Camera (CAM_FRONT)")
        ax_cam.axis("off")

    def close_debug_figure(self):
        """Close the persistent debug figure if open."""
        if self._debug_fig is not None:
            plt.close(self._debug_fig)
            self._debug_fig = None
            self._debug_axes = None

    def _plot_debug(self, xy, vel=None, labels=None, image=None, title="Radar points"):
        """Render / update radar points (left) + camera image (right).

        On the first call a figure window is opened and kept alive; subsequent
        calls clear the axes and redraw into the same window.

        Radar coordinate system mapping:
          - x (forward, 0–100 m)  → plot vertical axis
          - y (lateral, -50–50 m) → plot horizontal axis, positive on the left

        Points with |vx_comp| > 1 or |vy_comp| > 1 are drawn in red.

        Args:
            xy: (M, 2) array of (x, y) coordinates in radar frame.
            vel: (M, 2) array of [vx_comp, vy_comp], or None.
            labels: (M,) array of DBSCAN cluster labels (None = before clustering,
                    -1 = noise). When None, all points drawn in blue.
            image: (H, W, 3) uint8 BGR camera image, or None to skip.
            title: Figure suptitle.
        """
        # Lazy init — first call opens the window
        if self._debug_fig is None:
            self._init_debug_figure()

        ax_radar, ax_cam = self._debug_axes

        # ---- Clear previous frame ----
        ax_radar.cla()
        ax_cam.cla()

        # ---- Restore left-panel decoration ----
        ax_radar.set_aspect("equal")
        ax_radar.set_xlabel("y (m) — lateral")
        ax_radar.set_ylabel("x (m) — forward")
        ax_radar.set_title("Radar (FOV)")
        ax_radar.grid(True, alpha=0.3)
        ax_radar.set_xlim(50, -50)
        ax_radar.set_ylim(0, 100)

        # ---- Restore right-panel decoration ----
        ax_cam.set_title("Camera (CAM_FRONT)")
        ax_cam.axis("off")

        # ---- Draw radar points ----
        fast_mask = None
        if vel is not None:
        # 速度绝对值大于 1 的点视为高速点，以红色标注
            fast_mask = (np.abs(vel[:, 0]) > 1) | (np.abs(vel[:, 1]) > 1)

        if labels is None:
            # Before DBSCAN: slow = steelblue, fast = red
            if fast_mask is not None and fast_mask.any():
                slow = ~fast_mask
                if slow.any():
                    ax_radar.scatter(xy[slow, 1], xy[slow, 0], c="steelblue", s=12,
                                     alpha=0.7, edgecolors="none",
                                     label=f"slow ({slow.sum()})")
                ax_radar.scatter(xy[fast_mask, 1], xy[fast_mask, 0], c="red", s=16,
                                 alpha=0.9, edgecolors="darkred", linewidths=0.5,
                                 label=f"fast |v|>1 ({fast_mask.sum()})")
            else:
                ax_radar.scatter(xy[:, 1], xy[:, 0], c="steelblue", s=12, alpha=0.7,
                                 edgecolors="none", label=f"all points ({xy.shape[0]})")
            ax_radar.legend(loc="upper right")
        else:
            unique_labels = np.unique(labels)

            for label in unique_labels:
                mask = labels == label
                pts = xy[mask]
                pts_fast = fast_mask[mask] if fast_mask is not None else np.zeros(mask.sum(), dtype=bool)

                if label == -1:
                    ax_radar.scatter(pts[:, 1], pts[:, 0], c="lightgray", s=10,
                                     alpha=0.5, marker="x",
                                     label=f"noise ({pts.shape[0]})")
                else:
                    color = self._PALETTE[label % len(self._PALETTE)]
                    ax_radar.scatter(pts[:, 1], pts[:, 0], c=[color], s=16, alpha=0.8,
                                     edgecolors="none",
                                     label=f"cluster {label} ({pts.shape[0]})")
                    # Enclosing circle + centroid
                    centroid_xy = pts.mean(axis=0)
                    radius = np.max(np.linalg.norm(pts - centroid_xy, axis=1))
                    circle = patches.Circle(
                        (centroid_xy[1], centroid_xy[0]), radius,
                        fill=False, edgecolor=color, linewidth=1.8,
                        linestyle="--", alpha=0.8,
                    )
                    ax_radar.add_patch(circle)
                    ax_radar.plot(centroid_xy[1], centroid_xy[0], marker="+", color=color,
                                  markersize=10, markeredgewidth=1.5)

                # Overlay red edge on fast-moving points within this label group
                if pts_fast.any():
                    ax_radar.scatter(pts[pts_fast, 1], pts[pts_fast, 0],
                                     facecolors="none", edgecolors="red",
                                     s=28, linewidths=1.5, alpha=0.9)

            ax_radar.legend(loc="upper right", fontsize="small", ncol=2)

        # ---- Draw camera image ----
        if image is not None:
            ax_cam.imshow(image[:, :, ::-1])  # BGR → RGB

        # ---- Refresh ----
        self._debug_fig.suptitle(title, fontsize=13)
        self._debug_fig.tight_layout()
        self._debug_fig.canvas.draw_idle()
        plt.pause(0.05)  # process GUI events, keep window responsive
