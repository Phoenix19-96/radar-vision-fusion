"""Visualization utilities for the radar-camera fusion system.

Renders two views per frame:
  1. Camera image with overlays (radar projections, detections, tracks)
  2. BEV (Bird's Eye View) with radar points, fused objects, track trails
"""

import os
import numpy as np
import cv2
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peaks_info import PeaksInfo


CLASS_COLORS = {
    -1: (128, 128, 128),   # unknown: gray
    0:  (255, 100, 100),   # person: light red
    1:  (100, 255, 100),   # bicycle: light green
    2:  (100, 100, 255),   # car: blue
    3:  (255, 100, 255),   # motorcycle: pink
    5:  (255, 255, 100),   # bus: yellow
    7:  (100, 255, 255),   # truck: cyan
}

CLASS_NAMES = {
    -1: "unknown", 0: "pedestrian", 1: "bicycle",
    2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}


def render_frame(image_bgr, radar_dets=None, camera_dets=None,
                 fused_objects=None, tracks=None,
                 cs_radar=None, cs_cam=None, camera_intrinsic=None):
    """Draw detection and tracking overlays on the camera image."""
    img = image_bgr.copy()

    # Draw camera detections
    if camera_dets is not None and len(camera_dets) > 0:
        for det in camera_dets:
            u1, v1, u2, v2 = map(int, det[:4])
            cls_id = int(det[4])
            score = det[5]
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (u1, v1), (u2, v2), color, 2)
            label = f"{CLASS_NAMES.get(cls_id, '?')}:{score:.2f}"
            cv2.putText(img, label, (u1, v1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1)

    # Project fused objects to pixel and draw
    if fused_objects is not None and len(fused_objects) > 0 and cs_radar is not None:
        from utils.coordinate import radar_to_pixel
        fused_xyz = fused_objects[:, :3].T
        uv, _ = radar_to_pixel(fused_xyz, cs_radar, cs_cam, camera_intrinsic)
        for j in range(uv.shape[1]):
            u, v = int(uv[0, j]), int(uv[1, j])
            cls_id = int(fused_objects[j, 5])
            color = CLASS_COLORS.get(cls_id, (128, 128, 128))
            cv2.circle(img, (u, v), 6, color, -1)
            cv2.circle(img, (u, v), 8, (255, 255, 255), 1)

    # Draw track IDs
    if tracks is not None:
        for t in tracks:
            state = t["state"]
            tid = t["id"]
            cls_id = t["class"]
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            y_pos = 20 + (tid % 20) * 25
            if y_pos < img.shape[0]:
                vx, vy = state[3], state[4]
                speed = np.sqrt(vx ** 2 + vy ** 2)
                status = t["status"][:1].upper()
                label = f"T{tid} {CLASS_NAMES.get(cls_id, '?')} {speed:.1f}m/s [{status}]"
                cv2.putText(img, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 2)

    return img


def render_bev(peaks=None, fused_objects=None, tracks=None,
               radar_dets=None, figsize=(8, 8), xlim=(0, 80), ylim=(-30, 30)):
    """Render a Bird's Eye View of the scene."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X (m) forward")
    ax.set_ylabel("Y (m) left")
    ax.set_title("Bird's Eye View")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    if peaks is not None and len(peaks) > 0:
        points = PeaksInfo.to_array(peaks)
        x, y = points[0, :], points[1, :]
        ax.scatter(x, y, c="blue", s=2, alpha=0.3, label="radar points")

    if radar_dets is not None and len(radar_dets) > 0:
        ax.scatter(radar_dets[:, 0], radar_dets[:, 1], c="red", s=30,
                   marker="o", edgecolors="darkred", linewidths=0.5,
                   label="radar clusters")

    if fused_objects is not None and len(fused_objects) > 0:
        for obj in fused_objects:
            x, y, vx, vy = obj[0], obj[1], obj[3], obj[4]
            cls_id = int(obj[5])
            color_rgb = np.array(CLASS_COLORS.get(cls_id, (128, 128, 128))) / 255.0
            ax.scatter(x, y, c=[color_rgb], s=50, marker="s",
                       edgecolors="black", linewidths=0.5, zorder=5)
            speed = np.sqrt(vx ** 2 + vy ** 2)
            if speed > 0.1:
                ax.arrow(x, y, vx * 0.2, vy * 0.2, head_width=0.5,
                         head_length=0.3, fc="red", ec="red", alpha=0.7)

    if tracks is not None:
        for t in tracks:
            hist = t.get("history", [])
            if len(hist) >= 2:
                hx = [p[0] for p in hist]
                hy = [p[1] for p in hist]
                ax.plot(hx, hy, "-", linewidth=1.5, alpha=0.6)
                state = t["state"]
                ax.text(state[0], state[1] + 0.5, str(t["id"]),
                        fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf
