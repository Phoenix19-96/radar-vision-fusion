"""Camera-based 2D object detection using YOLOv8n.

Wraps ultralytics YOLOv8n as a black-box detector. Outputs 2D bounding boxes
with COCO class IDs and confidence scores.

Debug interface: set camera_debug_plot=True in config, or pass debug=True to
detect(), to render the camera image with YOLO detection boxes overlaid.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ultralytics import YOLO

from config import CONFIG

# COCO class mapping for nuScenes-relevant classes
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color palette for detection boxes
_CLASS_COLORS = {
    2:  "#00FF00",  # car — green
    5:  "#FFA500",  # bus — orange
    7:  "#FF6347",  # truck — tomato
    0:  "#00BFFF",  # person — deep sky blue
    1:  "#FFD700",  # bicycle — gold
    3:  "#FF69B4",  # motorcycle — hot pink
}


class CameraDetector:
    """YOLOv8n-based 2D object detector.

    Loads COCO-pretrained YOLOv8n, runs inference, filters by confidence
    and target classes, and returns 2D bounding boxes in image coordinates.
    """

    def __init__(self, config=None):
        cfg = config if config is not None else CONFIG
        self.conf_threshold = cfg.get("camera_confidence", 0.3)
        self.nms_iou = cfg.get("camera_nms_iou", 0.45)
        self.target_classes = set(cfg.get("camera_classes", [2, 5, 7, 0, 1, 3]))
        self.debug_plot = cfg.get("camera_debug_plot", False)
        self._model = YOLO("yolov8n.pt")
        self._debug_fig = None   # reused across frames
        self._debug_ax = None

    def detect(self, image_bgr, debug=None):
        """Run YOLOv8n detection on a BGR image.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image.
            debug: If True, render debug plot.
                   Falls back to self.debug_plot (from config) when None.

        Returns:
            detections: (N, 6) float64 array, columns:
                [u1, v1, u2, v2, class_id, score]
            Returns (0, 6) empty array if no detections.
        """
        show_plot = debug if debug is not None else self.debug_plot

        # 初版软件采用yoloV8的2D检测器，不做任何自定义训练，直接用COCO预训练权值跑推理，输出2D边界
        # 核心步骤为：加载模型->推理->过滤，只保留
        # nms_iou代表重叠框去重，0.45代表重叠超过45%就认为是同一个框
        results = self._model(image_bgr, verbose=False, conf=self.conf_threshold,
                              iou=self.nms_iou, device="cpu")

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                detections.append([x1, y1, x2, y2, cls_id, score])

        if not detections:
            dets = np.empty((0, 6))
        else:
            dets = np.array(detections)

        # --- Debug: render detection boxes on camera image ---
        if show_plot:
            self._plot_debug(image_bgr, dets,
                             title=f"Camera detections ({len(dets)} objects)")

        return dets

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _init_debug_figure(self):
        """Create the debug figure once; reused across frames."""
        plt.ion()  # enable interactive mode so plt.pause() refreshes in-place
        self._debug_fig, self._debug_ax = plt.subplots(1, 1, figsize=(12, 8))
        self._debug_ax.set_title("Camera (CAM_FRONT)")
        self._debug_ax.axis("off")

    def close_debug_figure(self):
        """Close the persistent debug figure if open."""
        if self._debug_fig is not None:
            plt.close(self._debug_fig)
            self._debug_fig = None
            self._debug_ax = None

    def _plot_debug(self, image_bgr, detections, title="Camera detections"):
        """Render camera image with YOLO detection boxes overlaid.

        On the first call a figure window is opened and kept alive; subsequent
        calls clear the axes and redraw into the same window.

        Detection boxes are color-coded by class and labeled with class name
        + confidence score.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image.
            detections: (N, 6) array [u1, v1, u2, v2, class_id, score].
            title: Figure suptitle.
        """
        # Lazy init — first call opens the window
        if self._debug_fig is None:
            self._init_debug_figure()

        # ---- Clear previous frame ----
        self._debug_ax.cla()

        # ---- Restore decoration ----
        self._debug_ax.set_title("Camera (CAM_FRONT)")
        self._debug_ax.axis("off")

        # ---- Draw image (BGR → RGB) ----
        self._debug_ax.imshow(image_bgr[:, :, ::-1])

        # ---- Draw detection boxes ----
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, cls_id, score = det
                cls_id = int(cls_id)
                cls_name = COCO_CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                color = _CLASS_COLORS.get(cls_id, "#ffffff")

                # Bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor=color, linewidth=2, alpha=0.9,
                )
                self._debug_ax.add_patch(rect)

                # Label with class name + confidence
                label = f"{cls_name} {score:.2f}"
                self._debug_ax.text(
                    x1, max(y1 - 4, 0), label,
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

        # ---- Refresh ----
        self._debug_fig.suptitle(title, fontsize=13)
        self._debug_fig.tight_layout()
        self._debug_fig.canvas.draw_idle()
        plt.pause(0.05)  # process GUI events, keep window responsive
