"""Camera-based 2D object detection using YOLOv8n.

Wraps ultralytics YOLOv8n as a black-box detector. Outputs 2D bounding boxes
with COCO class IDs and confidence scores.
"""

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
        self._model = YOLO("yolov8n.pt")

    def detect(self, image_bgr):
        """Run YOLOv8n detection on a BGR image.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image.

        Returns:
            detections: (N, 6) float64 array, columns:
                [u1, v1, u2, v2, class_id, score]
            Returns (0, 6) empty array if no detections.
        """
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
            return np.empty((0, 6))
        return np.array(detections)
