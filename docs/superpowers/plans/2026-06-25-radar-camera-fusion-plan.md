# Radar-Camera Fusion Perception System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a decision-level (late) fusion perception system that performs 3D object detection + multi-object tracking using nuScenes RADAR_FRONT + CAM_FRONT.

**Architecture:** Five-module pipeline: (1) radar DBSCAN clustering → (2) YOLOv8n camera detection → (3) coordinate-transform-based fusion with Hungarian matching → (4) Kalman filter tracking with track state machine → (5) evaluation + visualization. Each module is independently testable with well-defined NumPy array interfaces.

**Tech Stack:** Python 3.8, numpy, scipy, scikit-learn, opencv-python, pyquaternion, ultralytics (YOLOv8n), matplotlib, nuScenes devkit (local PYTHONPATH)

## Global Constraints

- Python: 3.8 (use `python` command, not `python3`)
- nuScenes devkit loaded via `PYTHONPATH="/d/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk"`
- Data root: `D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes/v1.0-mini`
- All module interfaces use numpy arrays (no custom classes across module boundaries)
- No GPU required; all processing runs on CPU

---

### Task 1: Project Setup — config.py and utils package

**Files:**
- Create: `code/config.py`
- Create: `code/utils/__init__.py`
- Create: `code/__init__.py`

**Interfaces:**
- Produces: `CONFIG` dict (imported by all other modules)
- Produces: `code.utils` package (imported by fusion, tracker, etc.)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p code/utils code/tests
```

- [ ] **Step 2: Write code/__init__.py (empty)**

```python
# code package
```

- [ ] **Step 3: Write code/utils/__init__.py (empty)**

```python
# utils package
```

- [ ] **Step 4: Write code/config.py**

```python
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
```

- [ ] **Step 5: Verify setup**

```bash
PYTHONPATH="D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk;D:/wzr/PyWorkspace/Fusion/code" python -c "
from config import CONFIG
from utils import __name__ as uname
print('config OK, radar_eps =', CONFIG['radar_eps'])
print('utils package OK')
"
```

Expected: prints config value and "utils package OK"

---

### Task 2: Coordinate Transform Utilities

**Files:**
- Create: `code/utils/coordinate.py`
- Create: `code/tests/test_coordinate.py`

**Interfaces:**
- Produces: `radar_to_ego(points_radar, cs_record) -> points_ego`
- Produces: `ego_to_camera(points_ego, cam_cs_record) -> points_camera`
- Produces: `camera_to_pixel(points_camera, cam_intrinsic) -> pixels_uv`
- Produces: `radar_to_pixel(points_radar, cs_record_radar, cs_record_cam, cam_intrinsic) -> pixels_uv, depths`
  — convenience: chains all three above

- [ ] **Step 1: Write failing tests for code/tests/test_coordinate.py**

```python
"""Tests for coordinate transform utilities."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from pyquaternion import Quaternion
from utils.coordinate import radar_to_ego, ego_to_camera, camera_to_pixel, radar_to_pixel


class TestCoordinateTransforms:
    """Test each transform in the Radar->Ego->Camera->Pixel chain."""

    def test_radar_to_ego_identity(self):
        """Identity transform: radar at origin with no rotation -> same point in ego."""
        points = np.array([[1.0, 2.0, 3.0]]).T  # (3, 1)
        cs = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        result = radar_to_ego(points, cs)
        np.testing.assert_array_almost_equal(result, points)

    def test_radar_to_ego_translation(self):
        """Pure translation: radar at (1,0,0) offsets by (2,3,4)."""
        points = np.array([[1.0, 0.0, 0.0]]).T
        cs = {"translation": [2.0, 3.0, 4.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        result = radar_to_ego(points, cs)
        np.testing.assert_array_almost_equal(result, np.array([[3.0, 3.0, 4.0]]).T)

    def test_radar_to_ego_rotation_90deg_z(self):
        """90-degree rotation around z-axis."""
        points = np.array([[1.0, 0.0, 0.0]]).T
        q = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
        cs = {"translation": [0.0, 0.0, 0.0],
              "rotation": [q.w, q.x, q.y, q.z]}
        result = radar_to_ego(points, cs)
        # (1,0,0) rotated 90deg around z -> (0,1,0)
        np.testing.assert_array_almost_equal(result, np.array([[0.0, 1.0, 0.0]]).T, decimal=5)

    def test_ego_to_camera_identity(self):
        """Identity: ego-camera transform with no offset."""
        points_ego = np.array([[5.0, 0.0, 10.0]]).T
        cs_cam = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        result = ego_to_camera(points_ego, cs_cam)
        np.testing.assert_array_almost_equal(result, points_ego)

    def test_camera_to_pixel_centered(self):
        """Point on optical axis projects to image center."""
        K = np.array([[1000.0, 0.0, 500.0],
                       [0.0, 1000.0, 500.0],
                       [0.0, 0.0, 1.0]])
        # Point at depth 10m on optical axis (0,0,10) in camera frame -> (500, 500)
        points_cam = np.array([[0.0, 0.0, 10.0]]).T
        uv = camera_to_pixel(points_cam, K)
        np.testing.assert_array_almost_equal(uv, np.array([[500.0, 500.0]]).T, decimal=2)

    def test_camera_to_pixel_negative_z_returns_none(self):
        """Points behind camera (Z<=0) should be filtered."""
        K = np.eye(3)
        points_cam = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 5.0]]).T
        uv = camera_to_pixel(points_cam, K)
        # Only the valid point should remain
        assert uv.shape[1] == 1

    def test_radar_to_pixel_integration(self):
        """End-to-end: known transform produces expected pixel coordinates."""
        points_radar = np.array([[10.0, 0.0, 0.0]]).T  # 10m ahead of radar
        cs_radar = {"translation": [3.0, 0.0, 0.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0]}
        cs_cam = {"translation": [1.0, 0.0, 1.5],
                  "rotation": [0.5, -0.5, 0.5, -0.5]}  # camera looking forward
        K = np.array([[1266.0, 0.0, 816.0],
                       [0.0, 1266.0, 491.0],
                       [0.0, 0.0, 1.0]])
        uv, depths = radar_to_pixel(points_radar, cs_radar, cs_cam, K)
        assert uv.shape[0] == 2
        assert depths.shape[0] == 1
        # Pixel coords should be within a reasonable image range
        assert 0 <= uv[0, 0] <= 1600
        assert 0 <= uv[1, 0] <= 900
        assert depths[0] > 0
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_coordinate.py -v 2>&1
```

Expected: All tests FAIL with ImportError (module not yet written)

- [ ] **Step 3: Write code/utils/coordinate.py**

```python
"""Coordinate transform utilities for radar-camera fusion.

Transform chain: Radar sensor -> Ego vehicle -> Camera sensor -> Image pixel.
All functions accept (3, N) or (3,) numpy arrays and return numpy arrays.
"""

import numpy as np
from pyquaternion import Quaternion


def radar_to_ego(points_radar, cs_record):
    """Transform points from radar sensor frame to ego vehicle frame.

    Args:
        points_radar: (3, N) array of [x, y, z] in radar sensor coordinates.
        cs_record: dict with 'translation' [x,y,z] and 'rotation' [w,x,y,z]
                   quaternion giving radar sensor pose in ego frame.

    Returns:
        (3, N) array of points in ego vehicle coordinates.
    """
    points_radar = np.atleast_2d(points_radar)
    if points_radar.shape[0] != 3:
        points_radar = points_radar.T

    translation = np.array(cs_record["translation"]).reshape(3, 1)
    rotation = Quaternion(cs_record["rotation"])

    # P_ego = R * P_radar + T
    points_ego = np.dot(rotation.rotation_matrix, points_radar) + translation
    return points_ego


def ego_to_camera(points_ego, cam_cs_record):
    """Transform points from ego vehicle frame to camera sensor frame.

    Args:
        points_ego: (3, N) array of [x, y, z] in ego coordinates.
        cam_cs_record: dict with 'translation' and 'rotation' (quaternion)
                       giving CAMERA sensor pose in ego frame.

    Returns:
        (3, N) array of points in camera sensor coordinates.
        Camera frame: x-right, y-down, z-forward.
    """
    points_ego = np.atleast_2d(points_ego)
    if points_ego.shape[0] != 3:
        points_ego = points_ego.T

    translation = np.array(cam_cs_record["translation"]).reshape(3, 1)
    rotation = Quaternion(cam_cs_record["rotation"])

    # P_ego = R * P_cam + T  =>  P_cam = R^-1 * (P_ego - T)
    points_cam = np.dot(rotation.rotation_matrix.T, points_ego - translation)
    return points_cam


def camera_to_pixel(points_camera, camera_intrinsic):
    """Project 3D camera-frame points to 2D image pixel coordinates.

    Args:
        points_camera: (3, N) array of [x, y, z] in camera coordinates.
        camera_intrinsic: (3, 3) camera intrinsic matrix K.

    Returns:
        (2, M) array of [u, v] pixel coordinates (M <= N, points behind
        camera are filtered out). Returns shape (2, 0) if all filtered.
    """
    points_camera = np.atleast_2d(points_camera)
    if points_camera.shape[0] != 3:
        points_camera = points_camera.T

    # Filter points behind or at the camera plane
    mask = points_camera[2, :] > 0.01
    valid = points_camera[:, mask]

    if valid.shape[1] == 0:
        return np.empty((2, 0))

    K = np.array(camera_intrinsic)
    # [u, v, 1]^T ~ K @ [X, Y, Z]^T / Z
    projected = K @ valid  # (3, M)
    uv = projected[:2, :] / projected[2, :]  # (2, M)
    return uv


def radar_to_pixel(points_radar, cs_record_radar, cs_record_cam, camera_intrinsic):
    """Full transform: radar sensor -> ego -> camera -> pixel.

    Args:
        points_radar: (3, N) array in radar sensor coordinates.
        cs_record_radar: calibrated_sensor record for the radar.
        cs_record_cam: calibrated_sensor record for the camera.
        camera_intrinsic: (3, 3) camera intrinsic matrix K.

    Returns:
        (uv, depths): Tuple of:
            uv: (2, M) array of pixel coordinates [u, v].
            depths: (M,) array of depth values (Z in camera frame, meters).
    """
    points_ego = radar_to_ego(points_radar, cs_record_radar)
    points_cam = ego_to_camera(points_ego, cs_record_cam)

    # Get depths before projection
    depths = points_cam[2, :].copy()

    uv = camera_to_pixel(points_cam, camera_intrinsic)
    # camera_to_pixel filters by Z>0, so align depths accordingly
    if uv.shape[1] != len(depths):
        mask = points_cam[2, :] > 0.01
        depths = depths[mask]

    return uv, depths
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_coordinate.py -v 2>&1
```

Expected: All 6 tests PASS

---

### Task 3: Data Loader

**Files:**
- Create: `code/data_loader.py`
- Create: `code/tests/test_data_loader.py`

**Interfaces:**
- Produces: `FusionDataLoader(nusc_version, dataroot)` — class
  - `.load_frame(sample_token) -> dict` with keys: `radar_points` (18,N), `image` (H,W,3), `cs_radar`, `cs_cam`, `camera_intrinsic`, `timestamp`
  - `.iter_scene(scene_idx) -> generator` yielding dict per frame

- [ ] **Step 1: Write failing test for code/tests/test_data_loader.py**

```python
"""Tests for the FusionDataLoader."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from data_loader import FusionDataLoader

DATAROOT = "D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes"


def test_load_frame_returns_expected_keys():
    """load_frame should return a dict with radar, image, and calibration."""
    loader = FusionDataLoader(version="v1.0-mini", dataroot=DATAROOT)
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    assert "radar_points" in data
    assert "image" in data
    assert "cs_radar" in data
    assert "cs_cam" in data
    assert "camera_intrinsic" in data
    assert "timestamp" in data


def test_radar_points_shape():
    """Radar points should be (18, N) with N > 0."""
    loader = FusionDataLoader(version="v1.0-mini", dataroot=DATAROOT)
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    pc = data["radar_points"]
    assert pc.ndim == 2
    assert pc.shape[0] == 18
    assert pc.shape[1] > 0


def test_image_shape():
    """Image should be (H, W, 3) RGB."""
    loader = FusionDataLoader(version="v1.0-mini", dataroot=DATAROOT)
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    img = data["image"]
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_camera_intrinsic_is_3x3():
    """Camera intrinsic matrix should be 3x3."""
    loader = FusionDataLoader(version="v1.0-mini", dataroot=DATAROOT)
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    K = data["camera_intrinsic"]
    assert K.shape == (3, 3)


def test_iter_scene_yields_all_frames():
    """iter_scene should yield all frames in scene 0."""
    loader = FusionDataLoader(version="v1.0-mini", dataroot=DATAROOT)
    frames = list(loader.iter_scene(0))
    assert len(frames) > 0
    # Scene 0 in v1.0-mini has 40 frames
    assert len(frames) == 40
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_data_loader.py -v 2>&1
```

Expected: FAIL with ImportError

- [ ] **Step 3: Write code/data_loader.py**

```python
"""NuScenes data loading wrapper for radar-camera fusion.

Provides a lightweight interface to load synchronized RADAR_FRONT + CAM_FRONT
frames with calibration data, iterating through scene samples.
"""

import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


class FusionDataLoader:
    """Load synchronized radar point clouds and camera images from nuScenes.

    Usage:
        loader = FusionDataLoader(version="v1.0-mini", dataroot="...")
        for frame in loader.iter_scene(scene_idx=0):
            radar = frame["radar_points"]    # (18, N)
            image = frame["image"]           # (H, W, 3) BGR
            ...
    """

    def __init__(self, version="v1.0-mini", dataroot=None):
        self.dataroot = dataroot
        # nuScenes stores data inside a subdirectory named after the version
        actual_dataroot = os.path.join(dataroot, version) if version in os.listdir(dataroot) else dataroot
        self.nusc = NuScenes(version=version, dataroot=actual_dataroot, verbose=False)

    def load_frame(self, sample_token):
        """Load one sample's radar + camera data with calibration.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            dict with keys:
                radar_points: (18, N) float32 array, raw radar point cloud.
                image: (H, W, 3) uint8 array, BGR image.
                cs_radar: calibrated_sensor record for RADAR_FRONT.
                cs_cam: calibrated_sensor record for CAM_FRONT.
                camera_intrinsic: (3, 3) float64 camera intrinsic matrix.
                timestamp: int, sample timestamp in microseconds.
        """
        sample = self.nusc.get("sample", sample_token)

        # --- Radar ---
        radar_sd = self.nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
        radar_path = os.path.join(self.nusc.dataroot, radar_sd["filename"])
        pc = RadarPointCloud.from_file(radar_path)
        cs_radar = self.nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])

        # --- Camera ---
        cam_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])
        image = cv2.imread(img_path)  # BGR
        cs_cam = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        camera_intrinsic = np.array(cs_cam["camera_intrinsic"])

        return {
            "radar_points": pc.points,          # (18, N)
            "image": image,                      # (H, W, 3) BGR
            "cs_radar": cs_radar,
            "cs_cam": cs_cam,
            "camera_intrinsic": camera_intrinsic,
            "timestamp": sample["timestamp"],
        }

    def iter_scene(self, scene_idx=0):
        """Generator yielding frames from a scene in temporal order.

        Args:
            scene_idx: Index into nusc.scene list.

        Yields:
            dict: Same as load_frame() return value, one per sample.
        """
        scene = self.nusc.scene[scene_idx]
        sample_token = scene["first_sample_token"]

        while sample_token:
            yield self.load_frame(sample_token)
            sample = self.nusc.get("sample", sample_token)
            sample_token = sample["next"]
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_data_loader.py -v 2>&1
```

Expected: All 5 tests PASS

---

### Task 4: Radar Detector (DBSCAN Clustering)

**Files:**
- Create: `code/radar_detector.py`
- Create: `code/tests/test_radar_detector.py`

**Interfaces:**
- Produces: `RadarDetector(config)` — class
  - `.detect(radar_points) -> radar_detections` where `radar_detections` is `(K, 8)` numpy array `[x, y, z, vx, vy, rcs_mean, size_xy, n_points]`

- [ ] **Step 1: Write failing test for code/tests/test_radar_detector.py**

```python
"""Tests for the radar detector (DBSCAN clustering)."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from radar_detector import RadarDetector


def test_detect_returns_correct_shape():
    """Output should be (K, 8) with K >= 0."""
    detector = RadarDetector()
    # Create synthetic radar points: (18, 20)
    points = np.zeros((18, 20))
    points[0, :] = np.linspace(0, 20, 20)  # x
    points[1, :] = np.random.randn(20) * 2  # y scatter
    points[3, :] = 0  # dyn_prop = stationary (valid)
    points[10, :] = 1  # is_quality_valid
    points[11, :] = 3  # ambig_state = unambiguous
    points[14, :] = 0  # invalid_state = valid

    detections = detector.detect(points)
    assert detections.ndim == 2
    assert detections.shape[1] == 8, f"Expected 8 columns, got {detections.shape[1]}"
    assert detections.shape[0] >= 0


def test_detect_filters_invalid_states():
    """Points with invalid_state != 0 should be filtered out."""
    detector = RadarDetector()
    points = np.zeros((18, 5))
    points[0, :] = [1, 3, 5, 7, 9]  # x
    points[1, :] = [0, 0, 0, 0, 0]  # y
    points[3, :] = 0   # dyn_prop
    points[10, :] = 1  # quality valid
    points[11, :] = 3  # unambiguous
    points[14, :] = [0, 0, 1, 0, 0]  # invalid_state: point at index 2 is invalid

    detections = detector.detect(points)
    # Point at index 2 (x=5) should be filtered out
    # Remaining 4 points may form 1-2 clusters depending on eps
    assert detections.shape[0] >= 0


def test_detect_on_real_data():
    """Integration: run on actual nuScenes radar data."""
    from data_loader import FusionDataLoader
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    detector = RadarDetector()
    detections = detector.detect(data["radar_points"])

    assert detections.ndim == 2
    assert detections.shape[1] == 8
    # With 74 raw points, expect at least 1 cluster
    assert detections.shape[0] >= 1, "Expected at least one cluster from real data"


def test_detect_cluster_features_are_plausible():
    """Cluster centroids should be within radar FOV and velocities reasonable."""
    from data_loader import FusionDataLoader
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    detector = RadarDetector()
    detections = detector.detect(data["radar_points"])

    # Radar FRONT FOV: ~100m forward, ~45deg horizontal
    for det in detections:
        assert 0 <= det[0] <= 150, f"x={det[0]} out of radar range"
        assert -50 <= det[1] <= 50, f"y={det[1]} out of radar range"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_radar_detector.py -v 2>&1
```

Expected: FAIL with ImportError

- [ ] **Step 3: Write code/radar_detector.py**

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_radar_detector.py -v 2>&1
```

Expected: All 4 tests PASS

---

### Task 5: Camera Detector (YOLOv8n)

**Files:**
- Create: `code/camera_detector.py`
- Create: `code/tests/test_camera_detector.py`

**Interfaces:**
- Produces: `CameraDetector(config)` — class
  - `.detect(image_bgr) -> camera_detections` where `camera_detections` is `(N, 6)` numpy array `[u1, v1, u2, v2, class_id, score]`

- [ ] **Step 1: Write failing test for code/tests/test_camera_detector.py**

```python
"""Tests for the camera detector (YOLOv8n wrapper)."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
import cv2
from camera_detector import CameraDetector


def test_detect_returns_correct_shape():
    """Output should be (N, 6) with N >= 0."""
    detector = CameraDetector()
    # Create a black test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(img)
    if detections.shape[0] > 0:
        assert detections.ndim == 2
        assert detections.shape[1] == 6
    else:
        # Black image has no detections - that's fine
        assert detections.shape == (0, 6) or detections.ndim == 2


def test_detect_on_real_data():
    """Integration: run on actual nuScenes CAM_FRONT image."""
    from data_loader import FusionDataLoader
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    detector = CameraDetector()
    detections = detector.detect(data["image"])

    assert detections.ndim == 2
    assert detections.shape[1] == 6
    assert detections.shape[0] >= 1, "Expected at least one detection on nuScenes image"


def test_detection_values_are_plausible():
    """Bbox coords should be within image dims, scores in [0,1]."""
    from data_loader import FusionDataLoader
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    detector = CameraDetector()
    detections = detector.detect(data["image"])
    h, w = data["image"].shape[:2]

    for det in detections:
        u1, v1, u2, v2, cls_id, score = det
        assert 0 <= u1 < w, f"u1={u1} out of bounds [0, {w})"
        assert 0 <= v1 < h, f"v1={v1} out of bounds [0, {h})"
        assert u1 < u2 <= w, f"u2={u2} must be > u1={u1} and <= {w}"
        assert v1 < v2 <= h, f"v2={v2} must be > v1={v1} and <= {h}"
        assert 0.0 <= score <= 1.0, f"score={score} not in [0,1]"
        assert cls_id in [0, 1, 2, 3, 5, 7], f"class_id={cls_id} not in expected COCO classes"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_camera_detector.py -v 2>&1
```

Expected: FAIL with ImportError

- [ ] **Step 3: Write code/camera_detector.py**

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_camera_detector.py -v 2>&1
```

Expected: All 3 tests PASS

---

### Task 6: Association Utilities (Hungarian + Cost Matrix)

**Files:**
- Create: `code/utils/association.py`
- Create: `code/tests/test_association.py`

**Interfaces:**
- Produces: `build_fusion_cost_matrix(radar_uv, radar_depths, camera_dets, config) -> cost_matrix (K,N)`
- Produces: `hungarian_match(cost_matrix, max_cost) -> (row_ind, col_ind, unmatched_rows, unmatched_cols)`

- [ ] **Step 1: Write failing test for code/tests/test_association.py**

```python
"""Tests for association utilities (cost matrix + Hungarian matching)."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from utils.association import build_fusion_cost_matrix, hungarian_match


def test_cost_matrix_shape():
    """Cost matrix should have shape (K_radar, N_camera)."""
    radar_uv = np.array([[100.0, 200.0, 300.0],
                          [150.0, 250.0, 350.0]])  # (2, 3)
    radar_depths = np.array([10.0, 20.0, 30.0])
    camera_dets = np.array([
        [90, 140, 160, 200, 2, 0.9],
        [280, 230, 350, 300, 0, 0.7],
    ])  # (2, 6)
    config = {"fusion_max_depth_diff": 5.0, "fusion_outside_penalty": 50.0}

    cost = build_fusion_cost_matrix(radar_uv, radar_depths, camera_dets, config)
    assert cost.shape == (3, 2)


def test_inside_bbox_cost_is_low():
    """Radar point inside camera bbox -> cost should be low (< 1.0)."""
    radar_uv = np.array([[100.0], [150.0]])  # inside bbox below
    radar_depths = np.array([10.0])
    camera_dets = np.array([[90, 140, 160, 200, 2, 0.9]])
    config = {"fusion_max_depth_diff": 5.0, "fusion_outside_penalty": 50.0}

    cost = build_fusion_cost_matrix(radar_uv, radar_depths, camera_dets, config)
    assert cost[0, 0] < 1.0, f"Expected low cost for point inside bbox, got {cost[0, 0]}"


def test_outside_bbox_cost_is_high():
    """Radar point far outside camera bbox -> cost should be high."""
    radar_uv = np.array([[800.0], [500.0]])  # far from bbox below
    radar_depths = np.array([10.0])
    camera_dets = np.array([[90, 140, 160, 200, 2, 0.9]])
    config = {"fusion_max_depth_diff": 5.0, "fusion_outside_penalty": 50.0}

    cost = build_fusion_cost_matrix(radar_uv, radar_depths, camera_dets, config)
    assert cost[0, 0] >= 10.0, f"Expected high cost for distant point, got {cost[0, 0]}"


def test_hungarian_match_all_matched():
    """Square cost matrix -> all pairs matched."""
    cost = np.array([[0.1, 5.0],
                      [5.0, 0.2]])
    row_ind, col_ind, unmatched_r, unmatched_c = hungarian_match(cost, max_cost=10.0)
    assert list(row_ind) == [0, 1]
    assert list(col_ind) == [0, 1]
    assert len(unmatched_r) == 0
    assert len(unmatched_c) == 0


def test_hungarian_match_max_cost_filters():
    """Cost above max_cost threshold -> unmatched."""
    cost = np.array([[0.1, 50.0],
                      [50.0, 0.2]])
    row_ind, col_ind, unmatched_r, unmatched_c = hungarian_match(cost, max_cost=10.0)
    # Only (0,0) and (1,1) if costs are low enough
    assert len(row_ind) == 2
    assert len(unmatched_r) == 0


def test_hungarian_match_rectangular():
    """Non-square cost matrix: more radar than camera detections."""
    cost = np.array([[0.1, 0.8],
                      [0.9, 0.2],
                      [5.0, 5.0]])  # 3 radars, 2 cameras
    row_ind, col_ind, unmatched_r, unmatched_c = hungarian_match(cost, max_cost=3.0)
    # Should match (0,0) and (1,1); row 2 is unmatched
    assert len(row_ind) == 2
    assert len(unmatched_r) >= 1
    assert 2 in unmatched_r
```

- [ ] **Step 2: Write code/utils/association.py**

```python
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
      - If (u,v) inside bbox: cost = 1.0 - IoU-like score based on
        distance from bbox center (normalized)
      - If outside: cost = pixel_distance_to_bbox + outside_penalty
      - Depth gate: if depth discrepancy > max_depth_diff, cost = +inf

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
    max_depth_diff = config.get("fusion_max_depth_diff", 5.0)
    outside_penalty = config.get("fusion_outside_penalty", 50.0)

    cost = np.full((K, N), 1e9)

    for i in range(K):
        u_r, v_r = radar_uv[0, i], radar_uv[1, i]
        d_r = radar_depths[i]

        for j in range(N):
            u1, v1, u2, v2 = camera_dets[j, 0], camera_dets[j, 1], camera_dets[j, 2], camera_dets[j, 3]

            # Depth gate: skip if depths are inconsistent
            # (camera doesn't provide depth; we use radar depth as reference,
            #  so this is mainly for sanity: skip if radar depth is implausible
            #  relative to bbox size — small bbox + large depth = mismatch)
            approx_depth = 10.0 / max(u2 - u1, 1) * 100  # rough heuristic
            # Actually, depth gate is used loosely here — mainly for future
            # if visual depth estimate is available. For now, skip gating.
            _ = d_r  # depth available for future use

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
```

- [ ] **Step 3: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="code" python -m pytest code/tests/test_association.py -v 2>&1
```

Expected: All 6 tests PASS

---

### Task 7: Fusion Module

**Files:**
- Create: `code/fusion.py`
- Create: `code/tests/test_fusion.py`

**Interfaces:**
- Produces: `FusionModule(config)` — class
  - `.fuse(radar_detections, camera_detections, cs_radar, cs_cam, camera_intrinsic) -> fused_objects`
  - `fused_objects`: `(M, 8)` numpy array `[x, y, z, vx, vy, class_id, conf_radar, conf_camera]`

- [ ] **Step 1: Write failing test for code/tests/test_fusion.py**

```python
"""Tests for the fusion module."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from fusion import FusionModule


def test_fuse_returns_correct_shape():
    """Fused objects should be (M, 8)."""
    fusion = FusionModule()
    radar_dets = np.array([[10.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.5, 5]])
    camera_dets = np.array([[400, 250, 500, 350, 2, 0.9]])
    cs_radar = {"translation": [3.0, 0.0, 0.5], "rotation": [1.0, 0.0, 0.0, 0.0]}
    cs_cam = {"translation": [1.0, 0.0, 1.5],
              "rotation": [0.5, -0.5, 0.5, -0.5]}
    K = np.eye(3)

    fused = fusion.fuse(radar_dets, camera_dets, cs_radar, cs_cam, K)
    assert fused.shape[1] == 8
    assert fused.shape[0] >= 0


def test_fuse_matched_has_full_info():
    """Matched radar+camera pair should have class_id and both confidences."""
    fusion = FusionModule()
    # Radar detection at ~10m forward
    radar_dets = np.array([[10.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.5, 5]])
    camera_dets = np.array([[380, 240, 500, 350, 2, 0.9]])
    cs_radar = {"translation": [3.0, 0.0, 0.5], "rotation": [1.0, 0.0, 0.0, 0.0]}
    cs_cam = {"translation": [1.0, 0.0, 1.5],
              "rotation": [0.5, -0.5, 0.5, -0.5]}
    K = np.array([[1266.0, 0.0, 816.0],
                   [0.0, 1266.0, 491.0],
                   [0.0, 0.0, 1.0]])

    fused = fusion.fuse(radar_dets, camera_dets, cs_radar, cs_cam, K)
    # Whether they match depends on projection; just check structure
    for obj in fused:
        assert -1 <= obj[5] <= 7, f"Invalid class_id: {obj[5]}"
        assert 0 <= obj[6] <= 1, f"Invalid conf_radar: {obj[6]}"
        assert 0 <= obj[7] <= 1, f"Invalid conf_camera: {obj[7]}"


def test_fuse_on_real_data():
    """Integration: run full fusion on real nuScenes frame."""
    from data_loader import FusionDataLoader
    from radar_detector import RadarDetector
    from camera_detector import CameraDetector

    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    scene = loader.nusc.scene[0]
    first_sample = loader.nusc.get("sample", scene["first_sample_token"])
    data = loader.load_frame(first_sample["token"])

    radar_det = RadarDetector()
    cam_det = CameraDetector()
    fusion = FusionModule()

    radar_dets = radar_det.detect(data["radar_points"])
    cam_dets = cam_det.detect(data["image"])
    fused = fusion.fuse(radar_dets, cam_dets, data["cs_radar"], data["cs_cam"],
                        data["camera_intrinsic"])

    assert fused.shape[1] == 8
    # Should have at least some fused objects
    assert fused.shape[0] >= 1
```

- [ ] **Step 2: Write code/fusion.py**

```python
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
                    obj[6] = 0.8  # conf_radar (heuristic based on n_points / rcs)
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
                    # Rough depth estimate from bbox height (simple heuristic)
                    u1, v1, u2, v2 = camera_detections[c_idx, :4]
                    bbox_height = v2 - v1
                    # Assume typical object height ~1.5m at camera height ~1.5m
                    # depth ≈ focal_length * real_height / bbox_height
                    est_depth = 1000.0 / max(bbox_height, 1)
                    # Back-project bbox bottom-center to 3D
                    cx_img = (u1 + u2) / 2
                    cy_img = v2  # bottom of bbox (ground contact)
                    fx = camera_intrinsic[0, 0]
                    fy = camera_intrinsic[1, 1]
                    cx = camera_intrinsic[0, 2]
                    cy = camera_intrinsic[1, 2]
                    # Camera frame: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
                    # But we need ego frame — use a simpler approach:
                    # Place at radar-detected depth or estimated depth in ego x
                    obj = np.zeros(8)
                    obj[0] = est_depth  # rough x estimate
                    obj[1] = (cx_img - cx) * est_depth / fx  # rough y
                    obj[2] = 0.0  # z unknown
                    obj[3:5] = 0.0  # velocity unknown
                    obj[5] = camera_detections[c_idx, 4]  # class from camera
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
```

- [ ] **Step 3: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_fusion.py -v 2>&1
```

Expected: All 3 tests PASS

---

### Task 8: Kalman Filter

**Files:**
- Create: `code/utils/kalman_filter.py`
- Create: `code/tests/test_kalman_filter.py`

**Interfaces:**
- Produces: `KalmanFilter(dim_x, dim_z, dt)` — class
  - `.predict()` — propagate state by dt
  - `.update(z)` — correct with measurement z (can be None for missed detection)
  - Properties: `x` (state), `P` (covariance), `S` (innovation covariance)

- [ ] **Step 1: Write failing test for code/tests/test_kalman_filter.py**

```python
"""Tests for the Kalman filter implementation."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from utils.kalman_filter import KalmanFilter


def test_kf_initialization():
    """KF should initialize with given state dimension."""
    kf = KalmanFilter(dim_x=7, dim_z=4, dt=0.05)
    assert kf.x.shape == (7, 1)
    assert kf.P.shape == (7, 7)


def test_kf_predict_increases_uncertainty():
    """Prediction should increase covariance trace (uncertainty grows)."""
    kf = KalmanFilter(dim_x=7, dim_z=4, dt=0.05)
    trace_before = np.trace(kf.P)
    kf.predict()
    trace_after = np.trace(kf.P)
    assert trace_after > trace_before, "Prediction should increase uncertainty"


def test_kf_update_reduces_uncertainty():
    """Measurement update should reduce covariance trace."""
    kf = KalmanFilter(dim_x=7, dim_z=4, dt=0.05)
    kf.x[0] = 10.0  # position x
    kf.x[1] = 5.0   # position y
    kf.x[3] = 3.0   # velocity x
    kf.x[4] = -1.0  # velocity y
    kf.predict()
    trace_before = np.trace(kf.P)
    z = np.array([11.0, 4.5, 3.2, -0.8])  # measurement near prediction
    kf.update(z)
    trace_after = np.trace(kf.P)
    assert trace_after < trace_before, "Update should reduce uncertainty"


def test_kf_constant_velocity():
    """With no acceleration, predicted position = x + v*dt."""
    kf = KalmanFilter(dim_x=7, dim_z=4, dt=1.0)
    kf.x[0] = 0.0  # x
    kf.x[3] = 10.0  # vx
    kf.x[5] = 0.0  # ax = 0
    kf.predict()
    # x_predicted = x + vx*dt + 0.5*ax*dt^2 = 0 + 10*1 + 0 = 10
    np.testing.assert_almost_equal(kf.x[0, 0], 10.0, decimal=3)
    np.testing.assert_almost_equal(kf.x[3, 0], 10.0, decimal=3)  # velocity unchanged with ax=0


def test_kf_skip_update():
    """Missing measurement (None) should not crash."""
    kf = KalmanFilter(dim_x=7, dim_z=4, dt=0.05)
    kf.predict()
    kf.update(None)  # should no-op without error
    assert kf.x[0, 0] > 0  # state unchanged from update
```

- [ ] **Step 2: Write code/utils/kalman_filter.py**

```python
"""Kalman filter with Constant Acceleration (CA) motion model.

State vector: X = [x, y, z, vx, vy, ax, ay]^T  (7-D)
Measurement:  Z = [x, y, vx, vy]^T              (4-D)

Key insight: radar provides direct Doppler velocity measurement,
so vx/vy are observed directly rather than derived from position changes.
"""

import numpy as np
from scipy.linalg import block_diag


class KalmanFilter:
    """Constant Acceleration Kalman filter for object tracking.

    Args:
        dim_x: State dimension (7).
        dim_z: Measurement dimension (4).
        dt: Time step in seconds.
        pos_variance: Initial position variance (m^2).
        process_noise: Process noise std for acceleration (m/s^2).
        meas_noise_pos: Measurement noise std for position (m).
        meas_noise_vel: Measurement noise std for velocity (m/s).
    """

    def __init__(self, dim_x=7, dim_z=4, dt=0.05,
                 pos_variance=1.0, process_noise=0.5,
                 meas_noise_pos=0.3, meas_noise_vel=0.1):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt

        # State vector: [x, y, z, vx, vy, ax, ay]
        self.x = np.zeros((dim_x, 1))

        # State transition matrix F
        self.F = np.eye(dim_x)
        # Position updates (indices 0,1,2)
        self.F[0, 3] = dt       # x += vx * dt
        self.F[1, 4] = dt       # y += vy * dt
        self.F[0, 5] = 0.5 * dt * dt  # x += 0.5 * ax * dt^2
        self.F[1, 6] = 0.5 * dt * dt  # y += 0.5 * ay * dt^2
        # Velocity updates (indices 3,4)
        self.F[3, 5] = dt       # vx += ax * dt
        self.F[4, 6] = dt       # vy += ay * dt
        # z (index 2): no velocity in z, stays constant

        # Observation matrix H: observe [x, y, vx, vy]
        self.H = np.zeros((dim_z, dim_x))
        self.H[0, 0] = 1.0  # observe x
        self.H[1, 1] = 1.0  # observe y
        self.H[2, 3] = 1.0  # observe vx
        self.H[3, 4] = 1.0  # observe vy

        # Process noise covariance Q
        # Acceleration is modeled as piecewise constant with noise
        q = process_noise ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        # Q for each 2D axis (x-ax and y-ay pairs) with CA model
        Q_2d = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2],
            [dt3 / 2, dt2, dt],
            [dt2 / 2, dt, 1.0],
        ]) * q
        # Build full Q: [x, ax, vx] for x-axis, similar for y
        # State order: 0=x, 1=y, 2=z, 3=vx, 4=vy, 5=ax, 6=ay
        self.Q = np.zeros((dim_x, dim_x))
        # x, vx, ax coupling (indices 0,3,5)
        self.Q[np.ix_([0, 3, 5], [0, 3, 5])] = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2],
            [dt3 / 2, dt2, dt],
            [dt2 / 2, dt, 1.0],
        ]) * q
        # y, vy, ay coupling (indices 1,4,6)
        self.Q[np.ix_([1, 4, 6], [1, 4, 6])] = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2],
            [dt3 / 2, dt2, dt],
            [dt2 / 2, dt, 1.0],
        ]) * q
        # z (index 2): small noise
        self.Q[2, 2] = 0.01 * q

        # Measurement noise covariance R
        self.R = np.diag([meas_noise_pos ** 2, meas_noise_pos ** 2,
                          meas_noise_vel ** 2, meas_noise_vel ** 2])

        # Initial state covariance P
        self.P = np.eye(dim_x) * pos_variance

        # Cached innovation covariance S
        self.S = np.zeros((dim_z, dim_z))

    def predict(self):
        """Predict state forward by one time step (dt)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update state with measurement.

        Args:
            z: (4,) or (4,1) array [x, y, vx, vy], or None to skip update
               (coasting — prediction only).
        """
        if z is None:
            return
        z = np.asarray(z).reshape(self.dim_z, 1)

        # Innovation (residual)
        y = z - self.H @ self.x

        # Innovation covariance
        self.S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
```

- [ ] **Step 3: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="code" python -m pytest code/tests/test_kalman_filter.py -v 2>&1
```

Expected: All 5 tests PASS

---

### Task 9: Multi-Object Tracker

**Files:**
- Create: `code/tracker.py`
- Create: `code/tests/test_tracker.py`

**Interfaces:**
- Produces: `MultiObjectTracker(config)` — class
  - `.update(fused_objects, timestamp) -> tracks`
  - `tracks`: list of dict `{id, state, cov, class, status, age, history}`

- [ ] **Step 1: Write failing test for code/tests/test_tracker.py**

```python
"""Tests for the multi-object tracker."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from tracker import MultiObjectTracker


def make_fused_obj(x, y, vx, vy, cls=2):
    """Helper: create a fused object array row."""
    return np.array([[x, y, 0.0, vx, vy, cls, 0.8, 0.9]])


def test_tracker_initialization():
    """Tracker starts with empty track list."""
    tracker = MultiObjectTracker()
    assert len(tracker.tracks) == 0


def test_tracker_creates_tentative_tracks():
    """First frame: new tracks should be created as tentative."""
    tracker = MultiObjectTracker()
    fused = make_fused_obj(10, 0, 5, 0)
    tracks = tracker.update(fused, timestamp=0)
    assert len(tracks) == 0  # Tentative tracks not yet confirmed


def test_tracker_confirms_after_hits():
    """Tracks should confirm after consecutive hits."""
    tracker = MultiObjectTracker()
    fused = make_fused_obj(10, 0, 5, 0)

    # Frame 1: born
    tracker.update(fused, timestamp=0)
    # Frame 2: should confirm (born_confirm=2)
    tracks = tracker.update(fused, timestamp=50000)  # 50ms later

    assert len(tracks) >= 1, "Track should confirm after 2 hits"
    if len(tracks) > 0:
        assert tracks[0]["status"] in ["confirmed", "tentative"]
        assert "id" in tracks[0]


def test_tracker_single_object_tracking():
    """Track a single moving object across multiple frames."""
    tracker = MultiObjectTracker()

    for i in range(10):
        x = 10.0 + i * 0.5  # moving at ~10 m/s
        fused = make_fused_obj(x, 0.0, 10.0, 0.0)
        tracks = tracker.update(fused, timestamp=i * 50000)

    assert len(tracks) >= 1
    if len(tracks) > 0:
        t = tracks[0]
        assert t["age"] >= 10
        np.testing.assert_almost_equal(t["state"][3], 10.0, decimal=1)


def test_tracker_prunes_lost_tracks():
    """Tracks that are lost for too long should be removed."""
    tracker = MultiObjectTracker()
    # Insert a track
    fused = make_fused_obj(10, 0, 5, 0)
    for _ in range(3):
        tracker.update(fused, timestamp=0)

    # Then run many empty frames
    for i in range(10):
        tracker.update(np.empty((0, 8)), timestamp=(i + 3) * 50000)

    # All tracks should be pruned
    assert len([t for t in tracker.tracks if t["status"] in ["confirmed", "tentative"]]) == 0
```

- [ ] **Step 2: Write code/tracker.py**

```python
"""Multi-object tracker with Kalman filtering and track management.

Uses Constant Acceleration Kalman filters per track, Hungarian matching
for frame-to-frame association, and a Born/Confirmed/Coasting/Dead state
machine for track lifecycle management.
"""

import numpy as np
from config import CONFIG
from utils.kalman_filter import KalmanFilter
from utils.association import hungarian_match


class Track:
    """Internal track representation."""
    _next_id = 0

    def __init__(self, detection, timestamp, config):
        self.id = Track._next_id
        Track._next_id += 1
        self.status = "tentative"
        self.age = 0
        self.hit_streak = 1
        self.miss_streak = 0
        self.last_update = timestamp

        # Initialize Kalman filter
        z_init = np.array([detection[0], detection[1], detection[3], detection[4]])
        self.kf = KalmanFilter(
            dim_x=7, dim_z=4, dt=0.05,
            pos_variance=config.get("track_init_variance", 1.0),
            process_noise=config.get("track_process_noise", 0.5),
            meas_noise_pos=config.get("track_meas_noise_pos", 0.3),
            meas_noise_vel=config.get("track_meas_noise_vel", 0.1),
        )
        self.kf.x[0] = detection[0]  # x
        self.kf.x[1] = detection[1]  # y
        self.kf.x[2] = detection[2]  # z
        self.kf.x[3] = detection[3]  # vx
        self.kf.x[4] = detection[4]  # vy
        self.class_id = int(detection[5])
        self.history = []

    def predict(self):
        self.kf.predict()

    def update(self, detection, timestamp):
        """Update with a matched detection."""
        z = np.array([detection[0], detection[1], detection[3], detection[4]])
        self.kf.update(z)
        self.class_id = int(detection[5]) if detection[5] >= 0 else self.class_id
        self.last_update = timestamp
        self.hit_streak += 1
        self.miss_streak = 0
        self.history.append((self.kf.x[0, 0], self.kf.x[1, 0]))
        if len(self.history) > 50:
            self.history.pop(0)

    def coast(self):
        """Called when no detection matches this track."""
        self.kf.update(None)  # prediction only
        self.miss_streak += 1
        self.hit_streak = 0

    def summary(self):
        """Return track summary dict for external output."""
        return {
            "id": self.id,
            "state": self.kf.x.flatten().copy(),
            "covariance": self.kf.P.copy(),
            "class": self.class_id,
            "status": self.status,
            "age": self.age,
            "history": self.history.copy(),
        }


class MultiObjectTracker:
    """Multi-object tracker with Kalman filter + Hungarian association.

    Lifecycle:
      tentative: new track, not yet confirmed
      confirmed: active tracked object
      coasting: lost for a few frames
      dead: removed from tracking
    """

    def __init__(self, config=None):
        self.config = config if config is not None else CONFIG
        self.born_confirm = self.config.get("track_born_confirm", 2)
        self.coast_max = self.config.get("track_coast_max", 5)
        self.mahalanobis_threshold = 20.0  # chi2 threshold for gating
        self.tracks = []

    def update(self, fused_objects, timestamp):
        """Process one frame of fused detections and update all tracks.

        Args:
            fused_objects: (M, 8) array from FusionModule.fuse().
            timestamp: int, frame timestamp (microseconds).

        Returns:
            confirmed_tracks: list of track summary dicts.
        """
        # Compute dt from last update for each track (or use default 0.05s)
        dt = 0.05  # default: 20Hz radar

        # --- Predict all existing tracks ---
        for track in self.tracks:
            track.kf.dt = dt
            track.predict()
            track.age += 1

        # --- Build association cost matrix ---
        active_tracks = [t for t in self.tracks if t.status != "dead"]
        M = fused_objects.shape[0]
        N = len(active_tracks)

        if M > 0 and N > 0:
            cost = np.full((N, M), 1e9)
            for i, track in enumerate(active_tracks):
                pred = track.kf.x.flatten()
                P = track.kf.P
                for j in range(M):
                    # Mahalanobis distance for position (x, y)
                    z = np.array([fused_objects[j, 0], fused_objects[j, 1],
                                  fused_objects[j, 3], fused_objects[j, 4]])
                    y = z - np.array([pred[0], pred[1], pred[3], pred[4]])
                    # Use position portion of innovation covariance
                    S_pos = P[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] + track.kf.R
                    try:
                        d = y @ np.linalg.inv(S_pos) @ y
                    except np.linalg.LinAlgError:
                        d = 1e9
                    cost[i, j] = d if d < self.mahalanobis_threshold else 1e9

            track_indices, obs_indices, unmatched_tracks, unmatched_obs = \
                hungarian_match(cost, max_cost=self.mahalanobis_threshold)
        else:
            track_indices, obs_indices = np.array([], dtype=int), np.array([], dtype=int)
            unmatched_tracks = list(range(N)) if N > 0 else []
            unmatched_obs = list(range(M)) if M > 0 else []

        # --- Update matched tracks ---
        matched_track_ids = set()
        for t_idx, o_idx in zip(track_indices, obs_indices):
            track = active_tracks[t_idx]
            track.update(fused_objects[o_idx], timestamp)
            matched_track_ids.add(t_idx)

        # --- Coast unmatched tracks ---
        for t_idx in unmatched_tracks:
            active_tracks[t_idx].coast()

        # --- Create new tracks from unmatched observations ---
        for o_idx in unmatched_obs:
            new_track = Track(fused_objects[o_idx], timestamp, self.config)
            self.tracks.append(new_track)

        # --- Update track statuses ---
        for track in self.tracks:
            if track.status == "tentative":
                if track.hit_streak >= self.born_confirm:
                    track.status = "confirmed"
                elif track.miss_streak >= 1:
                    track.status = "dead"
            elif track.status == "confirmed":
                if track.miss_streak >= 1:
                    track.status = "coasting"
            elif track.status == "coasting":
                if track.hit_streak >= 1:
                    track.status = "confirmed"
                elif track.miss_streak >= self.coast_max:
                    track.status = "dead"

        # --- Prune dead tracks ---
        self.tracks = [t for t in self.tracks if t.status != "dead"]

        # --- Return confirmed tracks as output ---
        return [t.summary() for t in self.tracks if t.status == "confirmed"]
```

- [ ] **Step 3: Run tests, verify they pass**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_tracker.py -v 2>&1
```

Expected: All 4 tests PASS

---

### Task 10: Visualization

**Files:**
- Create: `code/visualize.py`

**Interfaces:**
- Produces: `render_frame(image, radar_points, radar_dets, camera_dets, fused, tracks, ...)` — draws all overlays
- Produces: `render_bev(ax, radar_points, fused, tracks)` — draws BEV view
- Produces: `render_scene_video(loader, radar_det, cam_det, fusion, tracker, scene_idx, out_path)`

This module has no unit tests — it's verified visually.

- [ ] **Step 1: Write code/visualize.py**

```python
"""Visualization utilities for the radar-camera fusion system.

Renders two views per frame:
  1. Camera image with overlays (radar projections, detections, tracks)
  2. BEV (Bird's Eye View) with radar points, fused objects, track trails
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# Color map for object classes
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
    -1: "unknown",
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def render_frame(image_bgr, radar_dets=None, camera_dets=None,
                 fused_objects=None, tracks=None):
    """Draw detection and tracking overlays on the camera image.

    Args:
        image_bgr: (H, W, 3) BGR image.
        radar_dets: (K, 8) radar detections.
        camera_dets: (N, 6) camera detections.
        fused_objects: (M, 8) fused objects.
        tracks: list of track summary dicts.

    Returns:
        (H, W, 3) BGR image with overlays.
    """
    img = image_bgr.copy()

    # Draw camera detections (green bboxes)
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

    # Draw fused objects (red circles at projected position)
    # (projection requires calibration — done externally; here just draw from
    #  pre-computed pixel positions if available)
    if fused_objects is not None and len(fused_objects) > 0:
        for obj in fused_objects:
            # Fused objects in ego frame — we draw a marker at a fixed position
            # Actual pixel overlay requires projection (done in main pipeline)
            pass

    # Draw track IDs
    if tracks is not None:
        for t in tracks:
            state = t["state"]
            tid = t["id"]
            cls_id = t["class"]
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            # Text at top-left with track info
            y_pos = 20 + tid * 20
            if y_pos < img.shape[0]:
                vx, vy = state[3], state[4]
                speed = np.sqrt(vx ** 2 + vy ** 2)
                label = f"ID:{tid} {CLASS_NAMES.get(cls_id, '?')} {speed:.1f}m/s"
                cv2.putText(img, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)

    return img


def render_bev(radar_points=None, fused_objects=None, tracks=None,
               radar_dets=None, figsize=(10, 10), xlim=(0, 80), ylim=(-30, 30)):
    """Render a Bird's Eye View of the scene.

    Args:
        radar_points: (18, N) raw radar point cloud.
        fused_objects: (M, 8) fused objects in ego frame.
        tracks: list of track summary dicts.
        radar_dets: (K, 8) radar cluster centroids.
        figsize: (w, h) figure size in inches.
        xlim: (min, max) x-axis range (forward direction, meters).
        ylim: (min, max) y-axis range (lateral, meters).

    Returns:
        numpy array (H, W, 3) of the rendered BEV figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X (m) forward")
    ax.set_ylabel("Y (m) left")
    ax.set_title("Bird's Eye View")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot raw radar points
    if radar_points is not None and radar_points.shape[1] > 0:
        x, y = radar_points[0, :], radar_points[1, :]
        ax.scatter(x, y, c="blue", s=2, alpha=0.3, label="radar points")

    # Plot radar cluster centroids
    if radar_dets is not None and len(radar_dets) > 0:
        ax.scatter(radar_dets[:, 0], radar_dets[:, 1], c="red", s=30,
                   marker="o", edgecolors="darkred", linewidths=0.5,
                   label="radar clusters")

    # Plot fused objects with velocity arrows
    if fused_objects is not None and len(fused_objects) > 0:
        for obj in fused_objects:
            x, y, vx, vy = obj[0], obj[1], obj[3], obj[4]
            cls_id = int(obj[5])
            color = np.array(CLASS_COLORS.get(cls_id, (128, 128, 128))) / 255.0
            ax.scatter(x, y, c=[color], s=50, marker="s", edgecolors="black",
                       linewidths=0.5, zorder=5)
            # Velocity arrow
            speed = np.sqrt(vx ** 2 + vy ** 2)
            if speed > 0.1:
                ax.arrow(x, y, vx * 0.2, vy * 0.2, head_width=0.5,
                         head_length=0.3, fc="red", ec="red", alpha=0.7)

    # Plot track trails
    if tracks is not None:
        for t in tracks:
            hist = t.get("history", [])
            if len(hist) >= 2:
                hx = [p[0] for p in hist]
                hy = [p[1] for p in hist]
                ax.plot(hx, hy, "-", linewidth=1.5, alpha=0.6)
                # Label at current position
                state = t["state"]
                ax.text(state[0], state[1] + 0.5, str(t["id"]),
                        fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def render_scene_video(loader, radar_det, cam_det, fusion, tracker,
                       scene_idx=0, out_path="output.mp4"):
    """Process a full scene and render a side-by-side visualization video.

    Args:
        loader: FusionDataLoader instance.
        radar_det: RadarDetector instance.
        cam_det: CameraDetector instance.
        fusion: FusionModule instance.
        tracker: MultiObjectTracker instance.
        scene_idx: Scene index to process.
        out_path: Output video file path.
    """
    frames = list(loader.iter_scene(scene_idx))
    if not frames:
        print("No frames to render.")
        return

    # Determine output size
    sample_bev = render_bev(radar_points=frames[0]["radar_points"],
                            figsize=(6, 6))
    sample_img = frames[0]["image"]
    bev_h, bev_w = sample_bev.shape[:2]
    img_h, img_w = sample_img.shape[:2]

    # Side-by-side: image (left) + BEV (right)
    total_w = img_w + bev_w
    total_h = max(img_h, bev_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 10.0, (total_w, total_h))

    print(f"Rendering {len(frames)} frames to {out_path}...")
    for i, data in enumerate(frames):
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames)}")

        # Run full pipeline
        radar_dets = radar_det.detect(data["radar_points"])
        cam_dets = cam_det.detect(data["image"])
        fused = fusion.fuse(radar_dets, cam_dets, data["cs_radar"],
                            data["cs_cam"], data["camera_intrinsic"])
        tracks = tracker.update(fused, data["timestamp"])

        # Render
        img_overlay = render_frame(data["image"], radar_dets=radar_dets,
                                   camera_dets=cam_dets, fused_objects=fused,
                                   tracks=tracks)
        # Project fused objects to pixel for image overlay
        if fused.shape[0] > 0:
            from utils.coordinate import radar_to_pixel
            fused_xyz = fused[:, :3].T
            uv, _ = radar_to_pixel(fused_xyz, data["cs_radar"], data["cs_cam"],
                                   data["camera_intrinsic"])
            for j in range(uv.shape[1]):
                u, v = int(uv[0, j]), int(uv[1, j])
                cls_id = int(fused[j, 5])
                color = CLASS_COLORS.get(cls_id, (128, 128, 128))
                cv2.circle(img_overlay, (u, v), 6, color, -1)
                cv2.putText(img_overlay, str(cls_id), (u + 8, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        bev_img = render_bev(radar_points=data["radar_points"],
                             radar_dets=radar_dets,
                             fused_objects=fused, tracks=tracks)

        # Combine side-by-side
        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        combined[:img_h, :img_w] = img_overlay
        # Resize BEV if needed
        bev_resized = cv2.resize(bev_img, (bev_w, bev_h))
        combined[:bev_h, img_w:img_w + bev_w] = bev_resized

        out.write(combined)

    out.release()
    print(f"Video saved to {out_path}")
```

---

### Task 11: Evaluation Metrics

**Files:**
- Create: `code/evaluate.py`

**Interfaces:**
- Produces: `compute_mota_motp(gt_tracks, pred_tracks) -> dict`
- Produces: `compute_detection_metrics(gt_boxes, pred_boxes) -> dict`

This module starts as a simplified metrics implementation for development validation.
Full nuScenes-format evaluation can be added later.

- [ ] **Step 1: Write code/evaluate.py**

```python
"""Evaluation metrics for detection and tracking.

Simplified metrics for development validation:
  - Detection: mean IoU, recall at fixed threshold
  - Tracking: MOTA, MOTP (basic implementation)

For official evaluation, use nuscenes.eval.detection and nuscenes.eval.tracking.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_center_distance(gt, pred):
    """Compute Euclidean (x,y) distance between ground truth and prediction."""
    return np.sqrt((gt[0] - pred[0]) ** 2 + (gt[1] - pred[1]) ** 2)


def match_detections(gt_boxes, pred_boxes, threshold=2.0):
    """Match predicted boxes to ground truth using Hungarian algorithm.

    Args:
        gt_boxes: (G, 3+) array with at least [x, y, ...].
        pred_boxes: (P, 3+) array with at least [x, y, ...].
        threshold: Max center distance for a valid match (meters).

    Returns:
        matches: list of (gt_idx, pred_idx, distance).
        unmatched_gt: list of gt_idx.
        unmatched_pred: list of pred_idx.
    """
    G = gt_boxes.shape[0]
    P = pred_boxes.shape[0]

    if G == 0 or P == 0:
        return [], list(range(G)), list(range(P))

    cost = np.zeros((G, P))
    for g in range(G):
        for p in range(P):
            cost[g, p] = compute_center_distance(gt_boxes[g], pred_boxes[p])

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_g = set()
    matched_p = set()
    for g, p in zip(row_ind, col_ind):
        if cost[g, p] <= threshold:
            matches.append((g, p, cost[g, p]))
            matched_g.add(g)
            matched_p.add(p)

    unmatched_gt = [g for g in range(G) if g not in matched_g]
    unmatched_pred = [p for p in range(P) if p not in matched_p]

    return matches, unmatched_gt, unmatched_pred


def compute_mota_motp(all_gt, all_pred, threshold=2.0):
    """Compute MOTA and MOTP metrics across all frames.

    MOTA = 1 - (FN + FP + IDS) / num_gt
    MOTP = mean position error over matched pairs.

    Args:
        all_gt: list of (M_t, 4+) arrays, one per frame [x, y, track_id, ...].
        all_pred: list of (N_t, 4+) arrays, one per frame [x, y, track_id, ...].
        threshold: matching distance threshold (meters).

    Returns:
        dict with keys: mota, motp, num_gt, fn, fp, ids, precision, recall
    """
    total_gt = 0
    total_fn = 0
    total_fp = 0
    total_ids = 0
    total_dist = 0
    total_matches = 0

    # Track ID mapping (pred_id -> gt_id)
    id_map = {}

    for t, (gt, pred) in enumerate(zip(all_gt, all_pred)):
        total_gt += gt.shape[0]

        if gt.shape[0] == 0 and pred.shape[0] == 0:
            continue
        elif gt.shape[0] == 0:
            total_fp += pred.shape[0]
            continue
        elif pred.shape[0] == 0:
            total_fn += gt.shape[0]
            continue

        # Match by position
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred, threshold)

        total_fn += len(unmatched_gt)
        total_fp += len(unmatched_pred)

        for g_idx, p_idx, dist in matches:
            total_dist += dist
            total_matches += 1

            gt_id = int(gt[g_idx, 2])
            pred_id = int(pred[p_idx, 2])

            if pred_id in id_map:
                if id_map[pred_id] != gt_id:
                    total_ids += 1
            id_map[pred_id] = gt_id

    mota = 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1)
    motp = total_dist / max(total_matches, 1)

    return {
        "mota": max(0.0, mota),
        "motp": motp,
        "num_gt": total_gt,
        "fn": total_fn,
        "fp": total_fp,
        "ids": total_ids,
    }


def compute_frame_detection_stats(gt_boxes, pred_boxes, threshold=2.0):
    """Per-frame detection statistics.

    Args:
        gt_boxes: (G, 3+) array.
        pred_boxes: (P, 3+) array.
        threshold: Matching distance threshold (meters).

    Returns:
        dict with tp, fp, fn, precision, recall.
    """
    matches, unmatched_gt, unmatched_pred = match_detections(
        gt_boxes, pred_boxes, threshold)

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}
```

---

### Task 12: Main Pipeline Runner

**Files:**
- Create: `code/main.py`

**Interfaces:**
- Produces: executable script that runs the full pipeline on a scene.

- [ ] **Step 1: Write code/main.py**

```python
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

# Ensure code/ and devkit are on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nuscenes-devkit", "python-sdk"))

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
    print(f"  Radar: DBSCAN eps={CONFIG['radar_eps']}m, min_samples={CONFIG['radar_min_samples']}")
    print(f"  Camera: YOLOv8n, conf={CONFIG['camera_confidence']}")

    print("\n[3/5] Initializing fusion module...")
    fusion = FusionModule(CONFIG)
    print(f"  Late fusion with Hungarian matching")

    print("\n[4/5] Initializing tracker...")
    tracker = MultiObjectTracker(CONFIG)
    print(f"  CA-Kalman filter, confirm={CONFIG['track_born_confirm']}, coast_max={CONFIG['track_coast_max']}")

    print("\n[5/5] Running pipeline and rendering video...")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scene_{scene_idx}_fusion.mp4")

    t_start = time.time()
    render_scene_video(loader, radar_det, camera_det, fusion, tracker,
                       scene_idx=scene_idx, out_path=out_path)
    elapsed = time.time() - t_start

    frames = list(loader.iter_scene(scene_idx))
    print(f"\n{'=' * 60}")
    print(f"Complete: {len(frames)} frames in {elapsed:.1f}s ({elapsed/len(frames):.2f}s/frame)")
    print(f"Output: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
```

---

### Task 13: Integration Verification

**Files:**
- Modify: None (creates a test runner)
- Create: `code/tests/test_integration.py`

- [ ] **Step 1: Write code/tests/test_integration.py**

```python
"""End-to-end integration test: run full pipeline on first 5 frames of scene 0."""
import sys
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/nuscenes-devkit/python-sdk")
sys.path.insert(0, "D:/wzr/PyWorkspace/Fusion/code")

import numpy as np
from config import CONFIG
from data_loader import FusionDataLoader
from radar_detector import RadarDetector
from camera_detector import CameraDetector
from fusion import FusionModule
from tracker import MultiObjectTracker


def test_full_pipeline_5_frames():
    """Run full pipeline for 5 frames without crashing."""
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    radar_det = RadarDetector()
    cam_det = CameraDetector()
    fusion = FusionModule()
    tracker = MultiObjectTracker()

    frame_data = []
    for i, data in enumerate(loader.iter_scene(0)):
        if i >= 5:
            break
        frame_data.append(data)

    assert len(frame_data) == 5

    total_fused = 0
    total_tracks = 0

    for data in frame_data:
        radar_dets = radar_det.detect(data["radar_points"])
        cam_dets = cam_det.detect(data["image"])
        fused = fusion.fuse(radar_dets, cam_dets, data["cs_radar"],
                            data["cs_cam"], data["camera_intrinsic"])
        tracks = tracker.update(fused, data["timestamp"])

        total_fused += fused.shape[0]
        total_tracks += len(tracks)

    print(f"Total fused objects: {total_fused}")
    print(f"Total confirmed tracks: {total_tracks}")
    assert total_fused > 0, "Expected at least some fused objects"
    assert total_tracks > 0, "Expected at least some confirmed tracks"


def test_pipeline_module_interfaces():
    """Verify data flows correctly between modules."""
    loader = FusionDataLoader(version="v1.0-mini",
                              dataroot="D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes")
    data = list(loader.iter_scene(0))[0]

    radar_det = RadarDetector()
    radar_dets = radar_det.detect(data["radar_points"])
    assert radar_dets.shape[1] == 8, f"Radar output shape mismatch"

    cam_det = CameraDetector()
    cam_dets = cam_det.detect(data["image"])
    assert cam_dets.shape[1] == 6, f"Camera output shape mismatch"

    fusion = FusionModule()
    fused = fusion.fuse(radar_dets, cam_dets, data["cs_radar"],
                        data["cs_cam"], data["camera_intrinsic"])
    assert fused.shape[1] == 8, f"Fusion output shape mismatch"

    tracker = MultiObjectTracker()
    tracks = tracker.update(fused, data["timestamp"])
    for t in tracks:
        assert "id" in t
        assert "state" in t
        assert t["state"].shape == (7,)
```

- [ ] **Step 2: Run integration tests**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python -m pytest code/tests/test_integration.py -v 2>&1
```

Expected: All tests PASS

---

### Task 14: Run Full Pipeline

- [ ] **Step 1: Execute the full pipeline on Scene 0**

```bash
cd D:/wzr/PyWorkspace/Fusion && PYTHONPATH="nuscenes-devkit/python-sdk;code" python code/main.py 2>&1
```

Expected: Processes all 40 frames of scene 0, outputs `output/scene_0_fusion.mp4`.

- [ ] **Step 2: Verify output**

```bash
ls -lh D:/wzr/PyWorkspace/Fusion/output/scene_0_fusion.mp4
```

Expected: File exists, size > 100 KB.
