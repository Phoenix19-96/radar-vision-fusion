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
        points_cam = np.array([[0.0, 0.0, 10.0]]).T
        uv = camera_to_pixel(points_cam, K)
        np.testing.assert_array_almost_equal(uv, np.array([[500.0, 500.0]]).T, decimal=2)

    def test_camera_to_pixel_negative_z_returns_none(self):
        """Points behind camera (Z<=0) should be filtered."""
        K = np.eye(3)
        points_cam = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 5.0]]).T
        uv = camera_to_pixel(points_cam, K)
        assert uv.shape[1] == 1

    def test_radar_to_pixel_integration(self):
        """End-to-end: known transform produces expected pixel coordinates."""
        points_radar = np.array([[10.0, 0.0, 0.0]]).T
        cs_radar = {"translation": [3.0, 0.0, 0.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0]}
        cs_cam = {"translation": [1.0, 0.0, 1.5],
                  "rotation": [0.5, -0.5, 0.5, -0.5]}
        K = np.array([[1266.0, 0.0, 816.0],
                       [0.0, 1266.0, 491.0],
                       [0.0, 0.0, 1.0]])
        uv, depths = radar_to_pixel(points_radar, cs_radar, cs_cam, K)
        assert uv.shape[0] == 2
        assert depths.shape[0] == 1
        assert 0 <= uv[0, 0] <= 1600
        assert 0 <= uv[1, 0] <= 900
        assert depths[0] > 0
