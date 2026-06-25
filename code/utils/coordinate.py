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
