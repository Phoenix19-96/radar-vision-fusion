"""NuScenes data loading wrapper for radar-camera fusion.

Provides a lightweight interface to load synchronized RADAR_FRONT + CAM_FRONT
frames with calibration data, iterating through scene samples.

Radar points are parsed into PeaksInfo instances before leaving the loader.
"""

import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from peaks_info import PeaksInfo


class FusionDataLoader:
    """Load synchronized radar point clouds and camera images from nuScenes.

    Usage:
        loader = FusionDataLoader(version="v1.0-mini", dataroot="...")
        for frame in loader.iter_scene(scene_idx=0):
            peaks = frame["peaks"]           # list[PeaksInfo]
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
                peaks: list[PeaksInfo], parsed radar peaks (one per point).
                image: (H, W, 3) uint8 array, BGR image.
                cs_radar: calibrated_sensor record for RADAR_FRONT.
                cs_cam: calibrated_sensor record for CAM_FRONT.
                camera_intrinsic: (3, 3) float64 camera intrinsic matrix.
                timestamp: int, sample timestamp in microseconds.
                sample_token: str, nuScenes sample token.
                ego_pose: dict, ego_pose record for the sample.
        """
        sample = self.nusc.get("sample", sample_token)

        # --- Radar ---
        radar_sd = self.nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
        ego_pose = self.nusc.get("ego_pose", radar_sd["ego_pose_token"])
        radar_path = os.path.join(self.nusc.dataroot, radar_sd["filename"])
        pc = RadarPointCloud.from_file(radar_path)
        cs_radar = self.nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])

        # --- Camera ---
        cam_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])
        image = cv2.imread(img_path)  # BGR
        cs_cam = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        camera_intrinsic = np.array(cs_cam["camera_intrinsic"])

        peaks = PeaksInfo.from_array(pc.points)

        return {
            "peaks": peaks,                      # list[PeaksInfo]
            "image": image,                      # (H, W, 3) BGR
            "cs_radar": cs_radar,
            "cs_cam": cs_cam,
            "camera_intrinsic": camera_intrinsic,
            "timestamp": sample["timestamp"],
            "sample_token": sample_token,
            "ego_pose": ego_pose,
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
