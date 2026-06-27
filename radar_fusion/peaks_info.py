"""PeaksInfo — a single nuScenes radar peak with 18 named attributes.

Each PeaksInfo instance corresponds to one column of the (18, N) raw radar
point cloud array loaded from a .pcd file.  The 18 fields match the nuScenes
RADAR PCD header:

    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid
           ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeaksInfo:
    """A single radar peak (point) with its 18-channel attributes."""

    x: float                # front distance (m)
    y: float                # left distance (m)
    z: float                # up (m) — typically 0 for ARS 408
    dyn_prop: int           # dynamic property (0=moving … 7=stopped)
    id: int                 # radar-internal cluster ID
    rcs: float              # radar cross-section (dBsm)
    vx: float               # uncompensated velocity x (m/s)
    vy: float               # uncompensated velocity y (m/s)
    vx_comp: float          # ego-motion-compensated velocity x (m/s) ★ recommended
    vy_comp: float          # ego-motion-compensated velocity y (m/s) ★ recommended
    is_quality_valid: int   # quality flag (0=bad, 1=good)
    ambig_state: int        # Doppler ambiguity state (0=invalid, 3=unambiguous)
    x_rms: float            # x position std (m)
    y_rms: float            # y position std (m)
    invalid_state: int      # validity bitmask (0x00=valid)
    pdh0: int               # false-alarm probability (0=invalid … 7=≤100%)
    vx_rms: float           # vx velocity std (m/s)
    vy_rms: float           # vy velocity std (m/s)

    # Order must match nuScenes PCD FIELDS line above
    _FIELD_ORDER = [
        "x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy",
        "vx_comp", "vy_comp", "is_quality_valid", "ambig_state",
        "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms",
    ]

    # ------------------------------------------------------------------
    # Batch conversion helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_array(cls, points: np.ndarray) -> list["PeaksInfo"]:
        """Convert a raw (18, N) nuScenes radar array to a list of PeaksInfo.

        Args:
            points: (18, N) float32 / float64 array from RadarPointCloud.points.

        Returns:
            List of N PeaksInfo instances.
        """
        if points.shape[1] == 0:
            return []
        return [cls(*points[:, i]) for i in range(points.shape[1])]

    @staticmethod
    def to_array(peaks: list["PeaksInfo"]) -> np.ndarray:
        """Convert a list of PeaksInfo back to an (18, N) float64 array.

        Args:
            peaks: list of PeaksInfo (may be empty).

        Returns:
            (18, N) float64 array in nuScenes column order.
        """
        if not peaks:
            return np.empty((18, 0))
        cols = [[getattr(p, f) for p in peaks] for f in PeaksInfo._FIELD_ORDER]
        return np.array(cols, dtype=np.float64)
