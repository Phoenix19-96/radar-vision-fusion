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
                    z = np.array([fused_objects[j, 0], fused_objects[j, 1],
                                  fused_objects[j, 3], fused_objects[j, 4]])
                    y = z - np.array([pred[0], pred[1], pred[3], pred[4]])
                    S_pos = P[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])] + track.kf.R
                    try:
                        d = y @ np.linalg.inv(S_pos) @ y
                    except np.linalg.LinAlgError:
                        d = 1e9
                    cost[i, j] = d if d < self.mahalanobis_threshold else 1e9

            track_indices, obs_indices, unmatched_tracks, unmatched_obs = \
                hungarian_match(cost, max_cost=self.mahalanobis_threshold)
        else:
            track_indices = np.array([], dtype=int)
            obs_indices = np.array([], dtype=int)
            unmatched_tracks = list(range(N)) if N > 0 else []
            unmatched_obs = list(range(M)) if M > 0 else []

        # --- Update matched tracks ---
        for t_idx, o_idx in zip(track_indices, obs_indices):
            track = active_tracks[t_idx]
            track.update(fused_objects[o_idx], timestamp)

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
