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
        q = process_noise ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

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
