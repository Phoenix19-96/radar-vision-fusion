# Radar-Camera Fusion Perception System Design

**Date:** 2026-06-24
**Status:** Approved
**Dataset:** nuScenes v1.0-mini (scalable to v1.0-trainval)

## 1. Overview

Build a radar-vision fusion perception system that performs **3D object detection + multi-object tracking** using the nuScenes dataset. The system uses a **decision-level (late) fusion** architecture with classical algorithms as the initial implementation, designed for gradual upgrade to deep-learning components.

### 1.1 Scope

- **Sensors:** RADAR_FRONT + CAM_FRONT only (single front-facing pair)
- **Task:** 3D detection + multi-object tracking
- **Approach:** Classical methods first (DBSCAN, YOLO as black-box detector, Kalman filter)
- **Output:** Tracked objects with ID, 3D position, velocity, class label

### 1.2 References

1. "MmWave Radar and Vision Fusion for Object Detection in Autonomous Driving" — five-hierarchy analysis framework (Why/What/Where/When/How)
2. "Radar and Camera Fusion for Object Detection and Tracking: A Comprehensive Survey" (Kun Shi et al., IEEE COMST 2025) — systematic taxonomy of detection & tracking fusion

## 2. System Architecture

### 2.1 High-Level Pipeline

```
nuScenes Data Layer
  RADAR_FRONT (.pcd) + CAM_FRONT (.jpg)
  + Calibrated Sensor + Ego Pose tables
         │
         ├──────────────────────┐
         ▼                      ▼
  Module 1: Radar          Module 2: Camera
  Detection Pipeline       Detection Pipeline
  ① Preprocessing/filter   ① Image preprocessing
  ② DBSCAN clustering      ② YOLOv8n inference
  ③ Cluster feature ext.   ③ NMS + thresholding
         │                      │
         │ RadarDetections       │ CameraDetections
         │ [K, 8]                │ [N, 6]
         └──────────┬────────────┘
                    ▼
         Module 3: Fusion
         ① Coordinate transform chain
            Radar → Ego → Camera → Pixel
         ② Distance + IoU cost matrix
         ③ Hungarian matching
         ④ Unmatched handling
                    │
                    │ FusedObjects [M, 8]
                    ▼
         Module 4: Multi-Object Tracking
         ① Kalman filter (CTRV model)
         ② Track management (Born/Confirmed/Coasting/Dead)
         ③ Hungarian data association
                    │
                    │ Tracks [T, 9+id]
                    ▼
         Module 5: Evaluation
         ① Detection: mAP, NDS
         ② Tracking: MOTA, MOTP
         ③ Visualization (BEV + Image overlay)
```

### 2.2 Key Data Structures

```
RadarDetection:  [x, y, z, vx, vy, rcs_mean, size_xy, n_points]   (8,)
CameraDetection: [u1, v1, u2, v2, class_id, score]                  (6,)
FusedObject:     [x, y, z, vx, vy, class_id, conf_radar, conf_camera] (8,)
Track:           {id, age, state[7], P[7×7], class, status, history (last N positions, capped)}
```

## 3. Module Details

### 3.1 Module 1: Radar Point Cloud Clustering

#### Input
nuScenes radar point cloud: `(18, N)` numpy array. Each point has 18 dimensions:
x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms.

#### Processing

**Step 1: Filtering** — exactly three conditions:
```python
mask = (
    (invalid_state == 0) &          # valid points only
    (ambig_state == 3) &            # unambiguous Doppler solution
    (pdh0 <= 2)                     # false-alarm probability ≤ 50%
)
```

| Filter | Field | Condition | Purpose |
|--------|-------|-----------|---------|
| invalid_state | idx 14 | `== 0` | Exclude low-RCS artifacts, near-field artifacts, mirror ghosts, FOV outliers, harmonics |
| ambig_state | idx 11 | `== 3` | Exclude points where Doppler ambiguity is not reliably resolved |
| pdh0 | idx 15 | `<= 2` | Exclude likely multipath reflections / ghost targets (≤50% false-alarm probability) |

The following fields are explicitly **not** used for filtering:
- `dyn_prop`: useful for downstream classification (moving vs stationary), not for validity filtering
- `rcs`: signal strength varies too widely across object types; not a reliable validity indicator
- `is_quality_valid`: redundant with invalid_state + pdh0

**Step 2: Point-wise Detection**
- **No clustering is applied.**  The Continental ARS 408 radar firmware already outputs
  pre-clustered targets — each "point" in the nuScenes PCD file is actually a radar-internal
  cluster.  Running a second DBSCAN on already-clustered data adds little value and risks
  incorrectly merging distinct objects.
- Each filtered point becomes one detection.
- Detection columns: `[x, y, z, vx_comp, vy_comp, rcs, 0, 1]` — the last two columns
  (size_xy=0, n_points=1) are placeholders since single-point clusters have no spatial extent.

#### Design Decisions
- No clustering on the application side — radar hardware already clusters targets
- Filtering limited to three validity signals: invalid_state, ambig_state, pdh0
- dyn_prop and rcs excluded from filtering — they are classification signals, not validity indicators

#### Output
`RadarDetections`: `(K, 8)` — `[x, y, z, vx, vy, rcs_mean, size_xy, n_points]`

#### Design Decisions
- DBSCAN chosen because cluster count is unknown a priori
- 2D clustering avoids z-noise contamination
- `vx_comp/vy_comp` used to avoid ego-motion compensation step
- Filtering limited to three validity signals: invalid_state, ambig_state, pdh0
- dyn_prop and rcs excluded from filtering — they are classification signals, not validity indicators

### 3.2 Module 2: Camera Object Detection

#### Input
CAM_FRONT image: 1600×900 RGB

#### Processing

**Step 1: Preprocessing**
- Resize to YOLO input size
- Normalize per model requirements

**Step 2: YOLOv8n Inference**
- COCO pretrained weights
- Target classes: car, truck, bus, pedestrian, bicycle, motorcycle
- CPU inference: ~50-80ms per frame

**Step 3: Post-processing**
- NMS with IoU threshold = 0.45
- Confidence threshold = 0.3

#### Output
`CameraDetections`: `(N, 6)` — `[u1, v1, u2, v2, class_id, score]`

#### Design Decisions
- YOLOv8n as black-box detector: minimal integration cost, runs on CPU
- COCO classes map naturally to nuScenes categories
- Can be replaced with nuScenes-finetuned model in future iteration

### 3.3 Module 3: Spatio-Temporal Fusion

#### Coordinate Transform Chain

```
Radar Sensor Frame → Ego Frame → Camera Sensor Frame → Image Pixel Frame

Step 1: Radar → Ego
  Apply cs_record['translation'] + cs_record['rotation'] (radar calibration)

Step 2: Ego → Camera
  Apply inverse of camera cs_record['translation'] + cs_record['rotation']

Step 3: Camera → Pixel
  Apply camera_intrinsic matrix K (3×3):
    [u, v, 1]^T = K @ [Xc, Yc, Zc]^T / Zc
```

#### Association Strategy: Dual-Metric Cost Matrix

For each radar detection projected to pixel `(u_rad, v_rad, depth)`:

```
Cost(radar_i, camera_j) =
  ┌ 1 - IoU(radar_proj_point, camera_bbox)        if radar_proj inside bbox
  └ distance_to_bbox_center + outside_penalty      if radar_proj outside bbox
```

Additional constraints:
- Depth gate: skip if `|radar_depth - estimated_depth| > threshold`
- Speed direction consistency: optional

#### Matching
- Hungarian algorithm for optimal min-cost assignment
- Three outcomes:
  1. **Matched** (Radar ∩ Camera): fused object — position from radar, class from camera
  2. **Radar only**: camera miss (occlusion/dark) — class = `unknown`, velocity available
  3. **Camera only**: radar miss (distant/low-RCS) — velocity = unavailable, depth estimated

#### Output
`FusedObjects`: `(M, 8)` — `[x, y, z, vx, vy, class_id, conf_radar, conf_camera]`

### 3.4 Module 4: Multi-Object Tracking

#### Motion Model: CA (Constant Acceleration)

```
State vector:  X = [x, y, z, vx, vy, ax, ay]^T   (7-D)
                 — z is estimated from process model (radar z accuracy is poor)
Measurement:   Z = [x, y, vx, vy]^T               (4-D, from radar)

Process model (linear, constant acceleration):
  x_{k+1}  = x_k + vx_k * Δt + 0.5 * ax_k * Δt²
  y_{k+1}  = y_k + vy_k * Δt + 0.5 * ay_k * Δt²
  z_{k+1}  = z_k                    (no z-velocity in state, constant)
  vx_{k+1} = vx_k + ax_k * Δt
  vy_{k+1} = vy_k + ay_k * Δt
  ax_{k+1} = ax_k + ν_ax           (process noise driven)
  ay_{k+1} = ay_k + ν_ay

Observation model:
  H = [I(4×4) | 0(4×2)]  — linear, direct observation of position + velocity
  Note: z is NOT directly observed; updated only through process model correlation via P matrix

Process noise covariance Q: tuned per ax/ay noise level
Measurement noise R: pos ~0.3m, vel ~0.1m/s (radar Doppler is precise)
```

**Key advantage:** Radar provides direct velocity measurement (Doppler), orders of magnitude more accurate than vision-based velocity estimation from position differencing. Tracking is also compatible with Radar-only matched targets (velocity known) and Camera-only targets (velocity unobserved, estimated from position history).

#### Track State Machine

```
                    ┌──────────┐
          detected  │   Born   │  2 consecutive matches
         ──────────→│ (tentative)│─────────────────────┐
                    └──────────┘                       ▼
                                           ┌──────────────────┐
                                           │   Confirmed      │
                                           │   (active)       │──→ output
                                           └──┬────────┬──────┘
                              N consecutive  │        │ matched
                              misses          │        │ (KF update)
                                           ▼          │
                              ┌──────────────┐        │
                              │  Coasting    │────────┘
                              └──────┬───────┘
                        M more misses   │
                                     ▼
                              ┌──────────────┐
                              │     Dead     │ → removed
                              └──────────────┘
```

Parameters:
- Born → Confirmed: 2 consecutive hits
- Confirmed → Coasting: 1 miss
- Coasting → Dead: 5 misses (0.25s at 20Hz)
- Coasting → Confirmed: 1 hit

#### Frame-to-Frame Data Association

Cost matrix components:
1. **Mahalanobis distance** from Kalman innovation: `(z_pred - z_obs)^T * S^{-1} * (z_pred - z_obs)`
2. **Velocity cosine similarity**: `cos(angle(track.v, obs.v))` — radar unique capability
3. **Class consistency**: `track.class == obs.class ? 0 : penalty`

Hungarian algorithm for optimal assignment.

#### Output
`Tracks`: List of `{id, age, state[7], covariance[7×7], class, status, history}`

### 3.5 Module 5: Evaluation

#### Detection Metrics
- mAP (mean Average Precision) — standard 3D detection
- NDS (nuScenes Detection Score) — nuScenes-specific composite metric
- Match thresholds: center distance 2m for BEV

#### Tracking Metrics
- MOTA (Multiple Object Tracking Accuracy) — misses, false positives, ID switches
- MOTP (Multiple Object Tracking Precision) — position accuracy
- Track fragmentation, ID switches

#### Visualization
1. **BEV plot**: Radar points + fused detections + tracks with velocity arrows
2. **Image overlay**: 2D bboxes from camera + projected radar detections + track IDs
3. **Video export**: Frame-by-frame animated visualization

## 4. Project Structure

```
code/
├── config.py                  # Global configuration
├── data_loader.py             # nuScenes data loading wrapper
├── radar_detector.py          # Module 1: radar clustering
├── camera_detector.py         # Module 2: YOLO-based detection
├── fusion.py                  # Module 3: coordinate transform + association
├── tracker.py                 # Module 4: Kalman filter + track management
├── evaluate.py                # Module 5: metrics
├── visualize.py               # Module 5: visualization
├── main.py                    # End-to-end pipeline runner
└── utils/
    ├── coordinate.py          # Coordinate transform utilities
    ├── kalman_filter.py       # Kalman filter implementation
    └── association.py         # Hungarian algorithm + cost matrix
```

## 5. Implementation Roadmap

### Phase 1: Data Pipeline & Radar Detection (Week 1)
- [ ] nuScenes devkit integration (direct JSON + binary reading if pip install fails)
- [ ] `data_loader.py`: load RADAR_FRONT + CAM_FRONT for a scene
- [ ] `radar_detector.py`: filtering + DBSCAN + feature extraction
- [ ] Unit test: verify radar detection count and feature shapes
- [ ] Quick visualization: scatter radar clusters in BEV

### Phase 2: Camera Detection (Week 1-2)
- [ ] `camera_detector.py`: YOLOv8n wrapper
- [ ] Class mapping COCO → nuScenes categories
- [ ] Unit test: verify detection output format
- [ ] Visualization: draw YOLO bboxes on CAM_FRONT images

### Phase 3: Fusion (Week 2)
- [ ] `utils/coordinate.py`: Radar → Ego → Camera → Pixel transform chain
- [ ] `utils/association.py`: cost matrix + Hungarian algorithm
- [ ] `fusion.py`: projection + association + matched/unmatched handling
- [ ] Integration test: verify fused objects on a few frames
- [ ] Visualization: radar projections overlaid on camera image

### Phase 4: Tracking (Week 2-3)
- [ ] `utils/kalman_filter.py`: CTRV Kalman filter implementation
- [ ] `tracker.py`: track state machine + frame-to-frame association
- [ ] Integration test: verify track continuity on a full scene
- [ ] Visualization: tracks with ID labels in BEV + image overlay

### Phase 5: Evaluation & Polish (Week 3)
- [ ] `evaluate.py`: mAP, NDS, MOTA, MOTP computation
- [ ] `visualize.py`: end-to-end scene rendering to video
- [ ] `main.py`: pipeline script processing a full scene
- [ ] Documentation: usage instructions, parameter tuning guide

### Phase 6: Iterative Improvement (Future)
- [ ] Velocity consistency constraint in fusion association
- [ ] Camera-detector upgrade: YOLO → nuScenes-finetuned model
- [ ] Radar features → light classifier to distinguish vehicle/pedestrian from RCS patterns
- [ ] IMM (Interacting Multiple Model) filter replacing single CTRV
- [ ] Multi-sensor extension: add side radars + cameras
- [ ] Feature-level fusion (CNN radar heatmap + image features)

## 6. Configuration Parameters

```python
# config.py
CONFIG = {
    # Radar
    "radar_debug_plot": False,   # Enable debug plot (radar + camera side-by-side)

    # Camera
    "camera_confidence": 0.3,   # YOLO confidence threshold
    "camera_nms_iou": 0.45,     # NMS IoU threshold
    "camera_classes": [2, 5, 7, 0, 1, 3],  # COCO class IDs for car/truck/bus/pedestrian/bicycle/motorcycle

    # Fusion
    "fusion_max_depth_diff": 5.0,    # Max depth discrepancy (meters)
    "fusion_outside_penalty": 50.0,  # Penalty for projection outside bbox

    # Tracking
    "track_born_confirm": 2,    # Frames to confirm new track
    "track_coast_max": 5,       # Max coasting frames before deletion
    "track_init_variance": 1.0, # Initial position variance (m²)
    "track_process_noise": 0.5, # Process noise std (m/s²)
    "track_meas_noise_pos": 0.3,# Position measurement noise std (m)
    "track_meas_noise_vel": 0.1,# Velocity measurement noise std (m/s)

    # Data
    "dataroot": "D:/wzr/PyWorkspace/Fusion/data/sets/nuscenes",
    "version": "v1.0-mini",
    "scene_idx": 0,             # Scene index to process
}
```

## 7. Dependencies

```
numpy, scipy, matplotlib, opencv-python
scikit-learn (DBSCAN)
ultralytics (YOLOv8n)
nuscenes-devkit (local, editable install)
```

## 8. Success Criteria (Phase 1-5)

1. **Data integrity**: Correctly load and align RADAR_FRONT + CAM_FRONT timestamps
2. **Coordinate transform**: Radar projection errors < 5 pixels (visual inspection)
3. **Fusion ratio**: >50% matched (radar+camera), <30% unmatched-only at close range (<50m)
4. **Tracking continuity**: < 5 ID switches per scene, track lifetimes > 10 frames for near-range targets
5. **Speed**: Process 1 scene (40 frames) in < 5 minutes on CPU
