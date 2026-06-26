# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a workspace built around the **nuScenes devkit** (v1.2.0) — the official development kit for the nuScenes and nuImages autonomous driving datasets by Motional. The devkit is vendored from [nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and lives under `nuscenes-devkit/`. User-owned experiment/script code lives in `radar_fusion/`.

**Python version:** 3.8. Use `python` (not `python3`).

## Commands

### Run the fusion perception pipeline
```bash
cd D:/wzr/PyWorkspace/Fusion
PYTHONPATH="nuscenes-devkit/python-sdk;radar_fusion" python radar_fusion/main.py
```
Output video saved to `output/`.

### Run inline tests (no pytest)
```bash
PYTHONPATH="nuscenes-devkit/python-sdk;radar_fusion" python -c "
<test code>
"
```

### Run a tutorial (smoke test)
```bash
cd nuscenes-devkit/setup && bash test_tutorial.sh
```
Expects data under `data/sets/nuscenes/`. On headless systems, uses `xvfb-run` to avoid display errors.

## Architecture

### `nuscenes-devkit/python-sdk/` — The devkit source

Two top-level packages:

- **`nuscenes/`** — Core database class (`nuscenes.py: NuScenes`) that loads JSON table files and provides query/retrieve/render methods. This is the main entry point for all nuScenes operations. It aggregates utility modules:
  - `utils/data_classes.py` — `LidarPointCloud`, `RadarPointCloud`, `Box` (3D bounding boxes). Abstract `PointCloud` base.
  - `utils/data_io.py` — Binary file loading helpers.
  - `utils/geometry_utils.py` — Coordinate transforms, projection (`view_points`), visibility checks.
  - `utils/map_mask.py` — Map resolution and masking.
  - `lidarseg/` — LiDAR semantic segmentation coloring, statistics, legend generation.
  - `panoptic/` — Panoptic segmentation label generation and utilities.
  - `map_expansion/` — Map API (`map_api.py`) for querying lane geometry, plus arcline path utilities and bitmap maps.
  - `can_bus/` — CAN bus data API.
  - `prediction/` — Input representation builders (agents, static layers, combinators) for prediction models.

- **`nuimages/`** — Standalone package for the nuImages (2D image) dataset. Same overall pattern as `nuscenes/`.

### `nuscenes/eval/` — Evaluation code for each challenge

Each eval subpackage follows the same pattern: `config.py` (classes, thresholds), `data_classes.py` (metric containers), `evaluate.py` (main entry point), `render.py` (visualization), `algo.py` (baseline algorithm), `utils.py` (helpers), `tests/`.

- `eval/detection/` — 3D object detection evaluation (mAP, NDS, TP metrics).
- `eval/tracking/` — Multi-object tracking (AMOTA, AMOTP). Uses `mot.py` for core MOT metrics.
- `eval/prediction/` — Trajectory prediction challenge. Metric: minADE/minFDE.
- `eval/lidarseg/` — LiDAR semantic segmentation evaluation.
- `eval/panoptic/` — Panoptic segmentation and panoptic tracking evaluation.
- `eval/common/` — Shared evaluation code (config base, data classes, loaders, render utils).

### `tutorials/` — Jupyter notebook tutorials

Each is a canonical walkthrough (nuscenes, nuimages, can_bus, map_expansion, prediction). The `test_tutorial.sh` script converts them to `.py` and runs them as integration tests.

### `data/` — Dataset storage

Dataset archives are stored at the project root level. The devkit expects data under `data/sets/nuscenes/<version>/` (e.g., `v1.0-mini/`). The `.tgz` archives at `data/` level are source packages to be extracted.

### `radar_fusion/` — Radar-Camera Fusion Perception System

Contains the **radar-camera fusion perception system** (v0.1) — decision-level late fusion with classical algorithms:

| File | Role |
|------|------|
| `main.py` | End-to-end pipeline entry |
| `config.py` | All tunable parameters (DBSCAN eps, KF noise, etc.) |
| `data_loader.py` | Wraps nuScenes RADAR_FRONT + CAM_FRONT loading |
| `radar_detector.py` | DBSCAN clustering on (x,y) plane, 8-D cluster features |
| `camera_detector.py` | YOLOv8n wrapper (COCO pretrained, CPU inference) |
| `fusion.py` | Decision-level fusion: coordinate projection + Hungarian matching |
| `tracker.py` | CA-Kalman filter + Born/Confirmed/Coasting/Dead state machine |
| `visualize.py` | BEV plot + camera overlay, renders scene to .mp4 |
| `utils/coordinate.py` | Radar→Ego→Camera→Pixel transform chain |
| `utils/kalman_filter.py` | 7-D CA model, 4-D measurement [x,y,vx,vy] |
| `utils/association.py` | Hungarian matching via scipy, cost matrix builder |
| `tests/` | Test scripts using `python -c` inline (no pytest available) |

## Key dependencies

`numpy`, `scipy`, `matplotlib`, `opencv-python`, `pyquaternion`, `scikit-learn`, `tqdm`, `shapely`, `pycocotools`, `fire`, `descartes`, `ultralytics` (YOLOv8n), `torch`, `torchvision`.

## Environment Notes

- **PYTHONPATH** must include both `nuscenes-devkit/python-sdk` and `radar_fusion` (Windows semicolon separator: `;`)
- **No pytest** — tests run as inline `python -c` scripts
- **Git repo** only tracks `radar_fusion/`, `docs/`, `output/` (devkit, data, PDFs, .pt files excluded via .gitignore)
- **Remote:** `git@github.com:Phoenix19-96/radar-vision-fusion.git`
