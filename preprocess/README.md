# Panorama + LiDAR Preprocessing

This directory hosts the tooling that turns the raw scan in `2025-09-30_Test-L-UG2_491058901020/` into a NeRF-style dataset (`transforms_*.json` + perspective RGB + LiDAR-derived `points3d.ply`) that LOD-3DGS can ingest without COLMAP.

## 1. Python environment
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python pillow pyyaml tqdm py360convert laspy plyfile
```

> Tip: add the `source .venv/bin/activate` line to your shell profile if you switch contexts frequently.

## 2. Configure paths & offsets
The defaults live in `extrinsic_estimation.yaml`:
- `data.dataset_root` – folder with the raw LiDAR, panoramas, and pose logs
- `data.processed_root` – where the processed dataset will be written (default: `preprocess/output/lidar_extensive_full`)
- `camera_offset` / `camera_*_offset_deg` – rigid transform from LiDAR base to the panorama rig
- View settings (`horizontal_views`, `vertical_views`, `cube_face_fov_deg`, etc.)

Update these fields if you relocate the raw data or change the capture rig.

## 3. Run the converter
```bash
.venv/bin/python preprocess/prepare_dataset.py \
  --config extrinsic_estimation.yaml \
  --overwrite \
  --test-every 8
```
Key optional flags:
- `--output-dir PREPROCESSED_DIR` – override `processed_root`
- `--max-images N` – process only the first *N* panoramas (debug)
- `--max-time-diff SEC` – tighten/loosen LiDAR ↔ panorama timestamp matching
- `--test-every K` – hold out every Kth panorama for `transforms_test.json`
- `--write-depths` – also rasterize metric depth maps from the LiDAR for every perspective view (saved under `depths/`)
- `--depth-max-distance M` – ignore LiDAR points farther than `M` meters when creating depth (helps reduce clutter)
- `--depth-splat-radius R` – splat each LiDAR sample onto a `(2R+1)²` pixel block to densify depth maps (e.g., `R=1` for 3×3)

Outputs (within the chosen processed directory):
- `images/` – 18 perspective cube views per panorama (6 yaw angles × {−45°, 0°, +45°}) at `cube_face_resolution`
- `depths/` (optional) – 16-bit mono PNGs storing depth in millimeters (toggle via `--write-depths`)
- `transforms_train.json`, `transforms_test.json` – NeRF-compatible camera metadata
- `points3d.ply` – subsampled LiDAR point cloud (default cap: 200k points)
- `metadata/summary.json` – bookkeeping (pose source, timestamps, etc.)

## 4. Train with LOD-3DGS
Point `train.py` to the processed folder (Blender/NeRF-style loader):
```bash
python train.py -s preprocess/output/lidar_extensive_full --use_lod --sh_degree 2 --resolution 1
```
Add your usual flags (`--depths`, `--iterations`, etc.). Because the dataset now exposes `transforms_*.json`, you do **not** need COLMAP outputs.

## 5. Troubleshooting
- To verify alignment numerically, inspect `metadata/summary.json` – the `matched_with_trajectory` count should be high if timestamps line up (<0.75 s delta by default).
- Use `--max-images 1` to ensure the pipeline works before processing the entire scan (≈6,462 cube faces).
- Re-run with `--overwrite` whenever you tweak offsets or fov; the script refuses to clobber outputs silently.
