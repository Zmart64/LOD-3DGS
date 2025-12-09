# 3DGS LiDAR-Initialized Experiments (office scan)

## Scene & Data
- **Capture**: indoor office building with a LiDAR scan (`2025-09-30_Test-L-UG2_491058901020.bag__subsampled_colored.las`) and 359 panoramic RGB images (`images360`).
- **Goal**: test whether LiDAR-initialized 3D Gaussian Splatting (3DGS) can reconstruct the scene well enough to render synthetic images from novel viewpoints for downstream visual localization.
- **Processed datasets**:
  - Initial quick run used `pose_stride: 10`, resulting in 36 panoramas and 648 cube faces (saved under `data/processed/lidar_extensive_v2`).
  - Latest preprocessing (stride 1) keeps all 359 panoramas, yielding 6,462 cube faces (`data/processed/lidar_extensive_full`). The metadata reports 204 poses aligned via trajectory, 155 via metadata, RMSE alignment error ≈ 6.4 cm.
- **Alignment diagnostics**:
  - `scripts/check_alignment.py` generated overlays in `data/processed/lidar_extensive_v2/overlays_pano` and `overlays_persp` to visually confirm LiDAR ↔ RGB alignment.
  - When projecting Gaussians in COLMAP frame looked empty, we exported raw LiDAR to `lidar_world_points.npz` and used that for overlays/rendering sanity checks, as the processed Gaussians are normalized.

## Viewer & Inspection Tooling
- `simple_viewer.py` patched to accept `--point_size` for clearer point-cloud previews; confirmed LiDAR-colored points look coherent even when Gaussian renderings do not.
- Exported PLYs via `colmap model_converter` (renaming outputs to `.ply`) to inspect sparse reconstructions in MeshLab/CloudCompare.

## Training Pipeline
- Base script: `torchrun --nproc_per_node=8 scripts/train_gaussians.py --config configs/lidar_extensive.yaml`.
- Key infrastructure changes:
  - Added cosine LR decay with configurable floor (`min_learning_rate`), and resume support via `--resume` flag.
  - Added scale clamping inside the renderer plus tighter regularizers (opacity & distortion) to combat gigantic splats.
  - Exposed DataLoader worker count via config; currently using 6 workers and gradient accumulation (2) to balance throughput and shared-memory limits.
  - Training config now uses: batch size 1, 262 k rays, cosine LR 8e-4→2e-4, densify window [4 k, 450 k], prune every 2.5 k, `absgrad` enabled, `scale_clamp_max: 0.018`, and the full processed dataset.

## Results & Issues
1. **Original 36-panorama run**  
   - Metrics peaked around step ~262 k with PSNR ≈21 dB, but renders still showed “painted” blobs: splats grew too large once LR stayed at 2.5e-3, leaving blurry translucent geometry (see `gaussian_render.png`).  
   - Viewer point cloud looked fine (`viser_view.png`), highlighting the training issue rather than data corruption.

2. **Config iterations**  
   - Sequential tweaks included: lowering base LR, enabling cosine decay, densification/pruning schedules, scale clamps, and SH degree handling (kept SH=0 until color features are padded properly).
   - Added LiDAR-world overlays, improved alignment checking, and verified all 122 k COLMAP sparse points were present (outliers were clipped before viewing).

3. **Full dataset attempt (359 panoramas)**  
   - Training from scratch with the larger dataset (before current config) never exceeded PSNR 13 dB (best at step 642 k). Losses oscillated between 0.05–0.20, indicating optimizer thrash due to the heavy view count and insufficient regularization.
   - Giant splats reappeared; renders unusable. Concluded the earlier checkpoints (36 views) are mismatched and cannot be reused after reprocessing all panoramas.

4. **Current run (in progress)**  
   - Using updated config (stride 1 dataset, stronger regularization, gradient accumulation, more rays per step). GPUs now operate at 100 % utilization while staying cool (≈40 °C) with moderate VRAM (~2 GB/GPU).  
   - Awaiting new metrics/renderings to see whether PSNR improves and whether splats remain compact.

## Takeaways (so far)
- LiDAR-initialized 3DGS can register all panoramas (alignment overlays look good), but stabilizing splat sizes is crucial; otherwise renders degrade into translucent streaks despite accurate point colors.
- The 36-pose dataset is too sparse for reliable novel-view rendering; expanding to all 359 panoramas was necessary but exposed training instabilities that demanded scheduler/regularization upgrades.
- We now have a reproducible preprocessing + training pipeline, viewer tooling, and diagnostics to continue iterating. Even though satisfying renders aren’t yet produced, the experiment groundwork (data prep, alignment checks, training infrastructure) is ready for further trials or comparison with alternate reconstruction methods.
