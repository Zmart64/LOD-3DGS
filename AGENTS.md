# Agent Brief

## Thesis Context
- Working on master thesis implementation on the shared cluster; primary milestone is a high-fidelity 3D Gaussian Splatting (3dgs) pipeline.
- Stage one: use colored LiDAR as the Gaussian initialization and co-registered 360° robot captures (camera mounted ~40 cm above the floor, so viewpoints are sparse) for scene reconstruction.

## Repository Layout
- All repos live alongside this file; the 3d Gaussian Splatting codebase is `3dgs`.
- Larger features must land through dedicated branches per feature before merging back.
- Each repository ships its own Python virtual environment under `.venv`; activate it before running scripts.

## Data Sources
- Core dataset: `3dgs/2025-09-30_Test-L-UG2_491058901020` (contains colored LiDAR + co-registered 360° imagery).
- No explicit LiDAR/image extrinsics are provided; an estimated alignment & offset already lives inside the `3dgs` repo—re-use those transforms.

## Compute Notes
- Cluster node exposes 8×16 GB NVIDIA V100 GPUs (verify with `nvidia-smi`).
- Code should maximize GPU utilization for faster turnaround (multi-GPU data loading, batched splatting, etc.).
- Cluster environment offers no module system; rely on the repo-local `.venv` setups.

## Coding Guidelines
- Keep code structure clean, prefer descriptive docstrings, and avoid redundant comments.
- Document assumptions near the code via docstrings or README snippets instead of inline chatter.
- When extending 3dgs, respect the existing alignment assets and data organization.
