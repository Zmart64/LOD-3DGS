#!/usr/bin/env python3
"""Preprocess 360° LiDAR + RGB captures into a NeRF-style dataset for LOD-3DGS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import py360convert  # type: ignore
import yaml
from plyfile import PlyData, PlyElement
from tqdm import tqdm

try:
    import laspy
except ImportError:  # pragma: no cover - optional dependency
    laspy = None


@dataclass
class PanoramaPose:
    image_name: str
    timestamp: float
    position: np.ndarray  # meters, (x, y, z)
    yaw: float  # degrees
    pitch: float  # degrees
    roll: float  # degrees

    def to_dict(self) -> Dict[str, float]:
        return {
            "timestamp": self.timestamp,
            "position": self.position.tolist(),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert panoramic LiDAR captures to perspective images + transforms.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("extrinsic_estimation.yaml"),
        help="Path to extrinsic/config YAML (default: extrinsic_estimation.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for the processed dataset (overrides config.data.processed_root)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of panoramas for debugging",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=8,
        help="Assign every Nth panorama to transforms_test.json (default: 8, like LLFF)",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.75,
        help="Max timestamp difference (seconds) for matching LiDAR trajectory to panorama",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel panorama workers (default: 1)",
    )
    parser.add_argument(
        "--write-depths",
        action="store_true",
        help="Also generate per-view depth maps from LiDAR (stored under depths/)",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Meters-to-integer scale when saving depth PNGs (default: 1000 => millimeters)",
    )
    parser.add_argument(
        "--depth-max-distance",
        type=float,
        default=None,
        help="Ignore LiDAR points farther than this distance in meters when rasterizing depth (default: no limit)",
    )
    parser.add_argument(
        "--depth-splat-radius",
        type=int,
        default=0,
        help="Pixel radius for depth splatting (0 = single pixel, 1 = 3x3, etc.)",
    )
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "data" not in cfg:
        raise ValueError("Config must contain a top-level 'data' section")
    return cfg


def read_panorama_metadata(path: Path) -> List[PanoramaPose]:
    records: List[PanoramaPose] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip().strip('"')
            try:
                timestamp = float(row[1])
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                yaw = float(row[5])
                pitch = float(row[6])
                roll = float(row[7])
            except (ValueError, IndexError):
                continue
            records.append(
                PanoramaPose(
                    image_name=name,
                    timestamp=timestamp,
                    position=np.array([x, y, z], dtype=np.float64),
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )
    return records


def read_base_poses(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    positions = []
    yaws = []
    pitchs = []
    rolls = []
    float_pat = re.compile(r"-?\d+\.\d+|-?\d+")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            nums = [float(x) for x in float_pat.findall(line)]
            if len(nums) < 7:
                continue
            timestamps.append(nums[0])
            positions.append(nums[1:4])
            yaws.append(nums[4])
            pitchs.append(nums[5])
            rolls.append(nums[6])
    if not timestamps:
        raise ValueError(f"No poses parsed from {path}")
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(yaws, dtype=np.float64),
        np.asarray(pitchs, dtype=np.float64),
        np.asarray(rolls, dtype=np.float64),
    )


def euler_deg_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return rotation matrix (camera-to-world) using ZYX order."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def format_angle(angle: float) -> str:
    sign = "p" if angle >= 0 else "m"
    return f"{sign}{abs(int(round(angle))):03d}"


def build_view_grid(horizontal_views: int, vertical_views: int, step_deg: float) -> List[Tuple[float, float]]:
    if horizontal_views <= 0:
        raise ValueError("horizontal_views must be > 0")
    yaw_step = 360.0 / horizontal_views
    yaw_offsets = [i * yaw_step for i in range(horizontal_views)]

    pitch_offsets = [0.0]
    if vertical_views > 0:
        direction = 1
        level = 1
        for _ in range(vertical_views):
            pitch_offsets.append(direction * level * step_deg)
            direction *= -1
            if direction > 0:
                level += 1
    pitch_offsets = sorted(set(pitch_offsets))

    combos = []
    for pitch in pitch_offsets:
        for yaw in yaw_offsets:
            combos.append((yaw, pitch))
    return combos


def compute_camera_pose(
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    camera_offset: np.ndarray,
    camera_rot_offset: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    position = base_position + base_rotation @ camera_offset
    rotation = base_rotation @ camera_rot_offset
    return position, rotation


def rotation_from_offsets(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return euler_deg_to_matrix(yaw, pitch, roll)


def make_transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"{path} already exists. Use --overwrite to rebuild.")
    path.mkdir(parents=True, exist_ok=True)


def load_lidar_points(lidar_path: Path, max_points: int, use_colors: bool) -> Tuple[np.ndarray, np.ndarray]:
    ext = lidar_path.suffix.lower()
    if ext == ".ply":
        ply = PlyData.read(lidar_path)
        verts = ply["vertex"]
        xyz = np.vstack([verts[axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
        if use_colors and {"red", "green", "blue"}.issubset(verts.data.dtype.names):
            rgb = np.vstack([verts[c] for c in ("red", "green", "blue")]).T.astype(np.float32)
        else:
            rgb = np.ones_like(xyz) * 127
    elif ext in {".las", ".laz"}:
        if laspy is None:
            raise RuntimeError("laspy is required to read LAS/LAZ files")
        las = laspy.read(str(lidar_path))
        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        if use_colors and hasattr(las, "red"):
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
            if rgb.max() > 255:
                rgb = rgb / 65535.0 * 255.0
        else:
            intensity = getattr(las, "intensity", np.full(len(xyz), 0.5))
            rgb = np.repeat(intensity[:, None], 3, axis=1)
    else:
        raise ValueError(f"Unsupported LiDAR file extension: {ext}")

    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    return xyz, np.clip(rgb, 0, 255)


def write_point_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    normals = np.zeros_like(xyz, dtype=np.float32)
    vertex_data = np.empty(len(xyz), dtype=[
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertex_data["red"], vertex_data["green"], vertex_data["blue"] = (
        rgb[:, 0].astype(np.uint8),
        rgb[:, 1].astype(np.uint8),
        rgb[:, 2].astype(np.uint8),
    )
    PlyData([PlyElement.describe(vertex_data, "vertex")]).write(path)


def render_depth_map(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    rotation_world: np.ndarray,
    focal_length: float,
    image_size: int,
    max_distance: Optional[float],
    splat_radius: int,
) -> np.ndarray:
    if points_world is None or len(points_world) == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    rel = points_world - camera_position[None, :]
    if max_distance is not None:
        mask = np.sum(rel * rel, axis=1) <= max_distance * max_distance
        if not np.any(mask):
            return np.zeros((image_size, image_size), dtype=np.float32)
        rel = rel[mask]

    rotation_world = rotation_world.astype(np.float64, copy=False)
    rel = rel.astype(np.float64, copy=False)
    rotation_w2c = rotation_world.T
    cam = rel @ rotation_w2c
    z = cam[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return np.zeros((image_size, image_size), dtype=np.float32)
    cam = cam[valid]
    z = cam[:, 2]
    x = cam[:, 0]
    y = cam[:, 1]
    pixels_x = focal_length * (x / z) + image_size / 2.0
    pixels_y = focal_length * (y / z) + image_size / 2.0
    inside = (
        (pixels_x >= 0)
        & (pixels_x < image_size)
        & (pixels_y >= 0)
        & (pixels_y < image_size)
    )
    if not np.any(inside):
        return np.zeros((image_size, image_size), dtype=np.float32)
    px = pixels_x[inside].astype(np.int32)
    py = pixels_y[inside].astype(np.int32)
    z = z[inside]
    z = z.astype(np.float32, copy=False)
    depth = np.full((image_size, image_size), np.inf, dtype=np.float32)
    np.minimum.at(depth, (py, px), z)
    if splat_radius > 0:
        offsets = [
            (dy, dx)
            for dy in range(-splat_radius, splat_radius + 1)
            for dx in range(-splat_radius, splat_radius + 1)
            if not (dx == 0 and dy == 0)
        ]
        for dy, dx in offsets:
            px_off = px + dx
            py_off = py + dy
            inside_off = (
                (px_off >= 0)
                & (px_off < image_size)
                & (py_off >= 0)
                & (py_off < image_size)
            )
            if not np.any(inside_off):
                continue
            np.minimum.at(depth, (py_off[inside_off], px_off[inside_off]), z[inside_off])
    depth[np.isinf(depth)] = 0.0
    return depth


def process_panorama_task(task: Dict) -> Optional[Dict]:
    pano = task["pano"]
    pano_name = pano["image_name"]
    pano_timestamp = pano["timestamp"]
    pano_position = np.array(pano["position"], dtype=np.float64)
    pano_yaw = pano["yaw"]
    pano_pitch = pano["pitch"]
    pano_roll = pano["roll"]

    images_dir = Path(task["images_dir"])
    output_images = Path(task["output_images"])
    max_time_diff = task["max_time_diff"]
    view_angles = task["view_angles"]
    cube_res = task["cube_res"]
    fov_deg = task["fov_deg"]
    focal_length = task.get("focal_length", cube_res / (2.0 * math.tan(math.radians(fov_deg) / 2.0)))
    write_depths = task.get("write_depths", False)
    depth_dir = Path(task["depth_dir"]) if write_depths and task.get("depth_dir") else None
    depth_scale = float(task.get("depth_scale", 1000.0))
    depth_max_distance = task.get("depth_max_distance")
    depth_splat_radius = int(task.get("depth_splat_radius", 0))
    lidar_points = task.get("lidar_points")

    base_ts = np.asarray(task["base_ts"], dtype=np.float64)
    base_positions = np.asarray(task["base_positions"], dtype=np.float64)
    base_yaws = np.asarray(task["base_yaws"], dtype=np.float64)
    base_pitchs = np.asarray(task["base_pitchs"], dtype=np.float64)
    base_rolls = np.asarray(task["base_rolls"], dtype=np.float64)
    camera_offset = np.asarray(task["camera_offset"], dtype=np.float64)
    camera_rot_offset = np.asarray(task["camera_rot_offset"], dtype=np.float64)

    pano_img_path = images_dir / pano_name
    if not pano_img_path.exists():
        print(f"[WARN] Missing panorama {pano_img_path}, skipping")
        return None

    base_match_idx = np.searchsorted(base_ts, pano_timestamp)
    best_idx = None
    best_dt = None
    for candidate in [base_match_idx - 1, base_match_idx, base_match_idx + 1]:
        if 0 <= candidate < len(base_ts):
            dt = abs(base_ts[candidate] - pano_timestamp)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = candidate

    use_base = best_dt is not None and best_dt <= max_time_diff and best_idx is not None
    if use_base:
        base_position = base_positions[best_idx]
        base_rotation = euler_deg_to_matrix(base_yaws[best_idx], base_pitchs[best_idx], base_rolls[best_idx])
        camera_position, camera_rotation = compute_camera_pose(
            base_position, base_rotation, camera_offset, camera_rot_offset
        )
        source = "trajectory"
    else:
        camera_position = pano_position
        camera_rotation = euler_deg_to_matrix(pano_yaw, pano_pitch, pano_roll)
        source = "metadata"

    pano_bgr = cv2.imread(str(pano_img_path), cv2.IMREAD_COLOR)
    if pano_bgr is None:
        print(f"[WARN] Failed to load {pano_img_path}, skipping")
        return None
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    pano_record = {
        "image_name": pano_name,
        "timestamp": pano_timestamp,
        "pose_source": source,
        "views": [],
    }

    for yaw_offset, pitch_offset in view_angles:
        rot_offset = rotation_from_offsets(yaw_offset, pitch_offset, 0.0)
        rotation_world = camera_rotation @ rot_offset
        transform = make_transform(rotation_world, camera_position)

        face = py360convert.e2p(
            pano_rgb,
            fov_deg=fov_deg,
            u_deg=yaw_offset,
            v_deg=pitch_offset,
            out_hw=(cube_res, cube_res),
        )
        face = np.clip(face, 0, 255).astype(np.uint8, copy=False)
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        yaw_tag = format_angle(yaw_offset)
        pitch_tag = format_angle(pitch_offset)
        face_name = f"{Path(pano_name).stem}_{yaw_tag}_{pitch_tag}.png"
        face_path = output_images / face_name
        cv2.imwrite(str(face_path), face_bgr)

        frame_entry = {
            "file_path": f"images/{face_name}",
            "transform_matrix": transform.tolist(),
            "timestamp": pano_timestamp,
            "panorama": pano_name,
            "pose_source": source,
        }

        if write_depths and depth_dir is not None and lidar_points is not None:
            depth = render_depth_map(
                lidar_points,
                camera_position,
                rotation_world,
                focal_length,
                cube_res,
                depth_max_distance,
                depth_splat_radius,
            )
            depth_int = np.clip(depth * depth_scale, 0, 65535).astype(np.uint16, copy=False)
            depth_path = depth_dir / face_name
            cv2.imwrite(str(depth_path), depth_int)
            frame_entry["depth_file_path"] = f"depths/{face_name}"

        pano_record["views"].append(frame_entry)

    return {
        "order": task["order"],
        "pano_record": pano_record,
        "matched_with_trajectory": 1 if source == "trajectory" else 0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg_dir = args.config.resolve().parent

    data_cfg = cfg["data"]
    dataset_root = resolve_path(cfg_dir, data_cfg["dataset_root"])
    processed_root = (
        resolve_path(cfg_dir, data_cfg["processed_root"])
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    ensure_empty_dir(processed_root, args.overwrite)

    images_dir = dataset_root / data_cfg.get("rgb_dir", "images360")
    pose_file = dataset_root / data_cfg.get("pose_file", "images360_coordinates.txt")
    base_pose_file = dataset_root / data_cfg.get("base_pose_file", "trajectory-ts.txt")

    panoramas = sorted(read_panorama_metadata(pose_file), key=lambda p: p.timestamp)
    if not panoramas:
        raise RuntimeError(f"No panoramas parsed from {pose_file}")

    base_ts, base_positions, base_yaws, base_pitchs, base_rolls = read_base_poses(base_pose_file)

    camera_offset = np.array(data_cfg.get("camera_offset", [0, 0, 0]), dtype=np.float64)
    camera_rot_offset = rotation_from_offsets(
        data_cfg.get("camera_yaw_offset_deg", 0.0),
        data_cfg.get("camera_pitch_offset_deg", 0.0),
        data_cfg.get("camera_roll_offset_deg", 0.0),
    )

    pose_stride = max(1, int(data_cfg.get("pose_stride", 1)))
    max_images = args.max_images or len(panoramas)

    horizontal_views = int(data_cfg.get("horizontal_views", 6))
    vertical_views = int(data_cfg.get("vertical_views", 0))
    vertical_pitch = float(data_cfg.get("vertical_pitch_deg", 45.0))
    view_angles = build_view_grid(horizontal_views, vertical_views, vertical_pitch)

    cube_res = int(data_cfg.get("cube_face_resolution", 1024))
    fov_deg = float(data_cfg.get("cube_face_fov_deg", 90.0))
    fl = cube_res / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

    output_images = processed_root / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    metadata_dir = processed_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    depth_dir = processed_root / "depths" if args.write_depths else None
    if depth_dir is not None:
        depth_dir.mkdir(parents=True, exist_ok=True)

    lidar_file = dataset_root / data_cfg.get("lidar_file", "")
    lidar_points: Optional[np.ndarray] = None
    lidar_colors: Optional[np.ndarray] = None
    if lidar_file.exists():
        lidar_points, lidar_colors = load_lidar_points(
            lidar_file,
            int(data_cfg.get("lidar_max_points", 200_000)),
            bool(data_cfg.get("use_lidar_colors", True)),
        )
    else:
        print(f"[WARN] LiDAR file {lidar_file} not found; skipping points3d.ply export")

    if args.write_depths and (lidar_points is None or len(lidar_points) == 0):
        raise RuntimeError("Cannot write depth maps because LiDAR points are unavailable.")

    selected: List[Tuple[int, PanoramaPose]] = []
    for pano_idx, pano in enumerate(panoramas):
        if pano_idx % pose_stride != 0:
            continue
        if len(selected) >= max_images:
            break
        selected.append((pano_idx, pano))

    base_ts_list = base_ts.tolist()
    base_positions_list = base_positions.tolist()
    base_yaws_list = base_yaws.tolist()
    base_pitchs_list = base_pitchs.tolist()
    base_rolls_list = base_rolls.tolist()
    camera_offset_list = camera_offset.tolist()
    camera_rot_offset_list = camera_rot_offset.tolist()

    tasks: List[Dict] = []
    for order_idx, (_, pano) in enumerate(selected):
        task = {
                "order": order_idx,
                "pano": {
                    "image_name": pano.image_name,
                    "timestamp": pano.timestamp,
                    "position": pano.position.tolist(),
                    "yaw": pano.yaw,
                    "pitch": pano.pitch,
                    "roll": pano.roll,
                },
                "images_dir": str(images_dir),
                "output_images": str(output_images),
                "max_time_diff": args.max_time_diff,
                "view_angles": view_angles,
                "cube_res": cube_res,
                "fov_deg": fov_deg,
                "base_ts": base_ts_list,
                "base_positions": base_positions_list,
                "base_yaws": base_yaws_list,
                "base_pitchs": base_pitchs_list,
                "base_rolls": base_rolls_list,
                "camera_offset": camera_offset_list,
                "camera_rot_offset": camera_rot_offset_list,
                "focal_length": fl,
                "write_depths": args.write_depths,
                "depth_dir": str(depth_dir) if depth_dir is not None else None,
                "depth_scale": args.depth_scale,
                "depth_max_distance": args.depth_max_distance,
                "depth_splat_radius": args.depth_splat_radius,
                "lidar_points": lidar_points,
            }
        tasks.append(task)

    results: List[Optional[Dict]] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Panoramas"):
            results.append(process_panorama_task(task))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_panorama_task, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Panoramas"):
                results.append(fut.result())

    processed = [res for res in results if res is not None]
    processed.sort(key=lambda r: r["order"])

    test_every = max(1, args.test_every)
    frames_train: List[Dict] = []
    frames_test: List[Dict] = []
    report: Dict[str, List] = {"panoramas": []}

    pano_count = len(processed)
    matched_with_trajectory = sum(res["matched_with_trajectory"] for res in processed)

    if len(processed) != len(tasks):
        skipped = len(tasks) - len(processed)
        print(f"[WARN] Skipped {skipped} panoramas due to missing inputs.")

    for rel_idx, result in enumerate(processed):
        pano_record = result["pano_record"]
        report["panoramas"].append(pano_record)
        target_frames = frames_test if (rel_idx % test_every == 0) else frames_train
        target_frames.extend(pano_record["views"])

    stats = {
        "total_panoramas": pano_count,
        "frames_train": len(frames_train),
        "frames_test": len(frames_test),
        "matched_with_trajectory": matched_with_trajectory,
        "matched_with_metadata": pano_count - matched_with_trajectory,
        "view_angles": view_angles,
        "cube_face_resolution": cube_res,
        "has_depths": bool(args.write_depths),
    }
    if args.write_depths:
        stats["depth_scale"] = args.depth_scale
        stats["depth_max_distance"] = args.depth_max_distance
        stats["depth_splat_radius"] = args.depth_splat_radius

    with open(metadata_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "panoramas": report["panoramas"]}, f, indent=2)

    transforms_common = {
        "w": cube_res,
        "h": cube_res,
        "camera_angle_x": math.radians(fov_deg),
        "camera_angle_y": math.radians(fov_deg),
        "fl_x": fl,
        "fl_y": fl,
        "cx": cube_res / 2.0,
        "cy": cube_res / 2.0,
    }

    aabb = None
    if lidar_points is not None and lidar_colors is not None:
        write_point_cloud(processed_root / "points3d.ply", lidar_points, lidar_colors)
        aabb = [lidar_points.min(axis=0).tolist(), lidar_points.max(axis=0).tolist()]

    if aabb is not None:
        transforms_common["aabb"] = aabb

    with open(processed_root / "transforms_train.json", "w", encoding="utf-8") as f:
        json.dump({**transforms_common, "frames": frames_train}, f, indent=2)

    if frames_test:
        with open(processed_root / "transforms_test.json", "w", encoding="utf-8") as f:
            json.dump({**transforms_common, "frames": frames_test}, f, indent=2)

    shutil.copy2(args.config, processed_root / Path(args.config).name)

    print("--- Preprocessing complete ---")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
