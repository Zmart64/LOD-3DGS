#!/usr/bin/env python3
"""Create a side-by-side gallery of rendered vs ground-truth images."""

import argparse
import random
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a PNG grid comparing renders and corresponding GT frames."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the trained model directory (e.g. .../3D-Gaussian-Splatting/<run_id>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "test"),
        help="Which split to visualize (default: test)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=700000,
        help="Iteration folder to use inside the split (default: 700000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="How many random pairs to include (default: 20)",
    )
    parser.add_argument(
        "--pairs-per-row",
        type=int,
        default=5,
        help="How many (render, gt) pairs per row (default: 5)",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=512,
        help="Width in pixels for each image cell (default: 512, scaled proportionally)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("render_gallery.png"),
        help="Output PNG path (default: render_gallery.png)",
    )
    return parser.parse_args()


def discover_pairs(renders_dir: Path, gt_dir: Path) -> list[Path]:
    render_files = sorted(p for p in renders_dir.glob("*.png") if p.is_file())
    pairs = []
    for render_path in render_files:
        gt_path = gt_dir / render_path.name
        if gt_path.exists():
            pairs.append((render_path, gt_path))
    return pairs


def load_and_resize(path: Path, target_width: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if target_width > 0 and img.width != target_width:
        ratio = target_width / img.width
        new_size = (target_width, max(1, int(round(img.height * ratio))))
        img = img.resize(new_size, Image.BILINEAR)
    return img


def main() -> None:
    args = parse_args()

    split_root = (
        Path(args.model_dir)
        / args.split
        / f"ours_{args.iteration}"
    )
    renders_dir = split_root / "renders"
    gt_dir = split_root / "gt"

    if not renders_dir.exists():
        raise FileNotFoundError(f"No renders found at {renders_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"No ground-truth directory at {gt_dir}")

    all_pairs = discover_pairs(renders_dir, gt_dir)
    if not all_pairs:
        raise RuntimeError(f"No matching render/GT pairs inside {renders_dir}")

    sample_count = min(args.samples, len(all_pairs))
    chosen = random.sample(all_pairs, sample_count)

    per_row = max(1, args.pairs_per_row)
    rows = (sample_count + per_row - 1) // per_row
    cell_width = max(1, args.cell_width)

    # Load one image to determine the cell height after scaling.
    first_render = load_and_resize(chosen[0][0], cell_width)
    cell_height = first_render.height
    canvas_width = per_row * 2 * cell_width
    canvas_height = rows * cell_height

    gallery = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))

    for idx, (render_path, gt_path) in enumerate(chosen):
        row = idx // per_row
        col_pair = idx % per_row
        x_render = col_pair * 2 * cell_width
        y = row * cell_height

        render_img = load_and_resize(render_path, cell_width)
        gt_img = load_and_resize(gt_path, cell_width)
        # Align heights if they differ due to rounding.
        if render_img.height != cell_height:
            render_img = render_img.resize((cell_width, cell_height), Image.BILINEAR)
        if gt_img.height != cell_height:
            gt_img = gt_img.resize((cell_width, cell_height), Image.BILINEAR)

        gallery.paste(render_img, (x_render, y))
        gallery.paste(gt_img, (x_render + cell_width, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(args.output)
    print(f"[ Gallery ] Saved {sample_count} pairs to {args.output}")


if __name__ == "__main__":
    main()
