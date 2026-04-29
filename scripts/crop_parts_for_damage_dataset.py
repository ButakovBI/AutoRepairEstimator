from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class CropBox:
    class_id: int
    left: int
    top: int
    right: int
    bottom: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop car parts from source images using YOLO labels "
            "(supports bbox and polygon segmentation)."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory with label .txt files (one file per image).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory with source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cropped parts will be saved.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image extensions for lookup by stem (default: common image extensions).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=8,
        help="Skip crops smaller than this size in pixels (default: 8).",
    )
    return parser.parse_args()


def _find_image(images_dir: Path, stem: str, exts: list[str]) -> Path | None:
    for ext in exts:
        image_path = images_dir / f"{stem}{ext.lower()}"
        if image_path.exists():
            return image_path
        image_path_upper = images_dir / f"{stem}{ext.upper()}"
        if image_path_upper.exists():
            return image_path_upper
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _line_to_box(line: str, width: int, height: int, min_size: int) -> CropBox | None:
    raw = line.strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 5:
        return None

    class_id = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]

    # YOLO bbox: class x_center y_center box_w box_h
    if len(coords) == 4:
        x_center, y_center, box_w, box_h = coords
        left_n = x_center - box_w / 2.0
        right_n = x_center + box_w / 2.0
        top_n = y_center - box_h / 2.0
        bottom_n = y_center + box_h / 2.0
    else:
        # YOLO segmentation polygon: class x1 y1 x2 y2 ...
        if len(coords) % 2 != 0:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        left_n = min(xs)
        right_n = max(xs)
        top_n = min(ys)
        bottom_n = max(ys)

    left = int(_clamp(left_n, 0.0, 1.0) * width)
    right = int(_clamp(right_n, 0.0, 1.0) * width)
    top = int(_clamp(top_n, 0.0, 1.0) * height)
    bottom = int(_clamp(bottom_n, 0.0, 1.0) * height)

    # Ensure non-empty valid crop
    if right <= left or bottom <= top:
        return None
    if right - left < min_size or bottom - top < min_size:
        return None

    return CropBox(class_id=class_id, left=left, top=top, right=right, bottom=bottom)


def main() -> None:
    args = parse_args()
    labels_dir: Path = args.labels_dir
    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir
    exts: list[str] = [e if e.startswith(".") else f".{e}" for e in args.exts]

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {labels_dir}")

    total_labels = 0
    total_saved = 0
    total_missing_images = 0
    total_bad_lines = 0

    for label_file in label_files:
        image_path = _find_image(images_dir=images_dir, stem=label_file.stem, exts=exts)
        if image_path is None:
            total_missing_images += 1
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            lines = label_file.read_text(encoding="utf-8").splitlines()
            total_labels += len(lines)

            for idx, line in enumerate(lines):
                box = _line_to_box(line=line, width=width, height=height, min_size=args.min_size)
                if box is None:
                    total_bad_lines += 1
                    continue

                crop = image.crop((box.left, box.top, box.right, box.bottom))
                out_name = f"{label_file.stem}__obj{idx:03d}__cls{box.class_id}.jpg"
                crop.save(output_dir / out_name, format="JPEG", quality=95)
                total_saved += 1

    print(f"Labels read: {total_labels}")
    print(f"Crops saved: {total_saved}")
    print(f"Missing source images: {total_missing_images}")
    print(f"Skipped/invalid lines: {total_bad_lines}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
