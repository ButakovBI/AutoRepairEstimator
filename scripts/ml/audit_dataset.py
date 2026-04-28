"""Audit a YOLO-format dataset for sanity issues before training.

Reports:

* Orphan images (no matching ``.txt`` label file).
* Orphan labels (``.txt`` present, image missing).
* Empty label files (0 bytes or only whitespace).
* Malformed label lines (wrong column count, bad class id, non-numeric values).
* Per-class instance counts and per-class image counts.
* Multi-label coverage (distinct classes per image distribution).

Why this lives under ``scripts/ml/`` and not in the application package:
it's a one-shot data-quality check run by hand before kicking off Colab
training. It has no runtime dependency from the bot / backend / worker.

Usage::

    python scripts/ml/audit_dataset.py \\
        --images test/images_details/Train \\
        --labels test/labels/Train \\
        --classes 12 \\
        --names door front_fender rear_fender trunk hood roof headlight \\
                front_windshield rear_windshield side_window wheel bumper

Exit code is ``0`` if no structural problems are found (orphan images or
malformed lines), ``1`` otherwise — so CI / pre-training hooks can gate on
it if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AuditReport:
    images_dir: Path
    labels_dir: Path
    classes: int
    names: list[str]

    total_images: int = 0
    total_labels: int = 0
    orphan_images: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    malformed_lines: list[tuple[str, int, str]] = field(default_factory=list)

    instances_per_class: Counter[int] = field(default_factory=Counter)
    images_per_class: Counter[int] = field(default_factory=Counter)
    classes_per_image: Counter[int] = field(default_factory=Counter)

    def has_structural_problems(self) -> bool:
        return bool(self.orphan_images or self.malformed_lines)

    def print(self) -> None:
        print(f"=== Audit: {self.images_dir} ===")
        print(f"Images: {self.total_images}")
        print(f"Labels: {self.total_labels}")
        print(f"Orphan images (no .txt): {len(self.orphan_images)}")
        print(f"Orphan labels (no image): {len(self.orphan_labels)}")
        print(f"Empty label files: {len(self.empty_labels)}")
        print(f"Malformed lines: {len(self.malformed_lines)}")
        print()
        print(f"{'class':<22}{'instances':>12}{'images':>12}")
        for i in range(self.classes):
            name = self.names[i] if i < len(self.names) else str(i)
            print(
                f"{name:<22}"
                f"{self.instances_per_class[i]:>12}"
                f"{self.images_per_class[i]:>12}"
            )
        print()
        print("Distinct classes per labelled image:")
        for k in sorted(self.classes_per_image):
            print(f"  {k} classes: {self.classes_per_image[k]} images")

        if self.orphan_images:
            print()
            print(
                "First 10 orphan images (extend by re-running with --list-orphans):"
            )
            for p in self.orphan_images[:10]:
                print(f"  {p}")

        if self.malformed_lines:
            print()
            print("First 10 malformed lines:")
            for path, lineno, line in self.malformed_lines[:10]:
                print(f"  {path}:{lineno}  {line!r}")


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _list_labels(labels_dir: Path) -> list[Path]:
    return sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt")


def _validate_line(line: str, classes: int) -> str | None:
    """Return ``None`` if line is a valid YOLO-seg entry, else error description."""

    parts = line.split()
    if len(parts) < 7:
        # class x y x y x y → at least 3 polygon points (6 coords) + class id
        return "too few values for a segmentation polygon (need class + ≥3 points)"
    try:
        cls = int(parts[0])
    except ValueError:
        return "class id is not an integer"
    if cls < 0 or cls >= classes:
        return f"class id {cls} outside [0, {classes})"
    if (len(parts) - 1) % 2 != 0:
        return "odd number of coordinate values (polygon needs pairs)"
    try:
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return "non-numeric coordinate"
    if any(not (0.0 <= v <= 1.0) for v in coords):
        return "coordinate outside [0, 1] (not normalised?)"
    return None


def audit(
    images_dir: Path,
    labels_dir: Path,
    classes: int,
    names: list[str] | None = None,
) -> AuditReport:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    report = AuditReport(
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes=classes,
        names=names or [str(i) for i in range(classes)],
    )

    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)
    report.total_images = len(images)
    report.total_labels = len(labels)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for img in images:
        if img.stem not in label_stems:
            report.orphan_images.append(str(img))

    for lbl in labels:
        if lbl.stem not in image_stems:
            report.orphan_labels.append(str(lbl))

    for lbl in labels:
        raw = lbl.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        if not lines:
            report.empty_labels.append(str(lbl))
            continue

        classes_this: set[int] = set()
        for lineno, line in enumerate(lines, start=1):
            err = _validate_line(line, classes)
            if err is not None:
                report.malformed_lines.append((str(lbl), lineno, err))
                continue
            cls = int(line.split()[0])
            report.instances_per_class[cls] += 1
            classes_this.add(cls)

        for cls in classes_this:
            report.images_per_class[cls] += 1
        report.classes_per_image[len(classes_this)] += 1

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument(
        "--list-orphans",
        action="store_true",
        help="Print full orphan lists (can be long).",
    )
    args = parser.parse_args()

    report = audit(args.images, args.labels, args.classes, args.names)
    report.print()

    if args.list_orphans:
        print()
        print("=== All orphan images ===")
        for p in report.orphan_images:
            print(p)
        print()
        print("=== All orphan labels ===")
        for p in report.orphan_labels:
            print(p)

    return 1 if report.has_structural_problems() else 0


if __name__ == "__main__":
    sys.exit(main())
