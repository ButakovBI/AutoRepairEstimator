"""Produce a stratified train/val/test split for a YOLO-format dataset.

For small multi-label segmentation datasets (<1000 images with class imbalance
~80×) the default "shuffle + slice" split is fragile — the rarest classes
can end up with 0–1 images in val, making val metrics meaningless. This
script uses a greedy iterative-stratification variant that:

1. Sorts classes by rarity (rarest first).
2. For each class, distributes its images across splits to match the target
   ratio, skipping images already placed.
3. Handles orphan images (no label file) by ignoring them — they're excluded
   from training.
4. Produces the **canonical Ultralytics layout** so training works locally
   and in Colab without any path edits.

Output layout (non-destructive — originals are copied, not moved)::

    <output>/
    ├─ images/
    │   ├─ train/
    │   ├─ val/
    │   └─ test/
    ├─ labels/
    │   ├─ train/
    │   ├─ val/
    │   └─ test/
    ├─ train.txt       ← list of train image paths, relative to <output>
    ├─ val.txt
    ├─ test.txt
    └─ data.yaml

This is the layout Ultralytics probes for automatically (it derives each
label path by replacing ``/images/`` with ``/labels/`` in the image path),
so the resulting ``data.yaml`` is portable across machines — you can zip the
whole ``<output>`` folder, upload to Colab, and training "just works".

The ``train.txt``/``val.txt``/``test.txt`` files are also emitted for
compatibility with the legacy YOLO workflow (one image path per line) and
as a convenient way to spot-check which files ended up in each split
without walking the directory tree.

Usage::

    python scripts/ml/split_dataset.py \\
        --images test/images_details/Train \\
        --labels test/labels/Train \\
        --output test/parts \\
        --classes 12 \\
        --names door front_fender rear_fender trunk hood roof headlight \\
                front_windshield rear_windshield side_window wheel bumper \\
        --ratios 0.8 0.1 0.1 \\
        --seed 42

``--oversample`` can duplicate images+labels containing specified rare
classes to compensate for class imbalance. It only affects the **train**
split — duplicating into val would inflate metrics. Duplicated files get a
numeric suffix (``xxxx__dup1.jpg``).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabelledImage:
    image_path: Path
    label_path: Path
    classes: frozenset[int]


def _collect_labelled_images(
    images_dir: Path, labels_dir: Path
) -> list[LabelledImage]:
    """Return images that have a matching label file with ≥1 class entry.

    Orphan images (no label) and empty labels are silently skipped; audit
    them separately via ``audit_dataset.py``.
    """

    label_by_stem = {p.stem: p for p in labels_dir.iterdir() if p.suffix == ".txt"}
    out: list[LabelledImage] = []

    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        lbl = label_by_stem.get(img.stem)
        if lbl is None:
            continue
        classes: set[int] = set()
        for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                classes.add(int(line.split()[0]))
            except ValueError:
                continue
        if not classes:
            continue
        out.append(LabelledImage(image_path=img.resolve(), label_path=lbl, classes=frozenset(classes)))

    return out


def _stratified_split(
    items: list[LabelledImage],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[LabelledImage], list[LabelledImage], list[LabelledImage]]:
    """Greedy iterative stratification for multi-label data.

    Algorithm:

    * ``need[split][class]`` = how many images containing ``class`` still
      need to go into ``split`` to hit the target ratio.
    * For each item (processed in order of its rarest class, to "spend"
      rare-class budget first), choose the split whose biggest class-level
      deficit is largest.
    * Tie-break by smallest current split size (so ratios stay close to
      target on small datasets).
    """

    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "ratios must sum to 1.0"

    rng = random.Random(seed)

    class_to_items: dict[int, list[LabelledImage]] = defaultdict(list)
    for it in items:
        for c in it.classes:
            class_to_items[c].append(it)

    total_per_class: Counter[int] = Counter({c: len(v) for c, v in class_to_items.items()})
    split_ratios = {"train": train_r, "val": val_r, "test": test_r}

    # Target image-count per split per class.
    targets = {
        s: {c: total_per_class[c] * r for c in total_per_class}
        for s, r in split_ratios.items()
    }
    current: dict[str, Counter[int]] = {s: Counter() for s in split_ratios}
    split_size: Counter[str] = Counter()

    # Process items ordered by (rarest class they contain, then random tiebreak).
    # This front-loads images with rare classes so they get placed while
    # budget is still large everywhere — crucial for classes with <30 images.
    rarity = {c: total_per_class[c] for c in total_per_class}
    keyed = [(min(rarity[c] for c in it.classes), rng.random(), it) for it in items]
    keyed.sort(key=lambda k: (k[0], k[1]))

    placement: dict[str, list[LabelledImage]] = {s: [] for s in split_ratios}

    for _, _, it in keyed:
        # Ratio-normalised deficit: (target - current) / ratio. Without this
        # normalisation train's absolute deficit (total*0.8) always dominates
        # val/test at the start, and all rare-class images end up in train.
        # Empty splits (ratio=0) are filtered out. A zero-ratio split gets no
        # items at all, which is what the user asked for.
        def deficit(split: str) -> float:
            r = split_ratios[split]
            if r <= 0:
                return float("-inf")
            return max(
                (targets[split][c] - current[split][c]) / r for c in it.classes
            )

        scored = sorted(
            split_ratios.keys(),
            # Prefer: (1) split whose normalised deficit is largest, (2) split
            # whose current size is furthest BELOW its expected pro-rata share,
            # (3) random. Step (2) stops val/test drifting above their ratio
            # once all per-class deficits go to zero.
            key=lambda s: (
                -deficit(s),
                split_size[s] - split_ratios[s] * sum(split_size.values() or [0]),
                rng.random(),
            ),
        )
        chosen = scored[0]
        placement[chosen].append(it)
        split_size[chosen] += 1
        for c in it.classes:
            current[chosen][c] += 1

    return placement["train"], placement["val"], placement["test"]


def _oversample_factors(
    train: list[LabelledImage],
    oversample: dict[int, int],
) -> list[int]:
    """Return per-item duplication factor for each train item.

    ``oversample = {class_id: factor}``. An image gets the MAX factor of its
    classes (not the product) so multi-class images don't explode.
    Factor of 1 = no duplication (original only). Factor of N = original
    + (N-1) copies.
    """

    return [
        max((oversample.get(c, 1) for c in it.classes), default=1)
        for it in train
    ]


def _materialise_split(
    items: list[LabelledImage],
    split_name: str,
    output: Path,
    duplicate_factors: list[int] | None = None,
) -> list[Path]:
    """Copy images + labels into ``output/{images,labels}/<split_name>/``.

    Using copy (not move / symlink) because:

    * On Windows, symlink creation requires admin privileges.
    * Zipping the output folder for Colab upload needs real files.
    * The source tree under ``test/`` stays intact as a backup.

    If ``duplicate_factors`` is provided, it must be the same length as
    ``items``; extra copies beyond the first are written with a
    ``__dupN`` suffix in their stem. ``None`` means factor 1 for all.

    Returns the list of written image paths (in materialisation order)
    so the caller can build the matching ``<split>.txt`` file without
    walking the tree again — keeps ordering deterministic, which matters
    for reproducible hash-based caches.
    """

    images_dir = output / "images" / split_name
    labels_dir = output / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    factors = duplicate_factors or [1] * len(items)
    assert len(factors) == len(items)

    written: list[Path] = []
    for it, factor in zip(items, factors):
        for copy_idx in range(max(1, factor)):
            suffix = "" if copy_idx == 0 else f"__dup{copy_idx}"
            img_dst = images_dir / f"{it.image_path.stem}{suffix}{it.image_path.suffix}"
            lbl_dst = labels_dir / f"{it.image_path.stem}{suffix}.txt"
            # ``copy2`` preserves mtime so YOLO's dataset cache invalidates
            # correctly only when the source actually changes.
            shutil.copy2(it.image_path, img_dst)
            shutil.copy2(it.label_path, lbl_dst)
            written.append(img_dst)
    return written


def _write_txt_list(
    txt_path: Path,
    image_paths: list[Path],
    dataset_root: Path,
) -> None:
    """Write ``train.txt`` / ``val.txt`` / ``test.txt`` with relative paths.

    Paths are stored relative to ``dataset_root`` so the file stays valid
    after the dataset folder is copied to a different machine (e.g. unzipped
    in Colab at ``/content/parts``). Uses POSIX separators (``images/train/1.jpg``)
    unconditionally — that works on Windows too because Ultralytics
    normalises separators internally.
    """

    lines = [
        img.relative_to(dataset_root).as_posix() for img in image_paths
    ]
    txt_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _clean_output(output: Path) -> None:
    """Wipe previous split artefacts under ``output``.

    Removes ``images/``, ``labels/``, the three ``<split>.txt`` files and
    any leftover ``splits/`` from pre-1.1 versions of this script. Leaves
    ``data.yaml`` in place (it gets overwritten unconditionally later) and
    does not touch any other sibling files — the user might keep notes or
    raw unlabelled images here.
    """

    for sub in ("images", "labels", "splits"):
        victim = output / sub
        if victim.exists():
            shutil.rmtree(victim)
    for txt in ("train.txt", "val.txt", "test.txt"):
        victim_txt = output / txt
        if victim_txt.exists():
            victim_txt.unlink()


def _write_data_yaml(
    yaml_path: Path,
    output: Path,
    names: list[str],
) -> None:
    """Emit Ultralytics data.yaml for the canonical images/{train,val,test} layout.

    We deliberately **omit** the ``path:`` key. In Ultralytics ≥8.4,
    ``check_det_dataset`` resolves the dataset root as::

        Path(data.get("path") or Path(data["yaml_file"]).parent)

    so when ``path`` is absent the folder containing ``data.yaml`` becomes
    the root. That makes a zip of ``<output>/`` portable across Windows,
    WSL, Linux, and Colab — no more stale ``D:\\...`` or ``/mnt/d/...``
    entries breaking training after upload.

    ``train`` / ``val`` / ``test`` are then resolved as
    ``<yaml_dir>/images/train`` etc.
    """

    _ = output  # kept for API symmetry with callers; root is yaml_path.parent
    body_names = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    content = (
        "# Auto-generated by scripts/ml/split_dataset.py — DO NOT edit by hand.\n"
        "# Re-run the script to refresh splits.\n"
        "# No `path:` key — dataset root = directory containing this file (Ultralytics 8.4+).\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        f"names:\n{body_names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def _print_split_stats(
    names: list[str],
    train: list[LabelledImage],
    val: list[LabelledImage],
    test: list[LabelledImage],
) -> None:
    def instance_counts(items: list[LabelledImage]) -> Counter[int]:
        # Re-read labels to count instances (not just distinct classes per image).
        counts: Counter[int] = Counter()
        for it in items:
            for line in it.label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    counts[int(line.split()[0])] += 1
                except ValueError:
                    continue
        return counts

    tr, va, te = (instance_counts(x) for x in (train, val, test))

    print()
    print("Split stats (image count | instances per class):")
    print(f"  train: {len(train):>4} images")
    print(f"  val:   {len(val):>4} images")
    print(f"  test:  {len(test):>4} images")
    print()
    print(f"{'class':<22}{'train':>10}{'val':>8}{'test':>8}{'train%':>10}")
    for i, n in enumerate(names):
        total = tr[i] + va[i] + te[i]
        frac = f"{(tr[i] / total * 100):.1f}" if total else "-"
        print(f"{n:<22}{tr[i]:>10}{va[i]:>8}{te[i]:>8}{frac:>10}")


def _parse_oversample(raw: list[str] | None, names: list[str]) -> dict[int, int]:
    """Parse ``--oversample`` spec: ``class=factor`` pairs."""

    if not raw:
        return {}
    name_to_id = {n: i for i, n in enumerate(names)}
    out: dict[int, int] = {}
    for spec in raw:
        key, _, value = spec.partition("=")
        if not key or not value:
            raise ValueError(f"bad --oversample spec {spec!r}, expected 'class=factor'")
        try:
            cls_id = int(key)
        except ValueError:
            if key not in name_to_id:
                raise ValueError(f"unknown class name {key!r} in --oversample")
            cls_id = name_to_id[key]
        out[cls_id] = int(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Dataset root (will contain splits/ and data.yaml)")
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--oversample",
        nargs="*",
        help="Duplicate train images containing given classes. Format: 'class=factor' "
             "(e.g. --oversample flat_tire=5 rust=3). Only affects train.",
    )
    args = parser.parse_args()

    if len(args.names) != args.classes:
        print(
            f"ERROR: --classes={args.classes} but got {len(args.names)} names",
            file=sys.stderr,
        )
        return 2

    items = _collect_labelled_images(args.images, args.labels)
    if not items:
        print("ERROR: no labelled images found", file=sys.stderr)
        return 2

    print(f"Labelled images to split: {len(items)}")

    train, val, test = _stratified_split(items, tuple(args.ratios), args.seed)

    oversample = _parse_oversample(args.oversample, args.names)
    train_factors: list[int] | None = None
    if oversample:
        train_factors = _oversample_factors(train, oversample)
        duplicated_total = sum(train_factors)
        print(
            f"Oversampling will expand train from {len(train)} originals "
            f"to {duplicated_total} image copies on disk"
        )

    _clean_output(args.output)
    train_imgs = _materialise_split(
        train, "train", args.output, duplicate_factors=train_factors
    )
    val_imgs = _materialise_split(val, "val", args.output)
    test_imgs = _materialise_split(test, "test", args.output)

    _write_txt_list(args.output / "train.txt", train_imgs, args.output)
    _write_txt_list(args.output / "val.txt", val_imgs, args.output)
    _write_txt_list(args.output / "test.txt", test_imgs, args.output)

    _write_data_yaml(args.output / "data.yaml", args.output, args.names)

    _print_split_stats(args.names, train, val, test)
    print()
    print(f"Wrote {args.output / 'data.yaml'}")
    print(
        f"Wrote images/, labels/, train.txt, val.txt, test.txt under "
        f"{args.output} — zip this folder for Colab."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
