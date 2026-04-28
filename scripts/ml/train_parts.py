"""Train the parts segmentation model.

Designed to run locally for smoke-testing (CPU, tiny ``epochs``) or on Google
Colab with a GPU. Loads hyperparameters from :mod:`scripts.ml.train_config`
so the two training scripts (parts + damages) stay in sync on shared tunings.

Usage — local sanity check (minutes, CPU, no real learning)::

    python scripts/ml/train_parts.py \\
        --data test/parts/data.yaml \\
        --epochs 2 --device cpu --batch 2 --imgsz 416

Usage — Colab full training (hours, GPU)::

    python scripts/ml/train_parts.py \\
        --data /content/parts/data.yaml \\
        --device 0

Output goes to ``runs/segment/parts_<timestamp>/`` relative to cwd. The
``best.pt`` weight is the one to copy into ``test/details<version>_learned/``
and wire up in ``MLWorkerConfig.parts_model_path``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from train_config import PARTS_HYPERPARAMS, resolve_runtime_overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m-seg.pt", help="Base weights")
    parser.add_argument("--epochs", type=int, default=PARTS_HYPERPARAMS["epochs"])
    parser.add_argument("--imgsz", type=int, default=PARTS_HYPERPARAMS["imgsz"])
    parser.add_argument("--batch", type=int, default=PARTS_HYPERPARAMS["batch"])
    parser.add_argument("--device", default="0", help="'cpu', '0', '0,1', etc.")
    parser.add_argument("--patience", type=int, default=PARTS_HYPERPARAMS["patience"])
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument(
        "--name",
        default=None,
        help="Run name (default: parts_<utc-timestamp>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt if run dir already exists.",
    )
    args = parser.parse_args()

    # Import late so --help works without ultralytics installed (useful on
    # machines where we only inspect config).
    from ultralytics import YOLO

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    run_name = args.name or f"parts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    overrides = resolve_runtime_overrides(PARTS_HYPERPARAMS, args)

    print(f"Training parts segmentation model")
    print(f"  data:   {args.data}")
    print(f"  base:   {args.model}")
    print(f"  run:    {args.project}/{run_name}")
    print(f"  device: {args.device}")
    print()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        project=args.project,
        name=run_name,
        resume=args.resume,
        **overrides,
    )

    print()
    print(
        f"Training finished. Best weights: "
        f"{Path(args.project) / run_name / 'weights' / 'best.pt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
