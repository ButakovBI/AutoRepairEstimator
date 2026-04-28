"""Train the damages segmentation model.

Mirror of :mod:`scripts.ml.train_parts` but with damage-specific hyperparams:
stronger ``cls_pw`` (inverse-frequency class weights), ``copy_paste``
augmentation, and slightly lower ``lr0`` to cope with noisier labels.

The data.yaml passed here must be produced by ``scripts/ml/split_dataset.py``
with ``--oversample`` for the rarest classes — otherwise ``flat_tire`` and
``rust`` will never converge.

Usage (Colab)::

    python scripts/ml/train_damages.py \\
        --data /content/damages/data.yaml \\
        --device 0
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from train_config import DAMAGES_HYPERPARAMS, resolve_runtime_overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m-seg.pt", help="Base weights")
    parser.add_argument("--epochs", type=int, default=DAMAGES_HYPERPARAMS["epochs"])
    parser.add_argument("--imgsz", type=int, default=DAMAGES_HYPERPARAMS["imgsz"])
    parser.add_argument("--batch", type=int, default=DAMAGES_HYPERPARAMS["batch"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=DAMAGES_HYPERPARAMS["patience"])
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    run_name = args.name or f"damages_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    overrides = resolve_runtime_overrides(DAMAGES_HYPERPARAMS, args)

    print("Training damages segmentation model")
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
