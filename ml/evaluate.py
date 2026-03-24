from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument("--model", required=True, help="Path to trained .pt file")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(args.model)
    metrics = model.val(data=str(Path(args.data).resolve()), split=args.split)
    print(f"mAP@0.5: {metrics.seg.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"Per-class results: {metrics.seg.maps}")


if __name__ == "__main__":
    main()
