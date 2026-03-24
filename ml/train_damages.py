from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for damage detection")
    parser.add_argument("--data", default="data/damages.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="damages_seg_v1")
    parser.add_argument("--project", default="runs/damages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(args.model)
    results = model.train(
        data=str(Path(__file__).parent / args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        task="segment",
    )
    print(f"Training completed. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
