from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model weights to models/ directory")
    parser.add_argument("--model", required=True, help="Path to best.pt from training run")
    parser.add_argument("--output", required=True, help="Output path (e.g. ../docker/models/parts.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.model)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Exported {src} -> {dst}")


if __name__ == "__main__":
    main()
