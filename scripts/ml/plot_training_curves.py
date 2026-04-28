"""Render unified loss + mAP curve plots from Ultralytics ``results.csv``.

Why this script exists
----------------------

Ultralytics' built-in ``results.png`` is a 10-panel grid (box-loss,
seg-loss, cls-loss, dfl-loss, precision, recall, mAP50, mAP50-95 —
each on its own subplot). For a thesis figure we usually want the
opposite: **one compact plot with all losses on the left axis and the
mAP family on the right axis**, so the reader can immediately see
whether the model has converged and where overfitting starts (val-loss
going up while train-loss keeps falling) without scanning eight tiny
panels.

This script reads one or more ``results.csv`` files produced by
Ultralytics training and writes a publication-ready PNG per run, plus a
side-by-side summary when both ``parts`` and ``damages`` runs are
passed via ``--combined``.

Inputs are flexible: pass any number of ``--run``s. Each one is a
directory containing ``results.csv``, exactly the layout Ultralytics
materialises under ``runs/<task>/<name>/``.

Usage
-----

    # One run
    python scripts/ml/plot_training_curves.py \\
        --run path/to/runs/parts_v1 \\
        --output reports/parts_training.png

    # Two runs side by side (one figure with two subplots)
    python scripts/ml/plot_training_curves.py \\
        --run path/to/runs/parts_v1 \\
        --run path/to/runs/damages_v1 \\
        --output reports/training_curves.png \\
        --combined

The script silently picks the right column names — Ultralytics renamed
them between minor versions (``train/box_loss`` → ``train/box_om``,
etc.), so it tries a small list of known aliases per metric.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt


# Column-name aliases. Ultralytics has shipped at least three variants
# across 8.0 → 8.4 (``box_loss`` / ``box_om`` / ``box``). The first
# matching column wins; missing metrics are silently skipped (the run
# might be detection-only, with no seg-loss).
_LOSS_ALIASES: dict[str, list[str]] = {
    "train/box_loss":   ["train/box_loss", "train/box_om", "train/box"],
    "train/seg_loss":   ["train/seg_loss", "train/mask_loss"],
    "train/cls_loss":   ["train/cls_loss", "train/cls"],
    "train/dfl_loss":   ["train/dfl_loss", "train/dfl"],
    "val/box_loss":     ["val/box_loss", "val/box_om", "val/box"],
    "val/seg_loss":     ["val/seg_loss", "val/mask_loss"],
    "val/cls_loss":     ["val/cls_loss", "val/cls"],
    "val/dfl_loss":     ["val/dfl_loss", "val/dfl"],
}

# Map metric → ``(column aliases, label, line style)``. Box and Mask
# share the figure: detection metrics get solid lines, mask metrics get
# dashed, so they remain distinguishable in B/W print.
_METRIC_ALIASES: dict[str, tuple[list[str], str, str]] = {
    "metrics/mAP50(B)":      (["metrics/mAP50(B)",     "metrics/mAP_0.5"],      "mAP@50 (box)",     "-"),
    "metrics/mAP50-95(B)":   (["metrics/mAP50-95(B)",  "metrics/mAP_0.5:0.95"], "mAP@50-95 (box)",  "-"),
    "metrics/mAP50(M)":      (["metrics/mAP50(M)"],                              "mAP@50 (mask)",    "--"),
    "metrics/mAP50-95(M)":   (["metrics/mAP50-95(M)"],                           "mAP@50-95 (mask)", "--"),
}


@dataclass
class RunData:
    name: str
    epochs: list[float]
    losses: dict[str, list[float]]
    metrics: dict[str, list[float]]


def _read_run(run_dir: Path) -> RunData:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if not rows:
            raise ValueError(f"results.csv is empty: {csv_path}")
        # Ultralytics often emits column names with leading/trailing
        # spaces (``" train/box_loss"``). Normalise once.
        rows = [{k.strip(): v for k, v in row.items()} for row in rows]

    # Epoch column also has variants — ``epoch`` in 8.0+, sometimes
    # 1-indexed, sometimes 0-indexed. We just take whatever's there.
    epoch_col = next(
        (c for c in ("epoch", "Epoch", "step") if c in rows[0]),
        None,
    )
    if epoch_col is None:
        # Fall back to row index — every line is one epoch in Ultralytics
        # results.csv anyway.
        epochs = [float(i + 1) for i in range(len(rows))]
    else:
        epochs = [float(r[epoch_col]) for r in rows]

    def _series(aliases: list[str]) -> list[float] | None:
        for alias in aliases:
            if alias in rows[0]:
                # Tolerate empty cells (some Ultralytics builds skip val
                # metrics on the last epoch).
                vals: list[float] = []
                for r in rows:
                    raw = r.get(alias, "")
                    try:
                        vals.append(float(raw))
                    except (TypeError, ValueError):
                        # NaN sentinel keeps the x-axis aligned without
                        # breaking matplotlib (it just leaves a gap).
                        vals.append(float("nan"))
                return vals
        return None

    losses: dict[str, list[float]] = {}
    for canonical, aliases in _LOSS_ALIASES.items():
        s = _series(aliases)
        if s is not None:
            losses[canonical] = s

    metrics: dict[str, list[float]] = {}
    for canonical, (aliases, _label, _style) in _METRIC_ALIASES.items():
        s = _series(aliases)
        if s is not None:
            metrics[canonical] = s

    if not losses and not metrics:
        raise ValueError(
            f"No known loss/metric columns in {csv_path}. "
            f"Got: {list(rows[0].keys())}"
        )

    return RunData(
        name=run_dir.name,
        epochs=epochs,
        losses=losses,
        metrics=metrics,
    )


# Loss palette: warm colors. Train solid, val dashed, so the reader's
# eye associates color with semantic group (box/seg/cls/dfl) and dash
# with phase (train/val).
_LOSS_PALETTE: dict[str, tuple[str, str, str]] = {
    "train/box_loss": ("tab:red",     "-",  "Train box loss"),
    "train/seg_loss": ("tab:orange",  "-",  "Train seg loss"),
    "train/cls_loss": ("tab:brown",   "-",  "Train cls loss"),
    "train/dfl_loss": ("tab:pink",    "-",  "Train dfl loss"),
    "val/box_loss":   ("tab:red",     "--", "Val box loss"),
    "val/seg_loss":   ("tab:orange",  "--", "Val seg loss"),
    "val/cls_loss":   ("tab:brown",   "--", "Val cls loss"),
    "val/dfl_loss":   ("tab:pink",    "--", "Val dfl loss"),
}

# Metric palette: cool colors for box, deeper greens for mask.
_METRIC_PALETTE: dict[str, str] = {
    "metrics/mAP50(B)":      "tab:blue",
    "metrics/mAP50-95(B)":   "tab:cyan",
    "metrics/mAP50(M)":      "tab:green",
    "metrics/mAP50-95(M)":   "darkgreen",
}

PlotKind = Literal["loss", "map", "combined"]

def _plot_loss_only(ax: "plt.Axes", run: RunData) -> None:
    for key, vals in run.losses.items():
        color, style, label = _LOSS_PALETTE[key]
        ax.plot(run.epochs, vals, color=color, linestyle=style, linewidth=1.6, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=8, frameon=False)


def _plot_map_only(ax: "plt.Axes", run: RunData) -> None:
    for key, vals in run.metrics.items():
        color = _METRIC_PALETTE[key]
        _, label, style = _METRIC_ALIASES[key]
        ax.plot(run.epochs, vals, color=color, linestyle=style, linewidth=1.6, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP (0–1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=8, frameon=False)


def _plot_combined(ax: "plt.Axes", run: RunData) -> None:
    ax2 = ax.twinx()

    for key, vals in run.losses.items():
        color, style, label = _LOSS_PALETTE[key]
        ax.plot(run.epochs, vals, color=color, linestyle=style, linewidth=1.6, label=label)

    for key, vals in run.metrics.items():
        color = _METRIC_PALETTE[key]
        _, label, style = _METRIC_ALIASES[key]
        ax2.plot(run.epochs, vals, color=color, linestyle=style, linewidth=1.6, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("mAP (0–1)")
    ax2.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_title(title or run.name)

    # Combine the two legends so they don't overlap. Anchored under
    # the plot — for thesis figures we want the curves to dominate.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8,
        frameon=False,
    )


def _plot_run(ax: "plt.Axes", run: RunData, kind: PlotKind, title: str | None = None) -> None:
    if kind == "loss":
        _plot_loss_only(ax, run)
    elif kind == "map":
        _plot_map_only(ax, run)
    elif kind == "combined":
        _plot_combined(ax, run)
    else:  # pragma: no cover - exhaustive guard
        raise ValueError(f"unsupported plot kind: {kind}")
    ax.set_title(title or run.name)


def _with_suffix(output: Path, suffix: str) -> Path:
    ext = output.suffix or ".png"
    return output.with_name(f"{output.stem}_{suffix}{ext}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="Path to a run directory containing results.csv. "
             "Pass --run multiple times for several runs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Base PNG path. Script writes 3 files: *_loss, *_map, *_combined.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="If multiple --run are given, draw them side-by-side in one figure. "
             "Without this flag, each run produces a separate PNG (suffix _1, _2, ...).",
    )
    parser.add_argument(
        "--title",
        action="append",
        default=None,
        help="Per-run subplot title. Repeat once per --run, in the same order. "
             "Defaults to the run directory name.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    runs = [_read_run(r) for r in args.run]
    titles = args.title or [r.name for r in runs]
    if len(titles) != len(runs):
        print(
            f"ERROR: --title was passed {len(titles)} times but there are "
            f"{len(runs)} --run entries",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for kind in ("loss", "map", "combined"):
        if args.combined or len(runs) == 1:
            ncols = len(runs)
            fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 5.5), squeeze=False)
            for ax, run, title in zip(axes[0], runs, titles):
                _plot_run(ax, run, kind=kind, title=title)
            fig.tight_layout()
            out = _with_suffix(args.output, kind)
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")
        else:
            stem = args.output.stem
            ext = args.output.suffix or ".png"
            for idx, (run, title) in enumerate(zip(runs, titles), start=1):
                fig, ax = plt.subplots(figsize=(9, 5.5))
                _plot_run(ax, run, kind=kind, title=title)
                fig.tight_layout()
                out = args.output.with_name(f"{stem}_{kind}_{idx}_{run.name}{ext}")
                fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
