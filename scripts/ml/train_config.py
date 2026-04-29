"""Hyperparameter config for the parts/damages YOLOv8-seg training runs.

Single source of truth shared by ``train_parts.py`` and ``train_damages.py``.
Values here are the baseline documented in
``docs/ml/annotation_and_training_guide.md`` §4 — change them only after you
have read that section and understood why each value was chosen.

Two dictionaries are exported:

* ``PARTS_HYPERPARAMS`` — tuned for the 12-class parts dataset (~300 images
  when augmented with the rare-class dozbor).
* ``DAMAGES_HYPERPARAMS`` — tuned for the 8-class damages dataset (~300
  crops). Stronger ``cls_pw`` (inverse-frequency class weights) +
  ``copy_paste`` because the class imbalance here is much worse
  (``scratch``:``flat_tire`` ≈ 80:1).

``fl_gamma`` was removed from Ultralytics ``default.yaml`` in recent
releases (8.4+); use ``cls_pw`` for class imbalance instead.

Both are plain dicts — not pydantic — because they're passed straight to
``model.train(**hparams)`` and Ultralytics accepts all these keys.
"""

from __future__ import annotations

import argparse
from typing import Any


_COMMON: dict[str, Any] = {
    # Training duration
    "epochs": 150,
    "patience": 30,
    "imgsz": 640,
    "batch": 16,
    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "cos_lr": True,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    # Regularisation / stop-condition
    "close_mosaic": 15,  # last N epochs disable mosaic so model "lands"
    "dropout": 0.0,
    # Augmentations — see guide §4 for justification
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 8.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,  # breaks segmentation masks
    # Bookkeeping
    "plots": True,
    "save": True,
    "save_period": 25,  # checkpoint every 25 epochs in case Colab disconnects
    "seed": 42,
    "deterministic": True,
    "verbose": True,
    "workers": 4,
}


PARTS_HYPERPARAMS: dict[str, Any] = {
    **_COMMON,
    # Class-weight power: 0=off, 1=full inverse-frequency weighting.
    # Mild value nudges rare parts (rear_windshield, roof) without the
    # removed ``fl_gamma`` knob from older Ultralytics.
    "cls_pw": 0.25,
    # copy_paste for parts is less useful: whole car parts are hard to
    # cut-and-paste convincingly. Keep low but non-zero — it still helps
    # rare-class recall without creating obvious collages.
    "copy_paste": 0.15,
}


DAMAGES_HYPERPARAMS: dict[str, Any] = {
    **_COMMON,
    # Damages labels are noisier; a slightly lower LR keeps the optimiser
    # from overreacting to label artefacts (large scratch polygons etc.).
    "lr0": 0.0008,
    # Full inverse-frequency class weights — replaces the old ``fl_gamma``
    # focal-loss knob (removed in Ultralytics 8.4+ default args).
    "cls_pw": 1.0,
    # copy_paste on crop images works very well — pasted rust/paint_chip
    # blobs look plausible over other parts' surfaces, and this is the
    # single most effective lever for the rare classes that --oversample
    # can't fix by itself.
    "copy_paste": 0.3,
}


def resolve_runtime_overrides(
    hparams: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge base ``hparams`` with CLI overrides on ``args``.

    Only the parameters that the training CLIs expose as flags
    (``epochs``, ``imgsz``, ``batch``, ``device``, ``patience``) are
    overridable at runtime — everything else is intentional and should be
    changed in code (with a commit) so training runs stay reproducible.
    """

    merged = dict(hparams)
    merged["epochs"] = args.epochs
    merged["imgsz"] = args.imgsz
    merged["batch"] = args.batch
    merged["device"] = args.device
    merged["patience"] = args.patience
    return merged
