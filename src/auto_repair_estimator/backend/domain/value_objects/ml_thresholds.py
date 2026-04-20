"""Single source of truth for ML confidence thresholds.

Both the parts detector and the damages detector filter out detections
below a confidence cutoff before anything they produce reaches the
system-of-record (Postgres) or the user-facing bot UI. Those cutoffs
are therefore **business decisions**, not ML hyperparameters — they
directly shape:

* which detected parts get cropped and sent to the damage stage;
* which damages appear in the user's "обнаруженные повреждения" list;
* which damages enter the priced estimate.

Keeping the numbers here — a single, grep-friendly module in the domain
layer — avoids the old trap where different teams guessed different
defaults in the ML worker, in env files, and in the docker-compose
``environment:`` block. There's now exactly one place to change the
cutoff. :class:`MLWorkerConfig` still allows an env override
(``PARTS_CONFIDENCE_THRESHOLD`` / ``DAMAGES_CONFIDENCE_THRESHOLD``) for
operators who need a runtime knob without rebuilding the image, but the
default they override comes from this module.

Current calibration (for the YOLOv8-seg weights in
``docker/models/parts.pt`` and ``docker/models/damages.pt``):

* Parts: 0.5 — bias towards catching borderline detections like
  bumpers on cluttered backgrounds. Any part reaching this bar is
  cropped and passed through to the damage stage; downstream part↔
  damage compatibility filtering (see
  :mod:`part_damage_compatibility`) absorbs most false positives
  before they become priced rows.
* Damages: 0.2 — deliberately low. We want the user to SEE every
  potential damage in the "found damages" list so they can either
  confirm it or press [Изменить] to remove a false positive. A higher
  damage threshold silently hides real damages and forces the user
  into manual-entry mode, which defeats the point of the ML pipeline.

When you raise/lower either constant, check that
``test_ml_thresholds_are_business_calibrated`` still reflects the
current policy — that test exists precisely so a bump of either
threshold is a deliberate, reviewed action.
"""

from __future__ import annotations

from typing import Final

PARTS_CONFIDENCE_THRESHOLD: Final[float] = 0.5
DAMAGES_CONFIDENCE_THRESHOLD: Final[float] = 0.2


__all__ = [
    "DAMAGES_CONFIDENCE_THRESHOLD",
    "PARTS_CONFIDENCE_THRESHOLD",
]
