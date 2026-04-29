"""Single source of truth for ML confidence thresholds.

This file is the **only** place where the business-level cutoffs that
gate ML output live. To raise / lower a threshold for any damage class
or for the parts model, edit the constants below and rebuild the
``ml_worker`` image — no other code changes are required.

Why these cutoffs are a domain concern (not ML hyperparameters)
---------------------------------------------------------------

Both detectors filter detections below a confidence cutoff before
anything they produce reaches Postgres or the user-facing bot UI, so
the cutoffs directly shape:

* which detected parts get cropped and sent to the damage stage,
* which damages appear in the user's "обнаруженные повреждения" list,
* which damages enter the priced estimate.

Keeping them here — a single grep-friendly module in the domain layer —
avoids the trap of different teams guessing different defaults in the
ML worker, in env files, and in the docker-compose ``environment:``
block. There is exactly one place to change a cutoff. An optional env
override (``DAMAGES_CONFIDENCE_THRESHOLD``) is still honoured for ops
who need a runtime panic-knob, but it acts uniformly on **all** damage
classes — the per-class policy below stays in code review.

EDIT HERE — current calibration
-------------------------------

Parts model: one global threshold (the model is uniform in difficulty
and downstream part↔damage compatibility filtering absorbs most false
positives anyway).

Damages model: per-class thresholds. The map below MUST be exhaustive
over :class:`DamageType` — an import-time assertion enforces this so
adding a new damage class is a deliberate, reviewed action, not a
silent default. Rationale for each tier:

* ``scratch`` / ``flat_tire``        — 0.50: well represented in
                                       training data, high confidence
                                       is reliable.
* ``broken_glass`` /
  ``broken_headlight``               — 0.40: distinctive but rarer.
* ``rust`` / ``paint_chip``          — 0.30: rare classes, lower bar
                                       to keep recall up.
* ``dent`` / ``crack``               — 0.25: rarest geometric defects;
                                       weakest model confidence; user
                                       still triages before pricing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType

# ─── Parts ────────────────────────────────────────────────────────────
PARTS_CONFIDENCE_THRESHOLD: Final[float] = 0.5


# ─── Damages: per-class thresholds — EDIT THESE NUMBERS ───────────────
DAMAGES_CONFIDENCE_BY_CLASS: Final[Mapping[DamageType, float]] = {
    DamageType.SCRATCH: 0.50,
    DamageType.DENT: 0.25,
    DamageType.PAINT_CHIP: 0.30,
    DamageType.RUST: 0.30,
    DamageType.CRACK: 0.25,
    DamageType.BROKEN_GLASS: 0.40,
    DamageType.FLAT_TIRE: 0.50,
    DamageType.BROKEN_HEADLIGHT: 0.40,
}

# Exhaustiveness guard: every DamageType must have an explicit
# threshold. Without this, adding a new class would silently fall back
# to the floor and could make the new detections invisible to users.
_missing_classes = set(DamageType) - set(DAMAGES_CONFIDENCE_BY_CLASS)
if _missing_classes:
    raise RuntimeError(
        "DAMAGES_CONFIDENCE_BY_CLASS is missing thresholds for: "
        f"{sorted(m.value for m in _missing_classes)}. "
        "Edit ml_thresholds.py to set explicit values."
    )

# Validate range once at import. Out-of-range values silently disable
# (≤0) or saturate (≥1) per-class filtering — both are confusing
# failure modes that we'd rather catch immediately on startup.
for _cls, _value in DAMAGES_CONFIDENCE_BY_CLASS.items():
    if not 0.0 < _value < 1.0:
        raise RuntimeError(
            f"DAMAGES_CONFIDENCE_BY_CLASS[{_cls.value}]={_value} is "
            "outside the (0, 1) probability range — fix in ml_thresholds.py."
        )


# Used as the YOLO ``predict(conf=...)`` argument. Keeping it at the
# minimum of all per-class thresholds guarantees the model returns
# every box that **could** clear at least one configured cutoff; the
# per-class filter in :class:`DamageDetector` then enforces the actual
# policy. Setting ``conf`` higher would mean low-threshold classes
# never reach our code and the per-class numbers above would be lies.
DAMAGES_CONFIDENCE_FLOOR: Final[float] = min(DAMAGES_CONFIDENCE_BY_CLASS.values())


def damages_threshold_for(damage_type: str | DamageType) -> float:
    """Return the per-class damage threshold from the SSOT.

    For unknown class names (a future model emits a label not yet in
    :class:`DamageType`) we fall back to :data:`DAMAGES_CONFIDENCE_FLOOR`.
    The detector's enum filter rejects such detections anyway, but
    having a numeric fallback lets callers compare safely without
    special-casing ``None``.
    """

    if isinstance(damage_type, DamageType):
        return DAMAGES_CONFIDENCE_BY_CLASS[damage_type]
    try:
        return DAMAGES_CONFIDENCE_BY_CLASS[DamageType(damage_type)]
    except ValueError:
        return DAMAGES_CONFIDENCE_FLOOR


__all__ = [
    "DAMAGES_CONFIDENCE_BY_CLASS",
    "DAMAGES_CONFIDENCE_FLOOR",
    "PARTS_CONFIDENCE_THRESHOLD",
    "damages_threshold_for",
]
