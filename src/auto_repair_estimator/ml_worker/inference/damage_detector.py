from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from auto_repair_estimator.backend.domain.value_objects.ml_thresholds import (
    DAMAGES_CONFIDENCE_BY_CLASS,
    DAMAGES_CONFIDENCE_FLOOR,
)
from auto_repair_estimator.backend.domain.value_objects.part_damage_compatibility import (
    PART_DAMAGE_COMPATIBILITY,
)
from auto_repair_estimator.backend.domain.value_objects.request_enums import DamageType, PartType

# Single source of truth: the backend only accepts damage labels that are
# members of the `DamageType` enum, so the detector must refuse (or at least
# silently drop) anything else rather than propagating unknown strings
# downstream through Kafka into the repair_requests table.
_SUPPORTED_DAMAGE_NAMES: frozenset[str] = frozenset(member.value for member in DamageType)


@dataclass
class DamageDetection:
    damage_type: str
    part_type: str
    confidence: float
    mask: Any | None
    mask_key: str | None = None
    # Pixel bbox of the parent crop inside the original image, propagated
    # here so the composer can place the (crop-local) mask back at the
    # correct location on the full photo. ``None`` means "mask is already
    # in original-image space" — used by legacy/unit-test callers.
    crop_box_pixels: tuple[int, int, int, int] | None = None


class DamageDetector:
    """YOLO-backed damage detector with **per-class** confidence cutoffs.

    Threshold resolution (highest priority first):

    1. ``thresholds`` kwarg — explicit per-class mapping. Used by the
       ML worker when an operator sets a uniform env-override (the
       worker expands the single value into a full mapping there) and
       by tests that want a fully custom calibration.
    2. ``confidence_threshold`` kwarg — single float treated as a
       *uniform* override across all damage classes. Kept for
       backward compatibility with older callers and unit tests.
    3. Default — :data:`DAMAGES_CONFIDENCE_BY_CLASS` from
       ``ml_thresholds.py`` (the production policy).

    The model itself is invoked with ``conf=floor`` where ``floor`` is
    the minimum cutoff in the resolved mapping, so low-threshold
    classes still survive Ultralytics' built-in NMS pre-filter and
    reach our per-class check.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float | None = None,
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self._model_path = model_path

        if thresholds is not None:
            resolved: dict[str, float] = dict(thresholds)
        elif confidence_threshold is not None:
            resolved = {dt.value: confidence_threshold for dt in DamageType}
        else:
            resolved = {dt.value: cutoff for dt, cutoff in DAMAGES_CONFIDENCE_BY_CLASS.items()}

        self._thresholds: Mapping[str, float] = resolved
        # Floor must never exceed the smallest configured cutoff,
        # otherwise YOLO would silently filter low-threshold classes
        # before our code sees them. Falling back to the SSOT floor
        # when ``resolved`` is empty (defensive — shouldn't happen).
        self._floor: float = min(resolved.values()) if resolved else DAMAGES_CONFIDENCE_FLOOR
        self._model: Any | None = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(self._model_path)
        self._assert_model_classes_match_enum()
        logger.info("Damages model loaded from {}", self._model_path)

    def _assert_model_classes_match_enum(self) -> None:
        """Warn loudly when model ``names`` diverge from :class:`DamageType`.

        We don't raise — a newer model may legitimately add classes we
        haven't wired up yet, and those are simply filtered out in
        :meth:`predict`. But loading a model that has **zero** overlap with
        the enum is almost certainly a deployment mistake (wrong weights
        copied into ``/app/models/damages.pt``), so we escalate that case
        to an error log.
        """
        names_attr = getattr(self._model, "names", None)
        if names_attr is None:
            logger.warning("Damages model has no `names` metadata; class filtering disabled")
            return

        model_names = set(names_attr.values()) if isinstance(names_attr, dict) else set(names_attr)
        unknown = model_names - _SUPPORTED_DAMAGE_NAMES
        missing = _SUPPORTED_DAMAGE_NAMES - model_names

        if unknown:
            logger.warning(
                "Damages model exposes classes not in DamageType enum; these will be dropped: {}",
                sorted(unknown),
            )
        if missing:
            logger.warning(
                "Damages model is missing these DamageType classes (they will never be detected): {}",
                sorted(missing),
            )
        if not (model_names & _SUPPORTED_DAMAGE_NAMES):
            logger.error(
                "Damages model classes {} share NO overlap with DamageType enum {} — wrong weights?",
                sorted(model_names),
                sorted(_SUPPORTED_DAMAGE_NAMES),
            )

    def predict(
        self,
        crop_bytes: bytes,
        part_type: str,
        request_id: str,
        crop_index: int,
        bucket: str,
        crop_box_pixels: tuple[int, int, int, int] | None = None,
    ) -> list[DamageDetection]:
        if self._model is None:
            raise RuntimeError("DamageDetector not loaded; call load() first")

        from PIL import Image

        img = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
        # ``conf=self._floor`` ensures Ultralytics' internal pre-filter
        # doesn't silently drop classes whose per-class threshold is
        # lower than YOLO's default 0.25 — without this we'd lie about
        # the per-class numbers configured in ml_thresholds.py.
        results = self._model(img, verbose=False, conf=self._floor)
        detections: list[DamageDetection] = []
        dropped_unknown = 0
        dropped_incompatible = 0
        # Full per-detection trace, aggregated into a single structured
        # log line at the end of the call. Operators routinely ask "why
        # didn't the model see that dent?" — with this trace the answer
        # is one ``docker logs | grep`` away instead of requiring a
        # re-run with a debugger. Each entry is (class, conf, cutoff,
        # verdict) so per-class thresholds are auditable post-mortem.
        raw_tally: list[tuple[str, float, float, str]] = []

        # Resolve the compatibility set for the host part once per call.
        # If ``part_type`` isn't a known :class:`PartType` (e.g. a crop
        # produced by an older parts model), we fall back to "allow
        # anything the enum validates" — the backend's AddDamage use case
        # will do a second compatibility check and reject stray pairs.
        allowed_damages = self._allowed_damages_for(part_type)

        for result in results:
            boxes = result.boxes
            masks = result.masks if result.masks is not None else None
            names = result.names

            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                damage_type = names[class_id]
                # Per-class cutoff lookup: unknown classes fall back to
                # the floor, but they're rejected by the enum filter
                # below anyway — the cutoff is recorded purely for the
                # audit log so operators can reason about the decision.
                cutoff = self._thresholds.get(damage_type, self._floor)

                if confidence < cutoff:
                    raw_tally.append((damage_type, confidence, cutoff, "REJECTED_LOW_CONF"))
                    continue

                if damage_type not in _SUPPORTED_DAMAGE_NAMES:
                    dropped_unknown += 1
                    raw_tally.append((damage_type, confidence, cutoff, "REJECTED_UNKNOWN_CLASS"))
                    continue

                # Part ↔ damage compatibility filter.
                # Example regression this guards against: the damage model
                # firing ``broken_headlight`` on a door crop or ``broken_glass``
                # on the hood. Such detections are physically impossible,
                # confuse the user, and have no matching pricing rule. We
                # drop them at the ML boundary so they never enter the
                # system-of-record or the user-facing "found damages" list.
                if allowed_damages is not None and damage_type not in allowed_damages:
                    dropped_incompatible += 1
                    raw_tally.append((damage_type, confidence, cutoff, "REJECTED_INCOMPATIBLE"))
                    continue

                mask = masks.data[i].cpu().numpy() if masks is not None else None
                mask_key = (
                    f"{bucket}/{request_id}_damage_{crop_index}_{i}_{damage_type}.png" if mask is not None else None
                )

                detections.append(
                    DamageDetection(
                        damage_type=damage_type,
                        part_type=part_type,
                        confidence=confidence,
                        mask=mask,
                        mask_key=mask_key,
                        crop_box_pixels=crop_box_pixels,
                    )
                )
                raw_tally.append((damage_type, confidence, cutoff, "ACCEPTED"))

        if dropped_unknown:
            logger.warning(
                "DamageDetector[request={} crop={} part={}] dropped {} unknown-class detections",
                request_id,
                crop_index,
                part_type,
                dropped_unknown,
            )
        if dropped_incompatible:
            logger.info(
                "DamageDetector[request={} crop={} part={}] dropped {} incompatible detections (allowed={})",
                request_id,
                crop_index,
                part_type,
                dropped_incompatible,
                sorted(allowed_damages) if allowed_damages is not None else "any",
            )

        summary = sorted(raw_tally, key=lambda entry: -entry[1])
        summary_str = (
            ", ".join(f"{cls}@{conf:.2f}/cut@{cut:.2f}:{verdict}" for cls, conf, cut, verdict in summary) or "<none>"
        )
        logger.info(
            "DamageDetector[request={} crop={} part={}] raw={} accepted={} floor={:.2f} | {}",
            request_id,
            crop_index,
            part_type,
            len(raw_tally),
            len(detections),
            self._floor,
            summary_str,
        )
        return detections

    @staticmethod
    def _allowed_damages_for(part_type: str) -> frozenset[str] | None:
        """Return the set of damage-type *strings* allowed on ``part_type``.

        Returns ``None`` when ``part_type`` is not a recognised
        :class:`PartType` — the caller then skips compatibility filtering
        and leans on downstream backend validation. Using a frozenset of
        ``.value`` strings (not enum members) keeps the hot loop free of
        per-detection enum coercion, which otherwise showed up in cProfile
        as a non-trivial cost when a crop fires 20+ boxes.
        """

        try:
            pt = PartType(part_type)
        except ValueError:
            return None
        return frozenset(dt.value for dt in PART_DAMAGE_COMPATIBILITY.get(pt, frozenset()))
