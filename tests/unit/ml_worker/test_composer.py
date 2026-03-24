from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from auto_repair_estimator.ml_worker.inference.composer import compose


@dataclass
class _FakeDamage:
    damage_type: str
    mask: Any | None


def _make_image(width: int = 640, height: int = 640) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = [100, 150, 200]
    return Image.fromarray(arr)


def _full_mask(h: int = 640, w: int = 640) -> np.ndarray:  # type: ignore[type-arg]
    return np.ones((h, w), dtype=np.float32)


def _empty_mask(h: int = 640, w: int = 640) -> np.ndarray:  # type: ignore[type-arg]
    return np.zeros((h, w), dtype=np.float32)


def test_compose_returns_jpeg_bytes() -> None:
    img = _make_image()
    result = compose(img, [_FakeDamage("scratch", _full_mask())])
    out = Image.open(io.BytesIO(result))
    assert out.format == "JPEG"


def test_compose_with_no_damages_returns_original_image() -> None:
    img = _make_image(100, 100)
    orig_arr = np.array(img)
    result = compose(img, [])
    result_arr = np.array(Image.open(io.BytesIO(result)).convert("RGB"))
    assert result_arr.shape == orig_arr.shape


def test_compose_with_full_mask_changes_pixels() -> None:
    img = _make_image()
    orig_arr = np.array(img, dtype=np.float32)
    result = compose(img, [_FakeDamage("scratch", _full_mask())])
    result_arr = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    diff = float(np.abs(result_arr - orig_arr).mean())
    assert diff > 0.5


def test_compose_with_none_mask_skips_damage() -> None:
    img = _make_image()
    orig_arr = np.array(img, dtype=np.float32)
    result = compose(img, [_FakeDamage("dent", None)])
    result_arr = np.array(Image.open(io.BytesIO(result)).convert("RGB"), dtype=np.float32)
    diff = float(np.abs(result_arr - orig_arr).mean())
    assert diff < 5.0


def test_compose_with_different_damage_types() -> None:
    img = _make_image(200, 200)
    damages = [
        _FakeDamage("scratch", _full_mask(200, 200)),
        _FakeDamage("dent", _empty_mask(200, 200)),
        _FakeDamage("rust", _full_mask(200, 200)),
    ]
    result = compose(img, damages)
    assert len(result) > 0
