from __future__ import annotations

import io

import pytest
from PIL import Image

from auto_repair_estimator.ml_worker.inference.preprocessor import INPUT_SIZE, preprocess


def _make_image_bytes(width: int, height: int, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_preprocess_returns_correct_tensor_shape() -> None:
    data = _make_image_bytes(800, 600)
    result = preprocess(data)
    assert result.tensor.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert result.original_size == (800, 600)


def test_preprocess_normalises_to_0_1() -> None:
    data = _make_image_bytes(640, 640)
    result = preprocess(data)
    assert float(result.tensor.min()) >= 0.0
    assert float(result.tensor.max()) <= 1.0


def test_preprocess_accepts_png() -> None:
    data = _make_image_bytes(800, 600, fmt="PNG")
    result = preprocess(data)
    assert result.tensor.shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_preprocess_rejects_too_large_image() -> None:
    data = _make_image_bytes(1000, 1000)
    with pytest.raises(ValueError, match="too large"):
        preprocess(data, max_bytes=100)


def test_preprocess_rejects_too_small_image() -> None:
    data = _make_image_bytes(100, 100)
    with pytest.raises(ValueError, match="too small"):
        preprocess(data)
