"""Load-test harness for the AutoRepairEstimator backend.

What this simulates
-------------------

A realistic ML-request cycle from the VK bot's perspective, for each
virtual user:

    1. ``POST /v1/requests`` with ``mode=ml`` — returns a presigned PUT
       URL for the raw image bucket in MinIO (via ``StorageGateway``).
    2. ``PUT <presigned_url>`` — uploads a synthetic JPEG directly to
       MinIO (bypassing the backend, exactly like the production bot).
    3. ``POST /v1/requests/{id}/photo`` with ``image_key`` — queues the
       inference job through the backend's outbox.

Why this matters
----------------

The bottleneck in the synchronous path is the three backend round-trips
per photo (steps 1 + 3) plus the object-storage PUT (step 2). Measuring
these together yields the end-to-end perceived latency, while step 2
alone tells us whether object storage is saturating before the backend
does.

Observability
-------------

Per-phase histograms (``p50 / p95 / p99 / max``), request rate, error
counts by HTTP status + exception type, and optional Kafka queue-depth
sampling via ``aiokafka`` admin client.

Dependencies
------------

Uses only libraries already in ``[project.dependencies]`` of
``pyproject.toml`` (``httpx``, ``loguru``, ``pydantic-settings``) plus
``aiokafka`` (already pinned) for the optional queue-depth sampling.

Usage
-----

    # Start the full stack first:
    docker compose -f docker/docker-compose.yml up -d

    # Run 200 RPS for 60 seconds, 32 workers:
    python scripts/load_test.py \
        --backend-url http://localhost:8000 \
        --target-rps 200 \
        --duration 60 \
        --workers 32

    # CSV output for regression tracking:
    python scripts/load_test.py --output-json reports/load_2026_04_19.json

The script exits with a non-zero status if the observed success rate
drops below ``--fail-below-success-rate`` (default 0.95), making it
suitable as a CI smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx
from loguru import logger
from PIL import Image

try:
    from aiokafka.admin import AIOKafkaAdminClient  # type: ignore[import-not-found]

    _KAFKA_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _KAFKA_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class LoadConfig:
    backend_url: str
    target_rps: int
    duration_seconds: int
    workers: int
    image_side_px: int
    fail_below_success_rate: float
    output_json: str | None
    kafka_bootstrap: str | None
    kafka_topics: list[str]


def _parse_args(argv: list[str] | None = None) -> LoadConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default="http://localhost:8000", help="Base URL of the backend API.")
    parser.add_argument("--target-rps", type=int, default=120, help="Target full-cycle requests per second.")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds.")
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent worker coroutines.")
    parser.add_argument(
        "--image-side-px",
        type=int,
        default=640,
        help="Side length of the synthetic JPEG payload. 640 matches the YOLO default.",
    )
    parser.add_argument(
        "--fail-below-success-rate",
        type=float,
        default=0.95,
        help="Exit non-zero if the observed success rate drops below this value.",
    )
    parser.add_argument("--output-json", default=None, help="Path to write the final report as JSON.")
    parser.add_argument(
        "--kafka-bootstrap",
        default=None,
        help="Optional Kafka bootstrap servers for queue-depth sampling (e.g. localhost:9092).",
    )
    parser.add_argument(
        "--kafka-topics",
        default="inference_requests,inference_results,notifications",
        help="Comma-separated topic list to sample.",
    )
    ns = parser.parse_args(argv)
    return LoadConfig(
        backend_url=ns.backend_url.rstrip("/"),
        target_rps=ns.target_rps,
        duration_seconds=ns.duration,
        workers=ns.workers,
        image_side_px=ns.image_side_px,
        fail_below_success_rate=ns.fail_below_success_rate,
        output_json=ns.output_json,
        kafka_bootstrap=ns.kafka_bootstrap,
        kafka_topics=[t.strip() for t in ns.kafka_topics.split(",") if t.strip()],
    )


# ---------------------------------------------------------------------------
# Payload generation
# ---------------------------------------------------------------------------


def _make_jpeg(side_px: int) -> bytes:
    """Build a valid JPEG of the configured side length.

    Uses random noise so compression produces realistic byte counts — a
    flat-colour JPEG would compress to ~2 KB and hide bandwidth effects.
    """

    rgb = bytes(random.getrandbits(8) for _ in range(side_px * side_px * 3))
    img = Image.frombytes("RGB", (side_px, side_px), rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimings:
    samples_ms: list[float] = field(default_factory=list)

    def observe(self, ms: float) -> None:
        self.samples_ms.append(ms)

    def summarise(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"count": 0}
        s = sorted(self.samples_ms)
        n = len(s)
        return {
            "count": n,
            "p50_ms": round(s[int(n * 0.50)], 2),
            "p95_ms": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99_ms": round(s[min(int(n * 0.99), n - 1)], 2),
            "max_ms": round(s[-1], 2),
            "mean_ms": round(statistics.fmean(s), 2),
        }


@dataclass
class LoadMetrics:
    full_cycle: PhaseTimings = field(default_factory=PhaseTimings)
    create_request: PhaseTimings = field(default_factory=PhaseTimings)
    s3_upload: PhaseTimings = field(default_factory=PhaseTimings)
    confirm_upload: PhaseTimings = field(default_factory=PhaseTimings)
    total_attempted: int = 0
    total_succeeded: int = 0
    errors_by_kind: Counter = field(default_factory=Counter)

    def success_rate(self) -> float:
        return self.total_succeeded / self.total_attempted if self.total_attempted else 0.0

    def to_report(self, cfg: LoadConfig, wall_seconds: float, kafka_report: dict | None) -> dict:
        return {
            "config": {
                "backend_url": cfg.backend_url,
                "target_rps": cfg.target_rps,
                "duration_seconds": cfg.duration_seconds,
                "workers": cfg.workers,
                "image_side_px": cfg.image_side_px,
            },
            "totals": {
                "attempted": self.total_attempted,
                "succeeded": self.total_succeeded,
                "success_rate": round(self.success_rate(), 4),
                "wall_seconds": round(wall_seconds, 2),
                "observed_rps": round(self.total_attempted / wall_seconds, 2) if wall_seconds else 0.0,
            },
            "latency_ms": {
                "full_cycle": self.full_cycle.summarise(),
                "create_request": self.create_request.summarise(),
                "s3_upload": self.s3_upload.summarise(),
                "confirm_upload": self.confirm_upload.summarise(),
            },
            "errors": dict(self.errors_by_kind),
            "kafka": kafka_report or {},
        }


# ---------------------------------------------------------------------------
# Load loop
# ---------------------------------------------------------------------------


async def _run_one_cycle(
    client: httpx.AsyncClient,
    s3_client: httpx.AsyncClient,
    image_bytes: bytes,
    metrics: LoadMetrics,
) -> None:
    metrics.total_attempted += 1
    full_cycle_t0 = time.perf_counter()

    try:
        # --- Step 1: create ML request ----------------------------------
        t0 = time.perf_counter()
        resp = await client.post(
            "/v1/requests",
            json={"chat_id": random.randint(1, 10_000_000), "user_id": random.randint(1, 10_000_000), "mode": "ml"},
            timeout=30.0,
        )
        metrics.create_request.observe((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            metrics.errors_by_kind[f"create_request_http_{resp.status_code}"] += 1
            return

        body = resp.json()
        request_id: str = body["id"]
        presigned_url: str | None = body.get("presigned_put_url")

        # --- Step 2: upload to object storage ---------------------------
        if presigned_url:
            t0 = time.perf_counter()
            put_resp = await s3_client.put(
                presigned_url,
                content=image_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=30.0,
            )
            metrics.s3_upload.observe((time.perf_counter() - t0) * 1000)
            if put_resp.status_code not in (200, 204):
                metrics.errors_by_kind[f"s3_put_http_{put_resp.status_code}"] += 1
                return
        else:
            # Backend running without a storage gateway (dev mode). Skip
            # the PUT but still measure the confirm endpoint.
            metrics.errors_by_kind["no_presigned_url"] += 1
            # Not fatal — the confirm endpoint still works in dev mode.

        # --- Step 3: confirm upload to the backend ----------------------
        t0 = time.perf_counter()
        confirm_resp = await client.post(
            f"/v1/requests/{request_id}/photo",
            json={"image_key": f"raw-images/{request_id}.jpg"},
            timeout=30.0,
        )
        metrics.confirm_upload.observe((time.perf_counter() - t0) * 1000)
        if confirm_resp.status_code != 200:
            metrics.errors_by_kind[f"confirm_upload_http_{confirm_resp.status_code}"] += 1
            return

        metrics.total_succeeded += 1
        metrics.full_cycle.observe((time.perf_counter() - full_cycle_t0) * 1000)
    except httpx.TimeoutException:
        metrics.errors_by_kind["timeout"] += 1
    except httpx.ConnectError:
        metrics.errors_by_kind["connect_error"] += 1
    except Exception as exc:
        metrics.errors_by_kind[f"unexpected:{type(exc).__name__}"] += 1


async def _worker(
    cfg: LoadConfig,
    queue: asyncio.Queue,
    metrics: LoadMetrics,
    image_bytes: bytes,
) -> None:
    """One worker = one HTTP connection pool. Each item in the queue is
    a request to run. The dispatcher feeds the queue at the target RPS."""

    # Shared clients per worker. ``limits`` is low because each worker
    # does one request at a time.
    limits = httpx.Limits(max_keepalive_connections=2, max_connections=4)
    async with (
        httpx.AsyncClient(base_url=cfg.backend_url, limits=limits) as backend,
        httpx.AsyncClient(limits=limits) as s3,
    ):
        while True:
            sentinel = await queue.get()
            try:
                if sentinel is None:
                    return
                await _run_one_cycle(backend, s3, image_bytes, metrics)
            finally:
                queue.task_done()


async def _dispatcher(cfg: LoadConfig, queue: asyncio.Queue) -> None:
    """Push ``target_rps`` work items per second until duration elapses."""
    interval = 1.0 / max(cfg.target_rps, 1)
    deadline = time.monotonic() + cfg.duration_seconds
    next_tick = time.monotonic()

    while time.monotonic() < deadline:
        await queue.put(object())
        next_tick += interval
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    for _ in range(cfg.workers):
        await queue.put(None)


# ---------------------------------------------------------------------------
# Kafka queue-depth sampling
# ---------------------------------------------------------------------------


async def _sample_kafka_lag(cfg: LoadConfig) -> dict | None:
    if not cfg.kafka_bootstrap or not _KAFKA_AVAILABLE:
        return None

    client = AIOKafkaAdminClient(bootstrap_servers=cfg.kafka_bootstrap)
    try:
        await client.start()
        # describe_topics returns per-topic metadata; we extract offsets via
        # the AdminClient's ``list_topics`` / ``describe_cluster``. For a
        # quick sampler we just report whether the topics exist and have
        # partitions — full offset tracking would require consumer groups.
        metadata = await client.describe_topics(topics=cfg.kafka_topics)
        topic_report: dict[str, dict] = {}
        for t in metadata:
            topic_name = t.get("topic") if isinstance(t, dict) else None
            partitions = t.get("partitions") if isinstance(t, dict) else None
            if topic_name and partitions is not None:
                topic_report[topic_name] = {"partition_count": len(partitions)}
        return {"topics": topic_report}
    except Exception as exc:  # pragma: no cover - depends on infra
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(cfg: LoadConfig) -> int:
    logger.info("Preparing synthetic JPEG ({}x{})...", cfg.image_side_px, cfg.image_side_px)
    image_bytes = _make_jpeg(cfg.image_side_px)
    logger.info("JPEG payload size: {} bytes", len(image_bytes))

    metrics = LoadMetrics()
    queue: asyncio.Queue = asyncio.Queue(maxsize=cfg.target_rps * 2)

    logger.info(
        "Starting load: target_rps={} duration={}s workers={} backend={}",
        cfg.target_rps,
        cfg.duration_seconds,
        cfg.workers,
        cfg.backend_url,
    )

    wall_start = time.monotonic()
    workers = [asyncio.create_task(_worker(cfg, queue, metrics, image_bytes)) for _ in range(cfg.workers)]
    dispatcher = asyncio.create_task(_dispatcher(cfg, queue))

    await dispatcher
    await queue.join()
    await asyncio.gather(*workers)
    wall_seconds = time.monotonic() - wall_start

    kafka_report = await _sample_kafka_lag(cfg)

    report = metrics.to_report(cfg, wall_seconds, kafka_report)
    _print_report(report)

    if cfg.output_json:
        with open(cfg.output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Wrote JSON report to {}", cfg.output_json)

    success_rate = report["totals"]["success_rate"]
    if success_rate < cfg.fail_below_success_rate:
        logger.error(
            "FAIL: success rate {:.2%} is below the {:.2%} threshold",
            success_rate,
            cfg.fail_below_success_rate,
        )
        return 1

    return 0


def _print_report(report: dict) -> None:
    logger.info("=" * 72)
    logger.info("LOAD TEST REPORT")
    logger.info("=" * 72)
    t = report["totals"]
    logger.info(
        "Attempted: {}  Succeeded: {}  Success rate: {:.2%}  Wall: {}s  Observed RPS: {}",
        t["attempted"],
        t["succeeded"],
        t["success_rate"],
        t["wall_seconds"],
        t["observed_rps"],
    )
    logger.info("-" * 72)
    for phase, stats in report["latency_ms"].items():
        logger.info("{:>16}  {}", phase, stats)
    if report["errors"]:
        logger.info("-" * 72)
        logger.info("Errors:")
        for kind, count in sorted(report["errors"].items(), key=lambda kv: -kv[1]):
            logger.info("  {:<40} {}", kind, count)
    if report.get("kafka"):
        logger.info("-" * 72)
        logger.info("Kafka: {}", report["kafka"])
    logger.info("=" * 72)


if __name__ == "__main__":
    cfg = _parse_args()
    rc = asyncio.run(main(cfg))
    sys.exit(rc)
