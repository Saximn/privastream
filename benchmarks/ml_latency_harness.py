#!/usr/bin/env python3
"""Small benchmark harness for Privastream ML components.

The harness intentionally avoids heavyweight model imports by default. It measures callable
latencies and writes machine-readable summaries that can be used in CI or release notes.
Production benchmarks can import actual model callables through lightweight wrapper modules.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class LatencySummary:
    name: str
    iterations: int
    failures: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    avg_ms: float


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def resolve_callable(path: str) -> Callable[..., Any]:
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"{path} is not callable")
    return func


def run_latency_benchmark(
    name: str,
    func: Callable[..., Any],
    iterations: int,
    warmup: int,
    payload: Optional[Any] = None,
) -> LatencySummary:
    for _ in range(warmup):
        func(payload) if payload is not None else func()

    latencies_ms: List[float] = []
    failures = 0
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func(payload) if payload is not None else func()
        except Exception:
            failures += 1
        finally:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return LatencySummary(
        name=name,
        iterations=iterations,
        failures=failures,
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        max_ms=max(latencies_ms) if latencies_ms else 0.0,
        avg_ms=statistics.fmean(latencies_ms) if latencies_ms else 0.0,
    )


def noop(_: Optional[Any] = None) -> None:
    return None


def load_payload(payload_path: Optional[str]) -> Optional[Any]:
    if not payload_path:
        return None
    path = Path(payload_path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_bytes()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Privastream ML callable latency")
    parser.add_argument("--callable", default="__main__:noop", help="Callable as module:function")
    parser.add_argument("--name", default="noop", help="Name for the benchmark result")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--payload", help="Optional JSON or binary payload path")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = load_payload(args.payload)
    func = resolve_callable(args.callable)
    summary = run_latency_benchmark(args.name, func, args.iterations, args.warmup, payload)
    data: Dict[str, Any] = asdict(summary)

    if args.output:
        Path(args.output).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    return 0 if summary.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
