#!/usr/bin/env python3
"""
Neural Blitz NG v7.1 — UDP Network Latency Benchmark Tool

Measures: latency (percentiles), jitter, packet loss, throughput
Modes: single test, batch multi-target, echo server, monitor
Exports: JSON, CSV

Changes from v7.0:
  - Fixed: timestamp now stamped at sendto(), not at packet build
  - Fixed: response wait honors full --timeout instead of hardcap 2s
  - Fixed: token bucket uses asyncio.Lock to prevent rate overshoot
  - Fixed: tasks are batched instead of all-at-once creation
  - Fixed: coordinated-omission correction clamps negatives to 0
  - Fixed: prometheus label escaping
  - Removed: fake count_retried field (no retry logic exists)
  - Improved: histogram caches sorted state
"""

__version__ = "7.1.0"

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import uvloop

    LOOP_ENGINE = "uvloop"
except ImportError:
    LOOP_ENGINE = "asyncio"

logger = logging.getLogger("neural_blitz_ng")


class LatencyHistogram:
    """Microsecond-resolution latency percentile tracker."""

    def __init__(self):
        self._values: List[float] = []
        self._sorted: Optional[List[float]] = None

    def record(self, value_us: float):
        self._values.append(value_us)
        self._sorted = None

    def _get_sorted(self) -> List[float]:
        if self._sorted is None:
            self._sorted = sorted(self._values)
        return self._sorted

    def percentile(self, p: float) -> float:
        s = self._get_sorted()
        if not s:
            return 0.0
        idx = max(0, int(math.ceil(p / 100.0 * len(s))) - 1)
        return s[idx]

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def stddev(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((v - m) ** 2 for v in self._values) / (len(self._values) - 1))

    def min_val(self) -> float:
        return min(self._values) if self._values else 0.0

    def max_val(self) -> float:
        return max(self._values) if self._values else 0.0

    def count(self) -> int:
        return len(self._values)

    def jitter(self) -> float:
        if len(self._values) < 2:
            return 0.0
        diffs = [abs(self._values[i] - self._values[i - 1]) for i in range(1, len(self._values))]
        return sum(diffs) / len(diffs)


def coordinated_omission_correction(values_us: List[float], expected_interval_us: float) -> List[float]:
    """Add stall-corrected samples for missed intervals. Clamps to >= 0."""
    if not values_us or expected_interval_us <= 0:
        return list(values_us)
    corrected = []
    for v in values_us:
        corrected.append(v)
        missed = int(v / expected_interval_us) - 1
        for i in range(missed):
            sample = v - (i + 1) * expected_interval_us
            if sample > 0:
                corrected.append(sample)
    return corrected


class TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = 0.0
        self.last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.last = now
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            await asyncio.sleep(0.5 / self.rate)


@dataclass
class TestResult:
    target_name: str = ""
    target_host: str = ""
    target_port: int = 9999
    count_sent: int = 0
    count_received: int = 0
    count_lost: int = 0
    success_rate: float = 0.0
    min_us: float = 0.0
    p50_us: float = 0.0
    p90_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    p999_us: float = 0.0
    p9999_us: float = 0.0
    max_us: float = 0.0
    mean_us: float = 0.0
    stddev_us: float = 0.0
    jitter_us: float = 0.0
    co_p50_us: float = 0.0
    co_p99_us: float = 0.0
    co_p999_us: float = 0.0
    duration_s: float = 0.0
    pps: float = 0.0
    loop_engine: str = LOOP_ENGINE
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MAGIC = b"NBLZ"
HEADER_FMT = "!4sQQd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PAYLOAD_SIZE = 64


def build_packet(seq: int, send_ts_ns: int) -> bytes:
    header = struct.pack(HEADER_FMT, MAGIC, seq, send_ts_ns, 0.0)
    pad_len = max(0, PAYLOAD_SIZE - HEADER_SIZE)
    pad = os.urandom(pad_len) if pad_len > 0 else b""
    return header + pad


def parse_packet(data: bytes):
    if len(data) < HEADER_SIZE:
        return None, None, None
    magic, seq, send_ts, _ = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None, None, None
    return seq, send_ts, time.monotonic_ns()


class EchoServerProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.count = 0

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.count += 1
        self.transport.sendto(data, addr)
        if self.count % 10000 == 0:
            logger.info(f"Echo server: {self.count} packets echoed")


async def run_echo_server(bind: str, port: int):
    loop = asyncio.get_event_loop()
    transport, _ = await loop.create_datagram_endpoint(EchoServerProtocol, local_addr=(bind, port))
    logger.info(f"Echo server listening on ('{bind}', {port})")
    try:
        await asyncio.Future()
    finally:
        transport.close()


class BlitzClient(asyncio.DatagramProtocol):
    def __init__(self):
        self.transport = None
        self.responses: Dict[int, float] = {}

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        seq, send_ts, recv_ts = parse_packet(data)
        if seq is not None:
            rtt_us = (recv_ts - send_ts) / 1000.0
            self.responses[seq] = rtt_us

    def error_received(self, exc):
        logger.debug(f"Protocol error: {exc}")


async def run_single_test(
    host: str,
    port: int,
    count: int,
    rate: float,
    concurrency: int,
    warmup: int,
    timeout: float,
    name: str = "",
) -> TestResult:
    loop = asyncio.get_event_loop()
    transport, client = await loop.create_datagram_endpoint(BlitzClient, remote_addr=(host, port))

    rate_limiter = TokenBucket(rate) if rate > 0 else None
    semaphore = asyncio.Semaphore(concurrency)
    total_to_send = count + warmup

    async def send_packet(seq: int):
        async with semaphore:
            if rate_limiter:
                await rate_limiter.acquire()
            send_ts = time.monotonic_ns()
            client.transport.sendto(build_packet(seq, send_ts))

    start_time = time.monotonic()

    batch_size = 1000
    for batch_start in range(0, total_to_send, batch_size):
        batch_end = min(batch_start + batch_size, total_to_send)
        tasks = [asyncio.create_task(send_packet(i)) for i in range(batch_start, batch_end)]
        await asyncio.gather(*tasks)

    expected_seqs = set(range(warmup, total_to_send))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = set(client.responses.keys()) & expected_seqs
        if len(got) >= len(expected_seqs):
            break
        await asyncio.sleep(0.05)

    end_time = time.monotonic()
    duration = end_time - start_time
    transport.close()

    histogram = LatencyHistogram()
    rtt_list = []
    for seq in range(warmup, total_to_send):
        if seq in client.responses:
            val = client.responses[seq]
            histogram.record(val)
            rtt_list.append(val)

    received = histogram.count()
    lost = count - received

    expected_interval = 1_000_000.0 / rate if rate > 0 else 1000.0
    co_values = coordinated_omission_correction(rtt_list, expected_interval)
    co_hist = LatencyHistogram()
    for v in co_values:
        co_hist.record(v)

    return TestResult(
        target_name=name or f"{host}:{port}",
        target_host=host,
        target_port=port,
        count_sent=count,
        count_received=received,
        count_lost=lost,
        success_rate=round(received / count * 100, 1) if count else 0.0,
        min_us=round(histogram.min_val(), 1),
        p50_us=round(histogram.percentile(50), 1),
        p90_us=round(histogram.percentile(90), 1),
        p95_us=round(histogram.percentile(95), 1),
        p99_us=round(histogram.percentile(99), 1),
        p999_us=round(histogram.percentile(99.9), 1),
        p9999_us=round(histogram.percentile(99.99), 1),
        max_us=round(histogram.max_val(), 1),
        mean_us=round(histogram.mean(), 1),
        stddev_us=round(histogram.stddev(), 1),
        jitter_us=round(histogram.jitter(), 1),
        co_p50_us=round(co_hist.percentile(50), 1),
        co_p99_us=round(co_hist.percentile(99), 1),
        co_p999_us=round(co_hist.percentile(99.9), 1),
        duration_s=round(duration, 3),
        pps=round(received / duration, 1) if duration > 0 else 0.0,
        loop_engine=LOOP_ENGINE,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


async def run_batch(config_path: str, output_path: Optional[str] = None) -> List[TestResult]:
    if not HAS_YAML:
        logger.error("PyYAML required for batch mode: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    targets = cfg.get("targets", [])
    test_cfg = cfg.get("test", {})

    coros = []
    for t in targets:
        if isinstance(t, dict):
            tname = t.get("name", "")
            host = t.get("host", "127.0.0.1")
            port = t.get("port", 9999)
        elif isinstance(t, str):
            tname = t
            host = "127.0.0.1"
            port = 9999
            if "(" in t and ")" in t:
                inner = t[t.index("(") + 1 : t.index(")")]
                parts = inner.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 9999
        else:
            continue

        coros.append(
            run_single_test(
                host,
                port,
                test_cfg.get("count", 300),
                test_cfg.get("rate", 3000),
                test_cfg.get("concurrency", 30),
                test_cfg.get("warmup", 0),
                test_cfg.get("timeout", 5.0),
                tname,
            )
        )

    results = await asyncio.gather(*coros)

    if output_path:
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Batch results saved to {output_path}")

    return list(results)


def save_result(result: TestResult, output_path: str):
    d = result.to_dict()
    if output_path.endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=d.keys())
            writer.writeheader()
            writer.writerow(d)
    else:
        with open(output_path, "w") as f:
            json.dump(d, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_result(r: TestResult):
    print(f"\n{'=' * 60}")
    print(f"  Neural Blitz NG v{__version__} — Results")
    print(f"{'=' * 60}")
    print(f"  Target:        {r.target_name}")
    print(f"  Packets:       {r.count_received}/{r.count_sent} ({r.success_rate}%)")
    print(f"  Lost:          {r.count_lost}")
    print(f"  Duration:      {r.duration_s}s")
    print(f"  Throughput:    {r.pps} pps")
    print(f"  Loop Engine:   {r.loop_engine}\n")
    print("  Latency (µs):")
    print(f"    min:    {r.min_us:>12.1f}")
    print(f"    p50:    {r.p50_us:>12.1f}")
    print(f"    p90:    {r.p90_us:>12.1f}")
    print(f"    p95:    {r.p95_us:>12.1f}")
    print(f"    p99:    {r.p99_us:>12.1f}")
    print(f"    p99.9:  {r.p999_us:>12.1f}")
    print(f"    max:    {r.max_us:>12.1f}")
    print(f"    mean:   {r.mean_us:>12.1f}")
    print(f"    stddev: {r.stddev_us:>12.1f}")
    print(f"    jitter: {r.jitter_us:>12.1f}\n")
    print("  Coordinated-Omission Corrected:")
    print(f"    co_p50:   {r.co_p50_us:>12.1f}")
    print(f"    co_p99:   {r.co_p99_us:>12.1f}")
    print(f"    co_p99.9: {r.co_p999_us:>12.1f}")
    print(f"{'=' * 60}\n")


def _prom_escape(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


async def run_monitor(config_path: str, bind: str, http_port: int, interval: int):
    try:
        from aiohttp import web
    except ImportError:
        logger.error("Monitor mode requires aiohttp: pip install aiohttp")
        sys.exit(1)

    if not HAS_YAML:
        logger.error("Monitor mode requires PyYAML: pip install pyyaml")
        sys.exit(1)

    timeseries: Dict[str, List[Dict]] = {}
    latest: Dict[str, TestResult] = {}

    async def run_cycle():
        results = await run_batch(config_path)
        for r in results:
            latest[r.target_name] = r
            timeseries.setdefault(r.target_name, []).append(r.to_dict())
            if len(timeseries[r.target_name]) > 1000:
                timeseries[r.target_name] = timeseries[r.target_name][-1000:]

    async def metrics_json(request):
        return web.json_response({k: v.to_dict() for k, v in latest.items()})

    async def metrics_prometheus(request):
        lines = []
        for tname, r in latest.items():
            safe = _prom_escape(tname)
            lines.append(f'nblitz_p50_us{{target="{safe}"}} {r.p50_us}')
            lines.append(f'nblitz_p99_us{{target="{safe}"}} {r.p99_us}')
            lines.append(f'nblitz_success_rate{{target="{safe}"}} {r.success_rate}')
            lines.append(f'nblitz_pps{{target="{safe}"}} {r.pps}')
            lines.append(f'nblitz_jitter_us{{target="{safe}"}} {r.jitter_us}')
            lines.append(f'nblitz_lost{{target="{safe}"}} {r.count_lost}')
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def health(request):
        return web.json_response({"status": "ok", "version": __version__})

    async def api_targets(request):
        return web.json_response(list(latest.keys()))

    async def api_target(request):
        return web.json_response(timeseries.get(request.match_info["name"], []))

    app = web.Application()
    app.router.add_get("/metrics", metrics_json)
    app.router.add_get("/metrics/prometheus", metrics_prometheus)
    app.router.add_get("/health", health)
    app.router.add_get("/api/targets", api_targets)
    app.router.add_get("/api/target/{name}", api_target)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, bind, http_port)
    await site.start()
    logger.info(f"Monitor HTTP on {bind}:{http_port}")

    while True:
        try:
            await run_cycle()
            logger.info(f"Monitor cycle done — {len(latest)} targets")
        except Exception as e:
            logger.error(f"Monitor cycle error: {e}")
        await asyncio.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural_blitz_ng",
        description=f"Neural Blitz NG v{__version__} — UDP Network Latency Benchmark",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", help="Command to run")

    t = sub.add_parser("test", help="Run single target test")
    t.add_argument("--host", default="127.0.0.1")
    t.add_argument("--port", type=int, default=9999)
    t.add_argument("--count", type=int, default=500)
    t.add_argument("--rate", type=float, default=5000)
    t.add_argument("--concurrency", type=int, default=50)
    t.add_argument("--warmup", type=int, default=0)
    t.add_argument("--timeout", type=float, default=5.0)
    t.add_argument("--output", "-o")
    t.add_argument("--name", default="")

    b = sub.add_parser("batch", help="Multi-target batch from YAML config")
    b.add_argument("--config", required=True)
    b.add_argument("--output", "-o")

    s = sub.add_parser("server", help="Run UDP echo server")
    s.add_argument("--bind", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9999)

    m = sub.add_parser("monitor", help="Continuous monitoring + HTTP metrics")
    m.add_argument("--config", required=True)
    m.add_argument("--bind", default="0.0.0.0")
    m.add_argument("--http-port", type=int, default=8888)
    m.add_argument("--interval", type=int, default=30)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if LOOP_ENGINE == "uvloop":
        uvloop.install()

    if args.command == "test":
        result = asyncio.run(
            run_single_test(
                host=args.host,
                port=args.port,
                count=args.count,
                rate=args.rate,
                concurrency=args.concurrency,
                warmup=args.warmup,
                timeout=args.timeout,
                name=args.name,
            )
        )
        print_result(result)
        if args.output:
            save_result(result, args.output)

    elif args.command == "batch":
        results = asyncio.run(run_batch(args.config, args.output))
        for r in results:
            print_result(r)

    elif args.command == "server":
        asyncio.run(run_echo_server(args.bind, args.port))

    elif args.command == "monitor":
        asyncio.run(run_monitor(args.config, args.bind, args.http_port, args.interval))


if __name__ == "__main__":
    main()
