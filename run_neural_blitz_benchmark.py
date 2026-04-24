#!/usr/bin/env python3
"""Run Neural Blitz NG v7.1 benchmark suite and generate a shareable benchmark card."""

from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Dict, List

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_blitz_ng.py")
OUT_JSON = "benchmark_summary.json"
OUT_MD = "benchmark_summary.md"
OUT_SVG = "benchmark_showcase.svg"


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def run_test(label: str, args: List[str], out_file: str) -> Dict:
    result = run([sys.executable, SCRIPT, "test", *args, "--output", out_file])
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with rc={result.returncode}")
    with open(out_file) as fh:
        payload = json.load(fh)
    payload["label"] = label
    payload["artifact"] = out_file
    return payload


def build_markdown(summary: Dict, rows: List[Dict]) -> str:
    lines = [
        "# Neural Blitz NG v7.1 — Benchmark Showcase",
        "",
        f"- **Generated:** {summary['generated_utc']}",
        f"- **Host:** {summary['target']}",
        f"- **Loop:** {summary['loop_engine']}",
        f"- **Overall success:** {summary['overall_success_rate']}%",
        "",
        "## Results",
        "",
        "| Scenario | Received/Sent | Success | Throughput (pps) | p50 (µs) | p99 (µs) | Jitter (µs) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['count_received']}/{r['count_sent']} | {r['success_rate']}% "
            f"| {r['pps']} | {r['p50_us']} | {r['p99_us']} | {r['jitter_us']} |"
        )

    lines += [
        "",
        "## Files",
        "",
        "- `benchmark_summary.json`",
        "- `benchmark_summary.md`",
        "- `benchmark_showcase.svg`",
    ]
    return "\n".join(lines) + "\n"


def build_svg(summary: Dict, rows: List[Dict]) -> str:
    row_y = [230, 290, 350]

    def badge(text: str, color: str, x: int) -> str:
        return (
            f'<rect x="{x}" y="122" rx="10" ry="10" width="205" height="38" fill="{color}" opacity="0.95"/>'
            f'<text x="{x + 16}" y="147" font-size="17" fill="#ffffff" font-family="Inter,Segoe UI,Arial" font-weight="700">{text}</text>'
        )

    badges = "".join(
        [
            badge(f"Overall Success: {summary['overall_success_rate']}%", "#0f766e", 40),
            badge(f"Avg Throughput: {summary['avg_pps']} pps", "#1d4ed8", 270),
            badge(f"Avg p99: {summary['avg_p99_us']} µs", "#7c3aed", 500),
        ]
    )

    row_blocks = []
    for idx, row in enumerate(rows):
        y = row_y[idx]
        fill = "#0b1220" if idx % 2 == 0 else "#0f172a"
        row_blocks.append(
            f'<rect x="40" y="{y-30}" width="740" height="56" rx="8" fill="{fill}"/>'
            f'<text x="56" y="{y}" fill="#e5e7eb" font-size="16" font-family="Inter,Segoe UI,Arial" font-weight="700">{row["label"]}</text>'
            f'<text x="320" y="{y}" fill="#93c5fd" font-size="16" font-family="Inter,Segoe UI,Arial">{row["count_received"]}/{row["count_sent"]}</text>'
            f'<text x="430" y="{y}" fill="#86efac" font-size="16" font-family="Inter,Segoe UI,Arial">{row["success_rate"]}%</text>'
            f'<text x="520" y="{y}" fill="#fef08a" font-size="16" font-family="Inter,Segoe UI,Arial">{row["pps"]} pps</text>'
            f'<text x="645" y="{y}" fill="#f9a8d4" font-size="16" font-family="Inter,Segoe UI,Arial">p99 {row["p99_us"]}µs</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="500" viewBox="0 0 820 500">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#020617"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <linearGradient id="hero" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>

  <rect width="820" height="500" fill="url(#bg)"/>
  <rect x="24" y="20" width="772" height="460" rx="18" fill="#0b1020" stroke="#1f2937"/>
  <rect x="24" y="20" width="772" height="88" rx="18" fill="url(#hero)"/>

  <text x="44" y="64" fill="#ffffff" font-size="30" font-family="Inter,Segoe UI,Arial" font-weight="800">Neural Blitz NG v7.1</text>
  <text x="44" y="90" fill="#e0f2fe" font-size="16" font-family="Inter,Segoe UI,Arial">Benchmark Showcase • {summary['generated_utc']}</text>

  {badges}

  <text x="56" y="198" fill="#94a3b8" font-size="13" font-family="Inter,Segoe UI,Arial" font-weight="700">Scenario</text>
  <text x="320" y="198" fill="#94a3b8" font-size="13" font-family="Inter,Segoe UI,Arial" font-weight="700">Rx/Tx</text>
  <text x="430" y="198" fill="#94a3b8" font-size="13" font-family="Inter,Segoe UI,Arial" font-weight="700">Success</text>
  <text x="520" y="198" fill="#94a3b8" font-size="13" font-family="Inter,Segoe UI,Arial" font-weight="700">Throughput</text>
  <text x="645" y="198" fill="#94a3b8" font-size="13" font-family="Inter,Segoe UI,Arial" font-weight="700">Latency</text>

  {''.join(row_blocks)}

  <text x="44" y="456" fill="#64748b" font-size="12" font-family="Inter,Segoe UI,Arial">target={summary['target']} • loop={summary['loop_engine']} • artifacts: {OUT_JSON}, {OUT_MD}, {OUT_SVG}</text>
</svg>
'''


def main():
    run([sys.executable, SCRIPT, "--version"])

    print("\n[*] Starting echo server on 127.0.0.1:9999 ...")
    server = subprocess.Popen(
        [sys.executable, SCRIPT, "server", "--bind", "127.0.0.1", "--port", "9999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)

    artifacts = ["result_standard.json", "result_export.json", "result_stress.json"]
    try:
        results = [
            run_test(
                "Standard • 500 @ 5kpps",
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9999",
                    "--count",
                    "500",
                    "--rate",
                    "5000",
                    "--warmup",
                    "25",
                ],
                "result_standard.json",
            ),
            run_test(
                "Export • 200 @ 5kpps",
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9999",
                    "--count",
                    "200",
                    "--rate",
                    "5000",
                ],
                "result_export.json",
            ),
            run_test(
                "Stress • 1k @ 20kpps",
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9999",
                    "--count",
                    "1000",
                    "--rate",
                    "20000",
                    "--concurrency",
                    "100",
                ],
                "result_stress.json",
            ),
        ]

        summary = {
            "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target": f"{results[0]['target_host']}:{results[0]['target_port']}",
            "loop_engine": results[0]["loop_engine"],
            "overall_success_rate": round(statistics.mean(r["success_rate"] for r in results), 2),
            "avg_pps": round(statistics.mean(r["pps"] for r in results), 1),
            "avg_p99_us": round(statistics.mean(r["p99_us"] for r in results), 1),
            "scenarios": results,
        }

        with open(OUT_JSON, "w") as fh:
            json.dump(summary, fh, indent=2)

        with open(OUT_MD, "w") as fh:
            fh.write(build_markdown(summary, results))

        with open(OUT_SVG, "w") as fh:
            fh.write(build_svg(summary, results))

        print("\n" + "=" * 70)
        print(" BENCHMARK SHOWCASE READY")
        print("=" * 70)
        print(f"  Summary JSON : {OUT_JSON}")
        print(f"  Summary MD   : {OUT_MD}")
        print(f"  Pro SVG Card : {OUT_SVG}")
        print("=" * 70)
        print("Tip: open benchmark_showcase.svg and take your screenshot.")

    finally:
        server.terminate()
        server.wait(timeout=5)
        for a in artifacts:
            if os.path.exists(a):
                os.remove(a)
        print("\n[*] Echo server stopped.")


if __name__ == "__main__":
    main()
