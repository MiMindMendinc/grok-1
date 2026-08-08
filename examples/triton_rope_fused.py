# Copyright 2024 X.AI Corp.
# Copyright 2026 MiMindMendinc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton implementation of Grok-1 style rotary embeddings.

This is an optional PyTorch/Triton reference and microbenchmark. It intentionally
lives under ``examples/`` and does not add PyTorch/Triton to Grok-1's JAX runtime
requirements.

The math matches ``model.py::RotaryEmbedding`` exactly:

    phase = tile(phase_half, reps=(1, 2))
    out = x * cos(phase) + rotate_half(x) * sin(phase)

Typical use:

    python examples/triton_rope_fused.py --check
    python examples/triton_rope_fused.py --benchmark --json results.json

Requires a CUDA GPU plus PyTorch and Triton.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_cos_t,
    stride_cos_d,
    stride_sin_t,
    stride_sin_d,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One Triton program owns one complete (B, T, H) feature row.

    Flattening B*T*H into a single launch dimension avoids relying on a fourth
    program-id axis and guarantees that an in-place store cannot race another
    program reading the paired half of the same row.
    """
    row = tl.program_id(0)
    h = row % H
    tmp = row // H
    t = tmp % T
    b = tmp // T

    d = tl.arange(0, BLOCK_D)
    mask = d < D
    half = D // 2
    paired_d = tl.where(d < half, d + half, d - half)

    x_base = b * stride_x_b + t * stride_x_t + h * stride_x_h
    o_base = b * stride_o_b + t * stride_o_t + h * stride_o_h

    x = tl.load(x_ptr + x_base + d * stride_x_d, mask=mask, other=0.0)
    paired = tl.load(
        x_ptr + x_base + paired_d * stride_x_d,
        mask=mask,
        other=0.0,
    )
    rotated = paired * tl.where(d < half, -1.0, 1.0)

    cos_v = tl.load(
        cos_ptr + t * stride_cos_t + d * stride_cos_d,
        mask=mask,
        other=0.0,
    )
    sin_v = tl.load(
        sin_ptr + t * stride_sin_t + d * stride_sin_d,
        mask=mask,
        other=0.0,
    )

    out = x * cos_v + rotated * sin_v
    tl.store(out_ptr + o_base + d * stride_o_d, out, mask=mask)


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    """PyTorch equivalent of Grok-1 ``model.py::rotate_half``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def make_grok_rope_cache(
    sequence_length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    base_exponent: float = 10000.0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin tables with the same frequency layout as Grok-1 JAX.

    Grok-1 computes frequencies for dimensions ``0, 2, ...`` and then tiles the
    half-width phase vector twice. ``torch.cat((phase, phase), dim=-1)`` is the
    PyTorch equivalent. This is deliberately *not* ``repeat_interleave``.
    """
    if dim <= 0 or dim % 2:
        raise ValueError("dim must be a positive even integer")

    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base_exponent ** (exponents / dim))
    positions = torch.arange(
        offset,
        offset + sequence_length,
        device=device,
        dtype=torch.float32,
    )
    phase_half = torch.outer(positions, inv_freq)
    phase = torch.cat((phase_half, phase_half), dim=-1)
    return torch.cos(phase).to(dtype), torch.sin(phase).to(dtype)


def pytorch_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Literal PyTorch translation of Grok-1 RotaryEmbedding math."""
    return x * cos[None, :, None, :] + rotate_half_reference(x) * sin[None, :, None, :]


def fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply Grok-style RoPE to ``x`` shaped ``(B, T, H, D)``.

    Supports contiguous and non-contiguous input layouts through explicit
    strides. ``cos`` and ``sin`` must be CUDA tensors shaped ``(T, D)``.
    """
    if not x.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise ValueError("x, cos, and sin must be CUDA tensors")
    if x.ndim != 4:
        raise ValueError(f"x must have shape (B, T, H, D); got {tuple(x.shape)}")

    B, T, H, D = x.shape
    if D % 2:
        raise ValueError("RoPE feature dimension must be even")
    if cos.shape != (T, D) or sin.shape != (T, D):
        raise ValueError(
            f"cos/sin must both have shape {(T, D)}; got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise ValueError("x, cos, and sin must use the same dtype")

    out = x if inplace else torch.empty_like(x, memory_format=torch.preserve_format)
    block_d = triton.next_power_of_2(D)
    if block_d > 65536:
        raise ValueError(f"unsupported feature dimension D={D}")

    # Grok-1 uses D=128. Four warps is a strong default at this width while
    # retaining reasonable occupancy for neighboring test dimensions.
    num_warps = 4 if block_d <= 256 else 8
    grid = (B * T * H,)

    _fused_rope_kernel[grid](
        x,
        cos,
        sin,
        out,
        *x.stride(),
        *out.stride(),
        *cos.stride(),
        *sin.stride(),
        T=T,
        H=H,
        D=D,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


def _make_x(
    batch: int,
    sequence_length: int,
    heads: int,
    dim: int,
    *,
    dtype: torch.dtype,
    noncontiguous: bool,
) -> torch.Tensor:
    if noncontiguous:
        src = torch.randn(
            batch,
            heads,
            sequence_length,
            dim,
            device="cuda",
            dtype=dtype,
        )
        return src.transpose(1, 2)
    return torch.randn(
        batch,
        sequence_length,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def check_correctness() -> None:
    """Exercise shape, dtype, layout, offset, and in-place correctness."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton correctness suite")

    cases: Iterable[tuple[int, int, int, int, torch.dtype, bool, int]] = (
        (1, 1, 1, 128, torch.float16, False, 0),
        (1, 17, 8, 128, torch.float16, False, 0),
        (1, 257, 8, 128, torch.float16, True, 0),
        (2, 129, 8, 128, torch.float16, True, 37),
        (1, 257, 8, 128, torch.bfloat16, False, 0),
        (1, 257, 48, 128, torch.bfloat16, True, 19),
        (1, 129, 8, 64, torch.float16, False, 0),
        (1, 129, 8, 256, torch.float16, False, 0),
    )

    for B, T, H, D, dtype, noncontiguous, offset in cases:
        x = _make_x(B, T, H, D, dtype=dtype, noncontiguous=noncontiguous)
        cos, sin = make_grok_rope_cache(
            T,
            D,
            device=x.device,
            dtype=dtype,
            offset=offset,
        )
        expected = pytorch_rope_reference(x, cos, sin)
        actual = fused_rope(x, cos, sin)
        atol, rtol = _tolerances(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        x_inplace = x.clone(memory_format=torch.preserve_format)
        expected_inplace = pytorch_rope_reference(x_inplace, cos, sin)
        actual_inplace = fused_rope(x_inplace, cos, sin, inplace=True)
        torch.testing.assert_close(actual_inplace, expected_inplace, atol=atol, rtol=rtol)
        if actual_inplace.data_ptr() != x_inplace.data_ptr():
            raise AssertionError("in-place path did not reuse input storage")

        print(
            "PASS",
            f"B={B} T={T} H={H} D={D}",
            str(dtype).removeprefix("torch."),
            f"noncontiguous={noncontiguous}",
            f"offset={offset}",
        )


@dataclass
class BenchmarkResult:
    batch: int
    sequence_length: int
    heads: int
    dim: int
    dtype: str
    noncontiguous: bool
    pytorch_ms: float
    triton_ms: float
    speedup: float
    max_abs_error: float


def _bench_ms(fn, *, warmup: int = 25, rep: int = 100) -> float:
    """Benchmark with Triton's CUDA-event based harness."""
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def benchmark_case(
    *,
    batch: int,
    sequence_length: int,
    heads: int,
    dim: int,
    dtype: torch.dtype,
    noncontiguous: bool,
    warmup: int,
    rep: int,
) -> BenchmarkResult:
    x = _make_x(
        batch,
        sequence_length,
        heads,
        dim,
        dtype=dtype,
        noncontiguous=noncontiguous,
    )
    cos, sin = make_grok_rope_cache(
        sequence_length,
        dim,
        device=x.device,
        dtype=dtype,
    )

    expected = pytorch_rope_reference(x, cos, sin)
    actual = fused_rope(x, cos, sin)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs_error = float((actual - expected).abs().max().float().item())

    pytorch_ms = _bench_ms(
        lambda: pytorch_rope_reference(x, cos, sin),
        warmup=warmup,
        rep=rep,
    )
    triton_ms = _bench_ms(
        lambda: fused_rope(x, cos, sin),
        warmup=warmup,
        rep=rep,
    )

    return BenchmarkResult(
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        dim=dim,
        dtype=str(dtype).removeprefix("torch."),
        noncontiguous=noncontiguous,
        pytorch_ms=pytorch_ms,
        triton_ms=triton_ms,
        speedup=pytorch_ms / triton_ms,
        max_abs_error=max_abs_error,
    )


def run_benchmarks(args: argparse.Namespace) -> list[BenchmarkResult]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    results = []
    for T in args.sequence_lengths:
        result = benchmark_case(
            batch=args.batch,
            sequence_length=T,
            heads=args.heads,
            dim=args.dim,
            dtype=dtype,
            noncontiguous=args.noncontiguous,
            warmup=args.warmup,
            rep=args.rep,
        )
        results.append(result)
        print(
            f"T={T:>6}  PyTorch={result.pytorch_ms:>9.4f} ms  "
            f"Triton={result.triton_ms:>9.4f} ms  "
            f"speedup={result.speedup:>6.2f}x  "
            f"max_err={result.max_abs_error:.6g}"
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness matrix")
    parser.add_argument("--benchmark", action="store_true", help="run latency sweep")
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None, help="write machine-readable results")
    args = parser.parse_args()
    if not args.check and not args.benchmark:
        args.check = True
        args.benchmark = True
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    if args.check:
        check_correctness()

    results: list[BenchmarkResult] = []
    if args.benchmark:
        results = run_benchmarks(args)

    if args.json is not None:
        payload = {
            "gpu": torch.cuda.get_device_name(),
            "pytorch": torch.__version__,
            "triton": triton.__version__,
            "results": [asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
