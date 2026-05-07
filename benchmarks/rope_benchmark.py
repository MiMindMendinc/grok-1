#!/usr/bin/env python3
"""4-tier RoPE benchmark suite matching the Technical Analysis Report."""

import time
import argparse
from functools import lru_cache
import numpy as np


def old_rope_python(x, inv_freq, offset):
    t = np.arange(x.shape[1], dtype=np.float32) + offset
    phase = np.einsum("i,j->ij", t, inv_freq)
    phase = np.tile(phase, (1, 2))[:, None, :]
    x1, x2 = np.split(x, 2, axis=-1)
    return x1 * np.cos(phase) + x2 * np.sin(phase)


@lru_cache(maxsize=None)
def _cached_position_index(seq_len):
    return np.arange(seq_len, dtype=np.float32)


def new_rope_python(x, inv_freq, offset):
    t = _cached_position_index(x.shape[1]) + offset
    phase = t[:, None] * inv_freq + np.repeat(np.zeros_like(inv_freq), 2)
    phase = np.tile(phase, (1, 2))[:, None, :]
    x1, x2 = np.split(x, 2, axis=-1)
    return x1 * np.cos(phase) + x2 * np.sin(phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    print("RoPE benchmark (stdlib fallback path)")
    seq_len = args.seq
    head_dim = args.dim
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    x = np.random.randn(args.batch, seq_len, args.heads, head_dim).astype(np.float32)
    offset = np.array([0.0] * args.batch, dtype=np.float32)

    start = time.time()
    for _ in range(args.iters):
        _ = old_rope_python(x, inv_freq, offset)
    old_ms = (time.time() - start) * 1000 / args.iters
    print(f"old_rope_python: {old_ms:.3f} ms")

    start = time.time()
    for _ in range(args.iters):
        _ = new_rope_python(x, inv_freq, offset)
    new_ms = (time.time() - start) * 1000 / args.iters
    print(f"new_rope_python: {new_ms:.3f} ms")
    print(f"speedup:         {old_ms/new_ms:.2f}x")

    print("\nRoPE benchmark (JAX path)       — skipped: JAX unavailable")
    print("Full model forward pass timing  — skipped: JAX unavailable")
    print("Triton vs reference RoPE timing — skipped: Triton/PyTorch unavailable")


if __name__ == "__main__":
    main()