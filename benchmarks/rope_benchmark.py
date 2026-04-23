import argparse
import math
import random
import time
from dataclasses import dataclass

def old_rope_python(head_dim: int, seq_len: int, offset: float, base_exponent: int = 10000) -> float:
    acc = 0.0
    for pos in range(seq_len):
        # Simulates old path overhead by rebuilding frequency terms each step.
        inv_freq = [1.0 / (base_exponent ** ((2 * i) / head_dim)) for i in range(head_dim // 2)]
        t = pos + offset
        for f in inv_freq:
            phase = t * f
            acc += math.sin(phase) + math.cos(phase)
    return acc


def new_rope_python(head_dim: int, seq_len: int, offset: float, inv_freq) -> float:
    acc = 0.0
    for pos in range(seq_len):
        t = pos + offset
        for f in inv_freq:
            phase = t * f
            acc += math.sin(phase) + math.cos(phase)
    return acc


def bench(fn, *args, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - start) * 1000.0 / iters


@dataclass
class RopeBenchmarkResult:
    old_ms: float
    new_ms: float

    @property
    def speedup(self) -> float:
        return self.old_ms / self.new_ms if self.new_ms > 0 else float("inf")


def run_rope_only_numpy(batch: int, seq: int, heads: int, dim: int, iters: int) -> RopeBenchmarkResult:
    # stdlib-only benchmark fallback (works even without numpy/jax)
    offset = random.random()
    inv_freq = [1.0 / (10000 ** ((2 * i) / dim)) for i in range(dim // 2)]
    tokens = batch * seq * heads
    old_ms = bench(old_rope_python, dim, tokens, offset, 10000, iters=iters)
    new_ms = bench(new_rope_python, dim, tokens, offset, inv_freq, iters=iters)
    return RopeBenchmarkResult(old_ms=old_ms, new_ms=new_ms)


def run_jax_rope_backend_benchmark(batch: int, seq: int, heads: int, dim: int, iters: int):
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        return None, f"JAX unavailable: {exc}"

    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((-x2, x1), axis=-1)

    def old_rope(x, offset):
        exponents = jnp.arange(0, dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(1.0 / (10000 ** (exponents / dim)), dtype=jnp.float32)
        t = jnp.arange(x.shape[1], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]
        return x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)

    @jax.jit
    def new_rope(x, offset, inv_freq):
        t = jnp.arange(x.shape[1], dtype=jnp.float32)[None, :] + offset[:, None]
        phase = t[:, :, None] * inv_freq[None, None, :]
        phase = jnp.repeat(phase, repeats=2, axis=-1)[:, :, None, :]
        return x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, heads, dim), dtype=jnp.float32)
    offset = jnp.zeros((batch,), dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    old_jit = jax.jit(old_rope)
    old_ms = bench(lambda a, b: jax.block_until_ready(old_jit(a, b)), x, offset, iters=iters)
    new_ms = bench(lambda a, b, c: jax.block_until_ready(new_rope(a, b, c)), x, offset, inv_freq, iters=iters)
    return RopeBenchmarkResult(old_ms=old_ms, new_ms=new_ms), None


def run_full_forward_pass_benchmark(iters: int):
    try:
        import jax
        import jax.numpy as jnp
        import haiku as hk

        from model import LanguageModelConfig, TransformerConfig
    except Exception as exc:
        return None, f"Full forward pass unavailable: {exc}"

    cfg = LanguageModelConfig(
        vocab_size=4096,
        pad_token=0,
        eos_token=2,
        sequence_len=128,
        model=TransformerConfig(
            emb_size=512,
            widening_factor=4,
            key_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            num_layers=4,
            num_experts=1,
            num_selected_experts=1,
            rope_backend="jax",
        ),
    ).initialize()

    def forward(tokens):
        return cfg.make(mesh=None)(tokens).logits

    fn = hk.transform(forward)
    tokens = jnp.ones((1, 128), dtype=jnp.int32)
    params = fn.init(jax.random.PRNGKey(0), tokens)
    apply_jit = jax.jit(lambda p, x: fn.apply(p, None, x))
    ms = bench(lambda p, x: jax.block_until_ready(apply_jit(p, x)), params, tokens, iters=iters)
    return ms, None


def run_triton_rope_benchmark(batch: int, seq: int, q_heads: int, kv_heads: int, dim: int, iters: int):
    try:
        import torch
        import rope_triton
    except Exception as exc:
        return None, f"Triton benchmark unavailable: {exc}"

    if not rope_triton.is_triton_available():
        return None, "Triton benchmark unavailable: Triton/PyTorch runtime not installed."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(batch, seq, q_heads, dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float32)
    offset = torch.zeros((batch,), device=device, dtype=torch.float32)

    def baseline(q_, k_, off_):
        return rope_triton.apply_rope_torch_reference(q_, k_, off_, base_exponent=10000)

    def triton_path(q_, k_, off_):
        # currently routes through rope_triton fused entrypoint
        return rope_triton.apply_rope_torch(q_, k_, off_, base_exponent=10000)

    if device == "cuda":
        torch.cuda.synchronize()
    old_ms = bench(baseline, q, k, offset, iters=iters)
    if device == "cuda":
        torch.cuda.synchronize()
    new_ms = bench(triton_path, q, k, offset, iters=iters)
    if device == "cuda":
        torch.cuda.synchronize()
    return RopeBenchmarkResult(old_ms=old_ms, new_ms=new_ms), None


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RoPE and forward-pass performance.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print_section("RoPE benchmark (stdlib fallback path)")
    np_res = run_rope_only_numpy(args.batch, args.seq, args.heads, args.dim, args.iters)
    print(f"old_rope_python: {np_res.old_ms:.3f} ms")
    print(f"new_rope_python: {np_res.new_ms:.3f} ms")
    print(f"speedup:        {np_res.speedup:.2f}x")

    print_section("RoPE benchmark (JAX path)")
    jax_res, jax_err = run_jax_rope_backend_benchmark(
        args.batch, args.seq, args.heads, args.dim, args.iters
    )
    if jax_res is None:
        print(f"skipped: {jax_err}")
    else:
        print(f"old_rope_jax:   {jax_res.old_ms:.3f} ms")
        print(f"new_rope_jax:   {jax_res.new_ms:.3f} ms")
        print(f"speedup:        {jax_res.speedup:.2f}x")

    print_section("Full model forward pass timing")
    forward_ms, forward_err = run_full_forward_pass_benchmark(args.iters)
    if forward_ms is None:
        print(f"skipped: {forward_err}")
    else:
        print(f"forward_pass_jax: {forward_ms:.3f} ms")

    print_section("Triton vs JAX/PyTorch RoPE timing")
    triton_res, triton_err = run_triton_rope_benchmark(
        args.batch, min(args.seq, 512), args.heads, max(1, args.heads // 4), args.dim, args.iters
    )
    if triton_res is None:
        print(f"skipped: {triton_err}")
    else:
        print(f"baseline_ms:     {triton_res.old_ms:.3f} ms")
        print(f"triton_ms:       {triton_res.new_ms:.3f} ms")
        print(f"speedup:         {triton_res.speedup:.2f}x")


if __name__ == "__main__":
    main()
