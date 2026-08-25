"""Regression tests for RoPE (JAX + Triton)."""

import argparse
import math

import numpy as np
import pytest

from run import validate_args

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from model import RotaryEmbedding
    from rope_triton import (
        apply_rope_jax_compatible,
        apply_rope_jax_reference,
        apply_rope_triton_jax,
        can_apply_rope_to_jax_array,
    )

    HAS_JAX_STACK = True
except (ModuleNotFoundError, ImportError, RuntimeError):
    HAS_JAX_STACK = False

try:
    from rope_triton import is_triton_available

    HAS_TRITON = is_triton_available()
except Exception:
    HAS_TRITON = False


def test_rope_test_environment_smoke():
    if not HAS_JAX_STACK:
        pytest.skip("JAX/Haiku not installed in this environment.")
    assert True


def test_python_rope_precomputed_freqs_match_rebuilt_freqs():
    """Stdlib RoPE paths must agree before any speedup is meaningful."""
    dim, seq_len, offset, base = 8, 16, 1.25, 10000
    inv_freq = [1.0 / (base ** ((2 * i) / dim)) for i in range(dim // 2)]

    rebuilt = 0.0
    for pos in range(seq_len):
        freqs = [1.0 / (base ** ((2 * i) / dim)) for i in range(dim // 2)]
        t = pos + offset
        for f in freqs:
            rebuilt += math.sin(t * f) + math.cos(t * f)

    precomputed = 0.0
    for pos in range(seq_len):
        t = pos + offset
        for f in inv_freq:
            precomputed += math.sin(t * f) + math.cos(t * f)

    assert rebuilt == pytest.approx(precomputed)


@pytest.mark.skipif(not HAS_JAX_STACK, reason="JAX/Haiku not installed.")
def test_jax_rope_benchmark_passes_correctness_gate():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "rope_benchmark.py"
    spec = importlib.util.spec_from_file_location("rope_benchmark_jax_gate", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    result, err = bench.run_jax_rope_backend_benchmark(
        batch=1, seq=8, heads=2, dim=8, iters=1, rtol=1e-5, atol=1e-5
    )
    assert err is None, err
    assert result.correctness_ok, result.correctness_message
    assert result.max_abs_diff < 1e-5


@pytest.mark.skipif(not HAS_JAX_STACK, reason="JAX/Haiku not installed.")
def test_rotary_embedding_const_position_zero_is_supported():
    def forward(x, offset):
        rope = RotaryEmbedding(dim=x.shape[-1], base_exponent=10000)
        return rope(x, seq_dim=1, offset=offset, const_position=0)

    fn = hk.transform(forward)
    x = jnp.ones((2, 4, 3, 8), dtype=jnp.bfloat16)
    offset = jnp.array([0.0, 0.0], dtype=jnp.float32)
    params = fn.init(jax.random.PRNGKey(0), x, offset)
    out = fn.apply(params, None, x, offset)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.skipif(not HAS_JAX_STACK, reason="JAX/Haiku not installed.")
def test_rotary_embedding_matches_reference_phase_construction():
    def forward(x, offset):
        rope = RotaryEmbedding(dim=x.shape[-1], base_exponent=10000)
        return rope(x, seq_dim=1, offset=offset)

    fn = hk.transform(forward)
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 8, 2, 8), dtype=jnp.float32)
    offset = jnp.array([3.0], dtype=jnp.float32)
    params = fn.init(jax.random.PRNGKey(2), x, offset)
    out = fn.apply(params, None, x, offset)

    exponents = jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (exponents / x.shape[-1]))
    t = jnp.arange(x.shape[1], dtype=jnp.float32)[None, :] + offset[:, None]
    phase = jnp.einsum("bi,j->bij", t, inv_freq)
    phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    ref = x * jnp.cos(phase) + rotated * jnp.sin(phase)

    assert jnp.allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not HAS_JAX_STACK, reason="JAX stack not installed.")
def test_rope_output_shape():
    x = jnp.ones((2, 128, 8, 64), dtype=jnp.float32)
    off = jnp.array([0.0, 0.0], dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))

    out = apply_rope_jax_compatible(x=x, offset=off, inv_freq=inv_freq)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.skipif(not (HAS_JAX_STACK and HAS_TRITON), reason="Triton CUDA backend not available")
def test_triton_matches_jax_reference():
    """Triton path must match the pure JAX reference within tolerance."""
    x = jax.random.normal(jax.random.PRNGKey(7), (1, 64, 8, 128), dtype=jnp.float32)
    off = jnp.array([0.0], dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))

    jax_out = apply_rope_jax_reference(x=x, offset=off, inv_freq=inv_freq)

    if not can_apply_rope_to_jax_array(x):
        pytest.skip("JAX DLPack bridge unavailable for Triton validation")

    triton_out = apply_rope_triton_jax(x=x, offset=off, inv_freq=inv_freq)
    np.testing.assert_allclose(
        np.array(jax_out),
        np.array(triton_out),
        rtol=1e-3,
        atol=1e-3,
        err_msg="Triton and JAX RoPE outputs differ significantly",
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton CUDA backend not available")
def test_triton_matches_torch_reference():
    """Fused Triton kernel must match the PyTorch reference within tolerance."""
    import torch

    from rope_triton import apply_rope_torch, apply_rope_torch_reference

    q = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 32, 2, 64, device="cuda", dtype=torch.float32)
    offset = torch.zeros((2,), device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))

    q_ref, k_ref = apply_rope_torch_reference(q, k, offset, inv_freq=inv_freq)
    q_tri, k_tri = apply_rope_torch(q, k, offset, inv_freq=inv_freq)
    np.testing.assert_allclose(
        q_tri.detach().cpu().numpy(),
        q_ref.detach().cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
        err_msg="Triton Q RoPE differs from the PyTorch reference",
    )
    np.testing.assert_allclose(
        k_tri.detach().cpu().numpy(),
        k_ref.detach().cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
        err_msg="Triton K RoPE differs from the PyTorch reference",
    )


def test_pad_size_validation():
    """Non-positive pad sizes should be rejected early."""
    good = argparse.Namespace(
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
        sequence_len=128,
        pad_sizes=[64, 128],
    )
    validate_args(good)

    with pytest.raises(ValueError, match="pad-sizes values must all be > 0"):
        validate_args(
            argparse.Namespace(
                max_new_tokens=16,
                temperature=0.7,
                top_p=0.95,
                sequence_len=128,
                pad_sizes=[0, 128],
            )
        )


def test_benchmark_withholds_speedup_when_gate_fails(capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "rope_benchmark.py"
    spec = importlib.util.spec_from_file_location("rope_benchmark", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    failed = bench.RopeBenchmarkResult(
        old_ms=10.0,
        new_ms=1.0,
        correctness_ok=False,
        correctness_message="mismatch",
        max_abs_diff=9.0,
    )
    assert bench.report_result(("old_ms", "new_ms"), failed) is False
    failed_out = capsys.readouterr().out
    assert "withheld" in failed_out
    assert "10.00x" not in failed_out

    passed = bench.RopeBenchmarkResult(
        old_ms=10.0,
        new_ms=2.0,
        correctness_ok=True,
        max_abs_diff=0.0,
    )
    assert bench.report_result(("old_ms", "new_ms"), passed) is True
    passed_out = capsys.readouterr().out
    assert "5.00x" in passed_out


def test_run_cli_exposes_rope_backend_and_mesh_flags():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    assert "--rope-backend" in help_text
    assert "--local-mesh-config" in help_text
    assert "--pad-sizes" in help_text
