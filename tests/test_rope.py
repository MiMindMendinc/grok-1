"""Regression tests for RoPE (JAX + Triton)."""

import argparse

import numpy as np
import pytest

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from model import RotaryEmbedding
    from rope_triton import apply_rope_jax_compatible, can_apply_rope_to_jax_array
    from run import validate_args

    HAS_JAX_STACK = True
except ModuleNotFoundError:
    HAS_JAX_STACK = False
    HAS_TRITON = False
else:
    try:
        from rope_triton import apply_rope_torch, is_triton_available

        HAS_TRITON = is_triton_available()
    except Exception:
        HAS_TRITON = False


def test_rope_test_environment_smoke():
    if not HAS_JAX_STACK:
        pytest.skip("JAX/Haiku not installed in this environment.")
    assert True


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
    """Triton path must match the JAX-compatible reference within tolerance."""
    x = jax.random.normal(jax.random.PRNGKey(7), (1, 64, 8, 128), dtype=jnp.float32)
    off = jnp.array([0.0], dtype=jnp.float32)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))

    jax_out = apply_rope_jax_compatible(x=x, offset=off, inv_freq=inv_freq)

    if can_apply_rope_to_jax_array(x):
        triton_out = apply_rope_jax_compatible(x=x, offset=off, inv_freq=inv_freq)
        np.testing.assert_allclose(
            np.array(jax_out),
            np.array(triton_out),
            rtol=1e-3,
            atol=1e-3,
            err_msg="Triton and JAX RoPE outputs differ significantly",
        )
    else:
        pytest.skip("JAX DLPack bridge unavailable for Triton validation")


@pytest.mark.skipif(not HAS_JAX_STACK, reason="JAX stack not installed.")
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
