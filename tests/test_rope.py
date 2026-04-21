import pytest

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp

    from model import RotaryEmbedding

    HAS_JAX_STACK = True
except ModuleNotFoundError:
    HAS_JAX_STACK = False


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
