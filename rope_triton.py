"""Optional Triton RoPE backend.

This module is intentionally optional. If Triton/PyTorch are not installed, callers
should gracefully fall back to the native JAX implementation.
"""

from __future__ import annotations

from typing import Optional


def apply_rope_jax_compatible(
    x,
    offset,
    inv_freq,
    const_position: Optional[int] = None,
    t=None,
):
    """NumPy/JAX-compatible fallback path used by the JAX model integration.

    This implementation keeps FP32 phase math for numerical stability and mirrors the
    phase construction used by the optimized JAX RoPE path.
    """
    import jax.numpy as jnp

    fprop_dtype = x.dtype
    sequence_len = x.shape[1]
    if jnp.shape(offset) == ():
        offset = jnp.full((x.shape[0],), offset, dtype=jnp.float32)
    else:
        offset = offset.astype(jnp.float32)

    if const_position is not None:
        t = jnp.full((x.shape[0], sequence_len), const_position, dtype=jnp.float32)
    elif t is None:
        t = jnp.arange(sequence_len, dtype=jnp.float32)[None, :] + jnp.expand_dims(offset, -1)

    phase = t[:, :, None] * inv_freq[None, None, :]
    phase = jnp.repeat(phase, repeats=2, axis=-1)[:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    return (x * jnp.cos(phase) + rotated * jnp.sin(phase)).astype(fprop_dtype)


def is_triton_available() -> bool:
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def apply_rope_torch(
    q,
    k,
    offset,
    base_exponent: int = 10000,
):
    """Fused Q+K RoPE application for PyTorch tensors.

    Works for GQA/MQA layouts as long as `q` and `k` end with head_dim.
    Input:
      q: [B, S, Hq, D]
      k: [B, S, Hk, D]
      offset: scalar or [B]
    """
    import torch

    head_dim = q.shape[-1]
    assert head_dim % 2 == 0, f"Expected even head_dim, got {head_dim}"
    exponents = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (base_exponent ** (exponents / head_dim))

    if not torch.is_tensor(offset):
        offset = torch.tensor(offset, device=q.device, dtype=torch.float32)
    offset = offset.to(dtype=torch.float32, device=q.device)
    if offset.ndim == 0:
        offset = offset.expand(q.shape[0])

    seq_len = q.shape[1]
    t = torch.arange(seq_len, device=q.device, dtype=torch.float32)[None, :] + offset[:, None]
    phase = t[:, :, None] * inv_freq[None, None, :]
    phase = torch.repeat_interleave(phase, repeats=2, dim=-1)[:, :, None, :]
    cos = torch.cos(phase)
    sin = torch.sin(phase)

    def rotate_half_t(x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos + rotate_half_t(q) * sin
    k_out = k * cos + rotate_half_t(k) * sin
    return q_out, k_out
