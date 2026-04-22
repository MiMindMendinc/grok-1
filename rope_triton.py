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
    """JAX-facing RoPE entrypoint.

    When a CUDA Triton runtime is available and the call uses the standard position mode,
    this routes through the actual Triton kernel bridge. Otherwise it falls back to the
    numerically equivalent JAX implementation below.
    """
    if const_position is None and t is None and can_apply_rope_to_jax_array(x):
        try:
            return apply_rope_triton_jax(x=x, offset=offset, inv_freq=inv_freq)
        except Exception:
            pass

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
        import torch
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def _torch_modules():
    import torch
    import triton
    import triton.language as tl

    return torch, triton, tl


def apply_rope_torch_reference(
    q,
    k,
    offset,
    inv_freq=None,
    base_exponent: int = 10000,
):
    """Reference PyTorch implementation for benchmarking and correctness checks."""
    import torch

    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"Expected even head_dim, got {head_dim}")

    if inv_freq is None:
        exponents = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
        inv_freq = 1.0 / (base_exponent ** (exponents / head_dim))
    else:
        inv_freq = inv_freq.to(device=q.device, dtype=torch.float32)

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


def _apply_rope_single_torch_reference(x, offset, inv_freq):
    q_out, _ = apply_rope_torch_reference(x, x, offset, inv_freq=inv_freq)
    return q_out


def _build_triton_kernel():
    _, triton, tl = _torch_modules()

    @triton.jit
    def rope_kernel(
        x_ptr,
        out_ptr,
        offset_ptr,
        inv_freq_ptr,
        stride_b,
        stride_s,
        stride_h,
        stride_d,
        batch,
        seq_len,
        num_heads,
        half_dim,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = cols < half_dim

        heads_per_batch = seq_len * num_heads
        batch_idx = row // heads_per_batch
        seq_head_idx = row % heads_per_batch
        seq_idx = seq_head_idx // num_heads
        head_idx = seq_head_idx % num_heads

        base_ptr = x_ptr + batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
        x1 = tl.load(base_ptr + cols, mask=mask, other=0.0)
        x2 = tl.load(base_ptr + half_dim + cols, mask=mask, other=0.0)

        offset = tl.load(offset_ptr + batch_idx)
        inv_freq = tl.load(inv_freq_ptr + cols, mask=mask, other=0.0)
        phase = (tl.full((BLOCK,), seq_idx, tl.float32) + offset) * inv_freq
        cosv = tl.cos(phase)
        sinv = tl.sin(phase)

        out1 = x1 * cosv - x2 * sinv
        out2 = x2 * cosv + x1 * sinv

        out_base = out_ptr + batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
        tl.store(out_base + cols, out1, mask=mask)
        tl.store(out_base + half_dim + cols, out2, mask=mask)

    return rope_kernel


def _apply_rope_single_torch_triton(x, offset, inv_freq):
    torch, triton, _ = _torch_modules()

    if x.device.type != "cuda":
        return _apply_rope_single_torch_reference(x, offset, inv_freq)

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Expected even head_dim, got {x.shape[-1]}")

    if not torch.is_tensor(offset):
        offset = torch.tensor(offset, device=x.device, dtype=torch.float32)
    offset = offset.to(device=x.device, dtype=torch.float32)
    if offset.ndim == 0:
        offset = offset.expand(x.shape[0]).contiguous()
    else:
        offset = offset.contiguous()

    inv_freq = inv_freq.to(device=x.device, dtype=torch.float32).contiguous()
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    half_dim = x_contig.shape[-1] // 2
    rows = x_contig.shape[0] * x_contig.shape[1] * x_contig.shape[2]
    kernel = _build_triton_kernel()
    grid = (rows, triton.cdiv(half_dim, 128))
    kernel[grid](
        x_contig,
        out,
        offset,
        inv_freq,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        x_contig.stride(3),
        x_contig.shape[0],
        x_contig.shape[1],
        x_contig.shape[2],
        half_dim,
        BLOCK=128,
    )
    return out


def can_apply_rope_to_jax_array(x) -> bool:
    try:
        import jax  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return is_triton_available() and hasattr(x, "__dlpack__")


def apply_rope_triton_jax(x, offset, inv_freq, const_position: Optional[int] = None, t=None):
    """Apply the Triton backend to a JAX array via DLPack bridging.

    For non-standard position modes (`const_position` / explicit `t`), callers should
    use the native JAX-compatible implementation instead.
    """
    if const_position is not None or t is not None:
        return apply_rope_jax_compatible(
            x=x,
            offset=offset,
            inv_freq=inv_freq,
            const_position=const_position,
            t=t,
        )

    import jax.dlpack as jdl
    from torch.utils import dlpack as torch_dlpack

    torch_x = torch_dlpack.from_dlpack(x)
    torch_offset = torch_dlpack.from_dlpack(offset) if hasattr(offset, "__dlpack__") else offset
    torch_inv_freq = torch_dlpack.from_dlpack(inv_freq)
    out = _apply_rope_single_torch_triton(torch_x, torch_offset, torch_inv_freq)
    return jdl.from_dlpack(torch_dlpack.to_dlpack(out))


def apply_rope_torch(
    q,
    k,
    offset,
    inv_freq=None,
    base_exponent: int = 10000,
):
    """Fused Q+K RoPE application for PyTorch tensors.

    Uses the actual Triton kernel on CUDA when available, otherwise falls back to the
    PyTorch reference implementation.
    """
    if inv_freq is None:
        import torch

        head_dim = q.shape[-1]
        exponents = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
        inv_freq = 1.0 / (base_exponent ** (exponents / head_dim))

    if not is_triton_available() or q.device.type != "cuda" or k.device.type != "cuda":
        return apply_rope_torch_reference(
            q,
            k,
            offset,
            inv_freq=inv_freq,
            base_exponent=base_exponent,
        )

    q_out = _apply_rope_single_torch_triton(q, offset, inv_freq)
    k_out = _apply_rope_single_torch_triton(k, offset, inv_freq)
    return q_out, k_out
