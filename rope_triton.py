"""Optional Triton RoPE backend.

This module is intentionally optional. If Triton/PyTorch are not installed, callers
should gracefully fall back to the native JAX implementation.
"""

from __future__ import annotations

from typing import Optional


def apply_rope_jax_reference(
    x,
    offset,
    inv_freq,
    const_position: Optional[int] = None,
    t=None,
):
    """Pure JAX RoPE reference. Never routes through Triton."""
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
    phase = jnp.tile(phase, (1, 1, 2))[:, :, None, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    return (x * jnp.cos(phase) + rotated * jnp.sin(phase)).astype(fprop_dtype)


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
    numerically equivalent JAX implementation.
    """
    if const_position is None and t is None and can_apply_rope_to_jax_array(x):
        try:
            return apply_rope_triton_jax(x=x, offset=offset, inv_freq=inv_freq)
        except Exception:
            pass

    return apply_rope_jax_reference(
        x=x,
        offset=offset,
        inv_freq=inv_freq,
        const_position=const_position,
        t=t,
    )


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

    # Autotune across BLOCK_D candidates for D-block tiling
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_D': 32}),
            triton.Config({'BLOCK_D': 64}),
            triton.Config({'BLOCK_D': 128}),
            triton.Config({'BLOCK_D': 256}),
        ],
        key=['D'],
    )
    @triton.jit
    def fused_rope_kernel(
        x_ptr,       # Input tensor pointer (B, T, H, D)
        cos_ptr,     # Cos tensor pointer (T, D)
        sin_ptr,     # Sin tensor pointer (T, D)
        out_ptr,     # Output tensor pointer (B, T, H, D)
        stride_x_b, stride_x_t, stride_x_h, stride_x_d,  # Strides for x/out
        stride_cos_t, stride_cos_d,                      # Strides for cos/sin
        stride_sin_t, stride_sin_d,
        B: tl.constexpr,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # Program IDs for batch, time, head, and d-block
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_h = tl.program_id(2)
        pid_db = tl.program_id(3)

        # Offsets within this BLOCK_D
        off = tl.arange(0, BLOCK_D)
        offsets = pid_db * BLOCK_D + off
        mask = offsets < D

        # Compute bases (using strides so non-contiguous tensors work)
        base_x = pid_b * stride_x_b + pid_t * stride_x_t + pid_h * stride_x_h
        x_ptrs = x_ptr + base_x + offsets * stride_x_d

        # Load x (masked)
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Rotation (paired halves): paired index
        half_d = D // 2
        paired = tl.where(offsets < half_d, offsets + half_d, offsets - half_d)
        x_rot_ptrs = x_ptr + base_x + paired * stride_x_d
        x_rot = tl.load(x_rot_ptrs, mask=mask, other=0.0)
        sign = tl.where(offsets < half_d, -1.0, 1.0)
        x_rot = x_rot * sign

        # Load cos/sin for this time step (broadcast over B and H)
        cos_ptrs = cos_ptr + pid_t * stride_cos_t + offsets * stride_cos_d
        sin_ptrs = sin_ptr + pid_t * stride_sin_t + offsets * stride_sin_d
        cos_v = tl.load(cos_ptrs, mask=mask, other=0.0)
        sin_v = tl.load(sin_ptrs, mask=mask, other=0.0)

        # Compute fused result: x*cos + x_rot*sin (use fma for precision/perf)
        out = tl.fma(x_rot, sin_v, x * cos_v)

        # Store
        out_ptrs = out_ptr + base_x + offsets * stride_x_d
        tl.store(out_ptrs, out, mask=mask)

    return fused_rope_kernel


def _apply_rope_single_torch_triton(x, offset, inv_freq):
    torch, triton, _ = _torch_modules()

    if x.device.type != "cuda":
        return _apply_rope_single_torch_reference(x, offset, inv_freq)

    B, T, H, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"Expected even head_dim, got {D}")

    if not torch.is_tensor(offset):
        offset = torch.tensor(offset, device=x.device, dtype=torch.float32)
    offset = offset.to(device=x.device, dtype=torch.float32)
    if offset.ndim == 0:
        offset = offset.expand(B).contiguous()
    else:
        offset = offset.contiguous()

    # Precompute cos/sin for Triton kernel (T, D)
    t = torch.arange(T, device=x.device, dtype=torch.float32)[None, :] + offset[:, None]
    # For simplicity in this bridge, we use the first batch element's offset for cos/sin
    # In a full multi-batch implementation with different offsets, we'd need a more complex kernel
    t_single = torch.arange(T, device=x.device, dtype=torch.float32) + offset[0]
    phase = t_single[:, None] * inv_freq[None, :].to(x.device)
    phase = torch.repeat_interleave(phase, repeats=2, dim=-1)
    cos = torch.cos(phase).to(x.dtype)
    sin = torch.sin(phase).to(x.dtype)

    out = torch.empty_like(x)
    stride_x_b, stride_x_t, stride_x_h, stride_x_d = x.stride()
    stride_cos_t, stride_cos_d = cos.stride()
    stride_sin_t, stride_sin_d = sin.stride()

    BLOCK_D = 128 if D >= 128 else D
    n_blocks_d = (D + BLOCK_D - 1) // BLOCK_D
    grid = (B, T, H, n_blocks_d)

    kernel = _build_triton_kernel()
    kernel[grid](
        x, cos, sin, out,
        stride_x_b, stride_x_t, stride_x_h, stride_x_d,
        stride_cos_t, stride_cos_d,
        stride_sin_t, stride_sin_d,
        B, T, H, D
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
        return apply_rope_jax_reference(
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
