import math
import time
import torch
import triton
import triton.language as tl

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
    out_ptr,     # Output tensor pointer (B, T, H, D) — can be same as x_ptr for in-place
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


def fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Apply fused Triton RoPE to x (B, T, H, D).
    cos/sin: (T, D) precomputed (repeated for pairs).
    Supports non-contiguous tensors via explicit strides.

    Target versions: Python 3.10+, PyTorch 2.0+, Triton 2.0+
    """
    assert x.dim() == 4, "x must be (B, T, H, D)"
    B, T, H, D = x.shape
    assert D % 2 == 0, "Dim must be even for RoPE pairs"
    assert cos.shape == (T, D) and sin.shape == (T, D)

    # Ensure same dtype (kernel expects matching dtype)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    out = x if inplace else torch.empty_like(x)

    # Strides (PyTorch gives stride for each dimension)
    stride_x_b, stride_x_t, stride_x_h, stride_x_d = x.stride()
    stride_cos_t, stride_cos_d = cos.stride()
    stride_sin_t, stride_sin_d = sin.stride()

    # Choose block size: autotune will pick the best BLOCK_D, but we must launch across D blocks
    # Grid will be (B, T, H, n_blocks_d)
    BLOCK_D = 128 if D >= 128 else D
    n_blocks_d = (D + BLOCK_D - 1) // BLOCK_D

    grid = (B, T, H, n_blocks_d)

    fused_rope_kernel[grid](
        x, cos, sin, out,
        stride_x_b, stride_x_t, stride_x_h, stride_x_d,
        stride_cos_t, stride_cos_d,
        stride_sin_t, stride_sin_d,
        B, T, H, D
    )

    return out


# ------------------ Smoke test & Benchmark ------------------
if __name__ == "__main__":
    # Basic smoke test and micro-benchmark. Keep sizes reasonable for CI/desktop testing.
    device = torch.device("cuda")
    D = 128
    T = 16384  # 16k tokens (changeable)
    H = 8
    B = 1
    base = 10000.0

    # Precompute rotations in Grok-style
    theta = base ** (-2.0 * torch.arange(0, D // 2, device=device) / D)
    m = torch.arange(T, device=device)
    freqs = m[:, None] * theta[None, :]
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(torch.float16)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(torch.float16)

    # Create a non-contiguous x by constructing a differently ordered tensor and transposing
    x_src = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    x = x_src.transpose(1, 2)  # now shape (B, T, H, D) but non-contiguous

    # Baseline PyTorch RoPE
    def pytorch_rope(x, cos, sin):
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]

    # Warm up
    out_torch = pytorch_rope(x, cos, sin)
    out_triton = fused_rope(x, cos, sin)

    # Correctness check
    if not torch.allclose(out_triton, out_torch, atol=1e-2, rtol=1e-2):
        diff = (out_triton - out_torch).abs().max()
        raise RuntimeError(f"Triton fused_rope mismatch: max abs diff = {diff}")
    print("Smoke test passed: Triton output matches PyTorch baseline (FP16, tol=1e-2).")

    # Micro-benchmark (100 iterations)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = pytorch_rope(x, cos, sin)
    torch.cuda.synchronize()
    baseline_ms = (time.time() - start) * 1000.0 / 100.0
    print(f"PyTorch baseline: {baseline_ms:.2f} ms")

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = fused_rope(x, cos, sin)
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) * 1000.0 / 100.0
    print(f"Triton fused: {triton_ms:.2f} ms")
    print(f"Speedup: {baseline_ms / triton_ms:.2f}x")
