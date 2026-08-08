# Grok-1 Triton RoPE validation

This directory contains an **optional** PyTorch/Triton implementation of the
rotary embedding math used by `model.py::RotaryEmbedding`. It does not modify
the JAX reference runtime or add dependencies to `requirements.txt`.

## Why this is reviewable

The benchmark now uses the exact Grok-1 frequency layout:

```python
phase = jnp.einsum("bi,j->bij", t, inv_freq)
phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]
x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
```

The PyTorch reference therefore constructs `phase` by concatenating the
half-width phase vector with itself. The Triton kernel is checked against that
literal translation before any timing is reported.

## Install the optional benchmark dependencies

Use an environment with a supported NVIDIA GPU and matching CUDA/PyTorch build.
For example:

```bash
python -m venv .venv-triton
source .venv-triton/bin/activate
pip install torch triton
```

No Grok-1 checkpoint is needed for the kernel microbenchmark.

## Correctness

```bash
python examples/triton_rope_fused.py --check
```

The matrix covers:

- FP16 and BF16
- contiguous and non-contiguous tensors
- Grok-1 head dimension (`D=128`) plus neighboring dimensions
- scalar position offsets
- out-of-place and in-place execution
- query-head-like widths up to 48 heads

Every timed benchmark performs a correctness comparison first.

## Reproducible latency sweep

Grok-1's published maximum context is 8192 tokens, so the default sweep includes
512 through 8192 and also 16384 as a stress point:

```bash
python examples/triton_rope_fused.py \
  --benchmark \
  --sequence-lengths 512 1024 2048 4096 8192 16384 \
  --heads 8 \
  --dim 128 \
  --dtype fp16 \
  --json rope-h100-fp16.json
```

For the 48 query-head shape:

```bash
python examples/triton_rope_fused.py \
  --benchmark \
  --sequence-lengths 512 1024 2048 4096 8192 \
  --heads 48 \
  --dim 128 \
  --dtype bf16 \
  --json rope-h100-bf16-q48.json
```

To exercise the strided path:

```bash
python examples/triton_rope_fused.py --benchmark --noncontiguous
```

## Measurement policy

- GPU, PyTorch, and Triton versions are printed with every run.
- Timing uses `triton.testing.do_bench`, which uses GPU timing rather than
  wall-clock Python timing.
- Warmup and repetition counts are configurable (`--warmup`, `--rep`).
- The JSON output includes absolute PyTorch latency, absolute Triton latency,
  speedup, and maximum absolute error for every shape.
- Results are not hard-coded into the source. Contributors and maintainers can
  reproduce them on their own hardware.

## Scope

This contribution is deliberately an isolated optimization example. Grok-1's
public repository is a JAX correctness/reference implementation, while Triton
is a PyTorch-oriented kernel DSL. Integrating this file into `model.py` would
change the runtime stack and is therefore outside this PR's scope.

The useful artifact here is a corrected, independently measurable fused RoPE
kernel that implements Grok-1's published RoPE convention and can be reused by
PyTorch/Triton ports of the model.
