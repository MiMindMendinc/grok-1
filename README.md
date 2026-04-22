# DominusUltra: Grok-1 JAX Inference Engine

Production-focused JAX/Haiku inference stack for **Grok-1 (314B MoE)** with optimized RoPE, sharding-aware execution, benchmark tooling, and an optional Triton RoPE backend path.

---

## Why this repo

This project is tuned for practical large-model inference workflows:

- clean inference CLI (`run.py`)
- explicit mesh + padding-bucket controls
- RoPE correctness fixes and optimizations
- reproducible benchmark entrypoints
- optional Triton-backed RoPE path (with safe fallback)

---

## Model specifications (Grok-1)

- **Total parameters**: 314B
- **Architecture**: MoE (8 experts, top-2 routing)
- **Layers**: 64
- **Hidden size**: 6144
- **Attention**: 48 Q heads / 8 KV heads (GQA)
- **Context length**: 8192
- **Tokenizer**: SentencePiece, vocab 131072

---

## Installation

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Model files

Required:

- `tokenizer.model` at repo root
- checkpoint files under `checkpoints/ckpt-0/...`

Download from Hugging Face:

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include "ckpt-0/*" \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

---

## Inference usage

### One-shot generation

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The answer to life, the universe, and everything is" \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.95 \
  --rope-backend jax
```

### Interactive generation

```bash
python run.py --interactive
```

### CLI highlights

- `--temperature` > 0
- `--top-p` in (0, 1]
- `--rope-backend {jax,triton}`
- `--local-mesh-config DATA MODEL`
- `--between-hosts-config DATA MODEL`
- `--pad-sizes ...` (ascending buckets)

---

## Optimized RoPE

The RoPE path includes:

1. Correct `const_position=0` behavior.
2. Cached inverse frequencies (`inv_freq`) instead of recomputing per call.
3. Cached position index reuse.
4. Backend switch (`jax`/`triton`) via config + CLI.

### Triton backend

`rope_triton.py` provides:

- fused Q+K RoPE application entrypoint for PyTorch tensors,
- GQA/MQA-compatible tensor shape handling,
- decode/prefill compatible offset handling,
- FP32 phase math for stable cos/sin generation,
- safe fallback behavior when Triton is unavailable.

Set backend from CLI:

```bash
python run.py --rope-backend triton
```

If Triton/PyTorch is not present, the implementation falls back to JAX RoPE.

---

## Benchmarking

Run:

```bash
python benchmarks/rope_benchmark.py --iters 8
```

The benchmark reports:

- RoPE-only comparison (`old` vs `new`)
- JAX RoPE timing (if JAX installed)
- full model forward-pass timing (if JAX+Haiku installed)

### Latest benchmark results (this environment)

Date: **April 21, 2026**  
Command: `python benchmarks/rope_benchmark.py --iters 8`

- `old_rope_python`: **2010.286 ms**
- `new_rope_python`: **1316.307 ms**
- speedup: **1.53x**
- JAX benchmark: skipped (JAX unavailable in environment)
- Full forward-pass benchmark: skipped (JAX unavailable in environment)

> Note: these numbers are from the stdlib fallback benchmark path due environment package constraints. Use a JAX-enabled runtime for production-quality accelerator timings.

---

## Performance notes

- RoPE optimization reduces repeated decode overhead by avoiding recomputation-heavy frequency setup.
- Throughput/latency for full Grok-1 depends heavily on sharding topology and hardware memory bandwidth.
- Tune `--pad-sizes`, mesh config, and batch settings to your cluster.

---

## Contribution guidelines

1. Keep changes modular and benchmarkable.
2. Add/extend tests for correctness-sensitive paths.
3. Include benchmark evidence for performance claims.
4. Keep docs updated with flags, defaults, and known limitations.
5. Use clear commit messages (scope + intent).

Recommended local checks:

```bash
python -m compileall model.py run.py runners.py tests benchmarks rope_triton.py
python -m pytest tests/test_rope.py -q
python benchmarks/rope_benchmark.py --iters 8
```

---

## License

Apache 2.0 (code and Grok-1 weights under upstream release terms).

## Triton RoPE Acceleration (New)

This fork adds a guarded Triton kernel path for Rotary Position Embeddings and fixes the runtime/benchmark wiring around it.

### Quick Demo

```bash
python -m pytest tests/test_rope.py -q
python benchmarks/rope_benchmark.py --iters 20
python run.py --rope-backend triton --max-new-tokens 64
```

### Notes

- The `--rope-backend triton` flag now routes through the guarded Triton kernel bridge when the CUDA Triton runtime is available.
- The benchmark now compares the Triton path against the reference implementation instead of benchmarking the same function twice.
- The `run.py` CLI now rejects non-positive `--pad-sizes` values early.
- The `1.53x` figure above is from the repo's existing stdlib fallback benchmark section, not a freshly verified A100 run in this session.
