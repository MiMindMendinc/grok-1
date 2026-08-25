# Grok-1 Research Fork — Triton RoPE + Correctness-Gated Benchmarks

**A focused research fork of [xai-org/grok-1](https://github.com/xai-org/grok-1).**
This is **not** an official xAI repository.

This fork improves the rotary position embedding (RoPE) path, adds a guarded Triton backend, and ships a correctness-first benchmark harness. The goal is clean, inspectable inference research on the real Grok-1 architecture — not hype.

> Upstream PR: [#434](https://github.com/xai-org/grok-1/pull/434) (fused RoPE work)

---

## Why this fork exists

The official Grok-1 release is excellent as a reference.
This repository exists to make the RoPE and attention-related paths more explicit, testable, and optimizable.

**What you get:**

- Cleaner RoPE implementation with a Triton acceleration path
- Correctness gates before any timing claims
- Reproducible benchmark entry points
- Explicit mesh / padding / sharding controls
- A research-friendly structure for kernel and systems work

**What you do not get:**

- A production serving stack
- Magic multi-x speedups without baselines
- Support for running the full 314B model on a laptop

---

## Model Specs (unchanged from upstream)

| Spec         | Value               |
| ------------ | ------------------- |
| Parameters   | 314B (MoE)          |
| Experts      | 8 (top-2 active)    |
| Layers       | 64                  |
| Hidden size  | 6144                |
| Attention    | 48 Q / 8 KV heads   |
| Context      | 8192                |
| Tokenizer    | SentencePiece 131k  |

---

## Quick Start

```bash
git clone https://github.com/MiMindMendinc/grok-1.git
cd grok-1
pip install -r requirements.txt
```

### Weights (required for full model)

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include "ckpt-0/*" \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

### Run (JAX backend)

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The answer to life, the universe, and everything is" \
  --max-new-tokens 64 \
  --rope-backend jax
```

### Triton RoPE path

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The answer to life, the universe, and everything is" \
  --max-new-tokens 64 \
  --rope-backend triton
```

The Triton path falls back safely when the kernel is unavailable.

Mesh, padding, and sharding are explicit CLI flags on `run.py`:

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --local-mesh-config 1 8 \
  --between-hosts-config 1 1 \
  --pad-sizes 1024
```

---

## Benchmarks & Correctness

All timing numbers are gated behind correctness checks.

```bash
# RoPE unit + regression tests
python -m pytest tests/test_rope.py -v

# RoPE performance harness
python benchmarks/rope_benchmark.py
```

We deliberately separate correctness from performance.
If the numerical gate fails, the benchmark does not report speedups.

---

## Repository Layout

```
rope_triton.py          # Triton RoPE implementation
benchmarks/             # Reproducible harnesses
tests/                  # Correctness & regression tests
model.py / runners.py   # Core inference stack (derived from upstream)
run.py                  # Clean CLI entry point
```

---

## Status & Roadmap

- [x] Triton RoPE backend + fallback
- [x] Correctness-gated benchmarks
- [x] Clean CLI and mesh controls
- [ ] Broader attention kernel experiments
- [ ] Quantization paths
- [ ] Better single-GPU developer experience

This is active research code. Expect sharp edges.

---

## Citation / Attribution

- Original model & code: [xai-org/grok-1](https://github.com/xai-org/grok-1) (Apache 2.0)
- This fork: research extensions by Michigan MindMend

If you use the RoPE or benchmark work, a link back is appreciated.

---

## Contributing

Serious PRs that improve correctness, clarity, or measured performance are welcome.
See [PROJECT_STRATEGY.md](PROJECT_STRATEGY.md) for direction.
