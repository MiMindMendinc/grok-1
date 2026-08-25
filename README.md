# Grok-1 Research Fork

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/framework-JAX-red.svg)](https://github.com/google/jax)
[![Tests](https://github.com/MiMindMendinc/grok-1/actions/workflows/tests.yml/badge.svg)](https://github.com/MiMindMendinc/grok-1/actions/workflows/tests.yml)

> [!IMPORTANT]
> **This is not an official xAI repository.** It is a research fork of [xai-org/grok-1](https://github.com/xai-org/grok-1), maintained by Michigan MindMend, under Apache 2.0.

Focused work on rotary position embeddings (RoPE): a guarded Triton backend, JAX fallback, and a benchmark harness that **will not report a speedup** unless the numerical gate passes.

Upstream fused-RoPE discussion: [xai-org/grok-1#434](https://github.com/xai-org/grok-1/pull/434).

---

## Why this fork exists

The upstream xAI Grok-1 code is an excellent reference. This repository exists to make the RoPE and attention-related paths more explicit, testable, and optimizable — on the real architecture, without serving-stack hype.

| Included | Not included |
| --- | --- |
| Triton RoPE path with JAX fallback | Production serving |
| Correctness gates before any timing | Unverified multi-x speedup claims |
| Reproducible micro-benchmarks | Laptop-scale 314B inference |
| Explicit mesh, padding, and sharding CLI | A hosted Grok API |

---

## Model specs

Unchanged from upstream.

| Spec | Value |
| --- | --- |
| Parameters | 314B (MoE) |
| Experts | 8 (top-2 active) |
| Layers | 64 |
| Hidden size | 6144 |
| Attention | 48 Q / 8 KV heads |
| Context | 8192 |
| Tokenizer | SentencePiece, 131k |

---

## Quick start

```bash
git clone https://github.com/MiMindMendinc/grok-1.git
cd grok-1
pip install -r requirements.txt
```

### Weights (full model only)

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include "ckpt-0/*" \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

RoPE tests and micro-benchmarks do **not** need the 314B checkpoint.

### Run — JAX backend

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The answer to life, the universe, and everything is" \
  --max-new-tokens 64 \
  --rope-backend jax
```

### Run — Triton RoPE

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The answer to life, the universe, and everything is" \
  --max-new-tokens 64 \
  --rope-backend triton
```

If Triton or CUDA is unavailable, the kernel falls back to JAX.

### Mesh, padding, sharding

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --local-mesh-config 1 8 \
  --between-hosts-config 1 1 \
  --pad-sizes 1024
```

```bash
python run.py --help
```

---

## Benchmarks and correctness

Timing is gated. A failing numerical check withholds the speedup.

```bash
python -m pytest tests/test_rope.py -v
python benchmarks/rope_benchmark.py
```

---

## Layout

```
run.py                  CLI entry point
model.py / runners.py   Inference stack (from upstream)
rope_triton.py          Optional Triton RoPE + JAX reference
benchmarks/             Correctness-gated harnesses
tests/                  RoPE and CLI regression tests
PROJECT_STRATEGY.md     Research direction
NOTICE                  Fork attribution
```

---

## Status

- [x] Triton RoPE backend with JAX fallback
- [x] Correctness-gated benchmarks
- [x] CLI for backend, mesh, padding, and sharding
- [ ] Broader attention kernel experiments
- [ ] Quantization paths
- [ ] Better single-GPU developer experience

Active research code. Expect sharp edges.

---

## License and attribution

- Original model and code: [xai-org/grok-1](https://github.com/xai-org/grok-1) ([Apache 2.0](LICENSE.txt))
- This fork: research extensions by Michigan MindMend — see [NOTICE](NOTICE)

If you use the RoPE or benchmark work, a citation or link back is appreciated.

---

## Contributing

PRs that improve correctness, clarity, or *measured* performance are welcome. Direction: [PROJECT_STRATEGY.md](PROJECT_STRATEGY.md).
