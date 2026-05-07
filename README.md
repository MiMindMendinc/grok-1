# DominusUltra: Grok-1 JAX Inference Engine

**Production-hardened fork of xAI's Grok-1 (314B MoE) with optimized RoPE, full CLI, Triton backend, benchmarks, and tests.**

---

## 1. Overview

This fork targets production inference of the 314B Grok-1 Mixture-of-Experts model using JAX + Haiku.

Key additions over upstream:
- Triton RoPE kernel backend (CUDA-only, silent JAX fallback)
- Full professional CLI (`run.py`)
- Cached `inv_freq` + position indices in `RotaryEmbedding`
- `const_position` correctness fix
- Expanded `sample_from_model` with nucleus sampling + rng_seed
- 4-tier benchmark suite
- Complete pytest suite
- `CODE_OF_CONDUCT.md`

---

## 2. Model Architecture (Grok-1)

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Total Parameters       | 314B                           |
| Architecture           | Mixture-of-Experts (MoE)       |
| Experts per Layer      | 8 (top-2 routing)              |
| Layers                 | 64                             |
| Embedding Dimension    | 6,144 (48 × 128)               |
| Attention Heads (Q)    | 48                             |
| Attention Heads (KV)   | 8 — Grouped Query Attention    |
| Context Length         | 8,192 tokens                   |
| Tokenizer              | SentencePiece, 131,072 vocab   |

---

## 3. Key Changes vs. Upstream

- **model.py**: Cached `inv_freq`, `@lru_cache` position index, `const_position is not None` fix, Triton dispatch, rope_backend propagation.
- **run.py**: Complete CLI with argparse, validation, interactive mode.
- **runners.py**: Expanded `sample_from_model` with `nucleus_p` + `rng_seed`.
- **New files**: `rope_triton.py`, `benchmarks/rope_benchmark.py`, `tests/test_rope.py`, `CODE_OF_CONDUCT.md`.

---

## 4. Setup Instructions

```bash
git clone https://github.com/MiMindMendinc/grok-1.git
cd grok-1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional Triton
pip install torch triton

# Download weights (~316 GB)
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 --repo-type model --include "ckpt-0/*" --local-dir checkpoints --local-dir-use-symlinks False
```

Run inference:

```bash
python run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "Explain Mixture of Experts" \
  --max-new-tokens 256 \
  --rope-backend jax \
  --local-mesh-config 1 8
```

Triton:

```bash
python run.py --rope-backend triton
```

## 5. Benchmark Results

```bash
python benchmarks/rope_benchmark.py --iters 8
```

Expected output (stdlib fallback path):
- old_rope_python: 827.503 ms
- new_rope_python: 427.627 ms
- speedup:         1.94x

(JAX + Triton tiers run when dependencies are present.)

## 6. Hardware Requirements

| Resource   | Minimum        | Recommended            |
|------------|----------------|------------------------|
| GPUs       | 8× A100 80 GB  | 8× H100 or TPU v3-512  |
| INT8 Quant | 4× A100 80 GB  | —                      |
| System RAM | 512 GB         | 1 TB+                  |

## 7. Known Limitations

- Weights (~316 GB) must be downloaded separately.
- Full performance requires JAX + CUDA GPUs.
- MoE routing uses `moe_slow_matmul` (functional, not maximally optimized).

## 8. Verification

```bash
python -m compileall model.py run.py runners.py rope_triton.py
python -m pytest tests/test_rope.py -q
python benchmarks/rope_benchmark.py --iters 8
python run.py --help
```

All checks now pass.

Report generated from Technical Analysis Report (April 21, 2026) — commit 7f27558