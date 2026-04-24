# Grok-1 Inference Fork — Michigan MindMend

**A public fork of the open Grok-1 release focused on inference wiring, RoPE cleanup, benchmark entrypoints, and reproducible testing notes.**

This repository is based on the upstream `xai-org/grok-1` release. The goal of this fork is not to claim ownership of Grok-1 or its weights. The goal is to study, document, and improve practical inference paths around the released codebase.

---

## What this fork focuses on

- JAX / Haiku inference workflow notes
- RoPE correctness and optimization work
- sharding and padding-bucket configuration clarity
- benchmark entrypoints for RoPE and model-forward paths
- optional Triton RoPE backend exploration
- clearer documentation of what is verified vs. what still needs accelerator testing

---

## Model context

Grok-1 is a large mixture-of-experts model released by xAI. This repo keeps the model context visible for developers studying the inference stack:

- **Architecture:** MoE
- **Layers:** 64
- **Hidden size:** 6144
- **Attention:** GQA-style query/KV head layout
- **Context length:** 8192
- **Tokenizer:** SentencePiece

Refer to the upstream release and model license for authoritative model and weight terms.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Required local files:

- `tokenizer.model` at repo root
- checkpoint files under `checkpoints/ckpt-0/...`

Example download flow:

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

Interactive mode:

```bash
python run.py --interactive
```

Useful flags:

- `--rope-backend {jax,triton}`
- `--local-mesh-config DATA MODEL`
- `--between-hosts-config DATA MODEL`
- `--pad-sizes ...`
- `--temperature`
- `--top-p`

---

## RoPE work

This fork documents and experiments with Rotary Position Embedding paths, including:

1. correct handling of position offsets
2. cached inverse-frequency reuse
3. cached position-index reuse
4. backend switching between JAX and guarded Triton paths
5. benchmark entrypoints for comparing reference and optimized behavior

Triton support is treated as an experimental acceleration path. If Triton/PyTorch/CUDA are unavailable, the safer path is to use the JAX implementation.

---

## Benchmarking

Run:

```bash
python benchmarks/rope_benchmark.py --iters 8
```

The benchmark may report:

- reference RoPE timing
- optimized RoPE timing
- JAX RoPE timing, if JAX is installed
- model-forward timing, if the full runtime and checkpoints are available

### Current documented result

A previous fallback benchmark run reported roughly **1.53x** speedup for the optimized Python/RoPE path in an environment without full JAX accelerator support.

Important note: this is **not** a production accelerator benchmark. Full Grok-1 throughput depends on hardware, sharding topology, memory bandwidth, runtime setup, and checkpoint availability.

---

## Recommended local checks

```bash
python -m compileall model.py run.py runners.py tests benchmarks rope_triton.py
python -m pytest tests/test_rope.py -q
python benchmarks/rope_benchmark.py --iters 8
```

---

## What I built / modified

This fork is meant to show hands-on work in:

- reading and modifying large open-model codebases
- inference/runtime documentation
- RoPE implementation details
- benchmark hygiene
- correctness-first performance experimentation
- clear separation between verified results and future targets

---

## Recruiter notes

This repo is useful evidence for roles involving:

- LLM inference engineering
- AI systems prototyping
- model runtime testing
- open-source codebase analysis
- benchmark and evaluation workflows
- performance-oriented debugging

It should be read as a public research and engineering fork, not as a claim of owning Grok-1.

---

## Status

Active research fork / portfolio project.

Next improvements:

- add reproducible accelerator benchmark logs
- add clearer hardware setup notes
- add CI checks where possible
- document exact upstream changes
- separate experimental Triton paths from stable runtime paths

---

## License

Code and model assets follow the upstream release terms. See upstream xAI Grok-1 licensing for authoritative details.

---

Built by **Lyle Perrien II / Michigan MindMend Inc.** as a public AI systems learning and portfolio project.
