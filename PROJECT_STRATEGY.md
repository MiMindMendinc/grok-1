# Project strategy

**Anvil** is an independent research fork of [xai-org/grok-1](https://github.com/xai-org/grok-1). It is not affiliated with xAI and is not a production inference engine.

## Purpose

Make the Grok-1 RoPE and attention-adjacent paths explicit, testable, and optimizable on the real architecture.

## Non-goals

- A serving stack (vLLM, TensorRT-LLM, FastAPI, web UI)
- Claims of large speedups without a passing numerical gate
- Running the 314B model on a workstation as a supported product

## Current focus

1. **Correctness** — reference-matching RoPE (JAX and Triton), with tests that compare distinct implementations.
2. **Measurement** — micro-benchmarks that withhold speedup when the gate fails.
3. **Control** — explicit CLI for backend, mesh, padding, and sharding.
4. **Fallback** — Triton is optional; JAX remains the default path.

## Stack

| Layer | Choice |
| --- | --- |
| Model / sharding | Upstream Grok-1 (JAX SPMD, Haiku) |
| Kernels | Optional OpenAI Triton RoPE |
| Weights | Hugging Face `xai-org/grok-1` |
| Tokenizer | SentencePiece (upstream) |

Do not reimplement the tokenizer, sharding runtime, or base Transformer from scratch.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| 1 | RoPE correctness, Triton fallback, gated benchmarks, clean CLI | In progress |
| 2 | Broader attention kernel work; quantization experiments | Planned |
| 3 | Single-GPU developer experience | Planned |

Production serving is out of scope unless a later, separately documented effort is started.

## How to contribute

Prefer small PRs with:

- a correctness test, or
- a gated benchmark change, or
- a clarity fix in the RoPE / CLI path

See the root [README](README.md) for commands.
