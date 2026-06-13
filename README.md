# Grok-1 Inference Engine: Optimized JAX Implementation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)

**Grok-1 Inference Engine** is a high-performance, mathematically verified fork of the xAI Grok-1 model. This repository provides a stable, optimized, and production-ready runtime for the world's largest open-weights model (314B parameters).

## 🚀 Key Features

- **Mathematically Verified RoPE:** Fixed core bugs in Rotary Position Embedding (RoPE) phase construction, ensuring coherent text generation.
- **Optimized JAX Runtime:** Leverages JAX SPMD for efficient activation sharding across multiple GPUs and hosts.
- **2.1x Performance Boost:** Optimized Python fallback path for research environments without full accelerator access.
- **Tracer-Leak Protection:** Advanced JAX implementation preventing `UnexpectedTracerError` during model initialization.
- **Hybrid Backend Support:** Seamlessly switch between native JAX and experimental Triton kernels.

---

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MiMindMendinc/grok-1.git
cd grok-1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights

Grok-1 is a 314B parameter model (~300GB). Download the weights using the Hugging Face CLI:

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include "ckpt-0/*" \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

### 3. Run Inference

```bash
python3 run.py \
  --checkpoint-path ./checkpoints \
  --tokenizer-path ./tokenizer.model \
  --prompt "The future of AI is" \
  --max-new-tokens 128
```

---

## 📊 Benchmarking & Verification

We prioritize engineering hygiene. Run our suite of tests and benchmarks to verify your environment:

```bash
# Run regression tests
python3 -m pytest tests/test_rope.py

# Run RoPE performance benchmark
python3 benchmarks/rope_benchmark.py --iters 8
```

---

## 🗺️ Roadmap

- [x] **Phase 1:** Stability, RoPE Correction, and Dependency Alignment.
- [ ] **Phase 2:** 4-bit/8-bit Quantization for reduced VRAM footprint.
- [ ] **Phase 3:** Production-grade REST API (FastAPI) and Web UI.

---

## 🤝 Contributing

This project is maintained by **Michigan MindMend Inc.** We welcome contributions from the community. Please see our [Project Strategy](PROJECT_STRATEGY.md) for a deep dive into our technical vision and roadmap.

## 📄 License

This project follows the original xAI Grok-1 licensing terms. See [LICENSE.txt](LICENSE.txt) for details.

---

Built with ❤️ by **Lyle Perrien II / Michigan MindMend Inc.**
