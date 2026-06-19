# Grok-1 Inference Engine: Optimized JAX Implementation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)

This is a research fork of the xAI Grok-1 release used for RoPE, inference-wiring, and benchmark experiments. It is not an official xAI repository or a production-ready Grok service. Full-model validation requires the original weights and accelerator resources well beyond a typical workstation.

## 🚀 Key Features

- **RoPE regression work:** Includes focused tests for the repository's rotary-position code paths.
- **Optimized JAX Runtime:** Leverages JAX SPMD for efficient activation sharding across multiple GPUs and hosts.
- **Benchmark harness:** Provides a reproducible entry point for measuring the included RoPE path on the user's own hardware.
- **Tracer-Leak Protection:** Advanced JAX implementation preventing `UnexpectedTracerError` during model initialization.
- **Experimental backends:** Contains native JAX and exploratory Triton-oriented code paths that require environment-specific validation.

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

Run the focused tests and benchmark to verify the code paths available in your environment. These commands do not validate full 314B-parameter inference:

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
