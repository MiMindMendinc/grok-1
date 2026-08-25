# Checkpoints

Place Grok-1 weight shards in this directory for full-model inference.

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 \
  --repo-type model \
  --include "ckpt-0/*" \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

RoPE unit tests and `benchmarks/rope_benchmark.py` do not require these files.
