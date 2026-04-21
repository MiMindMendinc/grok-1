# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging

from model import LanguageModelConfig, TransformerConfig
from runners import InferenceRunner, ModelRunner, sample_from_model

DEFAULT_CKPT_PATH = "./checkpoints/"
DEFAULT_TOKENIZER_PATH = "./tokenizer.model"


def build_model_config(sequence_len: int = 8192, rope_backend: str = "jax") -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=sequence_len,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            num_experts=8,
            num_selected_experts=2,
            data_axis="data",
            model_axis="model",
            rope_backend=rope_backend,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grok-1 JAX inference.")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", default="The answer to life, the universe, and everything is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-len", type=int, default=8192)
    parser.add_argument(
        "--rope-backend",
        choices=("jax", "triton"),
        default="jax",
        help="Rotary embedding backend. Triton backend falls back to JAX if unavailable.",
    )
    parser.add_argument(
        "--local-mesh-config",
        type=int,
        nargs=2,
        default=(1, 8),
        metavar=("DATA", "MODEL"),
        help="Local mesh shape: data_axis model_axis",
    )
    parser.add_argument(
        "--between-hosts-config",
        type=int,
        nargs=2,
        default=(1, 1),
        metavar=("DATA", "MODEL"),
        help="Cross-host mesh shape: data_axis model_axis",
    )
    parser.add_argument(
        "--pad-sizes",
        type=int,
        nargs="+",
        default=(1024,),
        help="Prompt prefill buckets used for compilation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode and keep generating for new prompts.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.sequence_len <= 0:
        raise ValueError("--sequence-len must be > 0.")
    if sorted(args.pad_sizes) != list(args.pad_sizes):
        raise ValueError("--pad-sizes must be sorted in ascending order.")


def generate(
    gen,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    logging.info(
        "Generating with max_new_tokens=%d temperature=%.3f top_p=%.3f seed=%d",
        max_new_tokens,
        temperature,
        top_p,
        seed,
    )
    return sample_from_model(
        gen,
        prompt=prompt,
        max_len=max_new_tokens,
        temperature=temperature,
        nucleus_p=top_p,
        rng_seed=seed,
    )


def interactive_loop(
    gen,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    print("Interactive mode. Enter prompt text (Ctrl+C to quit).")
    while True:
        prompt = input("\nPrompt> ").strip()
        if not prompt:
            print("Please enter a non-empty prompt.")
            continue
        output = generate(gen, prompt, max_new_tokens, temperature, top_p, seed)
        print(f"\nCompletion:\n{output}")


def main():
    args = parse_args()
    validate_args(args)

    model_config = build_model_config(sequence_len=args.sequence_len, rope_backend=args.rope_backend)
    inference_runner = InferenceRunner(
        pad_sizes=tuple(args.pad_sizes),
        runner=ModelRunner(
            model=model_config,
            bs_per_device=0.125,
            checkpoint_path=args.checkpoint_path,
        ),
        name="local",
        load=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        local_mesh_config=tuple(args.local_mesh_config),
        between_hosts_config=tuple(args.between_hosts_config),
    )

    logging.info("Initializing inference runner...")
    inference_runner.initialize()
    gen = inference_runner.run()

    if args.interactive:
        interactive_loop(
            gen,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        return

    output = generate(
        gen,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(f"\nPrompt:\n{args.prompt}\n")
    print(f"Completion:\n{output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
