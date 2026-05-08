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

"""Tiny browser chat demo for Grok-1.

This server intentionally uses only the Python standard library so the website can be
started before installing the heavy JAX/GPU stack. Use ``--backend demo`` for a fast
local smoke test, or ``--backend grok`` on a machine with the Grok-1 checkpoint and
sufficient accelerator memory.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable


CKPT_PATH = "./checkpoints/"
TOKENIZER_PATH = "./tokenizer.model"


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grok-1 Web Demo</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #05070d;
      color: #eef3ff;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 20% 20%, rgba(58, 134, 255, 0.30), transparent 28rem),
        radial-gradient(circle at 80% 10%, rgba(255, 0, 110, 0.22), transparent 22rem),
        linear-gradient(135deg, #05070d, #101728 65%, #060910);
    }
    main {
      width: min(920px, calc(100vw - 32px));
      padding: 32px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 28px;
      background: rgba(8, 13, 25, 0.78);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
      backdrop-filter: blur(18px);
    }
    .eyebrow {
      color: #8eb9ff;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    h1 {
      margin: 8px 0 12px;
      font-size: clamp(2.3rem, 7vw, 5.4rem);
      line-height: 0.9;
    }
    p { color: #bdc8dc; line-height: 1.6; }
    form {
      display: grid;
      gap: 16px;
      margin-top: 28px;
    }
    textarea, input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.06);
      color: inherit;
      padding: 16px;
      font: inherit;
    }
    textarea { min-height: 140px; resize: vertical; }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 16px 22px;
      color: #04111f;
      background: linear-gradient(135deg, #7cc7ff, #c1ff72);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: 0.65; transform: none; }
    pre {
      white-space: pre-wrap;
      min-height: 150px;
      margin: 22px 0 0;
      padding: 20px;
      border-radius: 18px;
      background: rgba(0, 0, 0, 0.33);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #ecf5ff;
      line-height: 1.55;
    }
    .status { min-height: 1.5em; color: #c1ff72; }
    @media (max-width: 720px) { .controls { grid-template-columns: 1fr; } main { padding: 22px; } }
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Local AI Website</div>
    <h1>Spin up Grok-1 in the browser.</h1>
    <p>
      This page posts prompts to a local Python server. Start in demo mode for a quick website
      preview, then switch to the Grok backend on GPU hardware with the checkpoint installed.
    </p>
    <form id="prompt-form">
      <label>
        Prompt
        <textarea id="prompt">The answer to life, the universe, and everything is</textarea>
      </label>
      <div class="controls">
        <label>Max tokens <input id="max-len" type="number" min="1" max="512" value="96"></label>
        <label>Temperature <input id="temperature" type="number" min="0.01" max="2" step="0.01" value="0.7"></label>
        <button id="submit" type="submit">Generate</button>
      </div>
    </form>
    <div class="status" id="status">Ready.</div>
    <pre id="output">Responses appear here.</pre>
  </main>
  <script>
    const form = document.querySelector('#prompt-form');
    const button = document.querySelector('#submit');
    const output = document.querySelector('#output');
    const status = document.querySelector('#status');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.textContent = 'Generating...';
      output.textContent = '';
      try {
        const response = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: document.querySelector('#prompt').value,
            max_len: Number(document.querySelector('#max-len').value),
            temperature: Number(document.querySelector('#temperature').value)
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        output.textContent = data.output;
        status.textContent = `Generated with ${data.backend} backend.`;
      } catch (error) {
        output.textContent = error.message;
        status.textContent = 'Generation failed.';
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def make_demo_generate(backend_name: str) -> Callable[[str, int, float], dict[str, str | float | int]]:
    """Create a deterministic placeholder generator for local website smoke tests."""

    def generate(prompt: str, max_len: int, temperature: float) -> dict[str, str | float | int]:
        safe_prompt = " ".join(prompt.split()) or "your prompt"
        output = (
            f"{safe_prompt}\n\n"
            "[demo] Grok-1 web plumbing is online. Replace demo mode with "
            "`--backend grok` on a GPU host with checkpoints/ckpt-0 installed to run the "
            "open-weights model behind this same page."
        )
        return {
            "backend": backend_name,
            "output": output[: max(1, max_len) * 8],
            "temperature": temperature,
            "max_len": max_len,
        }

    return generate


def make_grok_generate(
    checkpoint_path: str,
    tokenizer_path: str,
) -> Callable[[str, int, float], dict[str, str | float | int]]:
    """Initialize Grok-1 and return a callable suitable for the web API."""

    from model import LanguageModelConfig, TransformerConfig
    from runners import InferenceRunner, ModelRunner, sample_from_model

    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
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
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=checkpoint_path,
        ),
        name="web",
        load=checkpoint_path,
        tokenizer_path=tokenizer_path,
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    generator = inference_runner.run()

    def generate(prompt: str, max_len: int, temperature: float) -> dict[str, str | float | int]:
        output = sample_from_model(generator, prompt, max_len=max_len, temperature=temperature)
        return {
            "backend": "grok",
            "output": output,
            "temperature": temperature,
            "max_len": max_len,
        }

    return generate


class GrokWebHandler(BaseHTTPRequestHandler):
    """HTTP routes for the static page and JSON generation endpoint."""

    generate: Callable[[str, int, float], dict[str, str | float | int]]
    backend_name: str

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_html(INDEX_HTML)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json()
            prompt = str(payload.get("prompt", ""))
            max_len = int(payload.get("max_len", 96))
            temperature = float(payload.get("temperature", 0.7))
            if not prompt.strip():
                raise ValueError("Prompt is required.")
            if max_len < 1 or max_len > 512:
                raise ValueError("max_len must be between 1 and 512.")
            if temperature <= 0 or temperature > 2:
                raise ValueError("temperature must be greater than 0 and no more than 2.")
            self._send_json(self.generate(prompt, max_len, temperature))
        except Exception as exc:
            logging.exception("Generation request failed")
            self._send_json({"error": html.escape(str(exc))}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8") or "{}")

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a tiny Grok-1 browser demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--backend",
        choices=("demo", "grok"),
        default="demo",
        help="Use demo for a quick website smoke test or grok for the real model.",
    )
    parser.add_argument("--checkpoint-path", default=CKPT_PATH, help="Directory containing ckpt-0.")
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH, help="SentencePiece tokenizer path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = parse_args()

    if args.backend == "grok":
        ckpt_dir = Path(args.checkpoint_path) / "ckpt-0"
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Expected Grok-1 checkpoint at {ckpt_dir}. Start with --backend demo for UI only."
            )
        generate = make_grok_generate(args.checkpoint_path, args.tokenizer_path)
    else:
        generate = make_demo_generate(args.backend)

    GrokWebHandler.generate = staticmethod(generate)
    GrokWebHandler.backend_name = args.backend
    server = ThreadingHTTPServer((args.host, args.port), GrokWebHandler)
    logging.info("Serving Grok-1 web demo at http://%s:%s", args.host, args.port)
    logging.info("Backend: %s", args.backend)
    server.serve_forever()


if __name__ == "__main__":
    main()
