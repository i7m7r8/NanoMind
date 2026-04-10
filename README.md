# NanoMind — Ultra-Compressed Pure-Rust LLM Inference Engine

**Run any GGUF model on anything — even a phone. Zero Python. Zero dependencies at runtime.**

## Features

- **Direct GGUF v3 loading** — download any HuggingFace GGUF model, run it. No conversion needed.
- **Full llama.cpp quantization** — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K_M, Q5_K, Q6_K, IQ4_NL
- **Unlimited context** — ring buffer KV cache with automatic eviction. Extended context via RoPE scaling (NTK-aware, YARN, Linear).
- **Ollama-compatible HTTP API** — drop-in replacement for Ollama at `/api/generate`, `/api/chat`, `/api/embeddings`
- **Advanced sampling** — greedy, temperature, top-k, top-p, min-p, mirostat v2, repetition/presence/frequency penalty, logit bias
- **Multi-architecture** — LLaMA, Qwen2/2.5/3, Mistral, Phi-3, Gemma 2 with GQA, SwiGLU, MoE, sliding window, logit softcap
- **Cross-platform** — Linux x86_64/ARM64, Android ARM64, macOS, Windows

## Quick Start

```bash
# Download any GGUF model from HuggingFace
# Example: qwen2.5-0.5b-instruct-q4_k_m.gguf

# Single prompt
./nanomind --model ./model.gguf --prompt "Explain quantum computing"

# Interactive chat
./nanomind --model ./model.gguf --chat

# Ollama-compatible server
./nanomind --model ./model.gguf --server --host 127.0.0.1:8080

# Then use with any Ollama client:
# curl http://127.0.0.1:8080/api/generate -d '{"model":"nanomind","prompt":"Hello"}'

# Benchmark
./nanomind --model ./model.gguf --bench

# Extended context (128K with RoPE scaling)
./nanomind --model ./model.gguf --ctx-size 131072 --prompt "Summarize this..."
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--model <path>` | Path to GGUF model file | *required* |
| `--prompt <text>` | Single prompt | stdin |
| `--chat` | Interactive chat mode | off |
| `--server` | Start HTTP server | off |
| `--host <addr>` | Server listen address | `127.0.0.1:8080` |
| `--bench` | Benchmark 100 tokens | off |
| `--max-tokens <n>` | Max tokens (0 = unlimited) | 0 |
| `--temp <float>` | Temperature (0 = greedy) | 0.8 |
| `--top-k <n>` | Top-k filtering | 40 |
| `--top-p <float>` | Nucleus sampling | 0.95 |
| `--repeat-penalty <float>` | Repetition penalty | 1.1 |
| `--mirostat <0\|2>` | Mirostat sampling | 0 (off) |
| `--ctx-size <n>` | Context window | model default |
| `--seed <n>` | Seed for reproducibility | 42 |

## Ollama API Compatibility

When running with `--server`, NanoMind exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Text completion |
| `/api/chat` | POST | Chat completion |
| `/api/embeddings` | POST | Text embeddings |
| `/api/tags` | GET | List loaded models |
| `/api/show` | POST | Model info |

Use with any Ollama-compatible client:

```bash
# Open WebUI, AnythingLLM, Continue.dev, etc.
# Point them at http://127.0.0.1:8080
```

## Build

```bash
cargo build --release --all
# Binary: target/release/nanomind
```

## Project Structure

```
NanoMind/
├── crates/
│   ├── nanomind-core/      # GGML types, dequant kernels, RoPE, tensor ops
│   ├── nanomind-gguf/      # GGUF v3 file parser (mmap-backed, zero-copy)
│   ├── nanomind-model/     # Transformer forward pass
│   ├── nanomind-tokenizer/ # Token-level tokenizer
│   ├── nanomind-sampling/  # Advanced sampling (mirostat, penalties, etc.)
│   ├── nanomind-server/    # Ollama-compatible HTTP server
│   └── nanomind/           # CLI binary
└── .github/workflows/
    ├── ci.yml              # Build + test + clippy on every push
    └── release.yml         # Cross-compile release binaries (Linux, Android, macOS, Windows)
```

## Downloads

See [Releases](https://github.com/i7m7r8/NanoMind/releases) for pre-built binaries:

- `nanomind-linux-x86_64.tar.gz`
- `nanomind-linux-aarch64.tar.gz`
- `nanomind-android-arm64.tar.gz`
- `nanomind-macos-x86_64.tar.gz`
- `nanomind-windows-x86_64.tar.gz`

## License

MIT
