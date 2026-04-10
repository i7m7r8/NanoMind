# NanoMind — Ultra-Compressed Rust Language Model Inference Engine

<p align="center">
  <strong>Run a 1B parameter LLM on a phone with 500 MB RAM. Pure Rust. Zero Python.</strong>
</p>

## Features

- **INT4 Block Quantization** (Q4_K_M style) — 1B model fits in ~600 MB on disk, ~480 MB in RAM
- **Pure Rust** — no PyTorch, no ONNX, no Python runtime
- **Memory-mapped weights** — zero-copy loading via `memmap2`
- **Grouped Query Attention (GQA)** — efficient KV cache
- **BPE Tokenizer** — compatible with Qwen/LLaMA vocabularies
- **Cross-compiled for Android ARM64** — runs on Termux
- **CLI with streaming output** — chat mode, benchmark mode

## Quick Start

```bash
# Build
cargo build --release --all

# Run inference
nanomind --model ./model.nm --prompt "Hello, world!" --max-tokens 128

# Interactive chat mode
nanomind --model ./model.nm --chat

# Benchmark
nanomind --model ./model.nm --prompt "The meaning of life is" --bench
```

## Compression Targets

| Model Size | Params | Disk (Q4) | RAM Target |
|------------|--------|-----------|------------|
| nano       | 135M   | ~90 MB    | ~120 MB    |
| mini       | 500M   | ~330 MB   | ~380 MB    |
| small      | 1B     | ~650 MB   | ~480 MB    |

## Project Structure

```
NanoMind/
├── crates/
│   ├── nanomind-core/      # Q4 quantization, tensor ops, SIMD, RoPE
│   ├── nanomind-tokenizer/ # BPE tokenizer (no_std compatible)
│   ├── nanomind-model/     # Qwen2-style transformer, .nm file format
│   ├── nanomind-runtime/   # Inference loop, sampling, KV cache
│   └── nanomind-cli/       # Binary: chat, bench, streaming
├── tools/
│   └── convert_weights.py  # Desktop tool: HF → .nm format
├── models/                 # .gitignored weight files
└── .github/workflows/      # CI + release (Android cross-compile)
```

## License

MIT
