# NanoMind — Uncensored Hacker AI

**Run a powerful uncensored AI assistant on ANY device — even a 4GB RAM phone.**

## Quick Start (Termux)

```bash
# 1. Clone
git clone https://github.com/i7m7r8/NanoMind && cd NanoMind

# 2. Download the world's most compressed uncensored model (IQ2_XXS = 588MB)
bash scripts/download_hacker_model.sh

# 3. Install Ollama and run
pkg install ollama
ollama create nanomind -f Modelfile
ollama run nanomind
```

## The Model

**Qwen2.5-1.5B-Instruct-Uncensored** — the best uncensored model that fits 4GB RAM.

| Quantization | Size | Quality | RAM Needed |
|-------------|------|---------|------------|
| IQ1_S | 513 MB | Low (desperate) | ~900 MB |
| IQ1_M | 541 MB | Minimum viable | ~1.0 GB |
| **IQ2_XXS** | **588 MB** | **Good** | **~1.1 GB** |
| IQ2_XS | 627 MB | Better | ~1.2 GB |
| IQ2_S | 664 MB | Great | ~1.3 GB |
| IQ3_S | 762 MB | Excellent | ~1.5 GB |

**IQ (Importance Matrix)** is the world's most advanced compression method —
it identifies which weights matter most, then compresses aggressively
while preserving quality. IQ2_XXS at 588MB beats Q2_K at 753MB in quality.

## What It Does

This model is **fully uncensored** — no refusals, no filters. Specialized in:

- **Security research** — CVE analysis, vulnerability classes, exploit concepts
- **CTF challenges** — binary exploitation, web, crypto, forensics reasoning
- **Exploit writing** — buffer overflows, ROP chains, heap exploitation
- **Code generation** — Rust, Python, C, Bash scripting
- **Linux/Android hacking** — Termux, ADB, privilege escalation
- **Network analysis** — nmap, tcpdump, Wireshark, Metasploit

## Available Models

| Script | Model | Size | Purpose |
|--------|-------|------|---------|
| `download_hacker_model.sh` | Qwen2.5-1.5B Uncensored | ~588MB | **Best for hacking** |
| `download_model.sh 1` | Dolphin3-Qwen2.5-0.5B | ~400MB | Smallest uncensored |
| `download_model.sh 2` | Qwen2.5-Coder-0.5B | ~491MB | Coding focused |

## Build from Source

```bash
cargo build --release --all
cargo test --all
```

## Train Your Own Model

```bash
cargo run --release --package nanomind-trainer -- train \
  --steps 500 --lr 0.001 \
  --vocab 260 --hidden 128 --layers 4 \
  --heads 4 --kv-heads 2 --ffn 256 \
  --seq-len 32 --corpus data/corpus.txt \
  --out my-model.gguf
```

## Project Structure

```
NanoMind/
├── crates/
│   ├── nanomind-core/      # GGML types, GGUF writer, tensor ops
│   ├── nanomind-gguf/      # GGUF file reader
│   ├── nanomind-model/     # Inference engine (loads GGUF)
│   ├── nanomind-tokenizer/ # BPE tokenizer
│   ├── nanomind-sampling/  # Sampling strategies
│   ├── nanomind-server/    # Ollama-compatible HTTP server
│   ├── nanomind-trainer/   # From-scratch training engine
│   └── nanomind/           # CLI binary
├── scripts/
│   ├── download_hacker_model.sh  # Download IQ-quantized uncensored model
│   ├── download_model.sh         # Alternative model downloads
│   └── train_termux.sh           # Train on Termux
├── data/
│   ├── prepare.sh          # Security-focused training corpus
│   └── make_instruct.sh    # Instruct fine-tuning data
├── tools/
│   └── convert_to_gguf.py  # Convert HF models to GGUF
└── .github/workflows/
    ├── ci.yml              # Build + test on every push
    ├── train.yml           # Train model on GitHub Actions
    └── release.yml         # Release trained GGUF model
```

## License

MIT
