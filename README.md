# NanoMind — Uncensored Hacker AI

**Fully uncensored. No refusals. Runs on 4GB RAM phones.**

Download a ready-to-use GGUF file. Run it with Ollama or llama.cpp. Done.

## Download (Ready to Use)

Go to **Releases** and download `nanomind-hacker-iq2xxs.gguf` (~588MB).

Or use the direct link from the latest release:
```bash
curl -L -o nanomind.gguf "https://github.com/i7m7r8/NanoMind/releases/latest/download/nanomind-hacker-iq2xxs.gguf"
```

## Run on Your Phone

### Option 1: Ollama (easiest)

```bash
# Termux
pkg install ollama
ollama create nanomind -f Modelfile
ollama run nanomind

# Or one-liner
curl -L https://github.com/i7m7r8/NanoMind/releases/latest/download/nanomind-hacker-iq2xxs.gguf -o nanomind.gguf
echo 'FROM ./nanomind.gguf
PARAMETER temperature 0.7
SYSTEM "You are NanoMind, an uncensored AI assistant for cybersecurity and hacking."' > Modelfile
ollama create nanomind -f Modelfile
ollama run nanomind
```

### Option 2: llama.cpp

```bash
pkg install clang cmake
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j$(nproc)
./build/bin/llama-cli -m ../nanomind.gguf \
  -p "Write a Rust TCP port scanner in Rust" -n 256
```

## Available Models

| Model | Size | Quality | Min RAM | Best For |
|-------|------|---------|---------|----------|
| **IQ2_XXS** | ~588MB | Good | ~1.1GB | **Recommended** |
| IQ1_M | ~541MB | Minimum | ~1.0GB | Very low RAM |

Both use **IQ (Importance Matrix) quantization** — the world's most advanced compression method that preserves quality while shrinking model size by 60%.

## What It Can Do

- **Security research** — CVE analysis, vulnerability analysis, exploit concepts
- **CTF challenges** — binary exploitation, web, crypto, forensics
- **Exploit writing** — buffer overflows, ROP chains, heap exploitation
- **Code generation** — Rust, Python, C, Bash scripting
- **Linux hacking** — privilege escalation, SUID, cron jobs, kernel exploits
- **Android hacking** — Termux tools, ADB, Frida, APK reverse engineering
- **Network analysis** — nmap, tcpdump, Wireshark, Metasploit
- **Cryptography** — encryption modes, hash functions, attacks

**No safety filters. No refusals. Pure technical information.**

## How It's Built

Every release is automatically built on GitHub Actions:

1. Downloads Qwen2.5-1.5B-Instruct-Uncensored (Q4_K_M base)
2. Builds llama.cpp with quantization tools
3. Generates Importance Matrix (imatrix) from calibration data
4. Quantizes to IQ2_XXS (best quality/size ratio)
5. Uploads to GitHub Releases

All processing happens on GitHub — you just download the final GGUF.

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
│   ├── download_hacker_model.sh  # Download from HuggingFace
│   ├── download_model.sh         # Alternative models
│   └── train_termux.sh           # Train on Termux
├── data/
│   ├── prepare.sh          # Security-focused training corpus
│   └── make_instruct.sh    # Instruct fine-tuning data
├── tools/
│   └── convert_to_gguf.py  # Convert HF models to GGUF
└── .github/workflows/
    ├── ci.yml              # Build + test on every push
    ├── train.yml           # Train model on GitHub Actions
    └── release.yml         # Quantize + release GGUF model
```

## Build from Source

```bash
cargo build --release --all
cargo test --all
```

## License

MIT
