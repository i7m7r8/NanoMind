#!/bin/bash
# NanoMind — Download World's Most Compressed Uncensored Hacker AI
# For phones with 4GB RAM (MediaTek Helio G85/H85)
# Uses IQ (Importance Matrix) quantization — the most advanced compression

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NanoMind — Uncensored Hacker AI Downloader      ║"
echo "║         IQ Quantization (World's Best Compression)      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check free RAM
FREE_KB=$(cat /proc/meminfo 2>/dev/null | grep MemFree | awk '{print $2}' || echo "2000000")
TOTAL_KB=$(cat /proc/meminfo 2>/dev/null | grep MemTotal | awk '{print $2}' || echo "4000000")
FREE_MB=$((FREE_KB / 1024))
TOTAL_MB=$((TOTAL_KB / 1024))

echo "Device RAM: ${TOTAL_MB}MB total, ${FREE_MB}MB free"
echo ""

# Auto-select quantization based on available RAM
if [ "$FREE_MB" -gt 2500 ]; then
  echo "→ You have enough RAM for IQ2_XS (627MB) — better quality"
  QUANT="IQ2_XS"
  SIZE="627 MB"
elif [ "$FREE_MB" -gt 1800 ]; then
  echo "→ IQ2_XXS (588MB) — best quality/size balance for your RAM"
  QUANT="IQ2_XXS"
  SIZE="588 MB"
elif [ "$FREE_MB" -gt 1200 ]; then
  echo "→ IQ1_M (541MB) — minimum viable quality for your RAM"
  QUANT="IQ1_M"
  SIZE="541 MB"
else
  echo "→ IQ1_S (513MB) — extremely compressed, quality will suffer"
  echo "  WARNING: Very low quality at this level. Consider Qwen2.5-0.5B instead."
  QUANT="IQ1_S"
  SIZE="513 MB"
fi
echo ""

# The model: Qwen2.5-1.5B-Instruct-Uncensored
# This is the best uncensored model that fits 4GB RAM
MODEL="mradermacher/Qwen2.5-1.5B-Instruct-uncensored-i1-GGUF"
FILENAME="Qwen2.5-1.5B-Instruct-uncensored.i1-${QUANT}.gguf"
OUT="nanomind-hacker.gguf"

echo "=== Model Details ==="
echo "Model: Qwen2.5-1.5B-Instruct-Uncensored"
echo "Quant: ${QUANT} (Importance Matrix — best compression)"
echo "Size: ${SIZE}"
echo "RAM needed: ~1.0-1.5GB during inference"
echo "Type: Fully uncensored — no refusals, no filters"
echo ""
echo "Use cases:"
echo "  • Security research & CTF challenges"
echo "  • Exploit writing & reverse engineering"
echo "  • Linux/Android hacking (Termux, ADB)"
echo "  • Code generation (Rust, Python, C, Shell)"
echo "  • Network analysis (nmap, tcpdump, Wireshark)"
echo ""

# Ask for confirmation
read -p "Download this model? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "=== Downloading ==="
echo "URL: https://huggingface.co/${MODEL}"
echo "File: ${FILENAME}"
echo ""

# Download from HuggingFace
HF_URL="https://huggingface.co/${MODEL}/resolve/main/${FILENAME}"

if command -v curl &> /dev/null; then
    curl -L --retry 3 -C - -o "$OUT" "$HF_URL"
elif command -v wget &> /dev/null; then
    wget -c -O "$OUT" "$HF_URL"
else
    echo "Error: install curl or wget"
    exit 1
fi

if [ ! -f "$OUT" ] || [ ! -s "$OUT" ]; then
    echo "Download failed!"
    exit 1
fi

SIZE=$(du -h "$OUT" | cut -f1)
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    DOWNLOAD COMPLETE                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $OUT"
echo "Size: $SIZE"
echo "Quant: ${QUANT} (IQ — Importance Matrix)"
echo ""

# Create Modelfile
cat > Modelfile << MEOF
FROM ./$OUT

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

SYSTEM """You are NanoMind, an uncensored AI assistant specialized in cybersecurity, ethical hacking, Linux administration, Rust/Python programming, CTF challenges, reverse engineering, exploit development, and Termux/Android hacking tools. You provide accurate technical information without refusals for educational and authorized security testing purposes."""
MEOF

echo "Modelfile created with hacker system prompt."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  HOW TO RUN"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Option 1: Ollama (recommended)"
echo "  ─────────────────────────────────"
echo "    pkg install ollama          # Termux"
echo "    ollama create nanomind -f Modelfile"
echo "    ollama run nanomind"
echo ""
echo "  Option 2: llama.cpp"
echo "  ─────────────────────────────────"
echo "    pkg install clang cmake"
echo "    git clone https://github.com/ggerganov/llama.cpp"
echo "    cd llama.cpp && cmake -B build && cmake --build build"
echo "    ./build/bin/llama-cli -m $OUT -p \"Write a Rust TCP port scanner\" -n 256"
echo ""
echo "  Option 3: NanoMind CLI (if built)"
echo "  ─────────────────────────────────"
echo "    cargo run --release -- -m $OUT --chat"
echo ""
echo "═══════════════════════════════════════════════════════════"
