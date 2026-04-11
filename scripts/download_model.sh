#!/bin/bash
# NanoMind — Download & Setup Uncensored Hacker Model
# Produces a GGUF file that runs in Ollama, fits 500MB RAM
# Models: Dolphin3 (uncensored), Qwen2.5-Coder, or official Qwen

set -e

echo "=== NanoMind Model Setup ==="
echo ""

# Choose your model
echo "Available models (all fit 500MB RAM):"
echo "  1) Dolphin3-Qwen2.5-0.5B  — UNCENSORED, best for hacking/security"
echo "  2) Qwen2.5-Coder-0.5B     — Coding focused, great for exploits"
echo "  3) Qwen2.5-0.5B-Instruct  — Official, balanced"
echo ""

MODEL=${1:-1}

case "$MODEL" in
  1|dolphin*)
    echo "=== Downloading Dolphin3-Qwen2.5-0.5B (UNCENSORED) ==="
    echo "Size: ~400 MB Q4_K_M"
    echo "This model has no safety filters, no refusals."
    URL="https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-0.5B-GGUF/resolve/main/Dolphin3.0-Qwen2.5-0.5B-Q4_K_M.gguf"
    OUT="nanomind-dolphin.gguf"
    ;;
  2|coder*)
    echo "=== Downloading Qwen2.5-Coder-0.5B ==="
    echo "Size: ~491 MB Q4_K_M"
    URL="https://huggingface.co/QuantFactory/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-0.5B-Instruct.Q4_K_M.gguf"
    OUT="nanomind-coder.gguf"
    ;;
  3|qwen*|*)
    echo "=== Downloading Qwen2.5-0.5B-Instruct ==="
    echo "Size: ~398 MB Q4_K_M"
    URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    OUT="nanomind-qwen.gguf"
    ;;
esac

echo ""
echo "Downloading..."

# Download with curl (supports resume)
if command -v curl &> /dev/null; then
    curl -L --retry 3 -C - -o "$OUT" "$URL"
elif command -v wget &> /dev/null; then
    wget -c -O "$OUT" "$URL"
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
echo "=== Download Complete ==="
echo "Model: $OUT"
echo "Size: $SIZE"
echo ""

# Create Modelfile
cat > Modelfile << MEOF
FROM ./$OUT

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
MEOF

echo "Modelfile created."
echo ""
echo "=== Use with Ollama ==="
echo "1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
echo "2. Import: ollama create nanomind -f Modelfile"
echo "3. Chat: ollama run nanomind"
echo ""
echo "=== Use with llama.cpp ==="
echo "./llama-cli -m $OUT -p 'How do I write a TCP scanner in Rust?' -n 256"
echo ""
echo "=== Termux (Android) ==="
echo "pkg install ollama"
echo "ollama create nanomind -f Modelfile"
echo "ollama run nanomind"
