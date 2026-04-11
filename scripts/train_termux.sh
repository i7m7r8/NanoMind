#!/bin/bash
# Train NanoMind on Termux (optimized for phones with limited RAM/CPU)

set -e

echo "=== NanoMind Termux Training ==="
echo "Device RAM: $(free -m 2>/dev/null | awk 'NR==2{print $2}' || echo 'unknown')MB"

# Check available RAM and pick model size
FREE_MB=$(free -m 2>/dev/null | awk 'NR==2{print $4}' || echo 500)
if [ "$FREE_MB" -gt 1500 ]; then
  HIDDEN=512; LAYERS=8; HEADS=8; KV=2; FFN=1024; SEQ=256; VOCAB=512
  echo "Using: medium config (512 hidden, 8 layers)"
elif [ "$FREE_MB" -gt 800 ]; then
  HIDDEN=256; LAYERS=6; HEADS=4; KV=2; FFN=512; SEQ=128; VOCAB=512
  echo "Using: small config (256 hidden, 6 layers)"
else
  HIDDEN=128; LAYERS=4; HEADS=4; KV=2; FFN=256; SEQ=64; VOCAB=260
  echo "Using: micro config (128 hidden, 4 layers)"
fi

STEPS=${1:-2000}
LR=${2:-0.0005}

echo "Steps: $STEPS | LR: $LR"
echo ""

# Prepare corpus
if [ ! -f data/corpus.txt ] || [ $(wc -c < data/corpus.txt) -lt 1000 ]; then
  echo "Generating built-in security corpus..."
  bash data/make_instruct.sh 2>/dev/null || true
  cat data/raw/security_concepts.txt data/instruct.txt 2>/dev/null > data/corpus.txt || true
fi

echo "Corpus: $(wc -c < data/corpus.txt) bytes"
echo ""

# Build (release for speed)
echo "Building trainer..."
cargo build --release --package nanomind-trainer 2>&1 | tail -5

# Train
echo ""
echo "Training... (Ctrl+C to stop, checkpoint saved every 500 steps)"
cargo run --release --package nanomind-trainer -- train \
  --steps $STEPS \
  --lr $LR \
  --vocab $VOCAB \
  --hidden $HIDDEN \
  --layers $LAYERS \
  --heads $HEADS \
  --kv-heads $KV \
  --ffn $FFN \
  --seq-len $SEQ \
  --corpus data/corpus.txt \
  --out nanomind-local.gguf

echo ""
echo "=== Done ==="
echo "Model: nanomind-local.gguf ($(du -h nanomind-local.gguf | cut -f1))"
echo ""
echo "Load in Ollama:"
echo "  ollama create nanomind -f Modelfile"
echo "  ollama run nanomind"
