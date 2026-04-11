# NanoMind — Train an LLM from Scratch in Pure Rust

**A language model trained from zero, output as GGUF, runs with Ollama. No external weights. No HuggingFace. Pure Rust.**

## What This Is

NanoMind trains a transformer language model **from scratch** using only:
- Raw text data (public domain books)
- Pure Rust training code
- GitHub Actions free tier (2 CPU cores, 7GB RAM, 6 hours)

The output is a **real GGUF file** that works with Ollama.

## Quick Start

### Option 1: Train on GitHub Actions (free, no setup)

1. Go to **Actions → Train (Manual)**
2. Click **Run workflow**
3. Choose training steps (500-5000) and learning rate
4. Wait for completion
5. Download `nanomind-trained.tar.gz` from artifacts

```bash
# Use with Ollama:
tar xzf nanomind-trained.tar.gz
ollama create nanomind -f Modelfile
ollama run nanomind
```

### Option 2: Train Locally

```bash
# Download training data
bash data/prepare.sh

# Train model
cargo run --release --package nanomind-trainer -- \
  train \
  --steps 1000 \
  --lr 0.001 \
  --vocab 260 \
  --hidden 128 \
  --layers 4 \
  --heads 4 \
  --seq-len 32 \
  --out nanomind.gguf

# Use with Ollama:
echo 'FROM ./nanomind.gguf' > Modelfile
ollama create nanomind -f Modelfile
ollama run nanomind
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--steps <n>` | Training steps | 200 |
| `--lr <float>` | Learning rate | 0.001 |
| `--vocab <n>` | Vocabulary size | 260 (byte-level) |
| `--hidden <n>` | Hidden dimension | 128 |
| `--layers <n>` | Number of layers | 4 |
| `--heads <n>` | Attention heads | 4 |
| `--seq-len <n>` | Sequence length | 32 |
| `--out <path>` | Output GGUF path | `nanomind.gguf` |
| `--seed <n>` | Random seed | 42 |

## Model Configurations

| Config | Params | RAM | Use Case |
|--------|--------|-----|----------|
| nano (default) | ~2M | ~120 MB | CI / quick tests |
| mini | ~15M | ~340 MB | Desktop training |
| small | ~50M | ~480 MB | Quality model |

## Architecture

```
Transformer (decoder-only)
├── Byte-level tokenizer (vocab 260)
├── Embedding: [vocab × hidden]
├── N layers:
│   ├── RMSNorm + QKV projections
│   ├── Multi-Head Attention (GQA)
│   ├── RoPE positional encoding
│   ├── SwiGLU FFN
│   └── Residual connections
├── Final RMSNorm
└── Output projection (tied embeddings)
```

## Training Details

- **Optimizer:** AdamW with cosine LR schedule + warmup
- **Gradient clipping:** 1.0 max norm
- **Weight decay:** 0.01
- **Data:** Public domain books (Alice in Wonderland, Shakespeare, etc.)
- **Output:** GGUF v3 format (Ollama-compatible)

## Ollama Integration

The GGUF file produced is natively compatible with Ollama:

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM ./nanomind.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 256
PARAMETER stop "<|eos|>"

SYSTEM """You are NanoMind, a small AI assistant trained from scratch in pure Rust."""
EOF

# Import into Ollama
ollama create nanomind -f Modelfile

# Chat with your model
ollama run nanomind "Write a short story about a robot"
```

## Project Structure

```
NanoMind/
├── crates/
│   ├── nanomind-core/      # GGML types, GGUF writer, tensor ops
│   ├── nanomind-gguf/      # GGUF file reader
│   ├── nanomind-model/     # Inference engine (loads GGUF)
│   ├── nanomind-tokenizer/ # Tokenizer for inference
│   ├── nanomind-sampling/  # Sampling strategies
│   ├── nanomind-server/    # HTTP API server
│   ├── nanomind-trainer/   # FROM-SCRATCH TRAINING ENGINE
│   │   ├── autodiff.rs     # Reverse-mode autodiff (backprop)
│   │   ├── model.rs        # Transformer architecture
│   │   ├── optimizer.rs    # AdamW optimizer
│   │   ├── data_loader.rs  # Training data pipeline
│   │   └── train.rs        # Training loop + GGUF export
│   └── nanomind/           # CLI binary (inference)
├── data/
│   └── prepare.sh          # Download training data
├── .github/workflows/
│   ├── ci.yml              # Build + test on every push
│   ├── train.yml           # Manual training trigger
│   └── release.yml         # Cross-compile binaries
└── README.md
```

## Training Data

Public domain books (no copyright issues):
- Alice in Wonderland (Lewis Carroll)
- Shakespeare Complete Works
- The Art of War (Sun Tzu)
- Aesop's Fables

Download with: `bash data/prepare.sh`

## Expected Output Quality

After 1000 steps on the default corpus:
- **Coherent short sentences** (5-15 words)
- **Basic grammar patterns**
- **Repetitive but not nonsense**

This is a **tiny** model (~2M params) trained on CPU. Don't expect ChatGPT — expect a toy that produces English-like text.

## Build

```bash
cargo build --release --all
cargo test --all  # 43 tests pass
```

## License

MIT
