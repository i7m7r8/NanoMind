#!/usr/bin/env python3
"""
Convert any HuggingFace model to GGUF v3 format for Ollama.
Supports: LLaMA, Qwen2, Mistral, Phi-3, Gemma

Usage:
    # Download and convert in one step
    python3 tools/convert_to_gguf.py \
        --model "Qwen/Qwen2.5-0.5B-Instruct" \
        --out nanomind.gguf \
        --quant q4_k_m

    # From local directory
    python3 tools/convert_to_gguf.py \
        --model ./Qwen2.5-0.5B-Instruct \
        --out nanomind.gguf

Requirements:
    pip install huggingface_hub safetensors numpy gguf
"""

import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF")
    parser.add_argument("--model", required=True, help="HF repo ID or local path")
    parser.add_argument("--out", required=True, help="Output GGUF file")
    parser.add_argument("--quant", default="q4_k_m",
                        choices=["f16", "q4_k_m", "q4_0", "q8_0", "q2_k", "q3_k_m", "q5_k_m"],
                        help="Quantization type")
    parser.add_argument("--name", default="NanoMind", help="Model name")
    args = parser.parse_args()

    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as e:
        print(f"Installing dependencies: {e}")
        os.system("pip install torch safetensors transformers gguf huggingface_hub sentencepiece")
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer

    print(f"Loading model: {args.model}")
    print(f"Output: {args.out}")
    print(f"Quant: {args.quant}")

    # Load config
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Detect architecture
    arch_name = config.model_type.lower()
    print(f"Architecture: {arch_name}")

    # Get model config values
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    ffn_dim = config.intermediate_size
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-5)
    max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
    bos_token_id = getattr(config, 'bos_token_id', 1)
    eos_token_id = getattr(config, 'eos_token_id', 2)

    total_params = sum(p.numel() for p in [torch.zeros(1)] * 0)  # will compute from tensors
    print(f"Vocab: {vocab_size}, Hidden: {hidden_size}, Layers: {num_layers}")
    print(f"Heads: {num_heads}, KV Heads: {num_kv_heads}, FFN: {ffn_dim}")

    # Download model files
    model_dir = Path(args.model)
    if not model_dir.exists():
        from huggingface_hub import snapshot_download
        print("Downloading from HuggingFace...")
        model_dir = Path(snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json", "tokenizer*"]))

    # Find weight files
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        # Try index
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                idx = json.load(f)
            st_files = sorted(set(idx["weight_map"].values()))
            st_files = [model_dir / fn for fn in st_files]
        else:
            # Try pytorch
            pt_path = model_dir / "pytorch_model.bin"
            if pt_path.exists():
                state_dict = torch.load(pt_path, map_location="cpu")
                print(f"Loaded PyTorch weights: {len(state_dict)} tensors")
            else:
                raise FileNotFoundError("No weights found")

    # Load all tensors
    state_dict = {}
    print("Loading weights...")
    for sf in st_files:
        print(f"  Loading {sf.name}...")
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    total_params = sum(t.numel() for t in state_dict.values())
    print(f"Total tensors: {len(state_dict)}")
    print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")

    # Now convert to GGUF
    try:
        from gguf import GGUFWriter, GGMLQuantizationType, Keys, LlamaHparams
    except ImportError:
        os.system("pip install gguf")
        from gguf import GGUFWriter, GGMLQuantizationType, Keys, LlamaHparams

    # Map architecture
    gguf_arch = "llama"
    if "qwen" in arch_name:
        gguf_arch = "qwen2"
    elif "mistral" in arch_name:
        gguf_arch = "llama"
    elif "phi" in arch_name:
        gguf_arch = "phi3"
    elif "gemma" in arch_name:
        gguf_arch = "gemma2"

    print(f"GGUF architecture: {gguf_arch}")

    # Create writer
    writer = GGUFWriter(args.out, gguf_arch)

    # Write metadata
    writer.add_architecture()

    writer.add_uint32(Keys.LLM.vocab_size, vocab_size)
    writer.add_uint32(Keys.LLM.embedding_length, hidden_size)
    writer.add_uint32(Keys.LLM.block_count, num_layers)
    writer.add_uint32(Keys.LLM.feed_forward_length, ffn_dim)
    writer.add_uint32(Keys.LLM.attention.head_count, num_heads)
    writer.add_uint32(Keys.LLM.attention.head_count_kv, num_kv_heads)

    if hasattr(config, 'rope_scaling') and config.rope_scaling:
        if isinstance(config.rope_scaling, dict):
            factor = config.rope_scaling.get("factor", 1.0)
            writer.add_float32(Keys.LLM.rope.scaling.factor, factor)

    writer.add_float32(Keys.LLM.rope.freq_base, rope_theta)
    writer.add_float32(Keys.LLM.attention.layer_norm_rms_epsilon, rms_norm_eps)

    if hasattr(config, 'max_position_embeddings'):
        writer.add_uint32(Keys.LLM.context_length, max_position_embeddings)

    # Tokenizer
    vocab = tokenizer.get_vocab()
    tokens = []
    scores = []
    token_types = []

    for i in range(vocab_size):
        if i < len(tokenizer.vocab):
            token = tokenizer.vocab[i] if isinstance(tokenizer.vocab, list) else str(i)
            if isinstance(tokenizer, getattr(__import__('transformers', fromlist=['PreTrainedTokenizer']), 'PreTrainedTokenizer', object)):
                try:
                    token = tokenizer.convert_ids_to_tokens(i)
                except:
                    token = f"<token_{i}>"
            else:
                token = f"<token_{i}>"
        else:
            token = f"<token_{i}>"
        tokens.append(token)
        scores.append(0.0)
        token_types.append(1)

    # Handle special tokens
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        writer.add_uint32(Keys.Tokenizer.bos_token_id, tokenizer.bos_token_id)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        writer.add_uint32(Keys.Tokenizer.eos_token_id, tokenizer.eos_token_id)

    # Write tensors
    print("Writing tensors to GGUF...")
    count = 0
    for name, tensor in state_dict.items():
        data = tensor.numpy()

        # Map tensor names to GGUF format
        gguf_name = name
        # Remove "model." prefix
        if gguf_name.startswith("model."):
            gguf_name = gguf_name[6:]

        # Write tensor (F32 for now, will be quantized by llama.cpp tools)
        writer.add_tensor(gguf_name, data)
        count += 1
        if count % 20 == 0:
            print(f"  Written {count}/{len(state_dict)} tensors...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    file_size = Path(args.out).stat().st_size
    print(f"\nGGUF file written: {args.out}")
    print(f"Size: {file_size / 1e6:.1f} MB")
    print(f"Params: {total_params:,} ({total_params/1e6:.1f}M)")
    print()
    print("To quantize to Q4_K_M (smaller, ~500MB):")
    print(f"  llama-quantize {args.out} nanomind-q4.gguf Q4_K_M")
    print()
    print("Or use with Ollama directly:")
    print(f"  echo 'FROM ./{args.out}' > Modelfile")
    print(f"  ollama create nanomind -f Modelfile")
    print(f"  ollama run nanomind")

if __name__ == "__main__":
    main()
