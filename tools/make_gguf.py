#!/usr/bin/env python3
"""
Convert any HuggingFace model (safetensors or pytorch) to GGUF v3 format.

Usage:
    # From a local HuggingFace model directory
    python tools/make_gguf.py --model ./Qwen2.5-0.5B-Instruct --out nanomind-Q4_K_M.gguf --outtype q4_k_m

    # Download and convert automatically
    python tools/make_gguf.py --model Qwen/Qwen2.5-0.5B-Instruct --out nanomind.gguf --outtype q4_k_m

    # Available quantization types:
    #   f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k_m, q4_k_m, q5_k_m, q6_k

Requirements:
    pip install torch transformers safetensors numpy gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType, Keys
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False
    print("[WARN] gguf package not found. Installing...")
    os.system("pip install gguf")
    from gguf import GGUFWriter, GGMLQuantizationType, Keys


QUANT_MAP = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q2_k": GGMLQuantizationType.Q2_K,
    "q3_k_m": GGMLQuantizationType.Q3_K,
    "q4_k_m": GGMLQuantizationType.Q4_K,
    "q5_k_m": GGMLQuantizationType.Q5_K,
    "q6_k": GGMLQuantizationType.Q6_K,
}

ARCH_MAP = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "llama": "llama",
    "mistral": "mistral",
    "phi3": "phi3",
    "gemma2": "gemma2",
    "gemma": "gemma",
    "deepseek2": "deepseek2",
    "stablelm": "stablelm",
}


def load_hf_model(model_dir: str):
    """Load model config and weights from a HuggingFace directory."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        # Try downloading
        from transformers import AutoConfig, AutoModelForCausalLM
        print(f"[INFO] Downloading {model_dir}...")
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        config.save_pretrained(model_dir)
    else:
        with open(config_path) as f:
            config = json.load(f)
        config = argparse.Namespace(**config) if isinstance(config, dict) else config

    # Try safetensors
    tensors = {}
    st_path = Path(model_dir) / "model.safetensors"
    if st_path.exists():
        from safetensors import safe_open
        with safe_open(st_path, framework="numpy") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
    else:
        # Try index
        idx_path = Path(model_dir) / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
            loaded_files = set()
            for key, filename in weight_map.items():
                fp = Path(model_dir) / filename
                if filename not in loaded_files:
                    from safetensors import safe_open
                    with safe_open(fp, framework="numpy") as sf:
                        for k in sf.keys():
                            if k in weight_map:
                                tensors[k] = sf.get_tensor(k)
                    loaded_files.add(filename)
        else:
            # Try PyTorch
            import torch
            pt_path = Path(model_dir) / "pytorch_model.bin"
            if pt_path.exists():
                state_dict = torch.load(pt_path, map_location="cpu")
                for key, tensor in state_dict.items():
                    tensors[key] = tensor.numpy()
            else:
                raise FileNotFoundError("No weight files found in model directory")

    return config, tensors


def detect_architecture(config) -> str:
    """Detect architecture from config."""
    arch = getattr(config, 'architectures', [''])[0].lower()
    if 'qwen2' in arch or 'qwen3' in arch:
        return 'qwen2'
    elif 'llama' in arch:
        return 'llama'
    elif 'mistral' in arch:
        return 'mistral'
    elif 'phi3' in arch:
        return 'phi3'
    elif 'gemma2' in arch:
        return 'gemma2'
    elif 'gemma' in arch:
        return 'gemma'
    return 'llama'  # default


def write_gguf(config, tensors: dict, out_path: str, quant_type: str, arch: str):
    """Write tensors to GGUF v3 format."""
    if not HAS_GGUF:
        print("[ERROR] Please install the gguf package: pip install gguf")
        sys.exit(1)

    # Map architecture
    gguf_arch = ARCH_MAP.get(arch, arch)

    # Extract config values
    vocab_size = getattr(config, 'vocab_size', 0)
    hidden_size = getattr(config, 'hidden_size', 0)
    intermediate_size = getattr(config, 'intermediate_size', 0)
    num_hidden_layers = getattr(config, 'num_hidden_layers', 0)
    num_attention_heads = getattr(config, 'num_attention_heads', 0)
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    head_dim = getattr(config, 'head_dim', hidden_size // num_attention_heads)
    rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-5)
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
    bos_token_id = getattr(config, 'bos_token_id', 1)
    eos_token_id = getattr(config, 'eos_token_id', 2)

    quant = QUANT_MAP.get(quant_type, GGMLQuantizationType.Q4_K)

    print(f"[INFO] Writing GGUF file: {out_path}")
    print(f"[INFO] Architecture: {arch} ({gguf_arch})")
    print(f"[INFO] Parameters: {sum(np.prod(list(t.shape)) for t in tensors.values()) / 1e9:.1f}B")
    print(f"[INFO] Quantization: {quant_type}")
    print(f"[INFO] Layers: {num_hidden_layers}")

    # Create GGUF writer
    writer = GGUFWriter(out_path, gguf_arch)

    # Write architecture metadata
    writer.add_architecture()
    writer.add_uint32(Keys.LLM.vocab_size, vocab_size)
    writer.add_uint32(Keys.LLM.embedding_length, hidden_size)
    writer.add_uint32(Keys.LLM.feed_forward_length, intermediate_size)
    writer.add_uint32(Keys.LLM.block_count, num_hidden_layers)
    writer.add_uint32(Keys.LLM.attention.head_count, num_attention_heads)
    writer.add_uint32(Keys.LLM.attention.head_count_kv, num_key_value_heads)

    if head_dim != hidden_size // num_attention_heads:
        writer.add_uint32(Keys.LLM.attention.key_length, head_dim)
        writer.add_uint32(Keys.LLM.attention.value_length, head_dim)

    writer.add_float32(Keys.LLM.attention.layer_norm_rms_epsilon, rms_norm_eps)
    writer.add_float32(Keys.LLM.rope.freq_base, rope_theta)
    writer.add_uint32(Keys.LLM.context_length, max_position_embeddings)

    # Tokenizer metadata
    writer.add_uint32(Keys.Tokenizer.tokenizer_model, "llama")
    if bos_token_id:
        writer.add_uint32(Keys.Tokenizer.bos_token_id, bos_token_id)
    if eos_token_id:
        writer.add_uint32(Keys.Tokenizer.eos_token_id, eos_token_id)

    # Quantization type
    writer.add_uint32(Keys.General.quantization_type, quant.value if hasattr(quant, 'value') else quant)

    # Write tensors with quantization
    total_params = 0
    for name, tensor in tensors.items():
        # Convert numpy array
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)

        # GGUF stores dims in reverse order for some tensors
        data = tensor

        total_params += data.size
        writer.add_tensor(name, data, raw_dtype=False)

        if total_params % 500_000_000 == 0:
            print(f"  ... written {total_params / 1e9:.1f}B params")

    # Write to file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    file_size = Path(out_path).stat().st_size / (1024**3)
    print(f"\n[SUCCESS] GGUF model written: {out_path}")
    print(f"[SUCCESS] File size: {file_size:.2f} GB")
    print(f"[SUCCESS] Total parameters: {total_params:,}")
    print(f"\nUsage with Ollama:")
    print(f"  1. Create a Modelfile:")
    print(f'     echo "FROM ./nanomind.gguf" > Modelfile')
    print(f"  2. Import into Ollama:")
    print(f"     ollama create nanomind -f Modelfile")
    print(f"  3. Run:")
    print(f"     ollama run nanomind")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to GGUF")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID")
    parser.add_argument("--out", required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", default="q4_k_m",
                        choices=list(QUANT_MAP.keys()),
                        help="Output quantization type")

    args = parser.parse_args()

    model_dir = args.model

    # Download if needed
    if not Path(model_dir).exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[INFO] Downloading {model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    config, tensors = load_hf_model(model_dir)
    arch = detect_architecture(config)
    write_gguf(config, tensors, args.out, args.outtype, arch)


if __name__ == "__main__":
    main()
