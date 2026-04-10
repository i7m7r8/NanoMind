#!/usr/bin/env python3
"""
Convert HuggingFace Qwen2/LLaMA weights to NanoMind .nm format.

Usage:
    python convert_weights.py --model Qwen/Qwen2-0.5B-Instruct --out model.nm --quant q4

This runs on a desktop with Python + PyTorch. The output .nm file
can be copied to a phone and loaded by the NanoMind CLI.

Requirements:
    pip install torch transformers safetensors numpy
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


NM_MAGIC = b"NANO"
NM_VERSION = 1
DTYPE_Q4 = 2
QK4 = 32


def quantize_q4_block(data: np.ndarray) -> bytes:
    """
    Quantize a 1D float32 array to Q4 blocks.

    Each block (32 elements):
    - scale: f16 (2 bytes)
    - min: f16 (2 bytes)
    - quants: 16 bytes (32 x 4-bit packed)
    """
    assert len(data) % QK4 == 0, f"Data length {len(data)} not divisible by {QK4}"

    n_blocks = len(data) // QK4
    result = bytearray()

    for b in range(n_blocks):
        block = data[b * QK4:(b + 1) * QK4]

        min_val = float(np.min(block))
        max_val = float(np.max(block))

        # Asymmetric quantization
        range_val = max_val - min_val
        scale = range_val / 15.0 if range_val > 1e-8 else 1e-8

        # Quantize
        quantized = np.round((block + min_val) / scale).clip(0, 15).astype(np.uint8)

        # Pack two 4-bit values per byte
        packed = bytearray(16)
        for i in range(16):
            packed[i] = int(quantized[i * 2]) | (int(quantized[i * 2 + 1]) << 4)

        # Write scale as f16
        scale_f16 = np.float16(scale).tobytes()
        min_f16 = np.float16(min_val).tobytes()

        result.extend(scale_f16)
        result.extend(min_f16)
        result.extend(packed)

    return bytes(result)


def write_u32(f, v: int):
    f.write(struct.pack('<I', v))


def write_u8(f, v: int):
    f.write(struct.pack('B', v))


def write_string(f, s: str):
    data = s.encode('utf-8')
    write_u32(f, len(data))
    f.write(data)


def write_tensor(f, name: str, data: np.ndarray):
    """Write a tensor in Q4 format."""
    original_shape = data.shape

    # Flatten and pad to QK4 boundary
    flat = data.flatten().astype(np.float32)
    n = len(flat)
    padded_len = ((n + QK4 - 1) // QK4) * QK4
    if padded_len > n:
        flat = np.pad(flat, (0, padded_len - n), mode='constant')

    # Quantize
    block_data = quantize_q4_block(flat)

    # Write tensor header
    write_string(f, name)
    write_u8(f, DTYPE_Q4)
    write_u8(f, len(original_shape))
    for dim in original_shape:
        write_u32(f, int(dim))

    # Write data
    write_u32(f, len(block_data))
    f.write(block_data)


def load_config(model_path: Path) -> dict:
    """Load model config from config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def load_weights(model_path: Path) -> dict:
    """Load model weights from safetensors or pytorch_model.bin."""
    weights = {}

    # Try safetensors first
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists() and HAS_SAFETENSORS:
        with safe_open(safetensors_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights

    # Try index
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists() and HAS_SAFETENSORS:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})

        loaded_files = set()
        for key, filename in weight_map.items():
            file_path = model_path / filename
            if filename not in loaded_files:
                with safe_open(file_path, framework="numpy") as sf:
                    for k in sf.keys():
                        if k in weight_map:
                            weights[k] = sf.get_tensor(k)
                loaded_files.add(filename)
        return weights

    # Try PyTorch
    if HAS_TORCH:
        pt_path = model_path / "pytorch_model.bin"
        if pt_path.exists():
            state_dict = torch.load(pt_path, map_location="cpu")
            for key, tensor in state_dict.items():
                weights[key] = tensor.numpy()
            return weights

    raise RuntimeError(
        "No weight files found. Install 'safetensors' or 'torch' "
        "and ensure model.safetensors or pytorch_model.bin exists."
    )


def convert_weights(model_dir: str, output: str, quant: str = "q4"):
    """Main conversion function."""
    model_path = Path(model_dir)

    print(f"[INFO] Loading config from {model_dir}...")
    config = load_config(model_path)

    print(f"[INFO] Loading weights...")
    weights = load_weights(model_path)
    print(f"[INFO] Loaded {len(weights)} tensors")

    # Map HuggingFace config to NanoMind config
    hf_config = config

    nm_config = {
        "vocab_size": hf_config.get("vocab_size", 151936),
        "hidden_dim": hf_config.get("hidden_size", 896),
        "num_heads": hf_config.get("num_attention_heads", 14),
        "num_kv_heads": hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", 14)),
        "num_layers": hf_config.get("num_hidden_layers", 24),
        "intermediate_dim": hf_config.get("intermediate_size", 4864),
        "max_seq_len": hf_config.get("max_position_embeddings", 32768),
        "rope_theta": float(hf_config.get("rope_theta", 1000000.0)),
        "rms_norm_eps": float(hf_config.get("rms_norm_eps", 1e-6)),
    }

    print(f"[INFO] NanoMind config: {json.dumps(nm_config, indent=2)}")

    # Name mapping: HuggingFace → NanoMind
    def map_name(name: str, layer_idx: int) -> str:
        """Map HuggingFace tensor names to NanoMind format."""
        # Embedding
        if name == "model.embed_tokens.weight":
            return "token_embeddings"
        if name == "lm_head.weight":
            return "output_weights"
        if name == "model.norm.weight":
            return "final_norm"

        # Layer-specific mappings
        prefix = f"model.layers.{layer_idx}."
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            mapping = {
                "self_attn.q_proj.weight": "attn_q",
                "self_attn.k_proj.weight": "attn_k",
                "self_attn.v_proj.weight": "attn_v",
                "self_attn.o_proj.weight": "attn_o",
                "mlp.gate_proj.weight": "ffn_gate",
                "mlp.up_proj.weight": "ffn_up",
                "mlp.down_proj.weight": "ffn_down",
                "input_layernorm.weight": "attn_norm",
                "post_attention_layernorm.weight": "ffn_norm",
            }
            if suffix in mapping:
                return f"layers.{layer_idx}.{mapping[suffix]}"

        return None

    # Collect tensors for the .nm file
    tensors = {}
    num_layers = nm_config["num_layers"]

    for name, data in weights.items():
        # Check if it's a global tensor (not layer-specific)
        if name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
            tensors[name] = data
            continue

        # Check which layer this belongs to
        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}."
            if name.startswith(prefix):
                nm_name = map_name(name, layer_idx)
                if nm_name:
                    tensors[nm_name] = data
                break

    # Write .nm file
    print(f"[INFO] Writing {len(tensors)} tensors to {output}...")

    with open(output, "wb") as f:
        # Magic + version
        f.write(NM_MAGIC)
        write_u32(f, NM_VERSION)

        # Config
        config_json = json.dumps(nm_config).encode("utf-8")
        write_u32(f, len(config_json))
        f.write(config_json)

        # Tensor count
        write_u32(f, len(tensors))

        # Write each tensor
        for name, data in sorted(tensors.items()):
            print(f"  Writing {name} ({data.shape})")
            write_tensor(f, name, data)

    file_size = Path(output).stat().st_size / (1024 * 1024)
    print(f"[INFO] Done! Output size: {file_size:.1f} MB")
    print(f"[INFO] Copy {output} to your phone and run:")
    print(f"       nanomind --model ./model.nm --prompt 'Hello'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace weights to NanoMind .nm format"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to HuggingFace model directory or HF model ID"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output .nm file path"
    )
    parser.add_argument(
        "--quant", default="q4", choices=["q4", "q8"],
        help="Quantization type (default: q4)"
    )

    args = parser.parse_args()

    # If model is a HF model ID, try to download it first
    model_dir = args.model
    if not Path(model_dir).exists():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[INFO] Downloading {model_dir}...")
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"[INFO] Downloaded to {model_dir}")
        except ImportError:
            print(f"[ERROR] Model directory {model_dir} not found.")
            print("Install transformers: pip install transformers")
            sys.exit(1)

    convert_weights(model_dir, args.out, args.quant)


if __name__ == "__main__":
    main()
