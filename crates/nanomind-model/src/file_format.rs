//! .nm model file format — custom binary format for NanoMind.
//!
//! Format:
//! ```text
//! [magic: 4 bytes "NANO"]
//! [version: u32 LE]
//! [config_len: u32 LE]
//! [config: JSON bytes, length-prefixed]
//! [num_tensors: u32 LE]
//! for each tensor:
//!   [name_len: u32 LE]
//!   [name: UTF-8 bytes]
//!   [dtype: u8]  // 0=f32, 1=f16, 2=q4, 3=q8
//!   [ndim: u8]
//!   [dims: ndim x u32 LE]
//!   [data_len: u32 LE]
//!   [data: raw bytes]
//! ```

use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;
use std::vec::Vec;

use nanomind_core::quantization::{Q4Block, QuantizedTensor};

use crate::model::{Config, Model, Layer, QuantType};

/// Magic bytes for the .nm format.
pub const NM_MAGIC: [u8; 4] = *b"NANO";
/// File format version.
pub const NM_VERSION: u32 = 1;

/// Load a model from a .nm file.
///
/// Uses streaming read — does NOT memory-map the file.
/// For large models, consider mmap-backed loading.
pub fn load_model(path: &Path) -> Result<Model, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| format!("Read magic: {}", e))?;
    if magic != NM_MAGIC {
        return Err(format!("Invalid magic: {:?}, expected {:?}", magic, NM_MAGIC));
    }

    // Read version
    let version = read_u32(&mut reader).map_err(|e| format!("Read version: {}", e))?;
    if version != NM_VERSION {
        return Err(format!("Unsupported version: {}", version));
    }

    // Read config
    let config_len = read_u32(&mut reader).map_err(|e| format!("Read config len: {}", e))?;
    let mut config_bytes = vec![0u8; config_len as usize];
    reader.read_exact(&mut config_bytes).map_err(|e| format!("Read config: {}", e))?;
    let config: Config = serde_json::from_slice(&config_bytes)
        .map_err(|e| format!("Parse config: {}", e))?;

    // Read tensors
    let num_tensors = read_u32(&mut reader).map_err(|e| format!("Read num tensors: {}", e))?;

    let mut model = Model::new(config.clone());

    // Pre-allocate layers
    model.layers = (0..config.num_layers)
        .map(|_| Layer {
            attn_q: QuantizedTensor::new(vec![0], vec![]),
            attn_k: QuantizedTensor::new(vec![0], vec![]),
            attn_v: QuantizedTensor::new(vec![0], vec![]),
            attn_o: QuantizedTensor::new(vec![0], vec![]),
            ffn_gate: QuantizedTensor::new(vec![0], vec![]),
            ffn_up: QuantizedTensor::new(vec![0], vec![]),
            ffn_down: QuantizedTensor::new(vec![0], vec![]),
            attn_norm: vec![0.0; config.hidden_dim],
            ffn_norm: vec![0.0; config.hidden_dim],
        })
        .collect();

    // Parse RoPE tables
    let head_dim = config.head_dim();
    let half_dim = head_dim / 2;
    model.rope_cos = vec![0.0; config.max_seq_len * half_dim];
    model.rope_sin = vec![0.0; config.max_seq_len * half_dim];

    for _ in 0..num_tensors {
        let name = read_string(&mut reader).map_err(|e| format!("Read name: {}", e))?;
        let dtype_byte = read_u8(&mut reader).map_err(|e| format!("Read dtype: {}", e))?;
        let ndim = read_u8(&mut reader).map_err(|e| format!("Read ndim: {}", e))?;

        let mut shape = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            let dim = read_u32(&mut reader).map_err(|e| format!("Read dim: {}", e))?;
            shape.push(dim as usize);
        }

        let data_len = read_u32(&mut reader).map_err(|e| format!("Read data len: {}", e))?;
        let mut data = vec![0u8; data_len as usize];
        reader.read_exact(&mut data).map_err(|e| format!("Read data: {}", e))?;

        let quant_type = QuantType::from_u8(dtype_byte)
            .ok_or_else(|| format!("Unknown dtype: {}", dtype_byte))?;

        let tensor = deserialize_tensor(&shape, quant_type, &data)?;
        assign_tensor(&mut model, &name, tensor, &config)?;
    }

    Ok(model)
}

/// Save a model to a .nm file.
pub fn save_model(model: &Model, path: &Path) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path.display(), e))?;
    let mut writer = BufWriter::new(file);

    // Write magic
    writer.write_all(&NM_MAGIC).map_err(|e| format!("Write magic: {}", e))?;

    // Write version
    write_u32(&mut writer, NM_VERSION).map_err(|e| format!("Write version: {}", e))?;

    // Write config
    let config_bytes = serde_json::to_vec(&model.config)
        .map_err(|e| format!("Serialize config: {}", e))?;
    write_u32(&mut writer, config_bytes.len() as u32)
        .map_err(|e| format!("Write config len: {}", e))?;
    writer.write_all(&config_bytes).map_err(|e| format!("Write config: {}", e))?;

    // Collect all tensors
    let mut tensors: Vec<(String, QuantizedTensor)> = Vec::new();

    tensors.push(("token_embeddings".into(), model.token_embeddings.clone()));
    tensors.push(("output_weights".into(), model.output_weights.clone()));

    // Final norm as f32 tensor
    let fn_shape = vec![model.config.hidden_dim];
    let fn_data: Vec<f32> = model.final_norm.clone();
    tensors.push(("final_norm".into(), f32_to_qtensor(&fn_data, &fn_shape)));

    for (i, layer) in model.layers.iter().enumerate() {
        tensors.push((format!("layers.{}.attn_q", i), layer.attn_q.clone()));
        tensors.push((format!("layers.{}.attn_k", i), layer.attn_k.clone()));
        tensors.push((format!("layers.{}.attn_v", i), layer.attn_v.clone()));
        tensors.push((format!("layers.{}.attn_o", i), layer.attn_o.clone()));
        tensors.push((format!("layers.{}.ffn_gate", i), layer.ffn_gate.clone()));
        tensors.push((format!("layers.{}.ffn_up", i), layer.ffn_up.clone()));
        tensors.push((format!("layers.{}.ffn_down", i), layer.ffn_down.clone()));

        tensors.push((format!("layers.{}.attn_norm", i), f32_to_qtensor(&layer.attn_norm, &[model.config.hidden_dim])));
        tensors.push((format!("layers.{}.ffn_norm", i), f32_to_qtensor(&layer.ffn_norm, &[model.config.hidden_dim])));
    }

    // RoPE tables
    tensors.push(("rope_cos".into(), f32_to_qtensor(&model.rope_cos, &[model.rope_cos.len()])));
    tensors.push(("rope_sin".into(), f32_to_qtensor(&model.rope_sin, &[model.rope_sin.len()])));

    // Write tensor count
    write_u32(&mut writer, tensors.len() as u32)
        .map_err(|e| format!("Write num tensors: {}", e))?;

    // Write each tensor
    for (name, tensor) in tensors {
        write_string(&mut writer, &name).map_err(|e| format!("Write name: {}", e))?;

        // For now, always save as Q4
        let dtype_byte = 2u8; // Q4
        write_u8(&mut writer, dtype_byte).map_err(|e| format!("Write dtype: {}", e))?;

        let ndim = tensor.shape.len() as u8;
        write_u8(&mut writer, ndim).map_err(|e| format!("Write ndim: {}", e))?;

        for &dim in &tensor.shape {
            write_u32(&mut writer, dim as u32).map_err(|e| format!("Write dim: {}", e))?;
        }

        // Write block data
        let data_bytes = serialize_q4_blocks(&tensor.blocks);
        write_u32(&mut writer, data_bytes.len() as u32)
            .map_err(|e| format!("Write data len: {}", e))?;
        writer.write_all(&data_bytes).map_err(|e| format!("Write data: {}", e))?;
    }

    writer.flush().map_err(|e| format!("Flush: {}", e))?;
    Ok(())
}

// ─── Binary I/O Helpers ─────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u8<W: Write>(w: &mut W, v: u8) -> std::io::Result<()> {
    w.write_all(&[v])
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_string<R: Read>(r: &mut R) -> std::io::Result<String> {
    let len = read_u32(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

// ─── Tensor Serialization ───────────────────────────────────────────────────

fn serialize_q4_blocks(blocks: &[Q4Block]) -> Vec<u8> {
    // Each block: 2 bytes (scale f16) + 2 bytes (min f16) + 16 bytes (quants)
    let mut data = Vec::with_capacity(blocks.len() * 20);
    for block in blocks {
        data.extend_from_slice(&block.scale.to_le_bytes());
        data.extend_from_slice(&block.min.to_le_bytes());
        data.extend_from_slice(&block.quants);
    }
    data
}

fn deserialize_tensor(
    shape: &[usize],
    quant_type: QuantType,
    data: &[u8],
) -> Result<QuantizedTensor, String> {
    match quant_type {
        QuantType::Q4 => {
            let block_size = 20; // f16 + f16 + 16 bytes
            if !data.len().is_multiple_of(block_size) {
                return Err(format!("Q4 data length {} not aligned to {}", data.len(), block_size));
            }
            let num_blocks = data.len() / block_size;
            let mut blocks = Vec::with_capacity(num_blocks);
            for i in 0..num_blocks {
                let offset = i * block_size;
                let scale_bytes: [u8; 2] = data[offset..offset + 2].try_into().unwrap();
                let min_bytes: [u8; 2] = data[offset + 2..offset + 4].try_into().unwrap();
                let quants: [u8; 16] = data[offset + 4..offset + 20].try_into().unwrap();

                blocks.push(Q4Block {
                    scale: half::f16::from_le_bytes(scale_bytes),
                    min: half::f16::from_le_bytes(min_bytes),
                    quants,
                });
            }
            Ok(QuantizedTensor::new(shape.to_vec(), blocks))
        }
        QuantType::F32 => {
            // Convert f32 data to Q4 tensor
            let f32_count = data.len() / 4;
            let f32_data: Vec<f32> = (0..f32_count)
                .map(|i| {
                    let bytes: [u8; 4] = data[i * 4..i * 4 + 4].try_into().unwrap();
                    f32::from_le_bytes(bytes)
                })
                .collect();
            let blocks = nanomind_core::quantization::quantize_q4(&f32_data);
            Ok(QuantizedTensor::new(shape.to_vec(), blocks))
        }
        _ => Err(format!("Deserialization for {:?} not implemented yet", quant_type)),
    }
}

/// Convert f32 data to a Q4 quantized tensor.
fn f32_to_qtensor(data: &[f32], shape: &[usize]) -> QuantizedTensor {
    let blocks = nanomind_core::quantization::quantize_q4(data);
    QuantizedTensor::new(shape.to_vec(), blocks)
}

/// Assign a loaded tensor to the correct field in the model.
fn assign_tensor(
    model: &mut Model,
    name: &str,
    tensor: QuantizedTensor,
    config: &Config,
) -> Result<(), String> {
    match name {
        "token_embeddings" => model.token_embeddings = tensor,
        "output_weights" => model.output_weights = tensor,
        "final_norm" => {
            // Dequantize into f32 buffer
            let mut buf = vec![0.0f32; config.hidden_dim];
            tensor.dequantize_row(0, &mut buf);
            model.final_norm = buf;
        }
        "rope_cos" => {
            let mut buf = vec![0.0f32; tensor.num_elements()];
            tensor.dequantize_row(0, &mut buf);
            model.rope_cos = buf;
        }
        "rope_sin" => {
            let mut buf = vec![0.0f32; tensor.num_elements()];
            tensor.dequantize_row(0, &mut buf);
            model.rope_sin = buf;
        }
        _ => {
            if let Some(layer_name) = name.strip_prefix("layers.") {
                let parts: Vec<&str> = layer_name.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let layer_idx: usize = parts[0].parse().map_err(|_| format!("Bad layer index: {}", parts[0]))?;
                    let param_name = parts[1];

                    let layer = model.layers.get_mut(layer_idx)
                        .ok_or_else(|| format!("Layer {} out of range", layer_idx))?;

                    match param_name {
                        "attn_q" => layer.attn_q = tensor,
                        "attn_k" => layer.attn_k = tensor,
                        "attn_v" => layer.attn_v = tensor,
                        "attn_o" => layer.attn_o = tensor,
                        "ffn_gate" => layer.ffn_gate = tensor,
                        "ffn_up" => layer.ffn_up = tensor,
                        "ffn_down" => layer.ffn_down = tensor,
                        "attn_norm" => {
                            let mut buf = vec![0.0f32; config.hidden_dim];
                            tensor.dequantize_row(0, &mut buf);
                            layer.attn_norm = buf;
                        }
                        "ffn_norm" => {
                            let mut buf = vec![0.0f32; config.hidden_dim];
                            tensor.dequantize_row(0, &mut buf);
                            layer.ffn_norm = buf;
                        }
                        _ => return Err(format!("Unknown layer param: {}", param_name)),
                    }
                }
            } else {
                return Err(format!("Unknown tensor name: {}", name));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanomind_core::quantization::quantize_q4;

    #[test]
    fn test_save_load_roundtrip() {
        let config = Config {
            vocab_size: 100,
            hidden_dim: 64,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            intermediate_dim: 128,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let mut model = Model::new(config);

        // Helper to create a Q4 tensor with properly sized data
        let make_qt = |rows: usize, cols: usize| -> QuantizedTensor {
            let n = rows * cols;
            let data: Vec<f32> = (0..n).map(|i| (i % 256) as f32 * 0.01).collect();
            let blocks = quantize_q4(&data);
            QuantizedTensor::new(vec![rows, cols], blocks)
        };

        model.token_embeddings = make_qt(100, 64);
        model.output_weights = make_qt(100, 64);

        // Fill the single layer
        let layer = &mut model.layers[0];
        layer.attn_q = make_qt(256, 64);   // [num_heads * head_dim, hidden]
        layer.attn_k = make_qt(256, 64);   // [num_kv_heads * head_dim, hidden]
        layer.attn_v = make_qt(256, 64);   // [num_kv_heads * head_dim, hidden]
        layer.attn_o = make_qt(64, 256);   // [hidden, num_heads * head_dim]
        layer.ffn_gate = make_qt(128, 64); // [intermediate, hidden]
        layer.ffn_up = make_qt(128, 64);   // [intermediate, hidden]
        layer.ffn_down = make_qt(64, 128); // [hidden, intermediate]
        layer.attn_norm = vec![1.0; 64];
        layer.ffn_norm = vec![1.0; 64];

        // Save
        let path = std::env::temp_dir().join("test_model.nm");
        save_model(&model, &path).expect("Save should succeed");

        // Load
        let loaded = load_model(&path).expect("Load should succeed");

        // Verify config
        assert_eq!(loaded.config.vocab_size, 100);
        assert_eq!(loaded.config.hidden_dim, 64);
        assert_eq!(loaded.layers.len(), 1);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
