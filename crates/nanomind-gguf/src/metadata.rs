//! GGUF metadata key-value store.

use std::collections::HashMap;
use std::io;

use crate::reader::ReadLeExt;

/// Metadata value types.
#[derive(Clone, Debug)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF64(Vec<f64>),
    ArrayString(Vec<String>),
}

/// Parsed GGUF metadata.
pub struct GgufMetadata {
    pub kv: HashMap<String, MetadataValue>,
    pub architecture: String,
}

impl GgufMetadata {
    pub fn parse<R: io::Read>(cursor: &mut R, count: u64, _version: u32) -> io::Result<Self> {
        let mut kv = HashMap::new();

        for _ in 0..count {
            let key = cursor.read_string()?;
            let value = parse_metadata_value(cursor)?;
            kv.insert(key, value);
        }

        let architecture = kv
            .get("general.architecture")
            .and_then(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".into());

        Ok(Self { kv, architecture })
    }

    /// Get a string value by key.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| match v {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Get a u32 value by key.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.kv.get(key).and_then(|v| match v {
            MetadataValue::U32(v) => Some(*v),
            _ => None,
        })
    }

    /// Get an f32 value by key.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| match v {
            MetadataValue::F32(v) => Some(*v),
            _ => None,
        })
    }

    /// Total parameter count.
    pub fn param_count(&self) -> u64 {
        self.get_u32("general.parameter_count").unwrap_or(0) as u64
    }

    /// Quantization type string.
    pub fn quantization_type(&self) -> String {
        self.get_string("general.quantization_type")
            .unwrap_or("unknown")
            .to_string()
    }

    /// Architecture-specific helpers.
    pub fn arch_string(&self) -> &str {
        &self.architecture
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.vocab_size", arch))
            .or_else(|| {
                self.get_u32("tokenizer.ggml.tokens").map(|_| {
                    // Count tokens from array
                    if let Some(MetadataValue::ArrayString(tokens)) =
                        self.kv.get("tokenizer.ggml.tokens")
                    {
                        tokens.len() as u32
                    } else {
                        0
                    }
                })
            })
            .unwrap_or(0)
    }

    /// Context length.
    pub fn context_length(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.context_length", arch))
            .or_else(|| self.get_u32(&format!("{}.n_ctx_train", arch)))
            .unwrap_or(4096)
    }

    /// Embedding dimension.
    pub fn embedding_length(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.embedding_length", arch))
            .unwrap_or(0)
    }

    /// Number of layers.
    pub fn block_count(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.block_count", arch)).unwrap_or(0)
    }

    /// Number of attention heads.
    pub fn head_count(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.attention.head_count", arch))
            .unwrap_or(0)
    }

    /// Number of KV heads (for GQA).
    pub fn head_count_kv(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.attention.head_count_kv", arch))
            .unwrap_or_else(|| self.head_count())
    }

    /// FFN embedding dimension.
    pub fn ffn_dim(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.feed_forward_length", arch))
            .unwrap_or(0)
    }

    /// RMS norm epsilon.
    pub fn rms_norm_eps(&self) -> f32 {
        let arch = &self.architecture;
        self.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            .or_else(|| self.get_f32(&format!("{}.attention.norm_eps", arch)))
            .unwrap_or(1e-5)
    }

    /// RoPE theta.
    pub fn rope_theta(&self) -> f32 {
        let arch = &self.architecture;
        self.get_f32(&format!("{}.rope.freq_base", arch))
            .unwrap_or(10000.0)
    }

    /// RoPE dimension count.
    pub fn rope_dim(&self) -> Option<u32> {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.rope.dimension_count", arch))
    }

    /// Expert count (for MoE models).
    pub fn expert_count(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.expert_count", arch)).unwrap_or(0)
    }

    /// Expert used count (for MoE).
    pub fn expert_used_count(&self) -> u32 {
        let arch = &self.architecture;
        self.get_u32(&format!("{}.expert_used_count", arch))
            .unwrap_or(0)
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.get_u32("tokenizer.ggml.bos_token_id")
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.get_u32("tokenizer.ggml.eos_token_id")
    }
}

fn parse_metadata_value<R: io::Read>(cursor: &mut R) -> io::Result<MetadataValue> {
    let ty = cursor.read_u32_le()?;
    Ok(match ty {
        0 => MetadataValue::U8(cursor.read_u8()?),
        1 => MetadataValue::I8(cursor.read_u8()? as i8),
        2 => MetadataValue::U16(cursor.read_u16_le()?),
        3 => MetadataValue::I16(cursor.read_i16_le()?),
        4 => MetadataValue::U32(cursor.read_u32_le()?),
        5 => MetadataValue::I32(cursor.read_i32_le()?),
        6 => MetadataValue::F32(cursor.read_f32_le()?),
        7 => {
            let b = cursor.read_u8()?;
            MetadataValue::Bool(b != 0)
        }
        8 => MetadataValue::String(cursor.read_string()?),
        9 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<u8> = (0..n).map(|_| cursor.read_u8()).collect::<Result<_, _>>()?;
            MetadataValue::ArrayU8(items)
        }
        10 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<i8> = (0..n)
                .map(|_| cursor.read_u8().map(|b| b as i8))
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayI8(items)
        }
        11 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<u16> = (0..n)
                .map(|_| cursor.read_u16_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayU16(items)
        }
        12 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<i16> = (0..n)
                .map(|_| cursor.read_i16_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayI16(items)
        }
        13 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<u32> = (0..n)
                .map(|_| cursor.read_u32_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayU32(items)
        }
        14 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<i32> = (0..n)
                .map(|_| cursor.read_i32_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayI32(items)
        }
        15 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<f32> = (0..n)
                .map(|_| cursor.read_f32_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayF32(items)
        }
        16 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<u64> = (0..n)
                .map(|_| cursor.read_u64_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayU64(items)
        }
        17 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<i64> = (0..n)
                .map(|_| cursor.read_i64_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayI64(items)
        }
        18 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<f64> = (0..n)
                .map(|_| cursor.read_f64_le())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayF64(items)
        }
        19 => {
            let n = cursor.read_u64_le()? as usize;
            let items: Vec<String> = (0..n)
                .map(|_| cursor.read_string())
                .collect::<Result<_, _>>()?;
            MetadataValue::ArrayString(items)
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown metadata type: {}", ty),
            ))
        }
    })
}
