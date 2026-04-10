//! GGUF v3 file writer — produces Ollama-compatible GGUF files.
//!
//! Implements the exact GGUF binary format spec:
//! <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! Ollama/llama.cpp reads these files natively.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

// ─── GGUF Constants ────────────────────────────────────────────────────────

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

// ─── GGUF Value Types ──────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ─── GGUF Value ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Array of values — all must be the same type
    Array(Vec<GgufValue>),
}

// ─── GGUF Tensor Data Type ─────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12, // Q4_K_M
    Q5_K = 13,
    Q6_K = 14,
    IQ4_NL = 15,
    BF16 = 30,
}

impl GgufDType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::IQ4_NL),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Number of bytes per block
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q4_0 => 2 + 16,
            Self::Q4_1 => 2 + 2 + 32,
            Self::Q5_0 => 2 + 4 + 32,
            Self::Q5_1 => 2 + 2 + 4 + 32,
            Self::Q8_0 => 2 + 32,
            Self::Q2_K => 1 + 1 + 32 + 16,
            Self::Q3_K => 1 + 1 + 32 + 8 + 16,
            Self::Q4_K => 2 + 2 + 12 + 128,
            Self::Q5_K => 2 + 2 + 12 + 176,
            Self::Q6_K => 1 + 32 + 16 + 128,
            Self::IQ4_NL => 2 + 16,
        }
    }

    /// Number of elements per block
    pub fn blck_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::BF16 => 1,
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::IQ4_NL => 32,
        }
    }
}

// ─── Tensor Info ───────────────────────────────────────────────────────────

pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64,
}

// ─── GGUF Writer ───────────────────────────────────────────────────────────

pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensor_infos: Vec<GgufTensorInfo>,
    data_buf: Vec<u8>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensor_infos: Vec::new(),
            data_buf: Vec::new(),
        }
    }

    /// Add a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: GgufValue) {
        self.metadata.push((key.to_string(), value));
    }

    /// Add a tensor. `data` must be the raw bytes of the tensor
    /// (already quantized if using Q4_K_M etc).
    /// `dims` should be `[cols, rows]` for 2D tensors (GGUF reverse order).
    pub fn add_tensor(&mut self, name: &str, dims: Vec<u64>, dtype: GgufDType, data: &[u8]) {
        let offset = self.data_buf.len() as u64;
        self.data_buf.extend_from_slice(data);
        // Pad to alignment
        let pad = (ALIGNMENT - (self.data_buf.len() as u64 % ALIGNMENT)) % ALIGNMENT;
        self.data_buf.resize(self.data_buf.len() + pad as usize, 0);

        self.tensor_infos.push(GgufTensorInfo {
            name: name.to_string(),
            dims,
            dtype,
            offset,
        });
    }

    /// Write the complete GGUF file to disk.
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // ── Header ──
        writer.write_all(GGUF_MAGIC)?;
        writer.write_all(&GGUF_VERSION.to_le_bytes())?;
        writer.write_all(&(self.tensor_infos.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        // ── Metadata ──
        for (key, value) in &self.metadata {
            self.write_metadata_entry(&mut writer, key, value)?;
        }

        // ── Tensor Infos ──
        for info in &self.tensor_infos {
            writer.write_all(&(info.name.len() as u64).to_le_bytes())?;
            writer.write_all(info.name.as_bytes())?;
            writer.write_all(&(info.dims.len() as u32).to_le_bytes())?;
            for &dim in &info.dims {
                writer.write_all(&dim.to_le_bytes())?;
            }
            writer.write_all(&(info.dtype as u32).to_le_bytes())?;
            writer.write_all(&info.offset.to_le_bytes())?;
        }

        // ── Padding before tensor data ──
        let data_offset = self.data_offset();
        let current_offset = self.header_size() + self.metadata_size() + self.tensor_info_size();
        let pad = data_offset - current_offset;
        for _ in 0..pad {
            writer.write_all(&[0])?;
        }

        // ── Tensor Data ──
        writer.write_all(&self.data_buf)?;

        writer.flush()?;
        Ok(())
    }

    // ── Internal helpers ──

    fn write_metadata_entry<W: Write>(
        &self,
        w: &mut W,
        key: &str,
        value: &GgufValue,
    ) -> io::Result<()> {
        w.write_all(&(key.len() as u64).to_le_bytes())?;
        w.write_all(key.as_bytes())?;
        self.write_value(w, value)?;
        Ok(())
    }

    fn write_value<W: Write>(&self, w: &mut W, value: &GgufValue) -> io::Result<()> {
        match value {
            GgufValue::U8(v) => {
                w.write_all(&(GgufValueType::Uint8 as u32).to_le_bytes())?;
                w.write_all(&[*v])?;
            }
            GgufValue::I8(v) => {
                w.write_all(&(GgufValueType::Int8 as u32).to_le_bytes())?;
                w.write_all(&[*v as u8])?;
            }
            GgufValue::U16(v) => {
                w.write_all(&(GgufValueType::Uint16 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::I16(v) => {
                w.write_all(&(GgufValueType::Int16 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::U32(v) => {
                w.write_all(&(GgufValueType::Uint32 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::I32(v) => {
                w.write_all(&(GgufValueType::Int32 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::F32(v) => {
                w.write_all(&(GgufValueType::Float32 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::Bool(v) => {
                w.write_all(&(GgufValueType::Bool as u32).to_le_bytes())?;
                w.write_all(&[*v as u8])?;
            }
            GgufValue::String(v) => {
                w.write_all(&(GgufValueType::String as u32).to_le_bytes())?;
                w.write_all(&(v.len() as u64).to_le_bytes())?;
                w.write_all(v.as_bytes())?;
            }
            GgufValue::U64(v) => {
                w.write_all(&(GgufValueType::Uint64 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::I64(v) => {
                w.write_all(&(GgufValueType::Int64 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::F64(v) => {
                w.write_all(&(GgufValueType::Float64 as u32).to_le_bytes())?;
                w.write_all(&v.to_le_bytes())?;
            }
            GgufValue::Array(items) => {
                w.write_all(&(GgufValueType::Array as u32).to_le_bytes())?;
                if items.is_empty() {
                    w.write_all(&0u32.to_le_bytes())?; // array type
                    w.write_all(&0u64.to_le_bytes())?; // count
                } else {
                    // Determine array type from first element
                    let arr_type = Self::value_type(&items[0]);
                    w.write_all(&(arr_type as u32).to_le_bytes())?;
                    w.write_all(&(items.len() as u64).to_le_bytes())?;
                    for item in items {
                        // Write value without type prefix (array items don't repeat type)
                        self.write_array_item(w, item)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_array_item<W: Write>(&self, w: &mut W, value: &GgufValue) -> io::Result<()> {
        match value {
            GgufValue::U8(v) => w.write_all(&[*v]),
            GgufValue::I8(v) => w.write_all(&[*v as u8]),
            GgufValue::U16(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::I16(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::U32(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::I32(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::F32(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::Bool(v) => w.write_all(&[*v as u8]),
            GgufValue::String(v) => {
                w.write_all(&(v.len() as u64).to_le_bytes())?;
                w.write_all(v.as_bytes())
            }
            GgufValue::U64(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::I64(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::F64(v) => w.write_all(&v.to_le_bytes()),
            GgufValue::Array(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "nested arrays not supported",
            )),
        }
    }

    fn value_type(value: &GgufValue) -> GgufValueType {
        match value {
            GgufValue::U8(_) => GgufValueType::Uint8,
            GgufValue::I8(_) => GgufValueType::Int8,
            GgufValue::U16(_) => GgufValueType::Uint16,
            GgufValue::I16(_) => GgufValueType::Int16,
            GgufValue::U32(_) => GgufValueType::Uint32,
            GgufValue::I32(_) => GgufValueType::Int32,
            GgufValue::F32(_) => GgufValueType::Float32,
            GgufValue::Bool(_) => GgufValueType::Bool,
            GgufValue::String(_) => GgufValueType::String,
            GgufValue::U64(_) => GgufValueType::Uint64,
            GgufValue::I64(_) => GgufValueType::Int64,
            GgufValue::F64(_) => GgufValueType::Float64,
            GgufValue::Array(_) => GgufValueType::Array,
        }
    }

    fn header_size(&self) -> u64 {
        4 + 4 + 8 + 8 // magic + version + n_tensors + n_kv
    }

    fn metadata_size(&self) -> u64 {
        let mut size = 0u64;
        for (key, value) in &self.metadata {
            size += 8 + key.len() as u64; // key_len + key
            size += self.value_size(value);
        }
        size
    }

    fn value_size(&self, value: &GgufValue) -> u64 {
        match value {
            GgufValue::U8(_) | GgufValue::I8(_) | GgufValue::Bool(_) => 4 + 1,
            GgufValue::U16(_) | GgufValue::I16(_) => 4 + 2,
            GgufValue::U32(_) | GgufValue::I32(_) | GgufValue::F32(_) => 4 + 4,
            GgufValue::U64(_) | GgufValue::I64(_) | GgufValue::F64(_) => 4 + 8,
            GgufValue::String(v) => 4 + 8 + v.len() as u64,
            GgufValue::Array(items) => {
                if items.is_empty() {
                    4 + 4 + 8 // type + empty type + count
                } else {
                    let mut size = 4 + 4 + 8; // type + arr_type + count
                    for item in items {
                        size += self.array_item_size(item);
                    }
                    size
                }
            }
        }
    }

    fn array_item_size(&self, value: &GgufValue) -> u64 {
        match value {
            GgufValue::U8(_) | GgufValue::I8(_) | GgufValue::Bool(_) => 1,
            GgufValue::U16(_) | GgufValue::I16(_) => 2,
            GgufValue::U32(_) | GgufValue::I32(_) | GgufValue::F32(_) => 4,
            GgufValue::U64(_) | GgufValue::I64(_) | GgufValue::F64(_) => 8,
            GgufValue::String(v) => 8 + v.len() as u64,
            GgufValue::Array(_) => 0,
        }
    }

    fn tensor_info_size(&self) -> u64 {
        let mut size = 0u64;
        for info in &self.tensor_infos {
            size += 8 + info.name.len() as u64; // name_len + name
            size += 4; // n_dims
            size += info.dims.len() as u64 * 8; // dims
            size += 4; // dtype
            size += 8; // offset
        }
        size
    }

    fn data_offset(&self) -> u64 {
        let total = self.header_size() + self.metadata_size() + self.tensor_info_size();
        // Align to 32 bytes
        (total + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
    }

    /// Return the total size of the GGUF file in bytes.
    pub fn total_size(&self) -> u64 {
        self.data_offset() + self.data_buf.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_write_and_read_header() {
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            GgufValue::String("llama".to_string()),
        );
        writer.add_metadata("general.name", GgufValue::String("NanoMind".to_string()));
        writer.add_metadata("llama.context_length", GgufValue::U32(2048));
        writer.add_metadata("llama.embedding_length", GgufValue::U32(256));
        writer.add_metadata("llama.block_count", GgufValue::U32(4));

        // Add a small tensor
        let data = vec![0u8; 64];
        writer.add_tensor("token_embd.weight", vec![256, 128], GgufDType::F32, &data);

        let path = Path::new("test_nanomind.gguf");
        writer.write_to_file(path).unwrap();

        // Read back and verify magic
        let mut file = File::open(path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"GGUF");

        let mut version = [0u8; 4];
        file.read_exact(&mut version).unwrap();
        assert_eq!(u32::from_le_bytes(version), 3);
    }

    #[test]
    fn test_quantization_types() {
        assert_eq!(GgufDType::Q4_K as u32, 12);
        assert_eq!(GgufDType::F32 as u32, 0);
        assert_eq!(GgufDType::F16 as u32, 1);
        assert_eq!(GgufDType::Q8_0 as u32, 8);
    }

    #[test]
    fn test_metadata_types() {
        let mut writer = GgufWriter::new();
        writer.add_metadata("test.u8", GgufValue::U8(42));
        writer.add_metadata("test.u32", GgufValue::U32(12345));
        writer.add_metadata("test.f32", GgufValue::F32(3.14));
        writer.add_metadata("test.bool", GgufValue::Bool(true));
        writer.add_metadata("test.string", GgufValue::String("hello".to_string()));

        let path = Path::new("test_nanomind_types.gguf");
        writer.write_to_file(path).unwrap();

        let mut file = File::open(path).unwrap();
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"GGUF");
    }
}
