//! GGUF file reader — memory-mapped access to model files.

use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

use memmap2::Mmap;

use crate::metadata::GgufMetadata;
use crate::tensor::TensorInfo;

/// Memory-mapped GGUF file reader.
pub struct GgufReader {
    _file: File,
    mmap: Mmap,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
    version: u32,
    tensor_count: u64,
    /// Byte offset where tensor data begins.
    pub data_offset: u64,
}

impl GgufReader {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(file, mmap)
    }

    fn from_mmap(file: File, mmap: Mmap) -> io::Result<Self> {
        let mut cursor = Cursor::new(&mmap);

        // Magic: "GGUF"
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Not a GGUF file (magic: {:?})", magic),
            ));
        }

        // Version
        let version = cursor.read_u32_le()?;
        if version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {} (only v3 supported)", version),
            ));
        }

        // Counts
        let tensor_count = cursor.read_u64_le()?;
        let kv_count = cursor.read_u64_le()?;

        // Parse metadata
        let metadata = GgufMetadata::parse(&mut cursor, kv_count, version)?;

        // Parse tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let info = TensorInfo::parse(&mut cursor, version)?;
            tensors.push(info);
        }

        // Align to 32 bytes
        let alignment = 32;
        let offset = cursor.position();
        let padded = (offset + alignment - 1) / alignment * alignment;
        cursor.set_position(padded);
        let data_start = padded;

        Ok(Self {
            _file: file,
            mmap,
            metadata,
            tensors,
            version,
            tensor_count,
            data_offset: data_start,
        })
    }

    /// Get raw bytes for a tensor by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.iter().find(|t| t.name == name)?;

        // Calculate offset: data_offset + tensor.offset
        let start = self.data_offset as usize + tensor.offset as usize;
        let end = start + tensor.n_bytes();

        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Print summary of the model.
    pub fn summary(&self) -> String {
        let arch = self
            .metadata
            .get_string("general.architecture")
            .unwrap_or("unknown");
        let params = self.metadata.param_count();
        let quant = self.metadata.quantization_type();

        format!(
            "GGUF Model\n\
             ─────────\n\
             Architecture: {}\n\
             Parameters:   {} ({:.1}B)\n\
             Quantization: {}\n\
             Tensors:      {}\n\
             File size:    {} MB",
            arch,
            params,
            params as f64 / 1e9,
            quant,
            self.tensors.len(),
            self.mmap.len() / (1024 * 1024),
        )
    }
}

/// Extension trait for reading little-endian values.
pub trait ReadLeExt {
    fn read_u8(&mut self) -> io::Result<u8>;
    fn read_u16_le(&mut self) -> io::Result<u16>;
    fn read_i16_le(&mut self) -> io::Result<i16>;
    fn read_u32_le(&mut self) -> io::Result<u32>;
    fn read_i32_le(&mut self) -> io::Result<i32>;
    fn read_u64_le(&mut self) -> io::Result<u64>;
    fn read_i64_le(&mut self) -> io::Result<i64>;
    fn read_f32_le(&mut self) -> io::Result<f32>;
    fn read_f64_le(&mut self) -> io::Result<f64>;
    fn read_string(&mut self) -> io::Result<String>;
    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>>;
}

impl<R: Read> ReadLeExt for R {
    fn read_u8(&mut self) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16_le(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16_le(&mut self) -> io::Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32_le(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32_le(&mut self) -> io::Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64_le(&mut self) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64_le(&mut self) -> io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32_le(&mut self) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64_le(&mut self) -> io::Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64_le()?;
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }
}
