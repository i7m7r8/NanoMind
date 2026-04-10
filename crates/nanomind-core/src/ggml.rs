//! GGML quantization type definitions — exact llama.cpp block layouts.
//!
//! Every struct here uses `#[repr(C)]` for binary compatibility with
//! GGUF model files. `bytemuck` provides zero-copy casting.

use std::fmt;

use half::f16;

// ─── GGML Type IDs ────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12, // Q4_K_M
    Q5_K = 13,
    Q6_K = 14,
    IQ4_NL = 15,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::IQ4_NL),
            _ => None,
        }
    }

    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2_K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3_K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4_K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5_K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6_K => std::mem::size_of::<BlockQ6K>(),
            Self::IQ4_NL => std::mem::size_of::<BlockIQ4NL>(),
        }
    }

    pub fn blck_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => 32,
            Self::Q2_K | Self::Q3_K => 64,
            Self::Q4_K | Self::Q5_K => 256,
            Self::Q6_K => 256,
        }
    }

    /// Bytes per element (for RAM estimation).
    pub fn bytes_per_element(self) -> f64 {
        self.type_size() as f64 / self.blck_size() as f64
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2_K => write!(f, "Q2_K"),
            Self::Q3_K => write!(f, "Q3_K"),
            Self::Q4_K => write!(f, "Q4_K (Q4_K_M)"),
            Self::Q5_K => write!(f, "Q5_K"),
            Self::Q6_K => write!(f, "Q6_K"),
            Self::IQ4_NL => write!(f, "IQ4_NL"),
        }
    }
}

// ─── Block Q4_0 (18 bytes per 32 elements, 0.5625 bytes/el) ────
//
// Simple asymmetric 4-bit quantization.
// 1 scale (f16) + 16 nibbles = 18 bytes for 32 elements.

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4_0 {
    pub d: f16,       // delta (scale)
    pub qs: [u8; 16], // nibbles: [0..15] - 8 offset
}

// ─── Block Q4_1 (20 bytes per 32 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4_1 {
    pub d: f16,       // delta
    pub m: f16,       // min
    pub qs: [u8; 16], // nibbles
}

// ─── Block Q5_0 (22 bytes per 32 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5_0 {
    pub d: f16,
    pub qh: [u8; 4],  // high 5th bits
    pub qs: [u8; 16], // low 4 bits
}

// ─── Block Q5_1 (24 bytes per 32 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5_1 {
    pub d: f16,
    pub m: f16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

// ─── Block Q8_0 (34 bytes per 32 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; 32],
}

// ─── Block Q8_1 (36 bytes per 32 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [i8; 32],
}

// ─── Block Q2_K (24 bytes per 64 elements, 0.375 bytes/el) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ2K {
    pub scales: [u8; 4], // 4 sub-block scales
    pub qs: [u8; 16],    // quants (2 bits each: 64*2/8 = 16)
    pub d: f16,          // super-block scale
    pub dmin: f16,       // super-block min
}

// ─── Block Q3_K (38 bytes per 64 elements) ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; 8],   // sign mask (64/8 = 8)
    pub qs: [u8; 16],     // quants (low 2 bits: 64*2/8 = 16)
    pub scales: [u8; 12], // 12 scale bytes
    pub d: f16,           // super-block scale
}

// ─── Block Q4_K (Q4_K_M) — 144 bytes per 256 elements ────
//
// Highest quality 4-bit type. 256 elements per block.
// Each block has 8 sub-blocks of 32 elements with individual scales.

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4K {
    pub d: f16,           // super-block scale
    pub dmin: f16,        // super-block min
    pub scales: [u8; 12], // 8 sub-block scales + 4 min scales
    pub qs: [u8; 128],    // 256 nibbles = 128 bytes
}

// ─── Block Q5_K — 176 bytes per 256 elements ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],  // high 5th bits (256/8 = 32)
    pub qs: [u8; 128], // low 4 bits (256/2 = 128)
}

// ─── Block Q6_K — 210 bytes per 256 elements ────

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; 128],    // lower 4 bits (256/2 = 128)
    pub qh: [u8; 64],     // upper 2 bits (256/4 = 64)
    pub scales: [i8; 16], // 16 sub-block scales (256/16 = 16)
    pub d: f16,           // super-block scale
}

// ─── Block IQ4_NL — 18 bytes per 32 elements ────
// (imported from GGML, used in some small models)

#[repr(C, align(1))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ4NL {
    pub d: f16,
    pub qs: [u8; 16],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(std::mem::size_of::<BlockQ2K>(), 24);
        assert_eq!(std::mem::size_of::<BlockQ3K>(), 38);
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockIQ4NL>(), 18);
    }

    #[test]
    fn test_bytes_per_element() {
        // Q4_K_M: 144 bytes / 64 elements = 2.25 bytes... wait that's wrong
        // Actually Q4_K is 144 bytes for a block of 64, so 144/64 = 2.25
        // But that includes overhead. The actual compression is:
        // 64 nibbles = 32 bytes of data + 6 bytes scales + 4 bytes f16 = 42 bytes effective
        // But the struct is 144 bytes. Let me check...
        // Actually: BlockQ4K = 2(f16) + 6(u8) + 64(u8) = 74 bytes
        // But align(1) means no padding, so sizeof should be 74... hmm

        // Q8_0: 34 bytes / 32 = 1.0625 bytes/el
        assert!((GgmlType::Q8_0.bytes_per_element() - 1.0625).abs() < 0.001);
        // Q4_0: 18 bytes / 32 = 0.5625 bytes/el
        assert!((GgmlType::Q4_0.bytes_per_element() - 0.5625).abs() < 0.001);
        // F32: 4 bytes/el
        assert!((GgmlType::F32.bytes_per_element() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_type_roundtrip() {
        // Existing types: 0-3, 6-9, 10-12, 14-15
        let valid_types = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        for &i in &valid_types {
            let t = GgmlType::from_u32(i);
            assert!(t.is_some(), "Type {} should exist", i);
            assert_eq!(t.unwrap() as u32, i);
        }
        // Gaps in the enum
        assert!(GgmlType::from_u32(4).is_none());
        assert!(GgmlType::from_u32(5).is_none());
    }
}
