//! INT4 block quantization (Q4_K_M style, like llama.cpp).
//!
//! Block size = 32 elements. Each block stores:
//! - 1 scale factor (f16)
//! - 1 min value (f16) — for super-block compression
//! - 32 INT4 values packed into 16 bytes (2 per byte)
//!
//! During matmul, we dequantize on-the-fly — never store full f32 weights.

use alloc::vec::Vec;
use half::f16;

/// Number of elements per quantization block.
pub const QK4: usize = 32;
/// Number of packed bytes per block (32 INT4 values → 16 bytes).
pub const Q4_BLOCK_BYTES: usize = 16;

/// A single Q4 quantization block (32 elements).
///
/// Memory layout per block:
/// - `scale`: f16 scale factor
/// - `min`: f16 minimum offset (for asymmetric quantization)
/// - `quants`: 16 bytes holding 32 packed INT4 values
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q4Block {
    pub scale: f16,
    pub min: f16,
    pub quants: [u8; Q4_BLOCK_BYTES],
}

impl Q4Block {
    /// Create a new zero-initialized block.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            scale: f16::ZERO,
            min: f16::ZERO,
            quants: [0u8; Q4_BLOCK_BYTES],
        }
    }

    /// Dequantize this block into `out` (must have length >= 32).
    /// Each value = scale * (quants[i] as i8 - 8).
    #[inline]
    pub fn dequantize(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= QK4);
        let s = self.scale.to_f32();
        let m = self.min.to_f32();
        for i in 0..Q4_BLOCK_BYTES {
            let lo = self.quants[i] & 0x0F;
            let hi = (self.quants[i] >> 4) & 0x0F;
            // INT4 values are signed, offset by 8: val = raw - 8
            // But with min scale: val = scale * raw - min
            out[i * 2] = s * lo as f32 - m;
            out[i * 2 + 1] = s * hi as f32 - m;
        }
    }

    /// Fused dequantize + dot product with a row from `vec`.
    /// Computes dot(block_dequant, vec[row_start..row_start+32]).
    /// This avoids materializing the full dequantized row.
    #[inline]
    pub fn dot_with(&self, vec: &[f32], offset: usize) -> f32 {
        let s = self.scale.to_f32();
        let m = self.min.to_f32();
        let mut sum = 0.0f32;
        for i in 0..Q4_BLOCK_BYTES {
            let lo = (self.quants[i] & 0x0F) as f32;
            let hi = ((self.quants[i] >> 4) & 0x0F) as f32;
            sum += (s * lo - m) * vec[offset + i * 2];
            sum += (s * hi - m) * vec[offset + i * 2 + 1];
        }
        sum
    }
}

/// A quantized tensor stored as Q4 blocks.
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    pub shape: Vec<usize>,
    pub blocks: Vec<Q4Block>,
}

impl QuantizedTensor {
    /// Create a new quantized tensor from pre-quantized blocks.
    ///
    /// `total_elements` must be divisible by 32 (QK4).
    pub fn new(shape: Vec<usize>, blocks: Vec<Q4Block>) -> Self {
        Self { shape, blocks }
    }

    /// Total number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of Q4 blocks needed.
    pub fn num_blocks(&self) -> usize {
        self.num_elements().div_ceil(QK4)
    }

    /// Dequantize a single row (flat index range) into `out`.
    /// `row` is the starting element index, `out.len()` determines how many elements.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) {
        let n = out.len();
        let mut out_idx = 0;
        let mut elem_idx = row;

        while out_idx < n {
            let block_idx = elem_idx / QK4;
            let inner_offset = elem_idx % QK4;
            let block = &self.blocks[block_idx];

            // How many elements we can copy from this block
            let remaining_in_block = QK4 - inner_offset;
            let take = remaining_in_block.min(n - out_idx);

            let mut temp = [0.0f32; QK4];
            block.dequantize(&mut temp);

            out[out_idx..out_idx + take].copy_from_slice(&temp[inner_offset..inner_offset + take]);

            out_idx += take;
            elem_idx += take;
        }
    }

    /// Fused matmul: compute one output element = dot(dequant_row, vec).
    /// `row_start` is the starting element index in this tensor.
    /// `vec` must have length >= `n`.
    pub fn matmul_row(&self, row_start: usize, vec: &[f32]) -> f32 {
        let n = vec.len();
        let mut sum = 0.0f32;
        let mut elem_idx = row_start;
        let mut vec_offset = 0;

        while vec_offset < n {
            let block_idx = elem_idx / QK4;
            let inner_offset = elem_idx % QK4;
            let block = &self.blocks[block_idx];

            let remaining_in_block = QK4 - inner_offset;
            let take = remaining_in_block.min(n - vec_offset);

            // Dequantize only the portion we need
            if take == QK4 && inner_offset == 0 {
                // Full block, use fast dot
                sum += block.dot_with(vec, vec_offset);
            } else {
                // Partial block, dequantize into temp buffer
                let mut temp = [0.0f32; QK4];
                block.dequantize(&mut temp);
                for i in 0..take {
                    sum += temp[inner_offset + i] * vec[vec_offset + i];
                }
            }

            vec_offset += take;
            elem_idx += take;
        }

        sum
    }

    /// Estimate RAM usage for this tensor (in bytes).
    /// Each Q4 block = 2 (f16) + 2 (f16) + 16 (quants) = 20 bytes for 32 elements.
    pub fn ram_bytes(&self) -> usize {
        self.blocks.len() * 20
    }
}

/// Quantize an f32 slice into Q4 blocks (asymmetric quantization).
///
/// `data` must have length divisible by 32.
pub fn quantize_q4(data: &[f32]) -> Vec<Q4Block> {
    assert!(data.len().is_multiple_of(QK4), "data length must be divisible by {}", QK4);
    let n_blocks = data.len() / QK4;
    let mut blocks = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * QK4;
        let block_data = &data[start..start + QK4];

        // Find min and max in this block
        let mut min_val = block_data[0];
        let mut max_val = block_data[0];
        for &v in block_data.iter() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }

        // Asymmetric quantization: scale * quant - min
        // quant = round((val + min) / scale), but we use a simpler approach
        // scale = (max - min) / 15 (since INT4 range is 0..=15)
        let range = max_val - min_val;
        let scale = if range > 1e-8 { range / 15.0 } else { 1e-8 };

        let mut block = Q4Block::zero();
        block.scale = f16::from_f32(scale);
        block.min = f16::from_f32(min_val);

        for (i, &v) in block_data.iter().enumerate() {
            // quant = round((val + min) / scale) clamped to [0, 15]
            let qf = (v + min_val) / scale;
            let q = qf.round().clamp(0.0, 15.0) as u8;

            let byte_idx = i / 2;
            if i % 2 == 0 {
                block.quants[byte_idx] = q;
            } else {
                block.quants[byte_idx] |= q << 4;
            }
        }

        blocks.push(block);
    }

    blocks
}

/// Q8 quantization (fallback — 1 byte per element with per-block scale).
pub const QK8: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Q8Block {
    pub scale: f16,
    pub quants: [i8; QK8], // 32 bytes for 32 values
}

impl Q8Block {
    #[inline]
    pub const fn zero() -> Self {
        Self {
            scale: f16::ZERO,
            quants: [0i8; QK8],
        }
    }

    #[inline]
    pub fn dequantize(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= QK8);
        let s = self.scale.to_f32();
        for (i, &q) in self.quants.iter().enumerate() {
            out[i] = s * q as f32;
        }
    }

    #[inline]
    pub fn dot_with(&self, vec: &[f32], offset: usize) -> f32 {
        let s = self.scale.to_f32();
        let mut sum = 0.0f32;
        for i in 0..QK8 {
            sum += self.quants[i] as f32 * vec[offset + i];
        }
        sum * s
    }
}

/// Quantize f32 slice into Q8 blocks.
pub fn quantize_q8(data: &[f32]) -> Vec<Q8Block> {
    assert!(data.len().is_multiple_of(QK8), "data length must be divisible by {}", QK8);
    let n_blocks = data.len() / QK8;
    let mut blocks = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * QK8;
        let block_data = &data[start..start + QK8];

        let mut max_abs = 0.0f32;
        for &v in block_data.iter() {
            let a = v.abs();
            if a > max_abs { max_abs = a; }
        }

        let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1e-8 };
        let inv_scale = 1.0 / scale;

        let mut block = Q8Block::zero();
        block.scale = f16::from_f32(scale);

        for (i, &v) in block_data.iter().enumerate() {
            block.quants[i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }

        blocks.push(block);
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_q4_roundtrip() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let blocks = quantize_q4(&data);
        assert_eq!(blocks.len(), 1);

        let mut out = [0.0f32; 32];
        blocks[0].dequantize(&mut out);

        // Check approximate reconstruction
        for i in 0..32 {
            let expected = i as f32 * 0.1;
            let error = (out[i] - expected).abs();
            assert!(error < 0.15, "Element {}: expected {}, got {}, error {}", i, expected, out[i], error);
        }
    }

    #[test]
    fn test_q8_roundtrip() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let blocks = quantize_q8(&data);
        assert_eq!(blocks.len(), 1);

        let mut out = [0.0f32; 32];
        blocks[0].dequantize(&mut out);

        for i in 0..32 {
            let expected = (i as f32 - 16.0) * 0.5;
            let error = (out[i] - expected).abs();
            assert!(error < 0.05, "Element {}: expected {}, got {}", i, expected, out[i]);
        }
    }

    #[test]
    fn test_q4_dot() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let blocks = quantize_q4(&data);
        let qt = QuantizedTensor::new(vec![2, 32], blocks);

        let vec: Vec<f32> = (0..64).map(|_| 1.0).collect();
        let result = qt.matmul_row(0, &vec);
        assert!(result.is_finite(), "Dot product should be finite");
    }
}
