//! Rotary Position Embeddings (RoPE).
//!
//! RoPE encodes position information by rotating query/key vectors
//! in 2D planes. This is used in Qwen, LLaMA, and other modern transformers.

use alloc::vec;
use alloc::vec::Vec;

/// Apply rotary position embeddings to query and key vectors.
///
/// `q` and `k` are the query and key slices (each of length `head_dim`).
/// `pos` is the token position index.
/// `head_dim` is the dimension per attention head.
/// `theta` is the base frequency (typically 10000.0 for Qwen/LLaMA).
///
/// The rotation is applied in 2D planes: for each pair (q[i], q[i+1]),
/// we rotate by angle `pos * theta^(-2*i/head_dim)`.
#[inline]
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    debug_assert_eq!(q.len(), k.len());
    debug_assert_eq!(q.len() % head_dim, 0);
    debug_assert!(head_dim.is_multiple_of(2), "head_dim must be even for RoPE");

    let num_heads = q.len() / head_dim;

    for h in 0..num_heads {
        let base = h * head_dim;
        let q_slice = &mut q[base..base + head_dim];
        let k_slice = &mut k[base..base + head_dim];
        apply_rope_single(q_slice, k_slice, pos, theta);
    }
}

/// Apply RoPE to a single head's query and key.
#[inline]
fn apply_rope_single(q: &mut [f32], k: &mut [f32], pos: usize, theta: f32) {
    let half = q.len() / 2;

    for i in 0..half {
        // Frequency for this dimension
        let freq = 1.0 / theta.powf(2.0 * i as f32 / q.len() as f32);
        let angle = pos as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();

        // Rotate query
        let q0 = q[i];
        let q1 = q[i + half];
        q[i] = q0 * cos_val - q1 * sin_val;
        q[i + half] = q0 * sin_val + q1 * cos_val;

        // Rotate key
        let k0 = k[i];
        let k1 = k[i + half];
        k[i] = k0 * cos_val - k1 * sin_val;
        k[i + half] = k0 * sin_val + k1 * cos_val;
    }
}

/// Precompute RoPE frequency cos/sin tables for faster inference.
///
/// Returns `(cos_table, sin_table)` each of shape `[max_seq_len, head_dim/2]`.
/// The tables are stored as flat arrays: table[pos * half_dim + i].
pub fn precompute_rope(
    max_seq_len: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq_len * half];
    let mut sin_table = vec![0.0f32; max_seq_len * half];

    for pos in 0..max_seq_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            cos_table[pos * half + i] = c;
            sin_table[pos * half + i] = s;
        }
    }

    (cos_table, sin_table)
}

/// Apply RoPE to a single vector (Q or K independently) using precomputed tables.
#[inline]
pub fn apply_rope_single_vec(vec: &mut [f32], head_dim: usize, cos_table: &[f32], sin_table: &[f32]) {
    debug_assert!(head_dim.is_multiple_of(2));
    debug_assert_eq!(vec.len() % head_dim, 0);

    let half = head_dim / 2;
    let num_heads = vec.len() / head_dim;

    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let c = cos_table[i];
            let s = sin_table[i];

            let v0 = vec[base + i];
            let v1 = vec[base + i + half];
            vec[base + i] = v0 * c - v1 * s;
            vec[base + i + half] = v0 * s + v1 * c;
        }
    }
}

/// Apply RoPE using precomputed tables.
#[inline]
pub fn apply_rope_table(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    debug_assert_eq!(q.len(), k.len());
    debug_assert_eq!(q.len() % head_dim, 0);
    debug_assert!(head_dim.is_multiple_of(2));

    let half = head_dim / 2;
    let num_heads = q.len() / head_dim;

    for h in 0..num_heads {
        let base = h * head_dim;
        let q_slice = &mut q[base..base + head_dim];
        let k_slice = &mut k[base..base + head_dim];

        for i in 0..half {
            let c = cos_table[pos * half + i];
            let s = sin_table[pos * half + i];

            let q0 = q_slice[i];
            let q1 = q_slice[i + half];
            q_slice[i] = q0 * c - q1 * s;
            q_slice[i + half] = q0 * s + q1 * c;

            let k0 = k_slice[i];
            let k1 = k_slice[i + half];
            k_slice[i] = k0 * c - k1 * s;
            k_slice[i + half] = k0 * s + k1 * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_rotation() {
        let mut q = [1.0, 0.0, 0.0, 1.0];
        let mut k = [1.0, 0.0, 0.0, 1.0];
        let head_dim = 4;
        let theta = 10000.0;

        // At pos=0, no rotation should occur
        apply_rope(&mut q, &mut k, 0, head_dim, theta);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 0.0).abs() < 1e-6);

        // At pos=1, some rotation should occur
        let mut q2 = [1.0, 0.0, 0.0, 1.0];
        let mut k2 = [1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut q2, &mut k2, 1, head_dim, theta);

        // Values should have changed from pos=0
        assert!(q2 != q || k2 != k, "Rotation at pos=1 should differ from pos=0");

        // Magnitude should be preserved (rotation preserves norm)
        let orig_norm: f32 = q.iter().map(|x| x * x).sum();
        let new_norm: f32 = q2.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() < 1e-5, "RoPE should preserve vector norm");
    }

    #[test]
    fn test_precompute_rope() {
        let (cos, sin) = precompute_rope(10, 64, 10000.0);
        assert_eq!(cos.len(), 10 * 32);
        assert_eq!(sin.len(), 10 * 32);

        // At pos=0, cos should be 1.0, sin should be 0.0
        for i in 0..32 {
            assert!((cos[i] - 1.0).abs() < 1e-6);
            assert!(sin[i].abs() < 1e-6);
        }
    }
}
