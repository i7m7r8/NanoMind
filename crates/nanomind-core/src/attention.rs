//! Tensor operations: dot product, softmax, RMS norm, activations.
//!
//! All functions are allocation-free — operate on borrowed slices.
//! Optimized with loop unrolling for NEON/ARM performance.

/// Dot product: `a · b`.
/// Unrolled by 8 for throughput.
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum = 0.0f32;
    let chunks = n / 8;
    let rem = n % 8;

    for i in 0..chunks {
        let base = i * 8;
        sum += a[base] * b[base];
        sum += a[base + 1] * b[base + 1];
        sum += a[base + 2] * b[base + 2];
        sum += a[base + 3] * b[base + 3];
        sum += a[base + 4] * b[base + 4];
        sum += a[base + 5] * b[base + 5];
        sum += a[base + 6] * b[base + 6];
        sum += a[base + 7] * b[base + 7];
    }
    for i in 0..rem {
        sum += a[chunks * 8 + i] * b[chunks * 8 + i];
    }
    sum
}

/// Fused dot product of quantized block with f32 vector.
/// Dequantizes on-the-fly without materializing the full f32 array.
pub fn dot_q4_f32(q4_data: &[u8], q4_type: crate::GgmlType, vec: &[f32]) -> f32 {
    let mut tmp = vec![0.0f32; 256];
    let n = crate::dequantize_block(q4_data, q4_type, &mut tmp);
    dot_f32(&tmp[..n], &vec[..n])
}

/// In-place softmax with numerical stability.
#[inline(always)]
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Softmax with temperature.
#[inline(always)]
pub fn softmax_with_temp(x: &mut [f32], temperature: f32) {
    if temperature > 0.0 {
        let inv_temp = 1.0 / temperature;
        for v in x.iter_mut() {
            *v *= inv_temp;
        }
    }
    softmax_inplace(x);
}

/// Top-k masking: set all but top-k values to -inf.
pub fn softmax_top_k(x: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= x.len() {
        return;
    }
    // Find top-k threshold via partial sort
    let mut indexed: Vec<(f32, usize)> =
        x.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.select_nth_unstable_by(x.len() - top_k, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let threshold = indexed[x.len() - top_k].0;
    for v in x.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// RMS normalization in-place.
/// `x = x / sqrt(mean(x²) + eps) * weight`
#[inline(always)]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len() as f32;
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// RMS normalization with optional centering (no bias, just norm).
#[inline(always)]
pub fn rms_norm_no_weight(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_rms;
    }
}

/// SiLU (Swish) activation: `x * sigmoid(x)`.
#[inline(always)]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// GELU approximate: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))`.
#[inline(always)]
pub fn gelu_approx_inplace(x: &mut [f32]) {
    const C: f32 = 0.797_884_6;
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (C * (*v + 0.044_715 * x3)).tanh());
    }
}

/// ReLU in-place.
#[inline(always)]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Matrix-vector multiply: `out = W * x` where W is [rows × cols] row-major.
#[inline]
pub fn matmul_f32(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for (i, o) in out.iter_mut().enumerate().take(rows) {
        let row = &w[i * cols..(i + 1) * cols];
        *o = dot_f32(row, x);
    }
}

/// Add vectors: `a += b`.
#[inline(always)]
pub fn vec_add(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Scale vector: `x *= s`.
#[inline(always)]
pub fn vec_scale(x: &mut [f32], s: f32) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// Zero a vector.
#[inline(always)]
pub fn vec_zero(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.0;
    }
}

/// Copy a vector.
#[inline(always)]
pub fn vec_copy(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    dst.copy_from_slice(src);
}

/// Repetition penalty: penalize previously-generated tokens.
/// If logit > 0: logit /= penalty; else: logit *= penalty.
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &t in prev_tokens {
        let idx = t as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Presence penalty: penalize tokens that appeared at all.
pub fn apply_presence_penalty(logits: &mut [f32], counts: &[usize], penalty: f32) {
    if penalty == 0.0 {
        return;
    }
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 && i < logits.len() {
            logits[i] -= penalty;
        }
    }
}

/// Frequency penalty: penalize proportional to count.
pub fn apply_frequency_penalty(logits: &mut [f32], counts: &[usize], penalty: f32) {
    if penalty == 0.0 {
        return;
    }
    for (i, &c) in counts.iter().enumerate() {
        if i < logits.len() {
            logits[i] -= penalty * c as f32;
        }
    }
}

/// Logit bias: add a fixed value to specific token logits.
pub fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    for &(token, bias_val) in bias {
        let idx = token as usize;
        if idx < logits.len() {
            logits[idx] += bias_val;
        }
    }
}

/// Compute log probabilities from logits (after softmax).
pub fn log_probs(probs: &[f32]) -> Vec<f32> {
    probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .collect()
}

/// Compute perplexity from log probabilities.
pub fn perplexity(log_prob: f32, num_tokens: usize) -> f32 {
    (-log_prob / num_tokens as f32).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = dot_f32(&a, &b);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_sum_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        rms_norm(&mut x, &w, 1e-5);
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        assert!(((ss / n).sqrt() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        let mut x = [0.0];
        silu_inplace(&mut x);
        assert!(x[0].abs() < 1e-6);

        let mut x = [1.0];
        silu_inplace(&mut x);
        assert!((x[0] - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = [2.0, 1.0, 3.0, 0.5];
        apply_repetition_penalty(&mut logits, &[2], 1.2);
        assert!(logits[2] < 3.0); // should be reduced
        assert!((logits[2] - 3.0 / 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_logit_bias() {
        let mut logits = [0.0; 4];
        apply_logit_bias(&mut logits, &[(1, 5.0)]);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[0], 0.0);
    }
}
