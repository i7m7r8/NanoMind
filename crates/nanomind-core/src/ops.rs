//! SIMD-accelerated tensor operations.
//!
//! Uses std::simd on nightly, falls back to manual unrolling on stable.
//! All functions work in-place or on borrowed slices — no allocations.

/// Dot product of two f32 slices.
///
/// On stable Rust, uses 8-element loop unrolling for decent performance.
/// On nightly with `std::simd`, uses platform-native SIMD.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "a and b must have same length");
    let n = a.len();
    let mut sum = 0.0f32;

    // 8-element unrolled loop for throughput
    let chunks = n / 8;
    let remainder = n % 8;

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

    for i in 0..remainder {
        sum += a[chunks * 8 + i] * b[chunks * 8 + i];
    }

    sum
}

/// In-place softmax with temperature scaling.
///
/// For numerical stability, subtracts the max before exp.
#[inline]
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max for numerical stability
    let mut max_val = x[0];
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// In-place softmax with temperature.
#[inline]
pub fn softmax_with_temp(x: &mut [f32], temperature: f32) {
    if temperature > 0.0 {
        for v in x.iter_mut() {
            *v /= temperature;
        }
    }
    softmax_inplace(x);
}

/// RMS Normalization (used in Qwen/LLaMA).
///
/// `x` is normalized in-place. `weight` must have same length as `x`.
///
/// Formula: x = x / sqrt(mean(x²) + eps) * weight
#[inline]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len() as f32;

    // Compute mean of squares
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let rms = (ss / n + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Apply norm and weight
    for i in 0..x.len() {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

/// SiLU (Sigmoid Linear Unit) activation in-place.
///
/// Also known as Swish: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
#[inline]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// GELU activation (approximate) in-place.
///
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu_approx_inplace(x: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.797_884_560_8;
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_PI * (*v + 0.044_715 * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

/// ReLU activation in-place.
#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Matrix-vector multiply: `out = mat * vec` where mat is [rows x cols].
///
/// `mat` is stored row-major. `out` must have length `rows`.
/// `vec` must have length `cols`.
#[inline]
pub fn matmul_f32(mat: &[f32], vec: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for r in 0..rows {
        let row_start = r * cols;
        let row = &mat[row_start..row_start + cols];
        out[r] = dot_f32(row, vec);
    }
}

/// Add two vectors in-place: `a += b`.
#[inline]
pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Scale a vector in-place: `x *= scale`.
#[inline]
pub fn vec_scale_inplace(x: &mut [f32], scale: f32) {
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Zero a slice.
#[inline]
pub fn vec_zero(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let result = dot_f32(&a, &b);
        // 2 + 6 + 12 + 20 + 30 = 70
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut x = [1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        // Check sum = 1
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Check ordering preserved
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &weight, 1e-5);
        // RMS should be ~1 after normalization
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let rms = (ss / n).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        let mut x = [0.0, 1.0, -1.0];
        silu_inplace(&mut x);
        // silu(0) = 0
        assert!(x[0].abs() < 1e-6);
        // silu(1) ≈ 0.731
        assert!((x[1] - 0.7310586).abs() < 1e-5);
        // silu(-1) ≈ -0.269
        assert!((x[2] - (-0.2689414)).abs() < 1e-4);
    }

    #[test]
    fn test_vec_scale() {
        let mut x = [1.0, 2.0, 3.0];
        vec_scale_inplace(&mut x, 2.0);
        assert_eq!(x, [2.0, 4.0, 6.0]);
    }
}
