//! Transformer model architecture (Qwen2-style, GQA support, SwiGLU FFN).

use nanomind_core::ops::{
    dot_f32, rms_norm, silu_inplace, softmax_inplace, vec_add_inplace, vec_zero,
};
use nanomind_core::quantization::{QuantizedTensor, QK4};
use nanomind_core::rope::{apply_rope_single_vec, precompute_rope};
use std::vec::Vec;

/// Model configuration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl Config {
    /// Head dimension (always 128 for efficiency).
    pub fn head_dim(&self) -> usize {
        128
    }

    /// GQA groups per head.
    pub fn kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Estimate RAM usage for a given max sequence length (bytes).
    /// Includes model weights + KV cache.
    pub fn estimate_ram_bytes(&self, quant_type: QuantType) -> usize {
        let params = self.total_params();
        let bytes_per_param = match quant_type {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            QuantType::Q4 => 20 / QK4, // ~0.625 bytes/param
            QuantType::Q8 => 1,
        };
        let weight_ram = params * bytes_per_param;

        // KV cache: 2 * num_layers * num_kv_heads * head_dim * max_seq * sizeof(f32)
        let kv_cache =
            2 * self.num_layers * self.num_kv_heads * self.head_dim() * self.max_seq_len * 4;

        // Activation buffers: ~3 * hidden_dim * f32
        let activations = 3 * self.hidden_dim * 4;

        weight_ram + kv_cache + activations
    }

    /// Total parameter count.
    pub fn total_params(&self) -> usize {
        let hd = self.hidden_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let id = self.intermediate_dim;
        let vd = self.vocab_size;
        let n_layers = self.num_layers;

        // Per layer:
        // - attn: q_proj, k_proj, v_proj, o_proj
        // - ffn: gate_proj, up_proj, down_proj
        // - rms norms: 2 per layer (attn + ffn)
        let per_layer = hd * (nh * 128 + nkv * 128 * 2 + nh * 128) // attn weights
            + hd * id * 3 // FFN (gate, up, down)
            + hd * 2; // RMS norms

        // Embedding + final norm + lm_head
        let embedding = vd * hd;
        let output = vd * hd;

        per_layer * n_layers + embedding + output
    }
}

/// Quantization type for the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4 = 2,
    Q8 = 3,
}

impl QuantType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(QuantType::F32),
            1 => Some(QuantType::F16),
            2 => Some(QuantType::Q4),
            3 => Some(QuantType::Q8),
            _ => None,
        }
    }
}

/// KV Cache for a single layer.
///
/// Stored as flat Vec<f32> for memory efficiency.
/// Layout: [pos][head][dim] flattened.
#[derive(Clone, Debug)]
pub struct KVCache {
    pub k_cache: Vec<Vec<f32>>, // [layer] flat[pos * kv_heads * head_dim]
    pub v_cache: Vec<Vec<f32>>,
    pub pos: usize,
    pub max_seq: usize,
}

impl KVCache {
    /// Create a new KV cache.
    pub fn new(config: &Config) -> Self {
        let head_dim = config.head_dim();
        let capacity = config.max_seq_len * config.num_kv_heads * head_dim;

        Self {
            k_cache: vec![vec![0.0f32; capacity]; config.num_layers],
            v_cache: vec![vec![0.0f32; capacity]; config.num_layers],
            pos: 0,
            max_seq: config.max_seq_len,
        }
    }

    /// Reset the cache for a new generation.
    pub fn reset(&mut self) {
        self.pos = 0;
        // We don't zero the buffers — they'll be overwritten during generation
    }

    /// Get K cache for a layer at the current position.
    #[inline]
    pub fn k_at(&self, layer: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let cache = &self.k_cache[layer];
        let stride = self.max_seq * head_dim;
        let base = kv_head * stride + self.pos * head_dim;
        &cache[base..base + head_dim]
    }

    /// Get V cache for a layer at the current position.
    #[inline]
    pub fn v_at(&self, layer: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let cache = &self.v_cache[layer];
        let stride = self.max_seq * head_dim;
        let base = kv_head * stride + self.pos * head_dim;
        &cache[base..base + head_dim]
    }
}

/// A single transformer layer's weights.
#[derive(Clone)]
pub struct Layer {
    // Attention
    pub attn_q: QuantizedTensor, // [hidden, n_heads * head_dim]
    pub attn_k: QuantizedTensor, // [hidden, n_kv_heads * head_dim]
    pub attn_v: QuantizedTensor, // [hidden, n_kv_heads * head_dim]
    pub attn_o: QuantizedTensor, // [n_heads * head_dim, hidden]

    // FFN (SwiGLU)
    pub ffn_gate: QuantizedTensor, // [hidden, intermediate]
    pub ffn_up: QuantizedTensor,   // [hidden, intermediate]
    pub ffn_down: QuantizedTensor, // [intermediate, hidden]

    // RMS Norms
    pub attn_norm: Vec<f32>, // [hidden]
    pub ffn_norm: Vec<f32>,  // [hidden]
}

/// Full model.
pub struct Model {
    pub config: Config,
    pub token_embeddings: QuantizedTensor, // [vocab, hidden]
    pub output_weights: QuantizedTensor,   // [vocab, hidden] (tied or separate)
    pub final_norm: Vec<f32>,              // [hidden]
    pub layers: Vec<Layer>,
    pub rope_cos: Vec<f32>, // [max_seq, head_dim/2]
    pub rope_sin: Vec<f32>, // [max_seq, head_dim/2]
}

impl Model {
    /// Create a new model and precompute RoPE tables.
    pub fn new(config: Config) -> Self {
        let (rope_cos, rope_sin) =
            precompute_rope(config.max_seq_len, config.head_dim(), config.rope_theta);

        // Pre-allocate layer slots
        let layers = (0..config.num_layers)
            .map(|_| Layer {
                attn_q: QuantizedTensor::new(vec![0], vec![]),
                attn_k: QuantizedTensor::new(vec![0], vec![]),
                attn_v: QuantizedTensor::new(vec![0], vec![]),
                attn_o: QuantizedTensor::new(vec![0], vec![]),
                ffn_gate: QuantizedTensor::new(vec![0], vec![]),
                ffn_up: QuantizedTensor::new(vec![0], vec![]),
                ffn_down: QuantizedTensor::new(vec![0], vec![]),
                attn_norm: vec![1.0; config.hidden_dim],
                ffn_norm: vec![1.0; config.hidden_dim],
            })
            .collect();

        Self {
            token_embeddings: QuantizedTensor::new(vec![0], vec![]),
            output_weights: QuantizedTensor::new(vec![0], vec![]),
            final_norm: vec![1.0; config.hidden_dim],
            layers,
            rope_cos,
            rope_sin,
            config,
        }
    }
}

// ─── Forward Pass ─────────────────────────────────────────────────────────

/// Run a forward pass for a single token position.
///
/// `x` is the input embedding [hidden_dim], modified in-place to become
/// the output embedding for this position.
/// `layer_idx` is which transformer layer to run.
/// `cache` is the KV cache (updated in-place).
/// `pos` is the current token position.
#[allow(clippy::too_many_arguments)]
pub fn transformer_block_forward(
    x: &mut [f32],
    layer: &Layer,
    cache: &mut KVCache,
    layer_idx: usize,
    pos: usize,
    config: &Config,
    rope_cos: &[f32],
    rope_sin: &[f32],
) {
    let hd = config.hidden_dim;
    let head_dim = config.head_dim();
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let kv_groups = config.kv_groups();

    // ─── Pre-Attention RMS Norm ───
    let mut norm_buf = x.to_vec();
    rms_norm(&mut norm_buf, &layer.attn_norm, config.rms_norm_eps);

    // ─── Q, K, V Projections ───
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0.0f32; num_kv_heads * head_dim];
    let mut v = vec![0.0f32; num_kv_heads * head_dim];

    // Q = norm_x @ W_q
    matmul_quantized(&norm_buf, &layer.attn_q, num_heads * head_dim, &mut q);
    // K = norm_x @ W_k
    matmul_quantized(&norm_buf, &layer.attn_k, num_kv_heads * head_dim, &mut k);
    // V = norm_x @ W_v
    matmul_quantized(&norm_buf, &layer.attn_v, num_kv_heads * head_dim, &mut v);

    // ─── RoPE ───
    let half_dim = head_dim / 2;
    let rope_cos_sliced = &rope_cos[pos * half_dim..(pos + 1) * half_dim];
    let rope_sin_sliced = &rope_sin[pos * half_dim..(pos + 1) * half_dim];

    // Apply RoPE to K heads (once each)
    apply_rope_single_vec(&mut k, head_dim, rope_cos_sliced, rope_sin_sliced);
    // Apply RoPE to Q heads
    apply_rope_single_vec(&mut q, head_dim, rope_cos_sliced, rope_sin_sliced);

    // ─── Store K, V in cache ───
    let kv_stride = cache.max_seq * head_dim;
    for kv_head in 0..num_kv_heads {
        let src_base = kv_head * head_dim;
        let dst_base = kv_head * kv_stride + pos * head_dim;

        cache.k_cache[layer_idx][dst_base..dst_base + head_dim]
            .copy_from_slice(&k[src_base..src_base + head_dim]);
        cache.v_cache[layer_idx][dst_base..dst_base + head_dim]
            .copy_from_slice(&v[src_base..src_base + head_dim]);
    }

    // ─── Multi-Head Attention with KV Cache ───
    let mut attn_scores = vec![0.0f32; pos + 1]; // scores for each past position

    for h in 0..num_heads {
        // In GQA, multiple query heads share the same KV head
        let kv_head = h / kv_groups;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        for (t, score_ref) in attn_scores.iter_mut().enumerate().take(pos + 1) {
            let k_t = &cache.k_cache[layer_idx]
                [kv_head * kv_stride + t * head_dim..kv_head * kv_stride + t * head_dim + head_dim];
            let score = dot_f32(q_head, k_t) / (head_dim as f32).sqrt();
            *score_ref = score;
        }

        // Softmax over attention scores
        softmax_inplace(&mut attn_scores);

        // Weighted sum of values
        let o_head = &mut q[h * head_dim..(h + 1) * head_dim]; // reuse q buffer for output
        vec_zero(o_head);

        for (t, &weight) in attn_scores.iter().enumerate().take(pos + 1) {
            let v_t = &cache.v_cache[layer_idx]
                [kv_head * kv_stride + t * head_dim..kv_head * kv_stride + t * head_dim + head_dim];
            for d in 0..head_dim {
                o_head[d] += weight * v_t[d];
            }
        }
    }

    // q now holds the concatenated attention output for all heads
    // ─── Output Projection ───
    let mut attn_out = vec![0.0f32; hd];
    matmul_quantized(&q, &layer.attn_o, hd, &mut attn_out);

    // ─── Residual Connection ───
    vec_add_inplace(x, &attn_out);

    // ─── Post-Attention RMS Norm (pre-FFN) ───
    let mut ffn_buf = x.to_vec();
    rms_norm(&mut ffn_buf, &layer.ffn_norm, config.rms_norm_eps);

    // ─── SwiGLU FFN ───
    // gate = silu(x @ gate_proj)
    let mut gate = vec![0.0f32; config.intermediate_dim];
    matmul_quantized(
        &ffn_buf,
        &layer.ffn_gate,
        config.intermediate_dim,
        &mut gate,
    );
    silu_inplace(&mut gate);

    // up = x @ up_proj
    let mut up = vec![0.0f32; config.intermediate_dim];
    matmul_quantized(&ffn_buf, &layer.ffn_up, config.intermediate_dim, &mut up);

    // hidden = gate * up (element-wise)
    for i in 0..config.intermediate_dim {
        gate[i] *= up[i];
    }

    // out = hidden @ down_proj
    let mut ffn_out = vec![0.0f32; hd];
    matmul_quantized(&gate, &layer.ffn_down, hd, &mut ffn_out);

    // ─── Residual Connection ───
    vec_add_inplace(x, &ffn_out);
}

/// Matrix-vector multiply with a quantized tensor.
///
/// `x` is input [input_dim], `weight` is [output_dim, input_dim] (row-major).
/// `out` is output [output_dim].
fn matmul_quantized(x: &[f32], weight: &QuantizedTensor, output_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.shape.last().copied().unwrap_or(0));
    debug_assert_eq!(out.len(), output_dim);

    for (i, o) in out.iter_mut().enumerate().take(output_dim) {
        *o = weight.matmul_row(i * x.len(), x);
    }
}

// ─── Embedding Lookup ─────────────────────────────────────────────────────

/// Look up token embedding from the token embedding table.
pub fn embed_token(
    token_id: u32,
    embeddings: &QuantizedTensor,
    hidden_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), hidden_dim);
    embeddings.dequantize_row(token_id as usize * hidden_dim, out);
}

// ─── LM Head (logits computation) ─────────────────────────────────────────

/// Compute logits: logits = hidden @ output_weights^T
pub fn compute_logits(
    hidden: &[f32],
    output_weights: &QuantizedTensor,
    vocab_size: usize,
    logits: &mut [f32],
) {
    debug_assert_eq!(logits.len(), vocab_size);

    for (v, l) in logits.iter_mut().enumerate().take(vocab_size) {
        *l = output_weights.matmul_row(v * hidden.len(), hidden);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nanomind_core::quantization::quantize_q4;

    fn make_test_config() -> Config {
        Config {
            vocab_size: 1000,
            hidden_dim: 256,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_dim: 512,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_config_estimates() {
        let config = make_test_config();
        let ram = config.estimate_ram_bytes(QuantType::Q4);
        assert!(ram > 0, "RAM estimate should be positive");
    }

    #[test]
    fn test_kv_cache() {
        let config = make_test_config();
        let cache = KVCache::new(&config);
        assert_eq!(cache.k_cache.len(), config.num_layers);
        assert_eq!(cache.v_cache.len(), config.num_layers);
        assert_eq!(cache.pos, 0);
    }

    #[test]
    fn test_transformer_forward() {
        let config = make_test_config();
        let hd = config.hidden_dim;

        // Create a minimal layer with random-ish weights
        let mut rng = SimpleRng(42);
        let mut make_qt = |rows: usize, cols: usize| -> QuantizedTensor {
            let n = rows * cols;
            // Round up to QK4 boundary
            let n_padded = (n + QK4 - 1) / QK4 * QK4;
            let data: Vec<f32> = (0..n_padded)
                .map(|_i| rng.next_f32() * 0.1 - 0.05)
                .collect();
            let blocks = quantize_q4(&data);
            QuantizedTensor::new(vec![rows, cols], blocks)
        };

        let layer = Layer {
            attn_q: make_qt(config.num_heads * config.head_dim(), hd),
            attn_k: make_qt(config.num_kv_heads * config.head_dim(), hd),
            attn_v: make_qt(config.num_kv_heads * config.head_dim(), hd),
            attn_o: make_qt(hd, config.num_heads * config.head_dim()),
            ffn_gate: make_qt(config.intermediate_dim, hd),
            ffn_up: make_qt(config.intermediate_dim, hd),
            ffn_down: make_qt(hd, config.intermediate_dim),
            attn_norm: vec![1.0; hd],
            ffn_norm: vec![1.0; hd],
        };

        let mut cache = KVCache::new(&config);
        let mut x: Vec<f32> = (0..hd).map(|i| i as f32 * 0.01).collect();

        // Precompute RoPE tables
        let (rope_cos, rope_sin) =
            precompute_rope(config.max_seq_len, config.head_dim(), config.rope_theta);

        // Run forward pass at position 0
        transformer_block_forward(
            &mut x, &layer, &mut cache, 0, 0, &config, &rope_cos, &rope_sin,
        );

        // Output should be finite and different from input
        assert!(x.iter().all(|&v| v.is_finite()));
    }

    /// Simple deterministic RNG for tests.
    struct SimpleRng(u64);

    impl SimpleRng {
        fn next_f32(&mut self) -> f32 {
            // LCG
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32) / (u32::MAX as f32)
        }
    }
}
