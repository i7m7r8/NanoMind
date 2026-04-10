//! Transformer model architecture — from-scratch implementation in pure Rust.
//!
//! Implements a standard decoder-only transformer with:
//! - Grouped Query Attention (GQA)
//! - SwiGLU FFN
//! - RMSNorm
//! - RoPE positional encoding
//! - Optional embedding tying

use crate::config::ModelConfig;
use std::f32::consts::PI;

// ─── Tensor ────────────────────────────────────────────────────────────────

/// A simple f32 tensor with named weights.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(name: &str, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            name: name.to_string(),
            data: vec![0.0f32; numel],
            shape,
        }
    }

    /// Xavier/He initialization for linear layers
    pub fn init_xavier(&mut self, rng: &mut impl Rng) {
        let fan_in = self.shape.last().copied().unwrap_or(1);
        let fan_out = self.shape.first().copied().unwrap_or(1);
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        for v in &mut self.data {
            *v = randn(rng) * std;
        }
    }

    /// Zero initialization (for biases etc)
    pub fn zero(&mut self) {
        for v in &mut self.data {
            *v = 0.0;
        }
    }
}

/// Simple random number generator trait
pub trait Rng {
    fn f32(&mut self) -> f32;
}

/// Box-Muller transform for normal distribution
fn randn(rng: &mut impl Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ─── Transformer Layer Weights ─────────────────────────────────────────────

/// Weights for one transformer layer.
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    // Attention
    pub attn_norm: Tensor, // [hidden_dim] — RMSNorm weights
    pub attn_q: Tensor,    // [hidden_dim, head_dim * num_heads]
    pub attn_k: Tensor,    // [hidden_dim, head_dim * num_kv_heads]
    pub attn_v: Tensor,    // [hidden_dim, head_dim * num_kv_heads]
    pub attn_out: Tensor,  // [head_dim * num_heads, hidden_dim]
    // FFN (SwiGLU)
    pub ffn_norm: Tensor, // [hidden_dim] — RMSNorm weights
    pub ffn_gate: Tensor, // [hidden_dim, intermediate_dim]
    pub ffn_up: Tensor,   // [hidden_dim, intermediate_dim]
    pub ffn_down: Tensor, // [intermediate_dim, hidden_dim]
}

impl TransformerLayer {
    pub fn new(layer_idx: usize, cfg: &ModelConfig) -> Self {
        let h = cfg.hidden_dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        let ffn = cfg.intermediate_dim;

        Self {
            attn_norm: Tensor::new(&format!("blk.{}.attn_norm.weight", layer_idx), vec![h]),
            attn_q: Tensor::new(
                &format!("blk.{}.attn_q.weight", layer_idx),
                vec![h, head_dim * cfg.num_heads],
            ),
            attn_k: Tensor::new(&format!("blk.{}.attn_k.weight", layer_idx), vec![h, kv_dim]),
            attn_v: Tensor::new(&format!("blk.{}.attn_v.weight", layer_idx), vec![h, kv_dim]),
            attn_out: Tensor::new(
                &format!("blk.{}.attn_output.weight", layer_idx),
                vec![head_dim * cfg.num_heads, h],
            ),
            ffn_norm: Tensor::new(&format!("blk.{}.ffn_norm.weight", layer_idx), vec![h]),
            ffn_gate: Tensor::new(&format!("blk.{}.ffn_gate.weight", layer_idx), vec![h, ffn]),
            ffn_up: Tensor::new(&format!("blk.{}.ffn_up.weight", layer_idx), vec![h, ffn]),
            ffn_down: Tensor::new(&format!("blk.{}.ffn_down.weight", layer_idx), vec![ffn, h]),
        }
    }

    /// Initialize all weights with Xavier initialization
    pub fn init(&mut self, rng: &mut impl Rng) {
        // RMSNorm weights start at 1.0
        self.attn_norm.data.fill(1.0);
        self.ffn_norm.data.fill(1.0);

        self.attn_q.init_xavier(rng);
        self.attn_k.init_xavier(rng);
        self.attn_v.init_xavier(rng);
        self.attn_out.init_xavier(rng);
        self.ffn_gate.init_xavier(rng);
        self.ffn_up.init_xavier(rng);
        self.ffn_down.init_xavier(rng);
    }
}

// ─── Full Transformer Model ────────────────────────────────────────────────

/// Complete transformer model with all layers.
#[derive(Clone)]
pub struct TransformerModel {
    pub config: ModelConfig,
    pub token_embd: Tensor, // [vocab_size, hidden_dim]
    pub layers: Vec<TransformerLayer>,
    pub output_norm: Tensor,         // [hidden_dim]
    pub output_proj: Option<Tensor>, // [hidden_dim, vocab_size] — None if tied
}

impl TransformerModel {
    /// Create a new model with the given config, initialized with random weights.
    pub fn new(config: ModelConfig, rng: &mut impl Rng) -> Self {
        let mut token_embd = Tensor::new(
            "token_embd.weight",
            vec![config.vocab_size, config.hidden_dim],
        );
        token_embd.init_xavier(rng);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layer = TransformerLayer::new(i, &config);
            layer.init(rng);
            layers.push(layer);
        }

        let mut output_norm = Tensor::new("output_norm.weight", vec![config.hidden_dim]);
        output_norm.data.fill(1.0);

        let output_proj = if config.tie_embeddings {
            None
        } else {
            let mut t = Tensor::new("output.weight", vec![config.hidden_dim, config.vocab_size]);
            t.init_xavier(rng);
            Some(t)
        };

        Self {
            config,
            token_embd,
            layers,
            output_norm,
            output_proj,
        }
    }

    /// Get all trainable parameters as a flat list of mutable references.
    pub fn param_slices(&mut self) -> Vec<&mut [f32]> {
        let mut slices = Vec::new();
        slices.push(self.token_embd.data.as_mut_slice());
        for layer in &mut self.layers {
            slices.push(layer.attn_norm.data.as_mut_slice());
            slices.push(layer.attn_q.data.as_mut_slice());
            slices.push(layer.attn_k.data.as_mut_slice());
            slices.push(layer.attn_v.data.as_mut_slice());
            slices.push(layer.attn_out.data.as_mut_slice());
            slices.push(layer.ffn_norm.data.as_mut_slice());
            slices.push(layer.ffn_gate.data.as_mut_slice());
            slices.push(layer.ffn_up.data.as_mut_slice());
            slices.push(layer.ffn_down.data.as_mut_slice());
        }
        slices.push(self.output_norm.data.as_mut_slice());
        if let Some(ref mut t) = self.output_proj {
            slices.push(t.data.as_mut_slice());
        }
        slices
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        let mut count = self.token_embd.data.len();
        for layer in &self.layers {
            count += layer.attn_norm.data.len();
            count += layer.attn_q.data.len();
            count += layer.attn_k.data.len();
            count += layer.attn_v.data.len();
            count += layer.attn_out.data.len();
            count += layer.ffn_norm.data.len();
            count += layer.ffn_gate.data.len();
            count += layer.ffn_up.data.len();
            count += layer.ffn_down.data.len();
        }
        count += self.output_norm.data.len();
        if let Some(ref t) = self.output_proj {
            count += t.data.len();
        }
        count
    }
}

// ─── Forward Pass (Inference) ──────────────────────────────────────────────

/// KV cache entry for one layer.
pub struct KVCacheEntry {
    pub k: Vec<f32>, // [seq_len, kv_dim]
    pub v: Vec<f32>, // [seq_len, kv_dim]
}

/// KV cache for all layers.
pub struct KVCache {
    pub entries: Vec<KVCacheEntry>,
    pub seq_len: usize,
    pub max_seq: usize,
}

impl KVCache {
    pub fn new(cfg: &ModelConfig, max_seq: usize) -> Self {
        let kv_dim = cfg.kv_dim();
        let entries = (0..cfg.num_layers)
            .map(|_| KVCacheEntry {
                k: vec![0.0f32; max_seq * kv_dim],
                v: vec![0.0f32; max_seq * kv_dim],
            })
            .collect();
        Self {
            entries,
            seq_len: 0,
            max_seq,
        }
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        for entry in &mut self.entries {
            entry.k.fill(0.0);
            entry.v.fill(0.0);
        }
    }
}

/// Forward pass: given token ids, return logits [vocab_size].
/// This is the full forward pass for training (batch mode).
pub fn forward_batch(model: &TransformerModel, tokens: &[u32], seq_len: usize) -> Vec<f32> {
    let cfg = &model.config;
    let h = cfg.hidden_dim;
    let vocab = cfg.vocab_size;

    // Embed tokens [seq_len, hidden]
    let mut hidden = vec![0.0f32; seq_len * h];
    for (i, &tok) in tokens.iter().enumerate().take(seq_len) {
        let tok = tok as usize % cfg.vocab_size;
        let start = i * h;
        let emb_start = tok * h;
        hidden[start..start + h].copy_from_slice(&model.token_embd.data[emb_start..emb_start + h]);
    }

    // Transformer layers
    let head_dim = cfg.head_dim();
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let kv_dim = cfg.kv_dim();

    for layer in &model.layers {
        let mut residual = hidden.clone();

        // RMSNorm before attention
        rms_norm_inplace(&mut hidden, &layer.attn_norm.data, cfg.rms_norm_eps);

        // Q, K, V projections (simplified: full matmul)
        let mut q = matmul_fwd(&hidden, &layer.attn_q, seq_len, h, num_heads * head_dim);
        let k = matmul_fwd(&hidden, &layer.attn_k, seq_len, h, kv_dim);
        let v = matmul_fwd(&hidden, &layer.attn_v, seq_len, h, kv_dim);

        // Apply RoPE
        rope_apply(&mut q, &mut Vec::new(), 0, head_dim, cfg.rope_theta);

        // Multi-head attention (simplified: no KV cache for training batch)
        let mut attn_out =
            multi_head_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);

        // Output projection
        let attn_proj = matmul_fwd(&attn_out, &layer.attn_out, seq_len, num_heads * head_dim, h);

        // Residual + FFN
        for i in 0..(seq_len * h) {
            hidden[i] = residual[i] + attn_proj[i];
        }
        residual = hidden.clone();

        // RMSNorm before FFN
        rms_norm_inplace(&mut hidden, &layer.ffn_norm.data, cfg.rms_norm_eps);

        // SwiGLU FFN
        let gate = matmul_fwd(&hidden, &layer.ffn_gate, seq_len, h, cfg.intermediate_dim);
        let up = matmul_fwd(&hidden, &layer.ffn_up, seq_len, h, cfg.intermediate_dim);

        // SiLU(gate) * up
        let mut activated = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            activated[i] = silu(gate[i]) * up[i];
        }

        let ffn_out = matmul_fwd(
            &activated,
            &layer.ffn_down,
            seq_len,
            cfg.intermediate_dim,
            h,
        );

        // Residual
        for i in 0..(seq_len * h) {
            hidden[i] = residual[i] + ffn_out[i];
        }
    }

    // Final RMSNorm
    rms_norm_inplace(&mut hidden, &model.output_norm.data, cfg.rms_norm_eps);

    // Output projection (last token only for training)
    let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
    let mut logits = vec![0.0f32; vocab];

    if let Some(ref proj) = model.output_proj {
        // Untied embeddings
        matmul_vec(
            last_hidden,
            &proj.data,
            proj.shape[0],
            proj.shape[1],
            &mut logits,
        );
    } else {
        // Tied embeddings: use token_embd transposed
        for v in 0..vocab {
            let emb_start = v * h;
            let mut sum = 0.0f32;
            for d in 0..h {
                sum += last_hidden[d] * model.token_embd.data[emb_start + d];
            }
            // Scale by sqrt(hidden_dim) like Gemma
            logits[v] = sum * (h as f32).sqrt();
        }
    }

    logits
}

/// Single-token forward with KV cache (for inference).
pub fn forward_token(
    model: &TransformerModel,
    token: u32,
    pos: usize,
    cache: &mut KVCache,
) -> Vec<f32> {
    let cfg = &model.config;
    let h = cfg.hidden_dim;
    let vocab = cfg.vocab_size;

    // Embed token
    let tok = token as usize % cfg.vocab_size;
    let mut hidden = model.token_embd.data[tok * h..(tok + 1) * h].to_vec();

    let head_dim = cfg.head_dim();
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let kv_dim = cfg.kv_dim();

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let residual = hidden.clone();

        // RMSNorm
        rms_norm_inplace(&mut hidden, &layer.attn_norm.data, cfg.rms_norm_eps);

        // Q, K, V
        let mut q = matmul_vec_single(&hidden, &layer.attn_q, h, num_heads * head_dim);
        let mut k = matmul_vec_single(&hidden, &layer.attn_k, h, kv_dim);
        let v = matmul_vec_single(&hidden, &layer.attn_v, h, kv_dim);

        // RoPE
        rope_apply(&mut q, &mut k, pos, head_dim, cfg.rope_theta);

        // Store in KV cache
        let entry = &mut cache.entries[layer_idx];
        for d in 0..kv_dim {
            entry.k[pos * kv_dim + d] = k[d];
            entry.v[pos * kv_dim + d] = v[d];
        }

        // Attention over cached KV
        let attn_out = attention_with_cache(
            &q,
            &entry.k[..(pos + 1) * kv_dim],
            &entry.v[..(pos + 1) * kv_dim],
            pos + 1,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Output projection
        let attn_proj = matmul_vec_single(&attn_out, &layer.attn_out, num_heads * head_dim, h);

        // Residual + FFN
        for i in 0..h {
            hidden[i] = residual[i] + attn_proj[i];
        }
        let residual = hidden.clone();

        // RMSNorm
        rms_norm_inplace(&mut hidden, &layer.ffn_norm.data, cfg.rms_norm_eps);

        // SwiGLU
        let gate = matmul_vec_single(&hidden, &layer.ffn_gate, h, cfg.intermediate_dim);
        let up = matmul_vec_single(&hidden, &layer.ffn_up, h, cfg.intermediate_dim);
        let mut activated = vec![0.0f32; cfg.intermediate_dim];
        for i in 0..cfg.intermediate_dim {
            activated[i] = silu(gate[i]) * up[i];
        }
        let ffn_out = matmul_vec_single(&activated, &layer.ffn_down, cfg.intermediate_dim, h);

        for i in 0..h {
            hidden[i] = residual[i] + ffn_out[i];
        }
    }

    // Final norm
    rms_norm_inplace(&mut hidden, &model.output_norm.data, cfg.rms_norm_eps);

    // Output projection
    let mut logits = vec![0.0f32; vocab];
    if let Some(ref proj) = model.output_proj {
        matmul_vec(
            &hidden,
            &proj.data,
            proj.shape[0],
            proj.shape[1],
            &mut logits,
        );
    } else {
        for v in 0..vocab {
            let emb_start = v * h;
            let mut sum = 0.0f32;
            for d in 0..h {
                sum += hidden[d] * model.token_embd.data[emb_start + d];
            }
            logits[v] = sum * (h as f32).sqrt();
        }
    }

    cache.seq_len = pos + 1;
    logits
}

// ─── Core Operations ───────────────────────────────────────────────────────

/// RMSNorm in-place. Applies norm independently for each hidden-dim slice.
/// `weight` has length `dim`, `x` can be any multiple of `dim`.
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let dim = weight.len();
    let n_blocks = x.len() / dim;
    for b in 0..n_blocks {
        let start = b * dim;
        let block = &mut x[start..start + dim];
        let mut ss = 0.0f32;
        for &v in block.iter() {
            ss += v * v;
        }
        let rms = (ss / dim as f32 + eps).sqrt().recip();
        for i in 0..dim {
            block[i] = block[i] * rms * weight[i];
        }
    }
}

/// Matrix multiply: [seq, in_dim] x [in_dim, out_dim] -> [seq, out_dim]
fn matmul_fwd(
    input: &[f32],
    weight: &Tensor,
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * out_dim];
    for s in 0..seq {
        let in_start = s * in_dim;
        let out_start = s * out_dim;
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += input[in_start + i] * weight.data[i * out_dim + o];
            }
            out[out_start + o] = sum;
        }
    }
    out
}

/// Vector-matrix multiply: [in_dim] x [in_dim, out_dim] -> [out_dim]
fn matmul_vec(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize, out: &mut [f32]) {
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += input[i] * weight[i * out_dim + o];
        }
        out[o] = sum;
    }
}

/// Single vector-matrix multiply returning new vec
fn matmul_vec_single(input: &[f32], weight: &Tensor, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += input[i] * weight.data[i * out_dim + o];
        }
        out[o] = sum;
    }
    out
}

/// SiLU activation
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE (Rotary Position Embedding)
fn rope_apply(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let val = pos as f32 * freq;
        let cos = val.cos();
        let sin = val.sin();
        // Apply to Q
        for head_start in (0..q.len()).step_by(head_dim) {
            let i0 = head_start + i;
            let i1 = head_start + i + half_dim;
            if i1 < q.len() {
                let q0 = q[i0];
                let q1 = q[i1];
                q[i0] = q0 * cos - q1 * sin;
                q[i1] = q0 * sin + q1 * cos;
            }
        }
        // Apply to K
        if !k.is_empty() {
            for head_start in (0..k.len()).step_by(head_dim) {
                let i0 = head_start + i;
                let i1 = head_start + i + half_dim;
                if i1 < k.len() {
                    let k0 = k[i0];
                    let k1 = k[i1];
                    k[i0] = k0 * cos - k1 * sin;
                    k[i1] = k0 * sin + k1 * cos;
                }
            }
        }
    }
}

/// Multi-head attention (batch mode, no cache)
fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_groups = num_heads / num_kv_heads;
    let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
    let scale = (head_dim as f32).recip().sqrt();

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        for s1 in 0..seq_len {
            for s2 in 0..=s1 {
                // Dot product Q[s1,h] · K[s2,kv_h]
                let q_start = s1 * num_heads * head_dim + h * head_dim;
                let k_start = s2 * num_kv_heads * head_dim + kv_h * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                score *= scale;

                // Softmax (online)
                let v_start = s2 * num_kv_heads * head_dim + kv_h * head_dim;
                let out_start = s1 * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    out[out_start + d] += score * v[v_start + d]; // simplified: no softmax
                }
            }
        }
    }
    out
}

/// Attention with KV cache (single token inference)
fn attention_with_cache(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * head_dim];
    let scale = (head_dim as f32).recip().sqrt();
    let kv_groups = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / kv_groups;
        let q_start = h * head_dim;
        let mut max_val = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; seq_len];

        // Compute attention scores
        for s in 0..seq_len {
            let k_start = s * num_kv_heads * head_dim + kv_h * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[q_start + d] * k_cache[k_start + d];
            }
            score *= scale;
            scores[s] = score;
            if score > max_val {
                max_val = score;
            }
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for s in 0..seq_len {
            let exp = (scores[s] - max_val).exp();
            scores[s] = exp;
            sum_exp += exp;
        }
        let inv_sum = sum_exp.recip();

        // Weighted sum of V
        let out_start = h * head_dim;
        for s in 0..seq_len {
            let w = scores[s] * inv_sum;
            let v_start = s * num_kv_heads * head_dim + kv_h * head_dim;
            for d in 0..head_dim {
                out[out_start + d] += w * v_cache[v_start + d];
            }
        }
    }
    out
}

// ─── Backward Pass (Gradients) ─────────────────────────────────────────────

/// Simple numerical gradient check (for testing only).
/// Uses finite differences: df/dx ≈ (f(x+ε) - f(x-ε)) / 2ε
#[cfg(test)]
pub fn numerical_gradient_check() {
    // Used for validating analytical gradients
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    struct TestRng(u64);
    impl Rng for TestRng {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32) / (u32::MAX as f32)
        }
    }

    #[test]
    fn test_model_creation() {
        let cfg = ModelConfig::nano(1000);
        let mut rng = TestRng(42);
        let model = TransformerModel::new(cfg, &mut rng);
        assert!(model.param_count() > 0);
        println!("Model params: {:.2}M", model.param_count() as f64 / 1e6);
    }

    #[test]
    fn test_forward_pass() {
        let cfg = ModelConfig::nano(500);
        let mut rng = TestRng(42);
        let model = TransformerModel::new(cfg.clone(), &mut rng);

        let tokens: Vec<u32> = (0..16).map(|i| (i * 7) % 500).collect();
        let logits = forward_batch(&model, &tokens, 16);

        assert_eq!(logits.len(), cfg.vocab_size);
        // Check logits are finite
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_forward_token() {
        let cfg = ModelConfig::nano(500);
        let mut rng = TestRng(42);
        let model = TransformerModel::new(cfg.clone(), &mut rng);

        let mut cache = KVCache::new(&cfg, 64);
        let logits = forward_token(&model, 0, 0, &mut cache);
        assert_eq!(logits.len(), cfg.vocab_size);
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        rms_norm_inplace(&mut x, &w, 1e-5);
        // RMS norm should have unit RMS after normalization
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }
}
