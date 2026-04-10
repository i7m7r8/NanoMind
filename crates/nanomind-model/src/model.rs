//! Model loading from GGUF and transformer forward pass.

use std::collections::HashMap;
use std::path::Path;
use std::vec::Vec;

use nanomind_core::{
    dot_f32, rms_norm, silu_inplace, softmax_inplace, GgmlType, RopeCache, RopeConfig, RopeScaling,
};
use nanomind_gguf::GgufReader;

use crate::config::{Architecture, ModelConfig};
use crate::kv_cache::KvCache;
use crate::layers::{LayerTensor, LayerWeights};

/// Fully loaded model ready for inference.
pub struct Model {
    pub config: ModelConfig,
    pub rope_cache: RopeCache,
    pub token_embeddings: LayerTensor,
    pub output_norm: Vec<f32>,
    pub output_weights: Option<LayerTensor>, // None if tied with embeddings
    pub layers: Vec<LayerWeights>,
    /// Maximum context the model supports.
    pub max_seq: usize,
}

impl Model {
    /// Load a model from a GGUF file.
    pub fn from_gguf(path: &Path, max_ctx: Option<usize>) -> Result<Self, String> {
        let reader =
            GgufReader::open(path).map_err(|e| format!("Failed to open GGUF file: {}", e))?;

        let config = ModelConfig::from_gguf(&reader.metadata);
        let max_seq = max_ctx.unwrap_or(config.max_position_embeddings);

        println!("{}", reader.summary());

        // Build rope cache
        let rope_config = RopeConfig {
            dim: config.head_dim,
            theta: config.rope_theta,
            scaling_factor: max_seq as f32 / config.max_position_embeddings as f32,
            scaling_type: if config.rope_scaling > 1.0 {
                RopeScaling::NtkAware
            } else {
                RopeScaling::None
            },
        };
        let rope_cache = RopeCache::new(&rope_config, max_seq);

        // Load token embeddings
        let token_embeddings = load_tensor(&reader, "token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight".to_string())?;

        // Load output norm
        let output_norm_name = match config.arch {
            Architecture::Gemma2 => "token_embd_norm.weight",
            _ => "output_norm.weight",
        };
        let output_norm = load_f32_tensor(&reader, output_norm_name)
            .or_else(|| load_f32_tensor(&reader, "norm.weight"))
            .unwrap_or_else(|| vec![1.0; config.hidden_size]);

        // Load output weights (may be tied)
        let output_weights = load_tensor(&reader, "output.weight");

        // Load layers
        let num_layers = config.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{}", layer_idx);

            let layer = LayerWeights {
                attn_q: load_tensor(&reader, &format!("{}.attn_q.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.attn_q.weight", prefix))?,
                attn_k: load_tensor(&reader, &format!("{}.attn_k.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.attn_k.weight", prefix))?,
                attn_v: load_tensor(&reader, &format!("{}.attn_v.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.attn_v.weight", prefix))?,
                attn_o: load_tensor(&reader, &format!("{}.attn_o.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.attn_o.weight", prefix))?,
                ffn_gate: load_tensor(&reader, &format!("{}.ffn_gate.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.ffn_gate.weight", prefix))?,
                ffn_up: load_tensor(&reader, &format!("{}.ffn_up.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.ffn_up.weight", prefix))?,
                ffn_down: load_tensor(&reader, &format!("{}.ffn_down.weight", prefix))
                    .ok_or_else(|| format!("Missing {}.ffn_down.weight", prefix))?,
                attn_norm: load_f32_tensor(&reader, &format!("{}.attn_norm.weight", prefix))
                    .unwrap_or_else(|| vec![1.0; config.hidden_size]),
                ffn_norm: load_f32_tensor(&reader, &format!("{}.ffn_norm.weight", prefix))
                    .unwrap_or_else(|| vec![1.0; config.hidden_size]),
                ffn_gate_experts: None,
                ffn_up_experts: None,
                ffn_down_experts: None,
            };

            layers.push(layer);
        }

        println!(
            "[INFO] Model loaded: {} layers, {} params, {} context",
            num_layers,
            config.estimate_params(),
            max_seq,
        );

        Ok(Self {
            config,
            rope_cache,
            token_embeddings,
            output_norm,
            output_weights,
            layers,
            max_seq,
        })
    }

    /// Run a forward pass for a single token.
    ///
    /// `input` is the token embedding (modified in-place).
    /// `cache` is the KV cache.
    /// `pos` is the absolute token position.
    pub fn forward_token(&self, input: &mut [f32], cache: &mut KvCache, pos: usize) {
        let cfg = &self.config;
        let hd = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let kv_groups = cfg.kv_groups();

        let mut residual = input.to_vec();

        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = &self.layers[layer_idx];

            // Pre-attention RMS norm
            let mut normed = residual.clone();
            rms_norm(&mut normed, &layer.attn_norm, cfg.rms_norm_eps);

            // Q projection
            let mut q = vec![0.0f32; n_heads * head_dim];
            layer.attn_q.matmul(&normed, &mut q);

            // K projection
            let mut k = vec![0.0f32; n_kv_heads * head_dim];
            layer.attn_k.matmul(&normed, &mut k);

            // V projection
            let mut v = vec![0.0f32; n_kv_heads * head_dim];
            layer.attn_v.matmul(&normed, &mut v);

            // Apply RoPE
            self.rope_cache.apply(&mut q, pos, head_dim);
            self.rope_cache.apply(&mut k, pos, head_dim);

            // Store in KV cache
            cache.store(layer_idx, &k, &v);

            // Multi-head attention with KV cache
            let attn_out = self.attention_forward(&q, layer_idx, pos, cache, cfg);

            // Output projection
            let mut o_proj = vec![0.0f32; hd];
            layer.attn_o.matmul(&attn_out, &mut o_proj);

            // Residual
            for i in 0..hd {
                residual[i] += o_proj[i];
            }

            // Post-attention RMS norm
            let mut normed = residual.clone();
            rms_norm(&mut normed, &layer.ffn_norm, cfg.rms_norm_eps);

            // SwiGLU FFN
            let ffn_out = self.ffn_forward(&normed, layer, cfg);

            // Residual
            for i in 0..hd {
                residual[i] += ffn_out[i];
            }
        }

        input.copy_from_slice(&residual);
    }

    /// Multi-head attention with GQA.
    fn attention_forward(
        &self,
        q: &[f32],
        layer_idx: usize,
        pos: usize,
        cache: &KvCache,
        cfg: &ModelConfig,
    ) -> Vec<f32> {
        let head_dim = cfg.head_dim;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let kv_groups = cfg.kv_groups();
        let context = cache.attn_context();
        let inv_sqrt_d = cfg.attn_norm_factor;

        let mut output = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h / kv_groups;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];

            let mut scores = vec![0.0f32; context];

            for t in 0..context {
                let k_t = cache.get_k(layer_idx, t);
                let kv_offset = kv_h * head_dim;
                let k_slice = &k_t[kv_offset..kv_offset + head_dim];
                scores[t] = dot_f32(q_head, k_slice) * inv_sqrt_d;
            }

            // Softmax
            softmax_inplace(&mut scores);

            // Weighted sum of values
            let o_head = &mut output[h * head_dim..(h + 1) * head_dim];
            for t in 0..context {
                let v_t = cache.get_v(layer_idx, t);
                let kv_offset = kv_h * head_dim;
                let v_slice = &v_t[kv_offset..kv_offset + head_dim];
                let w = scores[t];
                for d in 0..head_dim {
                    o_head[d] += w * v_slice[d];
                }
            }
        }

        output
    }

    /// SwiGLU FFN forward.
    fn ffn_forward(&self, x: &[f32], layer: &LayerWeights, cfg: &ModelConfig) -> Vec<f32> {
        let intermediate = cfg.intermediate_size;

        // Gate: silu(x @ W_gate)
        let mut gate = vec![0.0f32; intermediate];
        layer.ffn_gate.matmul(x, &mut gate);
        silu_inplace(&mut gate);

        // Up: x @ W_up
        let mut up = vec![0.0f32; intermediate];
        layer.ffn_up.matmul(x, &mut up);

        // Element-wise multiply
        for i in 0..intermediate {
            gate[i] *= up[i];
        }

        // Down: (gate * up) @ W_down
        let mut out = vec![0.0f32; cfg.hidden_size];
        layer.ffn_down.matmul(&gate, &mut out);

        out
    }

    /// Embed a token ID into a vector.
    pub fn embed_token(&self, token_id: u32, out: &mut [f32]) {
        let hd = self.config.hidden_size;
        let start = token_id as usize * hd;
        // Dequantize the embedding row
        let emb_data = &self.token_embeddings.data;
        let blck = self.token_embeddings.ggml_type.blck_size();
        let type_size = self.token_embeddings.ggml_type.type_size();

        if self.token_embeddings.ggml_type == GgmlType::F32 {
            for i in 0..hd {
                let bytes: [u8; 4] = emb_data[(start + i) * 4..(start + i) * 4 + 4]
                    .try_into()
                    .unwrap();
                out[i] = f32::from_le_bytes(bytes);
            }
        } else {
            let row_start = (start / blck) * type_size;
            nanomind_core::dequantize_block(
                &emb_data[row_start..],
                self.token_embeddings.ggml_type,
                out,
            );
        }

        // Gemma2 uses sqrt(d) * embedding
        if self.config.arch == Architecture::Gemma2 {
            let scale = (hd as f32).sqrt();
            for v in out.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Compute logits from hidden state.
    pub fn compute_logits(&self, hidden: &[f32], logits: &mut [f32]) {
        let vocab = self.config.vocab_size;

        if let Some(ref output_weights) = self.output_weights {
            output_weights.matmul(hidden, logits);
        } else {
            // Tied weights: use transposed embeddings
            let hd = self.config.hidden_size;
            let blck = self.token_embeddings.ggml_type.blck_size();
            let type_size = self.token_embeddings.ggml_type.type_size();
            let emb = &self.token_embeddings.data;

            for v in 0..vocab {
                if self.token_embeddings.ggml_type == GgmlType::F32 {
                    let mut sum = 0.0f32;
                    for d in 0..hd {
                        let bytes: [u8; 4] = emb[(v * hd + d) * 4..(v * hd + d) * 4 + 4]
                            .try_into()
                            .unwrap();
                        sum += f32::from_le_bytes(bytes) * hidden[d];
                    }
                    logits[v] = sum;
                } else {
                    let row_start = (v * hd / blck) * type_size;
                    logits[v] = nanomind_core::dot_q4_f32(
                        &emb[row_start..],
                        self.token_embeddings.ggml_type,
                        hidden,
                    );
                }
            }
        }

        // Logit softcap (Gemma2)
        if let Some(softcap) = self.config.logit_softcap {
            for v in 0..vocab {
                logits[v] = softcap * (logits[v] / softcap).tanh();
            }
        }

        // Output norm (Gemma2)
        if self.config.arch == Architecture::Gemma2 && !self.output_norm.is_empty() {
            rms_norm(logits, &self.output_norm, self.config.rms_norm_eps);
        }
    }
}

/// Load a tensor from GGUF.
fn load_tensor(reader: &GgufReader, name: &str) -> Option<LayerTensor> {
    let info = reader.tensor_info(name)?;
    let data = reader.tensor_data(name)?;

    let shape: Vec<usize> = info.shape().iter().map(|&d| d as usize).collect();

    Some(LayerTensor {
        data: data.to_vec(),
        ggml_type: info.ty,
        shape,
    })
}

/// Load a tensor as f32 (for norm weights).
fn load_f32_tensor(reader: &GgufReader, name: &str) -> Option<Vec<f32>> {
    let info = reader.tensor_info(name)?;
    let data = reader.tensor_data(name)?;

    let n = info.n_elements() as usize;
    let mut out = vec![0.0f32; n];

    match info.ty {
        GgmlType::F32 => {
            for i in 0..n {
                let bytes: [u8; 4] = data[i * 4..i * 4 + 4].try_into().ok()?;
                out[i] = f32::from_le_bytes(bytes);
            }
        }
        GgmlType::F16 => {
            for i in 0..n {
                let bytes: [u8; 2] = data[i * 2..i * 2 + 2].try_into().ok()?;
                out[i] = half::f16::from_le_bytes(bytes).to_f32();
            }
        }
        _ => {
            nanomind_core::dequantize_block(data, info.ty, &mut out);
        }
    }

    Some(out)
}
