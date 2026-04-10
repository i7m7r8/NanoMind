//! Model configuration derived from GGUF metadata.

use nanomind_gguf::metadata::GgufMetadata;

/// Architecture type detected from GGUF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Qwen2,
    Mistral,
    Phi3,
    Gemma2,
    Unknown,
}

impl Architecture {
    pub fn from_name(s: &str) -> Self {
        match s {
            "llama" => Self::Llama,
            "qwen2" | "qwen3" => Self::Qwen2,
            "mistral" => Self::Mistral,
            "phi3" => Self::Phi3,
            "gemma2" => Self::Gemma2,
            _ => Self::Unknown,
        }
    }
}

/// Full model configuration.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: Architecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_scaling: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    /// Whether to use sliding window attention (Mistral-style).
    pub sliding_window: Option<usize>,
    /// Whether to use query/ key normalization.
    pub norm_qk: bool,
    /// Logit softcapping (Gemma-style).
    pub logit_softcap: Option<f32>,
    /// Attention normalization factor.
    pub attn_norm_factor: f32,
}

impl ModelConfig {
    /// Parse from GGUF metadata.
    pub fn from_gguf(meta: &GgufMetadata) -> Self {
        let arch = Architecture::from_name(meta.arch_string());

        let hidden_size = meta.embedding_length() as usize;
        let num_attention_heads = meta.head_count() as usize;
        let num_key_value_heads = meta.head_count_kv() as usize;
        let head_dim = if hidden_size > 0 && num_attention_heads > 0 {
            hidden_size / num_attention_heads
        } else {
            128
        };
        let rope_dim = meta.rope_dim().unwrap_or(head_dim as u32) as usize;
        // Some architectures have a different head_dim than hidden/heads
        let head_dim = rope_dim;

        let arch_str = meta.arch_string();
        let rms_norm_eps = meta.rms_norm_eps();
        let rope_theta = meta.rope_theta();

        // Sliding window (Mistral)
        let sliding_window = if arch_str == "mistral" {
            meta.get_u32("mistral.sliding_window").map(|v| v as usize)
        } else {
            None
        };

        // Logit softcap (Gemma2)
        let logit_softcap = if arch_str == "gemma2" {
            meta.get_f32("gemma2.attn_logit_softcapping")
        } else {
            None
        };

        // Attention norm factor
        let attn_norm_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            arch,
            vocab_size: meta.vocab_size() as usize,
            hidden_size,
            intermediate_size: meta.ffn_dim() as usize,
            num_hidden_layers: meta.block_count() as usize,
            num_attention_heads,
            num_key_value_heads: if num_key_value_heads > 0 {
                num_key_value_heads
            } else {
                num_attention_heads
            },
            head_dim,
            rope_theta,
            rope_scaling: 1.0,
            rms_norm_eps,
            max_position_embeddings: meta.context_length() as usize,
            bos_token_id: meta.bos_token_id().unwrap_or(1),
            eos_token_id: meta.eos_token_id().unwrap_or(2),
            pad_token_id: 0,
            sliding_window,
            norm_qk: arch_str == "phi3" || arch_str == "gemma2",
            logit_softcap,
            attn_norm_factor,
        }
    }

    /// GQA groups (num_query_heads / num_kv_heads).
    pub fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total parameter estimate (for RAM budgeting).
    pub fn estimate_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let n_heads = self.num_attention_heads as u64;
        let n_kv = self.num_key_value_heads as u64;
        let ffn = self.intermediate_size as u64;
        let v = self.vocab_size as u64;
        let n_layers = self.num_hidden_layers as u64;
        let hd = self.head_dim as u64;

        // Per layer
        let qkv = h * (n_heads + 2 * n_kv) * hd / h; // q + k + v projections
        let o_proj = h * n_heads * hd / h;
        let ffn_params = 3 * h * ffn; // gate + up + down
        let layer_params = qkv + o_proj + ffn_params + 2 * h; // 2 RMS norms

        // embedding + lm_head
        n_layers * layer_params + v * h * 2
    }
}
