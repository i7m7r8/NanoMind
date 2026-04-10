//! Model configuration for from-scratch training.

/// Configuration for a transformer model.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_embeddings: bool,
}

impl ModelConfig {
    /// nano — ~2M params, fits in CI training
    pub fn nano(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 4,
            intermediate_dim: 256,
            max_seq_len: 256,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// mini — ~15M params, good for desktop training
    pub fn mini(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 6,
            intermediate_dim: 512,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// small — ~50M params
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_dim: 384,
            num_heads: 8,
            num_kv_heads: 4,
            num_layers: 8,
            intermediate_dim: 768,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Dimension of the key/value vectors
    pub fn kv_dim(&self) -> usize {
        self.hidden_dim / self.num_heads * self.num_kv_heads
    }

    /// Dimension of each attention head
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Total parameter count (approximate)
    pub fn param_count(&self) -> usize {
        let h = self.hidden_dim;
        let v = self.vocab_size;
        let ffn = self.intermediate_dim;
        let layers = self.num_layers;

        // Embedding
        let embed_params = v * h;

        // Per layer: attention + FFN
        let head_dim = self.head_dim();
        let kv_dim = self.kv_dim();
        let qkv = h * (self.num_heads * head_dim + 2 * kv_dim);
        let o_proj = h * self.num_heads * head_dim;
        let ffn_params = 3 * h * ffn; // gate + up + down
        let norms = 2 * h; // attn_norm + ffn_norm
        let layer_params = qkv + o_proj + ffn_params + norms;

        // Output
        let output_params = if self.tie_embeddings { 0 } else { v * h };

        embed_params + layers * layer_params + output_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_config() {
        let cfg = ModelConfig::nano(1000);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.hidden_dim, 128);
        assert_eq!(cfg.num_layers, 4);
    }

    #[test]
    fn test_mini_config() {
        let cfg = ModelConfig::mini(2000);
        assert_eq!(cfg.vocab_size, 2000);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_layers, 6);
    }

    #[test]
    fn test_param_count() {
        let cfg = ModelConfig::nano(1000);
        let params = cfg.param_count();
        assert!(params > 0);
        println!("Nano config params: {:.2}M", params as f64 / 1e6);
    }

    #[test]
    fn test_mini_param_count() {
        let cfg = ModelConfig::mini(2000);
        let params = cfg.param_count();
        assert!(params > 0);
        println!("Mini config params: {:.2}M", params as f64 / 1e6);
    }
}
