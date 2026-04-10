//! NanoMind Runtime — Inference loop, sampling, RAM budget enforcement.

use nanomind_core::ops::softmax_with_temp;
use nanomind_model::model::{Config, KVCache, Model, QuantType, compute_logits, embed_token, transformer_block_forward};
use nanomind_tokenizer::BpeTokenizer;

use rand::{Rng, SeedableRng};

/// Sampling configuration.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,    // 0.0 = greedy
    pub top_p: f32,          // nucleus sampling
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            max_new_tokens: 256,
        }
    }
}

/// Inference engine.
pub struct InferenceEngine {
    pub model: Model,
    pub tokenizer: BpeTokenizer,
    pub cache: KVCache,
    pub config: SamplingConfig,
    rng: rand::rngs::StdRng,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(model: Model, tokenizer: BpeTokenizer, config: SamplingConfig) -> Self {
        let cache = KVCache::new(&model.config);
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        Self {
            model,
            tokenizer,
            cache,
            config,
            rng,
        }
    }

    /// Generate text from a prompt.
    ///
    /// Returns the generated token IDs. The `callback` is called with each token ID.
    pub fn generate(&mut self, prompt: &str, mut callback: impl FnMut(u32)) -> Vec<u32> {
        let vocab_size = self.model.config.vocab_size;
        let hd = self.model.config.hidden_dim;
        let num_layers = self.model.config.num_layers;
        let max_seq_len = self.model.config.max_seq_len;

        // Reset KV cache
        self.cache.reset();

        // Encode prompt
        let prompt_tokens = self.tokenizer.encode(prompt);

        // Pre-fill: run prompt through the model (KV cache builds up)
        let mut hidden = vec![0.0f32; hd];
        let mut last_token: Option<u32> = None;

        for (i, &token_id) in prompt_tokens.iter().enumerate() {
            let pos = i.min(max_seq_len - 1);
            embed_token(token_id, &self.model.token_embeddings, hd, &mut hidden);

            // Run through all transformer layers
            for layer_idx in 0..num_layers {
                transformer_block_forward(
                    &mut hidden,
                    &self.model.layers[layer_idx],
                    &mut self.cache,
                    layer_idx,
                    pos,
                    &self.model.config,
                    &self.model.rope_cos,
                    &self.model.rope_sin,
                );
            }

            last_token = Some(token_id);
        }

        // Generate new tokens
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        let mut current_hidden = hidden;

        for _gen_step in 0..self.config.max_new_tokens {
            // Compute logits
            let mut logits = vec![0.0f32; vocab_size];
            compute_logits(&current_hidden, &self.model.output_weights, vocab_size, &mut logits);

            // Apply repetition penalty
            if let Some(prev_token) = last_token {
                if self.config.repetition_penalty != 1.0 {
                    let prev_idx = prev_token as usize;
                    if prev_idx < vocab_size {
                        if logits[prev_idx] > 0.0 {
                            logits[prev_idx] /= self.config.repetition_penalty;
                        } else {
                            logits[prev_idx] *= self.config.repetition_penalty;
                        }
                    }
                }
            }

            // Sample
            let token_id = self.sample_token(&logits);

            // Check for EOS
            if let Some(eos_id) = self.tokenizer.special.eos_token.as_ref() {
                if let Some(eos_token_id) = self.tokenizer.token_to_id(eos_id) {
                    if token_id == eos_token_id {
                        break;
                    }
                }
            }

            generated.push(token_id);
            callback(token_id);

            // Embed the generated token for next iteration
            embed_token(token_id, &self.model.token_embeddings, hd, &mut current_hidden);

            // Run through all layers
            let gen_pos = pos.min(max_seq_len - 1);
            for layer_idx in 0..num_layers {
                transformer_block_forward(
                    &mut current_hidden,
                    &self.model.layers[layer_idx],
                    &mut self.cache,
                    layer_idx,
                    gen_pos,
                    &self.model.config,
                    &self.model.rope_cos,
                    &self.model.rope_sin,
                );
            }

            last_token = Some(token_id);
            pos += 1;
        }

        generated
    }

    /// Decode token IDs to text.
    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    /// Get a token string by ID.
    pub fn token_str(&self, token_id: u32) -> Option<String> {
        self.tokenizer.id_to_token_str(token_id)
    }

    /// Sample a token ID from logits using temperature, top-k, and top-p.
    fn sample_token(&mut self, logits: &[f32]) -> u32 {
        let mut probs = logits.to_vec();

        if self.config.temperature > 0.0 {
            // Top-k filtering
            if self.config.top_k > 0 && self.config.top_k < probs.len() {
                let mut indexed: Vec<(f32, usize)> = probs.iter().copied().enumerate().map(|(i, p)| (p, i)).collect();
                indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                // Keep only top-k
                for (i, &(_, orig_idx)) in indexed.iter().enumerate().skip(self.config.top_k) {
                    probs[orig_idx] = f32::NEG_INFINITY;
                    let _ = i;
                }
            }

            // Softmax with temperature
            softmax_with_temp(&mut probs, self.config.temperature);

            // Top-p (nucleus) sampling
            if self.config.top_p < 1.0 {
                let mut indexed: Vec<(f32, usize)> = probs.iter().copied().enumerate().map(|(i, p)| (p, i)).collect();
                indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                let mut cumulative = 0.0f32;
                for (i, &(p, _orig_idx)) in indexed.iter().enumerate() {
                    cumulative += p;
                    if cumulative > self.config.top_p {
                        // Set remaining to zero
                        for &(_, idx) in indexed.iter().skip(i + 1) {
                            probs[idx] = 0.0;
                        }
                        break;
                    }
                }

                // Re-normalize
                let sum: f32 = probs.iter().filter(|&&p| p > 0.0).sum();
                if sum > 0.0 {
                    for p in probs.iter_mut() {
                        *p /= sum;
                    }
                }
            }

            // Categorical sampling
            let r: f32 = self.rng.gen_range(0.0..1.0);
            let mut cumulative = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r <= cumulative {
                    return i as u32;
                }
            }

            // Fallback: return last token
            (probs.len() - 1) as u32
        } else {
            // Greedy: argmax
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &p) in probs.iter().enumerate() {
                if p > best_val {
                    best_val = p;
                    best_idx = i;
                }
            }
            best_idx as u32
        }
    }
}

/// Estimate RAM usage for a model configuration.
///
/// Returns estimated bytes. Refuses to load if > 480 MB.
pub fn estimate_ram_usage(config: &Config, quant: QuantType, _max_seq: usize) -> usize {
    config.estimate_ram_bytes(quant)
}

/// Check if a model fits within the RAM budget.
pub fn fits_ram(config: &Config, quant: QuantType) -> bool {
    estimate_ram_usage(config, quant, config.max_seq_len) <= nanomind_core::MAX_RAM_BUDGET
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ram_estimate() {
        let config = Config {
            vocab_size: 1000,
            hidden_dim: 256,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            intermediate_dim: 512,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let ram = estimate_ram_usage(&config, QuantType::Q4, 512);
        assert!(ram > 0);
        assert!(fits_ram(&config, QuantType::Q4));
    }
}
