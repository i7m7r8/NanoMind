//! Rotary Position Embeddings (RoPE) with scaling support.
//!
//! Supports: standard RoPE, NTK-aware scaling, YARN scaling,
//! and linear scaling for arbitrary context windows.

use std::vec::Vec;

/// RoPE configuration for context scaling.
#[derive(Clone, Debug)]
pub struct RopeConfig {
    pub dim: usize,
    pub theta: f32,
    pub scaling_factor: f32, // > 1.0 for extended context
    pub scaling_type: RopeScaling,
}

#[derive(Clone, Debug)]
pub enum RopeScaling {
    /// No scaling — standard RoPE.
    None,
    /// NTK-aware: scale theta by factor^dim/(dim-2).
    NtkAware,
    /// YARN: YaRN method with temperature and beta.
    Yarn {
        temperature: f32,
        beta_fast: f32,
        beta_slow: f32,
    },
    /// Linear: scale frequencies linearly.
    Linear,
}

/// Precomputed RoPE cos/sin tables.
pub struct RopeCache {
    pub cos: Vec<f32>, // [max_seq * half_dim]
    pub sin: Vec<f32>,
    pub half_dim: usize,
}

impl RopeCache {
    /// Build a RoPE cache for the given config and max sequence length.
    pub fn new(config: &RopeConfig, max_seq: usize) -> Self {
        let half_dim = config.dim / 2;
        let mut cos = vec![0.0f32; max_seq * half_dim];
        let mut sin = vec![0.0f32; max_seq * half_dim];

        // Compute effective frequencies based on scaling type
        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            inv_freq[i] = 1.0 / config.theta.powf(2.0 * i as f32 / config.dim as f32);
        }

        // Apply scaling
        match &config.scaling_type {
            RopeScaling::None => {}
            RopeScaling::NtkAware => {
                let factor = config.scaling_factor;
                let base =
                    config.theta * factor.powf(config.dim as f32 / (config.dim as f32 - 2.0));
                for i in 0..half_dim {
                    inv_freq[i] = 1.0 / base.powf(2.0 * i as f32 / config.dim as f32);
                }
            }
            RopeScaling::Yarn {
                temperature: _,
                beta_fast,
                beta_slow,
            } => {
                let factor = config.scaling_factor;
                for i in 0..half_dim {
                    let freq = inv_freq[i];
                    if freq < *beta_slow {
                        // No change
                    } else if freq > *beta_fast {
                        inv_freq[i] /= factor;
                    } else {
                        // Smooth interpolation
                        let t = (freq - beta_slow) / (beta_fast - beta_slow);
                        inv_freq[i] /= 1.0 + t * (factor - 1.0);
                    }
                }
            }
            RopeScaling::Linear => {
                let factor = config.scaling_factor;
                for i in 0..half_dim {
                    inv_freq[i] /= factor;
                }
            }
        }

        for pos in 0..max_seq {
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                let (s, c) = freq.sin_cos();
                cos[pos * half_dim + i] = c;
                sin[pos * half_dim + i] = s;
            }
        }

        Self { cos, sin, half_dim }
    }

    /// Apply RoPE to a vector in-place.
    /// `vec` contains concatenated heads: [head0, head1, ...] each of size `head_dim`.
    #[inline]
    pub fn apply(&self, vec: &mut [f32], pos: usize, head_dim: usize) {
        let half = head_dim / 2;
        let num_heads = vec.len() / head_dim;
        let cos_base = pos * half;
        let sin_base = pos * half;

        for h in 0..num_heads {
            let base = h * head_dim;
            for i in 0..half {
                let c = self.cos[cos_base + i];
                let s = self.sin[sin_base + i];

                let v0 = vec[base + i];
                let v1 = vec[base + i + half];
                vec[base + i] = v0 * c - v1 * s;
                vec[base + i + half] = v0 * s + v1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_cache_creation() {
        let config = RopeConfig {
            dim: 64,
            theta: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RopeScaling::None,
        };
        let cache = RopeCache::new(&config, 128);
        assert_eq!(cache.half_dim, 32);
        assert_eq!(cache.cos.len(), 128 * 32);

        // At pos=0, cos=1, sin=0
        for i in 0..32 {
            assert!((cache.cos[i] - 1.0).abs() < 1e-6);
            assert!(cache.sin[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let config = RopeConfig {
            dim: 64,
            theta: 10000.0,
            scaling_factor: 1.0,
            scaling_type: RopeScaling::None,
        };
        let cache = RopeCache::new(&config, 10);

        let orig: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut rotated = orig.clone();
        cache.apply(&mut rotated, 5, 64);

        let orig_norm: f32 = orig.iter().map(|x| x * x).sum();
        let rot_norm: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(
            (orig_norm - rot_norm).abs() < 0.01,
            "RoPE should preserve norm"
        );
    }
}
