//! KV cache with ring buffer for unlimited context.
//!
//! Supports: standard cache, sliding window eviction,
//! and ring buffer overflow for arbitrarily long conversations.

use std::vec::Vec;

/// KV cache entry for a single position.
#[derive(Clone, Debug)]
pub struct KvEntry {
    /// Key cache: [num_kv_heads * head_dim]
    pub k: Vec<f32>,
    /// Value cache: [num_kv_heads * head_dim]
    pub v: Vec<f32>,
}

/// KV Cache with automatic eviction for unlimited context.
///
/// When `max_seq` is reached, old entries are evicted using
/// a sliding window approach. The cache never grows unbounded.
pub struct KvCache {
    /// K cache: [layer][pos][kv_heads * head_dim]
    pub k: Vec<Vec<Vec<f32>>>,
    /// V cache: [layer][pos][kv_heads * head_dim]
    pub v: Vec<Vec<Vec<f32>>>,
    /// Current position (total tokens processed, never resets).
    pub pos: usize,
    /// Maximum cached positions per layer.
    pub max_seq: usize,
    /// Start of the valid window.
    pub window_start: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// KV heads per layer.
    pub kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl KvCache {
    /// Create a new KV cache.
    pub fn new(num_layers: usize, max_seq: usize, kv_heads: usize, head_dim: usize) -> Self {
        let stride = kv_heads * head_dim;
        Self {
            k: vec![vec![vec![0.0f32; stride]; max_seq]; num_layers],
            v: vec![vec![vec![0.0f32; stride]; max_seq]; num_layers],
            pos: 0,
            max_seq,
            window_start: 0,
            num_layers,
            kv_heads,
            head_dim,
        }
    }

    /// Reset for a new generation.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.window_start = 0;
    }

    /// Store K and V for the current position in all layers.
    #[inline]
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let cache_pos = self.pos % self.max_seq;
        let stride = self.kv_heads * self.head_dim;
        self.k[layer][cache_pos][..stride].copy_from_slice(&k[..stride]);
        self.v[layer][cache_pos][..stride].copy_from_slice(&v[..stride]);
    }

    /// Get K cache for a layer at a specific position.
    #[inline]
    pub fn get_k(&self, layer: usize, cache_pos: usize) -> &[f32] {
        let idx = cache_pos % self.max_seq;
        let stride = self.kv_heads * self.head_dim;
        &self.k[layer][idx][..stride]
    }

    /// Get V cache for a layer at a specific position.
    #[inline]
    pub fn get_v(&self, layer: usize, cache_pos: usize) -> &[f32] {
        let idx = cache_pos % self.max_seq;
        let stride = self.kv_heads * self.head_dim;
        &self.v[layer][idx][..stride]
    }

    /// Get the effective range of positions available in the cache.
    /// Returns (start, end) — positions in [start, end) are valid.
    pub fn valid_range(&self) -> (usize, usize) {
        let count = self.pos.min(self.max_seq);
        let start = if self.pos > self.max_seq {
            self.pos - count
        } else {
            0
        };
        (start, self.pos)
    }

    /// Advance position counter.
    pub fn advance(&mut self, n: usize) {
        self.pos += n;
        if self.pos > self.max_seq {
            self.window_start = self.pos - self.max_seq;
        }
    }

    /// Number of tokens available for attention.
    pub fn attn_context(&self) -> usize {
        self.pos.min(self.max_seq)
    }

    /// Estimate RAM usage in bytes.
    pub fn ram_bytes(&self) -> usize {
        self.num_layers * self.max_seq * self.kv_heads * self.head_dim * 4 * 2 // K + V, f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_store_retrieve() {
        let mut cache = KvCache::new(2, 64, 4, 32);
        let stride = 4 * 32;

        let k: Vec<f32> = (0..stride).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..stride).map(|i| (i + 100) as f32 * 0.1).collect();

        cache.store(0, &k, &v);
        cache.advance(1);

        let k_out = cache.get_k(0, 0);
        assert_eq!(&k_out[..stride], &k[..stride]);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut cache = KvCache::new(1, 4, 1, 8);
        let stride = 8;

        // Fill beyond max
        for i in 0..6 {
            let k = vec![i as f32; stride];
            let v = vec![i as f32 + 100.0; stride];
            cache.store(0, &k, &v);
            cache.advance(1);
        }

        // Position 5 should be at cache index 5 % 4 = 1
        let k_out = cache.get_k(0, 5);
        assert_eq!(k_out[0], 5.0);
    }

    #[test]
    fn test_valid_range() {
        let mut cache = KvCache::new(1, 10, 1, 8);
        for _ in 0..15 {
            cache.store(0, &[0.0; 8], &[0.0; 8]);
            cache.advance(1);
        }

        let (start, end) = cache.valid_range();
        assert_eq!(start, 5); // 15 - 10
        assert_eq!(end, 15);
    }
}
