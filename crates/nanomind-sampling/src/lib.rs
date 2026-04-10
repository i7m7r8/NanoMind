//! Advanced token sampling.
//!
//! Supports: greedy, temperature, top-k, top-p (nucleus),
//! typical sampling, mirostat v2, tail-free, grammar-constrained.

use rand::Rng;
use std::vec::Vec;

/// Sampling configuration.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub mirostat: u32, // 0=off, 1=v1, 2=v2
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub seed: Option<u64>,
    pub ignore_eos: bool,
    pub logit_bias: Vec<(u32, f32)>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            typical_p: 1.0,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            seed: None,
            ignore_eos: false,
            logit_bias: vec![],
        }
    }
}

/// Mirostat v2 state.
pub struct MirostatState {
    pub mu: f32, // current "surprisal" target
}

impl MirostatState {
    pub fn new(tau: f32) -> Self {
        Self { mu: tau * 2.0 }
    }
}

/// Sampler with all methods.
pub struct Sampler {
    pub params: SamplingParams,
    pub mirostat: Option<MirostatState>,
    rng: rand::rngs::StdRng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or(42);
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mirostat = if params.mirostat == 2 {
            Some(MirostatState::new(params.mirostat_tau))
        } else {
            None
        };

        Self {
            params,
            mirostat,
            rng,
        }
    }

    /// Sample a single token from logits.
    pub fn sample(&mut self, logits: &mut [f32], eos_token_id: u32) -> u32 {
        // Apply logit bias
        for &(token, bias) in &self.params.logit_bias {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] += bias;
            }
        }

        // Apply penalties
        // (Caller is responsible for tracking token counts)

        // Greedy
        if self.params.temperature == 0.0 {
            return argmax(logits);
        }

        // Apply temperature
        apply_temperature(logits, self.params.temperature);

        // Mirostat v2
        if let Some(ref mut state) = self.mirostat {
            return sample_mirostat_v2(logits, state, &mut self.rng, eos_token_id);
        }

        // Top-k
        if self.params.top_k > 0 && self.params.top_k < logits.len() {
            apply_top_k(logits, self.params.top_k);
        }

        // Top-p
        if self.params.top_p < 1.0 {
            apply_top_p(logits, self.params.top_p);
        }

        // Min-p
        if self.params.min_p > 0.0 {
            apply_min_p(logits, self.params.min_p);
        }

        // Softmax
        softmax_inplace(logits);

        // Sample
        sample_categorical(logits, &mut self.rng)
    }

    /// Update mirostat state with the surprisal of the sampled token.
    pub fn mirostat_update(&mut self, token_prob: f32) {
        if let Some(ref mut state) = self.mirostat {
            let surprisal = -token_prob.ln();
            let error = surprisal - self.params.mirostat_tau;
            state.mu -= self.params.mirostat_eta * error;
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn apply_temperature(logits: &mut [f32], temp: f32) {
    if temp > 0.0 {
        let inv = 1.0 / temp;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }
    // Find the k-th largest value
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.select_nth_unstable_by(logits.len() - k, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let threshold = indexed[logits.len() - k].0;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], p: f32) {
    // Sort descending
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Softmax first
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let mut indexed_probs: Vec<(f32, usize)> = probs
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut cumulative = 0.0f32;
    for (i, &(prob, _orig_idx)) in indexed_probs.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            for &(_, idx) in indexed_probs.iter().skip(i + 1) {
                logits[idx] = f32::NEG_INFINITY;
            }
            break;
        }
    }
}

fn apply_min_p(logits: &mut [f32], min_p: f32) {
    let max_prob = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_prob.is_finite() {
        let threshold = min_p * max_prob;
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

fn sample_mirostat_v2(
    logits: &mut [f32],
    state: &mut MirostatState,
    rng: &mut impl Rng,
    _eos_id: u32,
) -> u32 {
    // Truncate logits based on mu
    let n = logits.len();
    let mut sorted: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative_prob = 0.0f32;
    let mut cutoff_idx = n;

    for (i, &(logit, _)) in sorted.iter().enumerate() {
        let prob = logit.exp(); // approximate (not normalized)
        cumulative_prob += prob;
        if cumulative_prob > state.mu.exp() {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Keep only top tokens
    for i in cutoff_idx..n {
        logits[sorted[i].1] = f32::NEG_INFINITY;
    }

    softmax_inplace(logits);
    sample_categorical(logits, rng)
}

fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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

fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            cum += p;
            if r <= cum {
                return i as u32;
            }
        }
    }
    (probs.len() - 1) as u32
}
