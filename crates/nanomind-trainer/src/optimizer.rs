//! AdamW optimizer for training.

/// AdamW optimizer state.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    /// First moment estimates (same size as params)
    pub m: Vec<f32>,
    /// Second moment estimates (same size as params)
    pub v: Vec<f32>,
}

impl AdamW {
    pub fn new(param_count: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            step: 0,
            m: vec![0.0f32; param_count],
            v: vec![0.0f32; param_count],
        }
    }

    /// Perform one optimizer step.
    /// `params`: mutable model parameters
    /// `grads`: gradients (same shape as params, flattened)
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.step += 1;
        let n = params.len();
        let t = self.step as f32;

        // Bias correction
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            let g = grads[i];

            // Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // AdamW update with weight decay
            params[i] -=
                self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * params[i]);
        }
    }

    /// Apply gradient clipping by global norm.
    /// Returns the clipped gradients.
    pub fn clip_gradients(grads: &[f32], max_norm: f32) -> Vec<f32> {
        let mut norm = 0.0f32;
        for &g in grads {
            norm += g * g;
        }
        norm = norm.sqrt();

        if norm > max_norm {
            let scale = max_norm / norm;
            grads.iter().map(|g| g * scale).collect()
        } else {
            grads.to_vec()
        }
    }

    /// Cosine learning rate schedule with warmup.
    pub fn cosine_lr(
        step: usize,
        warmup_steps: usize,
        max_steps: usize,
        max_lr: f32,
        min_lr: f32,
    ) -> f32 {
        if step < warmup_steps {
            // Linear warmup
            max_lr * (step as f32 / warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
            let progress = progress.min(1.0);
            min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut adam = AdamW::new(10, 0.001);
        let mut params = vec![0.5f32; 10];
        let grads = vec![0.1f32; 10];

        for _ in 0..10 {
            adam.step(&mut params, &grads);
        }

        // Params should change
        assert_ne!(params, vec![0.5f32; 10]);
        // All params should be the same (symmetric update)
        for i in 1..10 {
            assert!((params[i] - params[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let grads = vec![10.0f32; 10];
        let clipped = AdamW::clip_gradients(&grads, 5.0);
        let norm: f32 = clipped.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((norm - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_lr_schedule() {
        // Warmup phase
        let lr = AdamW::cosine_lr(0, 100, 1000, 0.001, 0.0001);
        assert!((lr - 0.0).abs() < 1e-6);

        let lr = AdamW::cosine_lr(50, 100, 1000, 0.001, 0.0001);
        assert!(lr > 0.0 && lr < 0.001);

        // At max LR after warmup
        let lr = AdamW::cosine_lr(100, 100, 1000, 0.001, 0.0001);
        assert!((lr - 0.001).abs() < 1e-4);

        // At end, near min LR
        let lr = AdamW::cosine_lr(999, 100, 1000, 0.001, 0.0001);
        assert!(lr <= 0.001 && lr >= 0.0001);
    }
}
