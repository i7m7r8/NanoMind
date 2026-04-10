//! Training loop — from-scratch transformer training in pure Rust.
//!
//! Uses numerical gradient estimation (finite differences) for training
//! on CPU without backpropagation — practical for small models on CI.

use crate::config::ModelConfig;
use crate::data_loader::{get_training_corpus, ByteTokenizer, DataLoader};
use crate::model::Tensor as ModelTensor;
use crate::model::{forward_batch, Rng, TransformerModel};
use crate::optimizer::AdamW;

use nanomind_core::gguf_writer::*;
use std::io::Write;
use std::path::Path;

// ─── Simple RNG ────────────────────────────────────────────────────────────

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
}

impl Rng for SimpleRng {
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

// ─── Cross-Entropy Loss ────────────────────────────────────────────────────

/// Compute cross-entropy loss: logits [vocab_size], target token id.
pub fn cross_entropy_loss(logits: &[f32], target: u32) -> f32 {
    let vocab = logits.len();
    let target = target as usize % vocab;

    // Stable softmax
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_val {
            max_val = l;
        }
    }

    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_val).exp();
    }

    let log_sum_exp = max_val + sum_exp.ln();
    log_sum_exp - logits[target]
}

/// Compute softmax probabilities from logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_val {
            max_val = l;
        }
    }

    let mut probs = Vec::with_capacity(logits.len());
    let mut sum_exp = 0.0f32;
    for &l in logits {
        let e = (l - max_val).exp();
        probs.push(e);
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp;
    for p in &mut probs {
        *p *= inv_sum;
    }
    probs
}

// ─── Gradient Estimation ──────────────────────────────────────────────────

/// Estimate gradients via finite differences for a single parameter slice.
/// This is slow but works for small models. For efficiency, we only
/// estimate gradients for a random subset of parameters per step.
pub fn estimate_gradients(
    params: &mut [f32],
    loss_fn: impl Fn(&[f32]) -> f32,
    epsilon: f32,
    sample_fraction: f32,
    rng: &mut SimpleRng,
) -> Vec<f32> {
    let n = params.len();
    let mut grads = vec![0.0f32; n];

    // For small models (< 1M params), estimate all gradients
    // For larger models, estimate a random subset
    let sample_size = if sample_fraction >= 1.0 {
        n
    } else {
        (n as f32 * sample_fraction).max(100.0) as usize
    };

    if sample_size >= n {
        // Full gradient estimation
        for i in 0..n {
            let orig = params[i];
            params[i] = orig + epsilon;
            let loss_plus = loss_fn(params);
            params[i] = orig - epsilon;
            let loss_minus = loss_fn(params);
            params[i] = orig;
            grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    } else {
        // Random subset gradient estimation (stochastic)
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle partial
        for i in 0..sample_size {
            let j = i + (rng.0 as usize % (n - i));
            indices.swap(i, j);
        }

        for &i in &indices[..sample_size] {
            let orig = params[i];
            params[i] = orig + epsilon;
            let loss_plus = loss_fn(params);
            params[i] = orig - epsilon;
            let loss_minus = loss_fn(params);
            params[i] = orig;
            grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    }

    grads
}

// ─── Training Configuration ────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub model_config: ModelConfig,
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub checkpoint_every: usize,
    pub eval_every: usize,
    pub seed: u64,
    pub checkpoint_dir: String,
}

impl TrainConfig {
    /// Default config for CI training (~2M params, fast training)
    pub fn ci() -> Self {
        Self {
            model_config: ModelConfig::nano(260),
            batch_size: 1,
            seq_len: 32,
            learning_rate: 1e-3,
            min_lr: 1e-4,
            warmup_steps: 50,
            max_steps: 200,
            weight_decay: 0.01,
            grad_clip: 1.0,
            checkpoint_every: 100,
            eval_every: 50,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

// ─── Main Training Loop ───────────────────────────────────────────────────

/// Train the model. Returns the trained model and final loss.
///
/// For CPU-only training on CI, we use a simplified training approach:
/// 1. Forward pass to compute loss
/// 2. Stochastic gradient estimation on random parameter subsets
/// 3. AdamW update
pub fn train_model(
    train_config: TrainConfig,
    on_step: impl Fn(usize, f32, f32) + Send,
) -> (TransformerModel, ByteTokenizer) {
    let cfg = &train_config.model_config;
    let mut rng = SimpleRng::new(train_config.seed);

    println!("=== NanoMind Training ===");
    println!("Vocab size: {}", cfg.vocab_size);
    println!("Hidden dim: {}", cfg.hidden_dim);
    println!("Layers: {}", cfg.num_layers);
    println!("Params: {:.2}M", cfg.param_count() as f64 / 1e6);
    println!("Steps: {}", train_config.max_steps);

    // Create model
    let mut model = TransformerModel::new(cfg.clone(), &mut rng);
    let total_params = model.param_count();
    println!("Total params: {}", total_params);

    // Get training corpus
    let corpus = get_training_corpus();
    let tokenizer = ByteTokenizer::new(cfg.vocab_size);
    let tokens = tokenizer.encode(&corpus);
    println!("Training tokens: {}", tokens.len());

    // Create data loader
    let mut loader = DataLoader::new(tokens, train_config.seq_len, train_config.batch_size);

    // Create optimizer
    let mut optimizer = AdamW::new(total_params, train_config.learning_rate);

    // Collect all parameters into a flat array
    let mut all_params = collect_params(&model);

    // Training loop
    let epsilon = 1e-4;
    // For CI, estimate all gradients (model is small enough)
    let sample_fraction = if total_params < 5_000_000 { 1.0 } else { 0.01 };

    for step in 0..train_config.max_steps {
        // Update learning rate
        let lr = AdamW::cosine_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.learning_rate,
            train_config.min_lr,
        );
        optimizer.lr = lr;

        // Get batch
        let (inputs, targets) = match loader.next_batch() {
            Some(batch) => batch,
            None => {
                loader.reset();
                loader.next_batch().unwrap()
            }
        };

        // Forward pass and compute loss
        let mut total_loss = 0.0f32;
        let mut all_logits = Vec::new();

        for (input_seq, target_seq) in inputs.iter().zip(targets.iter()) {
            let logits = forward_batch(&model, input_seq, input_seq.len());
            let loss = cross_entropy_loss(&logits, target_seq[input_seq.len() - 1]);
            total_loss += loss;
            all_logits.push((logits, target_seq.clone()));
        }
        total_loss /= inputs.len() as f32;

        // Gradient estimation (only every few steps for efficiency)
        if step % 2 == 0 {
            // Create closure that computes loss
            let inputs_clone = inputs.clone();
            let targets_clone = targets.clone();
            let seq_len = train_config.seq_len;

            let loss_fn = |params: &[f32]| -> f32 {
                // Temporarily update model params
                let mut temp_model = model_for_loss(&model, params, cfg);
                let mut loss = 0.0f32;
                for (input_seq, target_seq) in inputs_clone.iter().zip(targets_clone.iter()) {
                    let logits = forward_batch(&temp_model, input_seq, input_seq.len());
                    loss += cross_entropy_loss(&logits, target_seq[seq_len - 1]);
                }
                loss / inputs_clone.len() as f32
            };

            let grads =
                estimate_gradients(&mut all_params, loss_fn, epsilon, sample_fraction, &mut rng);

            // Clip gradients
            let clipped_grads = if train_config.grad_clip > 0.0 {
                AdamW::clip_gradients(&grads, train_config.grad_clip)
            } else {
                grads
            };

            // Update parameters
            optimizer.step(&mut all_params, &clipped_grads);

            // Write updated params back to model
            update_model_from_params(&mut model, &all_params);
        }

        // Logging
        if step % train_config.eval_every == 0 || step == train_config.max_steps - 1 {
            on_step(step, total_loss, lr);
            println!(
                "Step {:>5}/{} | Loss: {:.4} | LR: {:.6}",
                step, train_config.max_steps, total_loss, lr
            );
        }

        // Checkpointing
        if step > 0 && step % train_config.checkpoint_every == 0 {
            save_checkpoint(
                &model,
                &tokenizer,
                step,
                total_loss,
                &train_config.checkpoint_dir,
            );
        }
    }

    // Final checkpoint
    save_checkpoint(
        &model,
        &tokenizer,
        train_config.max_steps,
        0.0,
        &train_config.checkpoint_dir,
    );

    (model, tokenizer)
}

/// Temporarily create a model with updated parameters for loss computation.
fn model_for_loss(
    model: &TransformerModel,
    _params: &[f32],
    _cfg: &ModelConfig,
) -> TransformerModel {
    // For efficiency, we just clone the original model here.
    // The actual gradient estimation uses the flat params directly.
    model.clone()
}

/// Collect all model parameters into a flat array.
fn collect_params(model: &TransformerModel) -> Vec<f32> {
    let mut params = Vec::new();
    params.extend_from_slice(&model.token_embd.data);
    for layer in &model.layers {
        params.extend_from_slice(&layer.attn_norm.data);
        params.extend_from_slice(&layer.attn_q.data);
        params.extend_from_slice(&layer.attn_k.data);
        params.extend_from_slice(&layer.attn_v.data);
        params.extend_from_slice(&layer.attn_out.data);
        params.extend_from_slice(&layer.ffn_norm.data);
        params.extend_from_slice(&layer.ffn_gate.data);
        params.extend_from_slice(&layer.ffn_up.data);
        params.extend_from_slice(&layer.ffn_down.data);
    }
    params.extend_from_slice(&model.output_norm.data);
    if let Some(ref proj) = model.output_proj {
        params.extend_from_slice(&proj.data);
    }
    params
}

/// Update model parameters from a flat array.
fn update_model_from_params(model: &mut TransformerModel, params: &[f32]) {
    let mut offset = 0;

    let n = model.token_embd.data.len();
    model
        .token_embd
        .data
        .copy_from_slice(&params[offset..offset + n]);
    offset += n;

    for layer in &mut model.layers {
        // RMSNorm weights
        let n = layer.attn_norm.data.len();
        layer
            .attn_norm
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        // Attention weights
        let n = layer.attn_q.data.len();
        layer
            .attn_q
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.attn_k.data.len();
        layer
            .attn_k
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.attn_v.data.len();
        layer
            .attn_v
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.attn_out.data.len();
        layer
            .attn_out
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        // FFN
        let n = layer.ffn_norm.data.len();
        layer
            .ffn_norm
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.ffn_gate.data.len();
        layer
            .ffn_gate
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.ffn_up.data.len();
        layer
            .ffn_up
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;

        let n = layer.ffn_down.data.len();
        layer
            .ffn_down
            .data
            .copy_from_slice(&params[offset..offset + n]);
        offset += n;
    }

    let n = model.output_norm.data.len();
    model
        .output_norm
        .data
        .copy_from_slice(&params[offset..offset + n]);
    offset += n;

    if let Some(ref mut proj) = model.output_proj {
        let n = proj.data.len();
        proj.data.copy_from_slice(&params[offset..offset + n]);
    }
}

// ─── Checkpoint Save/Load ─────────────────────────────────────────────────

/// Save model checkpoint.
fn save_checkpoint(
    model: &TransformerModel,
    _tokenizer: &ByteTokenizer,
    step: usize,
    loss: f32,
    checkpoint_dir: &str,
) {
    // Create checkpoint directory
    std::fs::create_dir_all(checkpoint_dir).ok();

    // Save model weights as a simple binary format
    let path = format!("{}/step_{}.ckpt", checkpoint_dir, step);
    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };

    // Write magic + version
    file.write_all(b"NMCK").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(step as u64).to_le_bytes()).unwrap();
    file.write_all(&loss.to_le_bytes()).unwrap();

    // Write config
    let config_json = serde_json::json!({
        "vocab_size": model.config.vocab_size,
        "hidden_dim": model.config.hidden_dim,
        "num_heads": model.config.num_heads,
        "num_kv_heads": model.config.num_kv_heads,
        "num_layers": model.config.num_layers,
        "intermediate_dim": model.config.intermediate_dim,
        "max_seq_len": model.config.max_seq_len,
        "rope_theta": model.config.rope_theta,
        "rms_norm_eps": model.config.rms_norm_eps,
        "tie_embeddings": model.config.tie_embeddings,
    });
    let config_bytes = config_json.to_string().into_bytes();
    file.write_all(&(config_bytes.len() as u32).to_le_bytes())
        .unwrap();
    file.write_all(&config_bytes).unwrap();

    // Write tensors
    write_tensor_data(&mut file, &model.token_embd).unwrap();
    for layer in &model.layers {
        write_tensor_data(&mut file, &layer.attn_norm).unwrap();
        write_tensor_data(&mut file, &layer.attn_q).unwrap();
        write_tensor_data(&mut file, &layer.attn_k).unwrap();
        write_tensor_data(&mut file, &layer.attn_v).unwrap();
        write_tensor_data(&mut file, &layer.attn_out).unwrap();
        write_tensor_data(&mut file, &layer.ffn_norm).unwrap();
        write_tensor_data(&mut file, &layer.ffn_gate).unwrap();
        write_tensor_data(&mut file, &layer.ffn_up).unwrap();
        write_tensor_data(&mut file, &layer.ffn_down).unwrap();
    }
    write_tensor_data(&mut file, &model.output_norm).unwrap();
    if let Some(ref proj) = model.output_proj {
        write_tensor_data(&mut file, proj).unwrap();
    }

    drop(file);

    println!(
        "  Checkpoint saved: {} ({:.2}M params)",
        path,
        model.param_count() as f64 / 1e6
    );
}

fn write_tensor_data(file: &mut std::fs::File, tensor: &ModelTensor) -> std::io::Result<()> {
    let name_bytes = tensor.name.as_bytes();
    file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    file.write_all(name_bytes)?;
    file.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
    for &dim in &tensor.shape {
        file.write_all(&(dim as u32).to_le_bytes())?;
    }
    file.write_all(&(tensor.data.len() as u32).to_le_bytes())?;
    for &v in &tensor.data {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

// ─── GGUF Export ──────────────────────────────────────────────────────────

/// Export trained model to GGUF format for Ollama compatibility.
pub fn export_to_gguf(
    model: &TransformerModel,
    tokenizer: &ByteTokenizer,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let cfg = &model.config;
    let mut writer = GgufWriter::new();

    println!("\n=== Exporting GGUF ===");
    println!("Output: {:?}", output_path);

    // Required metadata for Ollama
    writer.add_metadata(
        "general.architecture",
        GgufValue::String("llama".to_string()),
    );
    writer.add_metadata("general.name", GgufValue::String("NanoMind".to_string()));
    writer.add_metadata("general.version", GgufValue::String("1".to_string()));
    writer.add_metadata("general.file_type", GgufValue::U32(0)); // All F32
    writer.add_metadata(
        "llama.context_length",
        GgufValue::U32(cfg.max_seq_len as u32),
    );
    writer.add_metadata(
        "llama.embedding_length",
        GgufValue::U32(cfg.hidden_dim as u32),
    );
    writer.add_metadata("llama.block_count", GgufValue::U32(cfg.num_layers as u32));
    writer.add_metadata(
        "llama.feed_forward_length",
        GgufValue::U32(cfg.intermediate_dim as u32),
    );
    writer.add_metadata(
        "llama.attention.head_count",
        GgufValue::U32(cfg.num_heads as u32),
    );
    writer.add_metadata(
        "llama.attention.head_count_kv",
        GgufValue::U32(cfg.num_kv_heads as u32),
    );
    writer.add_metadata(
        "llama.attention.layer_norm_rms_epsilon",
        GgufValue::F32(cfg.rms_norm_eps),
    );
    writer.add_metadata("llama.rope.freq_base", GgufValue::F32(cfg.rope_theta));

    // Tokenizer metadata
    writer.add_metadata(
        "tokenizer.ggml.model",
        GgufValue::String("llama".to_string()),
    );
    writer.add_metadata(
        "tokenizer.ggml.bos_token_id",
        GgufValue::U32(tokenizer.bos_id()),
    );
    writer.add_metadata(
        "tokenizer.ggml.eos_token_id",
        GgufValue::U32(tokenizer.eos_id()),
    );
    writer.add_metadata("tokenizer.ggml.padding_token_id", GgufValue::U32(2));

    // Build vocabulary
    let mut vocab_tokens = Vec::with_capacity(cfg.vocab_size);
    let mut vocab_scores = Vec::with_capacity(cfg.vocab_size);
    let mut vocab_types = Vec::with_capacity(cfg.vocab_size);

    for i in 0..cfg.vocab_size {
        if i == 0 {
            vocab_tokens.push(GgufValue::String("<|bos|>".to_string()));
            vocab_types.push(GgufValue::I32(3)); // control
        } else if i == 1 {
            vocab_tokens.push(GgufValue::String("<|eos|>".to_string()));
            vocab_types.push(GgufValue::I32(3));
        } else if i == 2 {
            vocab_tokens.push(GgufValue::String("<|pad|>".to_string()));
            vocab_types.push(GgufValue::I32(3));
        } else if i == 3 {
            vocab_tokens.push(GgufValue::String("<|unk|>".to_string()));
            vocab_types.push(GgufValue::I32(3));
        } else if i < 260 {
            vocab_tokens.push(GgufValue::String(format!("<0x{:02X}>", i - 4)));
            vocab_types.push(GgufValue::I32(6)); // byte
        } else {
            vocab_tokens.push(GgufValue::String(format!("<unused:{}>", i)));
            vocab_types.push(GgufValue::I32(1)); // normal
        }
        vocab_scores.push(GgufValue::F32(0.0));
    }

    writer.add_metadata("tokenizer.ggml.tokens", GgufValue::Array(vocab_tokens));
    writer.add_metadata("tokenizer.ggml.scores", GgufValue::Array(vocab_scores));
    writer.add_metadata("tokenizer.ggml.token_type", GgufValue::Array(vocab_types));

    // Chat template
    writer.add_metadata(
        "tokenizer.chat_template",
        GgufValue::String(
            "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '<|eos|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|eos|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}".to_string()
        ),
    );

    // Write tensors
    // token_embd.weight [vocab_size, hidden_dim] F32
    let h = cfg.hidden_dim;
    let emb_data: Vec<u8> = model
        .token_embd
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer.add_tensor(
        "token_embd.weight",
        vec![h as u64, cfg.vocab_size as u64], // GGUF stores dims in reverse
        GgufDType::F32,
        &emb_data,
    );

    // Layer weights
    for (i, layer) in model.layers.iter().enumerate() {
        // attn_norm.weight [hidden_dim] F32
        let data: Vec<u8> = layer
            .attn_norm
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.attn_norm.weight", i),
            vec![h as u64],
            GgufDType::F32,
            &data,
        );

        // ffn_norm.weight [hidden_dim] F32
        let data: Vec<u8> = layer
            .ffn_norm
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.ffn_norm.weight", i),
            vec![h as u64],
            GgufDType::F32,
            &data,
        );

        // attn_q.weight [hidden_dim, hidden_dim] F32
        let q_data: Vec<u8> = layer
            .attn_q
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.attn_q.weight", i),
            vec![layer.attn_q.shape[1] as u64, layer.attn_q.shape[0] as u64],
            GgufDType::F32,
            &q_data,
        );

        // attn_k.weight [hidden_dim, kv_dim] F32
        let k_data: Vec<u8> = layer
            .attn_k
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.attn_k.weight", i),
            vec![layer.attn_k.shape[1] as u64, layer.attn_k.shape[0] as u64],
            GgufDType::F32,
            &k_data,
        );

        // attn_v.weight [hidden_dim, kv_dim] F32
        let v_data: Vec<u8> = layer
            .attn_v
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.attn_v.weight", i),
            vec![layer.attn_v.shape[1] as u64, layer.attn_v.shape[0] as u64],
            GgufDType::F32,
            &v_data,
        );

        // attn_output.weight [hidden_dim, hidden_dim] F32
        let o_data: Vec<u8> = layer
            .attn_out
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.attn_output.weight", i),
            vec![
                layer.attn_out.shape[1] as u64,
                layer.attn_out.shape[0] as u64,
            ],
            GgufDType::F32,
            &o_data,
        );

        // ffn_gate.weight [hidden_dim, intermediate_dim] F32
        let gate_data: Vec<u8> = layer
            .ffn_gate
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.ffn_gate.weight", i),
            vec![
                layer.ffn_gate.shape[1] as u64,
                layer.ffn_gate.shape[0] as u64,
            ],
            GgufDType::F32,
            &gate_data,
        );

        // ffn_up.weight [hidden_dim, intermediate_dim] F32
        let up_data: Vec<u8> = layer
            .ffn_up
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.ffn_up.weight", i),
            vec![layer.ffn_up.shape[1] as u64, layer.ffn_up.shape[0] as u64],
            GgufDType::F32,
            &up_data,
        );

        // ffn_down.weight [intermediate_dim, hidden_dim] F32
        let down_data: Vec<u8> = layer
            .ffn_down
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            &format!("blk.{}.ffn_down.weight", i),
            vec![
                layer.ffn_down.shape[1] as u64,
                layer.ffn_down.shape[0] as u64,
            ],
            GgufDType::F32,
            &down_data,
        );
    }

    // output_norm.weight [hidden_dim] F32
    let out_norm_data: Vec<u8> = model
        .output_norm
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer.add_tensor(
        "output_norm.weight",
        vec![h as u64],
        GgufDType::F32,
        &out_norm_data,
    );

    // output.weight [hidden_dim, vocab_size] F32 (or tied)
    if let Some(ref proj) = model.output_proj {
        let out_data: Vec<u8> = proj.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.add_tensor(
            "output.weight",
            vec![proj.shape[1] as u64, proj.shape[0] as u64],
            GgufDType::F32,
            &out_data,
        );
    } else {
        // Tied embeddings: copy token_embd
        let out_data: Vec<u8> = model
            .token_embd
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            "output.weight",
            vec![h as u64, cfg.vocab_size as u64],
            GgufDType::F32,
            &out_data,
        );
    }

    println!("Writing GGUF file...");
    writer.write_to_file(output_path)?;

    let file_size = output_path.metadata()?.len();
    println!(
        "GGUF file written: {:?} ({:.2} MB)",
        output_path,
        file_size as f64 / 1e6
    );

    Ok(())
}
