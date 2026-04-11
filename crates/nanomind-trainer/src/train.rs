//! Training loop — from-scratch transformer training in pure Rust.
//!
//! Uses proper backpropagation (reverse-mode autodiff) for efficient training.

use crate::autodiff::Tape;
use crate::config::ModelConfig;
use crate::data_loader::{get_training_corpus, ByteTokenizer};
use crate::model::TransformerModel;
use crate::optimizer::AdamW;

use nanomind_core::gguf_writer::*;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

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
    pub corpus_path: Option<String>,
}

impl TrainConfig {
    /// Default config for CI training (~50K params, fast training)
    pub fn ci() -> Self {
        Self {
            model_config: ModelConfig::nano(260),
            batch_size: 1,
            seq_len: 32,
            learning_rate: 1e-3,
            min_lr: 1e-4,
            warmup_steps: 100,
            max_steps: 500,
            weight_decay: 0.01,
            grad_clip: 1.0,
            checkpoint_every: 250,
            eval_every: 50,
            seed: 42,
            checkpoint_dir: "checkpoints".to_string(),
            corpus_path: None,
        }
    }
}

// ─── Training Engine with proper gradient tracking ─────────────────────────

struct ParamInfo {
    offset: usize, // offset in flat param array
    len: usize,    // length of this parameter
}

struct TrainingEngine {
    model: TransformerModel,
    tokenizer: ByteTokenizer,
    optimizer: AdamW,
    all_params: Vec<f32>,
    param_info: Vec<ParamInfo>,
}

impl TrainingEngine {
    fn new(model: TransformerModel, tokenizer: ByteTokenizer, lr: f32) -> Self {
        let total_params = model.param_count();
        let optimizer = AdamW::new(total_params, lr);
        let (all_params, param_info) = collect_params_with_info(&model);

        Self {
            model,
            tokenizer,
            optimizer,
            all_params,
            param_info,
        }
    }

    /// One training step: forward + backward + optimizer update.
    /// Returns the loss.
    fn step(&mut self, tokens: &[u32]) -> f32 {
        let seq_len = tokens.len() - 1;
        let cfg = &self.model.config;
        let h = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        let n_heads = cfg.num_heads;
        let n_kv_heads = cfg.num_kv_heads;
        let ffn = cfg.intermediate_dim;

        let mut tape = Tape::new();
        let mut var_to_param_idx: HashMap<usize, usize> = HashMap::new();
        let mut next_param_idx = 0;

        // Helper to register a parameter tensor with the gradient tracker
        let mut reg_param =
            |data: Vec<f32>, shape: Vec<usize>, tape: &mut Tape, next: &mut usize| -> usize {
                let pid = *next;
                let vid = tape.var(data, shape);
                var_to_param_idx.insert(vid, pid);
                *next += 1;
                vid
            };

        // ── Embedding ──
        let emb_id = reg_param(
            self.model.token_embd.data.clone(),
            vec![vocab, h],
            &mut tape,
            &mut next_param_idx,
        );

        // Lookup embeddings for input tokens
        let mut hidden = vec![0.0f32; seq_len * h];
        for (i, &tok) in tokens.iter().enumerate().take(seq_len) {
            let tok = tok as usize % vocab;
            let start = i * h;
            let emb_start = tok * h;
            hidden[start..start + h]
                .copy_from_slice(&self.model.token_embd.data[emb_start..emb_start + h]);
        }

        // ── Transformer layers ──
        let mut layer_hidden = hidden.clone();

        for layer in &self.model.layers {
            let residual = layer_hidden.clone();

            // ── RMSNorm before attention ──
            let norm_x_id = tape.var(layer_hidden.clone(), vec![seq_len, h]);
            let norm_w_id = reg_param(
                layer.attn_norm.data.clone(),
                vec![h],
                &mut tape,
                &mut next_param_idx,
            );
            let normed_id = tape.forward_rms_norm(norm_x_id, norm_w_id, cfg.rms_norm_eps);

            // ── Q projection ──
            let q_w_id = reg_param(
                layer.attn_q.data.clone(),
                vec![h, n_heads * head_dim],
                &mut tape,
                &mut next_param_idx,
            );
            let q_id = tape.forward_matmul(normed_id, q_w_id, seq_len, h, n_heads * head_dim);

            // ── K projection ──
            let k_w_id = reg_param(
                layer.attn_k.data.clone(),
                vec![h, kv_dim],
                &mut tape,
                &mut next_param_idx,
            );
            let k_id = tape.forward_matmul(normed_id, k_w_id, seq_len, h, kv_dim);

            // ── V projection ──
            let v_w_id = reg_param(
                layer.attn_v.data.clone(),
                vec![h, kv_dim],
                &mut tape,
                &mut next_param_idx,
            );
            let v_id = tape.forward_matmul(normed_id, v_w_id, seq_len, h, kv_dim);

            // ── RoPE ──
            let (q_rope_id, k_rope_id) =
                tape.forward_rope(q_id, k_id, 0, head_dim, cfg.rope_theta, n_heads, n_kv_heads);

            // ── Attention ──
            let attn_id = tape.forward_attention(
                q_rope_id, k_rope_id, v_id, seq_len, n_heads, n_kv_heads, head_dim,
            );

            // ── Output projection ──
            let o_w_id = reg_param(
                layer.attn_out.data.clone(),
                vec![n_heads * head_dim, h],
                &mut tape,
                &mut next_param_idx,
            );
            let attn_out_id = tape.forward_matmul(attn_id, o_w_id, seq_len, n_heads * head_dim, h);

            // ── Residual ──
            let resid_id = tape.var(residual.clone(), vec![seq_len, h]);
            let y_add_id = tape.forward_add_residual(attn_out_id, resid_id);

            // ── FFN ──
            let hidden_for_ffn = tape.vars[y_add_id].value.clone();
            layer_hidden = hidden_for_ffn.clone();

            // RMSNorm before FFN
            let ffn_norm_w_id = reg_param(
                layer.ffn_norm.data.clone(),
                vec![h],
                &mut tape,
                &mut next_param_idx,
            );
            let y_ffn_norm_id = tape.forward_rms_norm(y_add_id, ffn_norm_w_id, cfg.rms_norm_eps);

            // Gate projection
            let gate_w_id = reg_param(
                layer.ffn_gate.data.clone(),
                vec![h, ffn],
                &mut tape,
                &mut next_param_idx,
            );
            let gate_id = tape.forward_matmul(y_ffn_norm_id, gate_w_id, seq_len, h, ffn);

            // SiLU activation
            let gate_act_id = tape.forward_silu(gate_id);

            // Up projection
            let up_w_id = reg_param(
                layer.ffn_up.data.clone(),
                vec![h, ffn],
                &mut tape,
                &mut next_param_idx,
            );
            let up_id = tape.forward_matmul(y_ffn_norm_id, up_w_id, seq_len, h, ffn);

            // Element-wise multiply (gate * up)
            let gate_act = &tape.vars[gate_act_id].value;
            let up = &tape.vars[up_id].value;
            let activated: Vec<f32> = gate_act
                .iter()
                .zip(up.iter())
                .map(|(&a, &b)| a * b)
                .collect();
            let activated_id = tape.var(activated, vec![seq_len, ffn]);

            // Down projection
            let down_w_id = reg_param(
                layer.ffn_down.data.clone(),
                vec![ffn, h],
                &mut tape,
                &mut next_param_idx,
            );
            let ffn_out_id = tape.forward_matmul(activated_id, down_w_id, seq_len, ffn, h);

            // Residual
            let resid2_id = tape.var(layer_hidden.clone(), vec![seq_len, h]);
            let y_ffn_add_id = tape.forward_add_residual(ffn_out_id, resid2_id);

            layer_hidden = tape.vars[y_ffn_add_id].value.clone();
        }

        // ── Final RMSNorm ──
        let hidden_for_norm = layer_hidden.clone();
        let hidden_norm_id = tape.var(hidden_for_norm.clone(), vec![seq_len, h]);
        let out_norm_w_id = reg_param(
            self.model.output_norm.data.clone(),
            vec![h],
            &mut tape,
            &mut next_param_idx,
        );
        let y_final_norm_id =
            tape.forward_rms_norm(hidden_norm_id, out_norm_w_id, cfg.rms_norm_eps);

        // ── Output projection (last token) ──
        let last_hidden = &layer_hidden[(seq_len - 1) * h..seq_len * h];
        let last_hidden_id = tape.var(last_hidden.to_vec(), vec![1, h]);

        let mut logits: Vec<f32>;
        if let Some(ref proj) = self.model.output_proj {
            let out_w_id = reg_param(
                proj.data.clone(),
                vec![h, vocab],
                &mut tape,
                &mut next_param_idx,
            );
            let logits_id = tape.forward_matmul(last_hidden_id, out_w_id, 1, h, vocab);
            logits = tape.vars[logits_id].value.clone();
        } else {
            // Tied embeddings: use token_embd transposed
            let mut l = vec![0.0f32; vocab];
            for v in 0..vocab {
                let emb_start = v * h;
                let mut sum = 0.0f32;
                for d in 0..h {
                    sum += last_hidden[d] * self.model.token_embd.data[emb_start + d];
                }
                l[v] = sum * (h as f32).sqrt();
            }
            logits = l;
        }

        // ── Loss ──
        let target = tokens[seq_len];
        let logits_id = tape.var(logits.clone(), vec![vocab]);
        let loss_id = tape.forward_softmax_cross_entropy(logits_id, target);
        let loss = tape.vars[loss_id].value[0];

        // ── Backward pass ──
        let grads = tape.backward();

        // ── Map gradients back to flat param array ──
        let mut flat_grads = vec![0.0f32; self.all_params.len()];
        for (var_id, grad) in &grads {
            if let Some(&pidx) = var_to_param_idx.get(var_id) {
                if pidx < self.param_info.len() {
                    let pi = &self.param_info[pidx];
                    let copy_len = pi.len.min(grad.len());
                    for i in 0..copy_len {
                        if pi.offset + i < flat_grads.len() {
                            flat_grads[pi.offset + i] += grad[i];
                        }
                    }
                }
            }
        }

        // ── Clip gradients ──
        let clipped_grads = AdamW::clip_gradients(&flat_grads, 1.0);

        // ── Update parameters ──
        self.optimizer.step(&mut self.all_params, &clipped_grads);

        // ── Write updated params back to model ──
        update_model_from_params(&mut self.model, &self.all_params);

        loss
    }
}

// ─── Parameter collection with proper tracking ────────────────────────────

fn collect_params_with_info(model: &TransformerModel) -> (Vec<f32>, Vec<ParamInfo>) {
    let mut params = Vec::new();
    let mut info = Vec::new();
    let mut offset = 0;

    let add_param =
        |params: &mut Vec<f32>, info: &mut Vec<ParamInfo>, offset: &mut usize, data: &[f32]| {
            let start = *offset;
            let len = data.len();
            params.extend_from_slice(data);
            *offset += len;
            info.push(ParamInfo { offset: start, len });
        };

    add_param(&mut params, &mut info, &mut offset, &model.token_embd.data);

    for layer in &model.layers {
        add_param(&mut params, &mut info, &mut offset, &layer.attn_norm.data);
        add_param(&mut params, &mut info, &mut offset, &layer.attn_q.data);
        add_param(&mut params, &mut info, &mut offset, &layer.attn_k.data);
        add_param(&mut params, &mut info, &mut offset, &layer.attn_v.data);
        add_param(&mut params, &mut info, &mut offset, &layer.attn_out.data);
        add_param(&mut params, &mut info, &mut offset, &layer.ffn_norm.data);
        add_param(&mut params, &mut info, &mut offset, &layer.ffn_gate.data);
        add_param(&mut params, &mut info, &mut offset, &layer.ffn_up.data);
        add_param(&mut params, &mut info, &mut offset, &layer.ffn_down.data);
    }

    add_param(&mut params, &mut info, &mut offset, &model.output_norm.data);
    if let Some(ref proj) = model.output_proj {
        add_param(&mut params, &mut info, &mut offset, &proj.data);
    }

    (params, info)
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
        for tensor in [
            &mut layer.attn_norm.data,
            &mut layer.attn_q.data,
            &mut layer.attn_k.data,
            &mut layer.attn_v.data,
            &mut layer.attn_out.data,
            &mut layer.ffn_norm.data,
            &mut layer.ffn_gate.data,
            &mut layer.ffn_up.data,
            &mut layer.ffn_down.data,
        ] {
            let n = tensor.len();
            tensor.copy_from_slice(&params[offset..offset + n]);
            offset += n;
        }
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

// ─── Simple RNG ────────────────────────────────────────────────────────────

pub(crate) struct SimpleRng(u64);

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

impl crate::model::Rng for SimpleRng {
    fn f32(&mut self) -> f32 {
        let val = self.next();
        ((val >> 33) as f32) / (u32::MAX as f32)
    }
}

// ─── Main Training Loop ───────────────────────────────────────────────────

/// Train the model. Returns the trained model and tokenizer.
pub fn train_model(
    train_config: TrainConfig,
    on_step: impl Fn(usize, f32, f32),
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
    let model = TransformerModel::new(cfg.clone(), &mut rng);
    let total_params = model.param_count();
    println!("Total params: {}", total_params);

    // Load corpus
    let corpus = match &train_config.corpus_path {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) if text.len() > 10_000 => {
                println!("Loaded corpus: {} bytes from {}", text.len(), path);
                text
            }
            Ok(_) => {
                eprintln!("Corpus at {} too small, using built-in", path);
                get_training_corpus()
            }
            Err(e) => {
                eprintln!("Cannot load corpus {}: {}. Using built-in.", path, e);
                get_training_corpus()
            }
        },
        None => get_training_corpus(),
    };

    let tokenizer = ByteTokenizer::new(cfg.vocab_size);
    let tokens = tokenizer.encode(&corpus);
    println!("Training tokens: {}", tokens.len());

    // Create training engine
    let mut engine = TrainingEngine::new(model, tokenizer, train_config.learning_rate);

    // Training loop
    for step in 0..train_config.max_steps {
        // Update learning rate
        let lr = AdamW::cosine_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.learning_rate,
            train_config.min_lr,
        );
        engine.optimizer.lr = lr;

        // Get batch
        let batch_tokens = get_training_batch(&tokens, train_config.seq_len, &mut rng);
        let loss = engine.step(&batch_tokens);

        // Logging
        if step % train_config.eval_every == 0 || step == train_config.max_steps - 1 {
            on_step(step, loss, lr);
            println!(
                "Step {:>5}/{} | Loss: {:.4} | LR: {:.6}",
                step, train_config.max_steps, loss, lr
            );
        }

        // Checkpointing
        if step > 0 && step % train_config.checkpoint_every == 0 {
            save_checkpoint(&engine.model, step, loss, &train_config.checkpoint_dir);
        }
    }

    // Final checkpoint
    save_checkpoint(
        &engine.model,
        train_config.max_steps,
        0.0,
        &train_config.checkpoint_dir,
    );

    (engine.model, engine.tokenizer)
}

/// Get a training batch from the corpus.
fn get_training_batch(tokens: &[u32], seq_len: usize, rng: &mut SimpleRng) -> Vec<u32> {
    let n_tokens = tokens.len();
    let start = (rng.next() as usize) % n_tokens.max(seq_len + 1);
    let end = (start + seq_len + 1).min(n_tokens);
    if end - start < seq_len + 1 {
        let mut batch = tokens[start..].to_vec();
        let needed = seq_len + 1 - batch.len();
        batch.extend_from_slice(&tokens[..needed]);
        batch
    } else {
        tokens[start..start + seq_len + 1].to_vec()
    }
}

// ─── Checkpoint Save/Load ─────────────────────────────────────────────────

fn save_checkpoint(model: &TransformerModel, step: usize, loss: f32, checkpoint_dir: &str) {
    std::fs::create_dir_all(checkpoint_dir).ok();

    let path = format!("{}/step_{}.ckpt", checkpoint_dir, step);
    let mut file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(_) => return,
    };

    file.write_all(b"NMCK").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(step as u64).to_le_bytes()).unwrap();
    file.write_all(&loss.to_le_bytes()).unwrap();

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

    println!(
        "  Checkpoint saved: {} ({:.2}M params)",
        path,
        model.param_count() as f64 / 1e6
    );
}

use crate::model::Tensor as ModelTensor;

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

pub fn export_to_gguf(
    model: &TransformerModel,
    tokenizer: &ByteTokenizer,
    output_path: &Path,
) -> std::io::Result<()> {
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
    writer.add_metadata("general.file_type", GgufValue::U32(0));
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
            vocab_types.push(GgufValue::I32(3));
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
            vocab_types.push(GgufValue::I32(6));
        } else {
            vocab_tokens.push(GgufValue::String(format!("<unused:{}>", i)));
            vocab_types.push(GgufValue::I32(1));
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
    let h = cfg.hidden_dim;
    let emb_data: Vec<u8> = model
        .token_embd
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer.add_tensor(
        "token_embd.weight",
        vec![h as u64, cfg.vocab_size as u64],
        GgufDType::F32,
        &emb_data,
    );

    for (i, layer) in model.layers.iter().enumerate() {
        let norm_tensors = [
            ("attn_norm.weight", &layer.attn_norm),
            ("ffn_norm.weight", &layer.ffn_norm),
        ];
        for (name, tensor) in &norm_tensors {
            let data: Vec<u8> = tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.add_tensor(
                &format!("blk.{}.{}", i, name),
                vec![h as u64],
                GgufDType::F32,
                &data,
            );
        }

        let proj_tensors = [
            ("attn_q.weight", &layer.attn_q),
            ("attn_k.weight", &layer.attn_k),
            ("attn_v.weight", &layer.attn_v),
            ("attn_output.weight", &layer.attn_out),
            ("ffn_gate.weight", &layer.ffn_gate),
            ("ffn_up.weight", &layer.ffn_up),
            ("ffn_down.weight", &layer.ffn_down),
        ];
        for (name, tensor) in &proj_tensors {
            let data: Vec<u8> = tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.add_tensor(
                &format!("blk.{}.{}", i, name),
                vec![tensor.shape[1] as u64, tensor.shape[0] as u64],
                GgufDType::F32,
                &data,
            );
        }
    }

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

    if let Some(ref proj) = model.output_proj {
        let out_data: Vec<u8> = proj.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.add_tensor(
            "output.weight",
            vec![proj.shape[1] as u64, proj.shape[0] as u64],
            GgufDType::F32,
            &out_data,
        );
    } else {
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
