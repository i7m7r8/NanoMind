//! NanoMind Training CLI
//!
//! Train a language model from scratch and export to GGUF.

use std::path::PathBuf;
use std::time::Instant;

use nanomind_trainer::config::ModelConfig;
use nanomind_trainer::data_loader::get_training_corpus;
use nanomind_trainer::train::{export_to_gguf, train_model, TrainConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "export" => cmd_export(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("NanoMind Trainer — train from scratch → GGUF for Ollama");
    println!();
    println!("Usage:");
    println!("  nanomind-train train [OPTIONS]    Train a model from scratch");
    println!("  nanomind-train export --model <path> --out <path>  Export checkpoint to GGUF");
    println!();
    println!("Train Options:");
    println!("  --steps <n>       Training steps (default: 200)");
    println!("  --lr <float>      Learning rate (default: 0.001)");
    println!("  --seq-len <n>     Sequence length (default: 32)");
    println!("  --vocab <n>       Vocabulary size (default: 260)");
    println!("  --hidden <n>      Hidden dimension (default: 128)");
    println!("  --layers <n>      Number of layers (default: 4)");
    println!("  --heads <n>       Number of attention heads (default: 4)");
    println!("  --out <path>      Output GGUF path (default: nanomind.gguf)");
    println!("  --seed <n>        Random seed (default: 42)");
    println!();
    println!("Example:");
    println!("  nanomind-train train --steps 500 --out model.gguf");
}

fn cmd_train(args: &[String]) {
    let mut steps = 200usize;
    let mut lr = 1e-3f32;
    let mut seq_len = 32usize;
    let mut vocab = 260usize;
    let mut hidden = 128usize;
    let mut layers = 4usize;
    let mut heads = 4usize;
    let mut out_path = PathBuf::from("nanomind.gguf");
    let mut seed = 42u64;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("Invalid steps");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("Invalid learning rate");
            }
            "--seq-len" => {
                i += 1;
                seq_len = args[i].parse().expect("Invalid seq-len");
            }
            "--vocab" => {
                i += 1;
                vocab = args[i].parse().expect("Invalid vocab size");
            }
            "--hidden" => {
                i += 1;
                hidden = args[i].parse().expect("Invalid hidden dim");
            }
            "--layers" => {
                i += 1;
                layers = args[i].parse().expect("Invalid layers");
            }
            "--heads" => {
                i += 1;
                heads = args[i].parse().expect("Invalid heads");
            }
            "--out" => {
                i += 1;
                out_path = PathBuf::from(&args[i]);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid seed");
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("=== NanoMind Training ===");
    println!("Steps: {}", steps);
    println!("Learning rate: {}", lr);
    println!("Sequence length: {}", seq_len);
    println!("Vocab size: {}", vocab);
    println!("Hidden dim: {}", hidden);
    println!("Layers: {}", layers);
    println!("Heads: {}", heads);
    println!("Output: {:?}", out_path);
    println!();

    // Build config
    let model_config = ModelConfig {
        vocab_size: vocab,
        hidden_dim: hidden,
        num_heads: heads,
        num_kv_heads: heads / 2,
        num_layers: layers,
        intermediate_dim: hidden * 2,
        max_seq_len: seq_len,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        tie_embeddings: true,
    };

    let train_config = TrainConfig {
        model_config,
        batch_size: 1,
        seq_len,
        learning_rate: lr,
        min_lr: lr * 0.1,
        warmup_steps: steps / 10,
        max_steps: steps,
        weight_decay: 0.01,
        grad_clip: 1.0,
        checkpoint_every: steps / 4,
        eval_every: 10,
        seed,
        checkpoint_dir: "checkpoints".to_string(),
    };

    println!(
        "Model params: {:.2}M",
        train_config.model_config.param_count() as f64 / 1e6
    );
    println!("Corpus size: {} bytes", get_training_corpus().len());
    println!();

    let start = Instant::now();

    let (model, tokenizer) = train_model(train_config, |_step, loss, lr| {
        // Progress callback
        let _ = (loss, lr);
    });

    let elapsed = start.elapsed();
    println!();
    println!("Training completed in {:.1}s", elapsed.as_secs_f64());

    // Export to GGUF
    println!("\nExporting to GGUF...");
    export_to_gguf(&model, &tokenizer, &out_path).expect("Failed to export GGUF");

    println!("\n=== Done ===");
    println!("Model saved to: {:?}", out_path);
    println!();
    println!("Use with Ollama:");
    println!("  echo 'FROM {:?}' > Modelfile", out_path);
    println!("  ollama create nanomind -f Modelfile");
    println!("  ollama run nanomind");
}

fn cmd_export(_args: &[String]) {
    eprintln!("Export command not yet implemented. Use 'train --out <path>' instead.");
    std::process::exit(1);
}
