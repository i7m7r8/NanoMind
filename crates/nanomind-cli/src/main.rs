//! NanoMind CLI — Load model, chat loop, benchmark.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use nanomind_model::file_format::load_model;
use nanomind_runtime::{estimate_ram_usage, fits_ram, InferenceEngine, SamplingConfig};
use nanomind_tokenizer::BpeTokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "nanomind",
    about = "Ultra-compressed Rust LLM inference engine"
)]
struct Args {
    /// Path to .nm model file
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text (or read from stdin if not provided)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Interactive chat mode
    #[arg(long)]
    chat: bool,

    /// Run benchmark (100 tokens, report tok/s and RAM)
    #[arg(long)]
    bench: bool,

    /// Maximum tokens to generate
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value = "0.7")]
    temp: f32,

    /// Top-p nucleus sampling
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Top-k filtering
    #[arg(long, default_value = "40")]
    top_k: usize,

    /// Repetition penalty
    #[arg(long, default_value = "1.1")]
    rep_penalty: f32,
}

fn main() {
    let args = Args::parse();

    // Load model
    println!("[INFO] Loading model from {}...", args.model.display());
    let load_start = Instant::now();

    let model = match load_model(&args.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[ERROR] Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let load_time = load_start.elapsed();
    let ram_mb = estimate_ram_usage(
        &model.config,
        nanomind_model::model::QuantType::Q4,
        model.config.max_seq_len,
    ) / (1024 * 1024);
    println!(
        "[INFO] Model loaded: {} MB RAM ({:.2}s)",
        ram_mb,
        load_time.as_secs_f32()
    );

    // Check RAM budget
    if !fits_ram(&model.config, nanomind_model::model::QuantType::Q4) {
        eprintln!(
            "[WARN] Model estimated RAM usage ({} MB) exceeds 480 MB budget",
            ram_mb
        );
    }

    // Print config
    let config = &model.config;
    println!(
        "[INFO] Config: vocab={}, hidden={}, heads={}, kv_heads={}, layers={}, max_seq={}",
        config.vocab_size,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.num_layers,
        config.max_seq_len
    );

    // Load tokenizer
    let tokenizer_path = args.model.with_extension("json");
    let tokenizer = if tokenizer_path.exists() {
        println!(
            "[INFO] Loading tokenizer from {}...",
            tokenizer_path.display()
        );
        let content = fs::read_to_string(&tokenizer_path).unwrap_or_default();
        BpeTokenizer::from_json(&content).unwrap_or_else(|_| {
            eprintln!("[WARN] Failed to parse tokenizer JSON, using fallback");
            BpeTokenizer::new_fallback(config.vocab_size)
        })
    } else {
        println!(
            "[WARN] No tokenizer found at {}, using fallback",
            tokenizer_path.display()
        );
        BpeTokenizer::new_fallback(config.vocab_size)
    };

    let sampling_config = SamplingConfig {
        temperature: args.temp,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.rep_penalty,
        max_new_tokens: args.max_tokens,
    };

    let mut engine = InferenceEngine::new(model, tokenizer, sampling_config);

    if args.bench {
        run_benchmark(&mut engine, &args);
    } else if args.chat {
        run_chat_loop(&mut engine);
    } else {
        run_single_prompt(&mut engine, &args);
    }
}

/// Run a single prompt.
fn run_single_prompt(engine: &mut InferenceEngine, args: &Args) {
    let prompt = args.prompt.clone().unwrap_or_else(|| {
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).ok();
        input
    });

    if prompt.trim().is_empty() {
        eprintln!("[ERROR] No prompt provided. Use --prompt or pipe input.");
        std::process::exit(1);
    }

    println!("\n[PROMPT] {}", prompt);
    println!("[GENERATING]\n");

    let gen_start = Instant::now();
    let mut token_count = 0;

    let tokens = engine.generate(&prompt, |_token_id| {
        token_count += 1;
    });

    let gen_time = gen_start.elapsed();
    let tok_per_sec = token_count as f64 / gen_time.as_secs_f64();

    // Decode and print result
    let generated_text = engine.decode_tokens(&tokens);
    println!("{}", generated_text);
    println!(
        "\n[INFO] Generated {} tokens in {:.2}s ({:.1} tok/s)",
        token_count,
        gen_time.as_secs_f32(),
        tok_per_sec
    );
}

/// Run interactive chat loop.
fn run_chat_loop(engine: &mut InferenceEngine) {
    println!("\n=== NanoMind Chat ===");
    println!("Type 'quit' or 'exit' to leave.\n");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        let gen_start = Instant::now();
        let mut token_count = 0;

        println!("\n[ASSISTANT]");
        let tokens = engine.generate(input, |_token_id| {
            token_count += 1;
        });
        let gen_time = gen_start.elapsed();

        let generated_text = engine.decode_tokens(&tokens);
        println!("{}", generated_text);

        if token_count > 0 {
            let tok_per_sec = token_count as f64 / gen_time.as_secs_f64();
            println!("\n[INFO] {} tokens, {:.1} tok/s", token_count, tok_per_sec);
        } else {
            println!("\n[INFO] No tokens generated.");
        }
        println!();
    }
}

/// Run benchmark.
fn run_benchmark(engine: &mut InferenceEngine, args: &Args) {
    let prompt = args.prompt.as_deref().unwrap_or("The meaning of life is");

    println!("\n[INFO] Benchmarking 100 tokens...");
    println!("[INFO] Prompt: \"{}\"", prompt);

    engine.config.max_new_tokens = 100;
    engine.config.temperature = args.temp;

    let gen_start = Instant::now();
    let mut token_count = 0;
    let _tokens = engine.generate(prompt, |_token_id| {
        token_count += 1;
    });
    let gen_time = gen_start.elapsed();
    let tok_per_sec = token_count as f64 / gen_time.as_secs_f64();

    let ram_mb = estimate_ram_usage(
        &engine.model.config,
        nanomind_model::model::QuantType::Q4,
        engine.model.config.max_seq_len,
    ) / (1024 * 1024);

    println!("\n=== Benchmark Results ===");
    println!("  Tokens generated: {}", token_count);
    println!("  Time: {:.3}s", gen_time.as_secs_f64());
    println!("  Speed: {:.1} tokens/sec", tok_per_sec);
    println!("  Peak RAM: {} MB", ram_mb);
}
