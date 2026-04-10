//! NanoMind CLI — GGUF model loader, inference engine, Ollama server.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use nanomind_model::Model;
use nanomind_sampling::{Sampler, SamplingParams};
use nanomind_server::Server;
use nanomind_tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "nanomind",
    about = "Ultra-compressed pure-Rust LLM inference engine. Ollama-compatible.",
    version
)]
#[derive(Clone)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Prompt text
    #[arg(short, long)]
    prompt: Option<String>,

    /// Interactive chat mode
    #[arg(long)]
    chat: bool,

    /// Start Ollama-compatible HTTP server
    #[arg(long)]
    server: bool,

    /// Server listen address
    #[arg(long, default_value = "127.0.0.1:8080")]
    host: String,

    /// Benchmark mode
    #[arg(long)]
    bench: bool,

    /// Max tokens to generate (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value = "0.8")]
    temp: f32,

    /// Top-k
    #[arg(long, default_value = "40")]
    top_k: usize,

    /// Top-p
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// Mirostat (0=off, 2=v2)
    #[arg(long, default_value = "0")]
    mirostat: u32,

    /// Context window size
    #[arg(long)]
    ctx_size: Option<usize>,

    /// Seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,

    /// Ignore EOS token
    #[arg(long)]
    ignore_eos: bool,
}

fn main() {
    let args = Args::parse();

    if args.server {
        run_server(&args);
        return;
    }

    if args.model.is_none() {
        eprintln!("Error: --model is required (unless running in server mode)");
        std::process::exit(1);
    }

    let model_path = args.model.as_ref().unwrap();

    println!("[INFO] Loading model: {}", model_path.display());
    let load_start = Instant::now();

    let model = match Model::from_gguf(model_path, args.ctx_size) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[ERROR] Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let load_time = load_start.elapsed();
    let ram_mb = model.config.estimate_params() * 8 / (1024 * 1024); // rough estimate
    println!(
        "[INFO] Model loaded in {:.2}s (~{} MB estimated RAM)",
        load_time.as_secs_f32(),
        ram_mb,
    );

    // Create tokenizer from vocab
    let tokenizer = create_tokenizer(&model);

    if args.bench {
        run_benchmark(&model, &tokenizer, &args);
    } else if args.chat {
        run_chat(&model, &tokenizer, &args);
    } else {
        run_prompt(&model, &tokenizer, &args);
    }
}

/// Create a tokenizer from the model config.
fn create_tokenizer(_model: &Model) -> Tokenizer {
    // Create a fallback tokenizer
    // In production, load from the model's tokenizer file
    let mut tokens = Vec::new();

    // Add common tokens
    let common = [
        "<unk>", "<s>", "</s>", "<pad>", "the", "a", "an", "is", "are", "was", "were", "of", "and",
        "to", "in", "that", "it", "for", "on", "with", "as", "at", "by", "from", "I", "you", "he",
        "she", "we", "they", "this", "that", "these", "those", "Hello", "World", "hello", "world",
        "The", "A", "An", "In", "On", "At", "\n", " ", ".", ",", "!", "?", ":", ";",
    ];

    for (i, token) in common.iter().enumerate() {
        tokens.push(token.to_string());
        // Also add variations
        if i < 1000 {
            // Common English words
            let word = format!(" {}", token);
            if !tokens.contains(&word) {
                tokens.push(word);
            }
        }
    }

    // Fill with numbered tokens for the rest
    for i in tokens.len()..32000 {
        tokens.push(format!("<t{}>", i));
    }

    Tokenizer::from_vocab(tokens, 1, 2, 0)
}

/// Run a single prompt.
fn run_prompt(model: &Model, tokenizer: &Tokenizer, args: &Args) {
    let prompt = if let Some(ref p) = args.prompt {
        p.clone()
    } else {
        let mut input = String::new();
        print!("> ");
        io::stdout().flush().ok();
        io::stdin().lock().read_line(&mut input).ok();
        input.trim().to_string()
    };

    if prompt.is_empty() {
        eprintln!("No prompt provided");
        return;
    }

    println!("\n[PROMPT] {}", prompt);
    println!("[GENERATING]\n");

    let (text, stats) = generate_text(model, tokenizer, &prompt, args);
    println!("\n{}", text);
    println!(
        "\n[INFO] {} tokens in {:.2}s ({:.1} tok/s)",
        stats.tokens,
        stats.time.as_secs_f32(),
        stats.tokens as f64 / stats.time.as_secs_f64()
    );
}

/// Run interactive chat.
fn run_chat(model: &Model, tokenizer: &Tokenizer, args: &Args) {
    println!("\n=== NanoMind Chat ===");
    println!("Model: {:?}", model.config.arch);
    println!("Type 'quit' or 'exit' to leave.\n");

    let mut history = String::new();
    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Build prompt with history
        let prompt = format!("{}User: {}\nAssistant:", history, input);

        let (response, stats) = generate_text(model, tokenizer, &prompt, args);

        // Update history
        history.push_str(&format!("User: {}\nAssistant: {}\n\n", input, response));

        println!("\n{}", response);
        println!(
            "\n[INFO] {} tokens, {:.1} tok/s",
            stats.tokens,
            stats.tokens as f64 / stats.time.as_secs_f64()
        );
    }
}

/// Run benchmark.
fn run_benchmark(model: &Model, tokenizer: &Tokenizer, args: &Args) {
    let prompt = args.prompt.as_deref().unwrap_or("The meaning of life is");

    println!("\n[INFO] Benchmarking 100 tokens...");
    println!("[INFO] Prompt: \"{}\"", prompt);
    println!(
        "[INFO] Architecture: {:?}, Layers: {}",
        model.config.arch, model.config.num_hidden_layers,
    );

    let mut bench_args = args.clone();
    bench_args.max_tokens = 100;

    let (_, stats) = generate_text(model, tokenizer, prompt, &bench_args);

    let ram_mb = model.config.estimate_params() * 8 / (1024 * 1024);
    println!("\n=== Benchmark Results ===");
    println!("  Tokens generated: {}", stats.tokens);
    println!("  Time: {:.3}s", stats.time.as_secs_f64());
    println!(
        "  Speed: {:.1} tokens/sec",
        stats.tokens as f64 / stats.time.as_secs_f64()
    );
    println!("  Estimated peak RAM: {} MB", ram_mb);
}

struct GenStats {
    tokens: usize,
    time: std::time::Duration,
}

/// Generate text from a prompt.
fn generate_text(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    args: &Args,
) -> (String, GenStats) {
    let cfg = &model.config;
    let hd = cfg.hidden_size;

    // Setup sampler
    let params = SamplingParams {
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repeat_penalty,
        mirostat: args.mirostat,
        seed: args.seed,
        ignore_eos: args.ignore_eos,
        ..Default::default()
    };
    let mut sampler = Sampler::new(params);

    // Setup KV cache
    let max_ctx = args.ctx_size.unwrap_or(model.max_seq);
    let mut cache = nanomind_model::KvCache::new(
        cfg.num_hidden_layers,
        max_ctx,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt);
    println!("[INFO] Prompt tokens: {}", prompt_tokens.len());

    // Pre-fill
    let mut hidden = vec![0.0f32; hd];
    let gen_start = Instant::now();

    for (i, &token_id) in prompt_tokens.iter().enumerate() {
        model.embed_token(token_id, &mut hidden);
        model.forward_token(&mut hidden, &mut cache, i);
    }
    cache.advance(prompt_tokens.len());

    // Generate
    let max_tokens = if args.max_tokens > 0 {
        args.max_tokens
    } else {
        256
    };
    let mut generated = Vec::new();
    let mut pos = prompt_tokens.len();

    for _ in 0..max_tokens {
        // Compute logits
        let mut logits = vec![0.0f32; cfg.vocab_size];
        model.compute_logits(&hidden, &mut logits);

        // Sample
        let token_id = sampler.sample(&mut logits, cfg.eos_token_id);

        // Check EOS
        if !args.ignore_eos && token_id == cfg.eos_token_id {
            break;
        }

        generated.push(token_id);

        // Embed and forward
        model.embed_token(token_id, &mut hidden);
        model.forward_token(&mut hidden, &mut cache, pos);
        cache.advance(1);
        pos += 1;

        // Print token
        if let Some(token_str) = tokenizer.token_str(token_id) {
            print!("{}", token_str);
            io::stdout().flush().ok();
        }
    }

    let gen_time = gen_start.elapsed();
    let text = tokenizer.decode(&generated);

    (
        text,
        GenStats {
            tokens: generated.len(),
            time: gen_time,
        },
    )
}

/// Run the Ollama-compatible HTTP server.
fn run_server(args: &Args) {
    println!("[INFO] Starting NanoMind server on {}", args.host);

    if let Some(ref model_path) = args.model {
        println!("[INFO] Loading model: {}", model_path.display());
        match Model::from_gguf(model_path, args.ctx_size) {
            Ok(_) => println!("[INFO] Model loaded successfully"),
            Err(e) => eprintln!("[WARN] Could not load model: {}", e),
        }
    }

    let server = Server::new(&args.host);
    server.serve().unwrap_or_else(|e| {
        eprintln!("[ERROR] Server failed: {}", e);
        std::process::exit(1);
    });
}
