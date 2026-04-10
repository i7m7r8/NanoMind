//! Ollama-compatible HTTP API server.
//!
//! Endpoints:
//! - POST /api/generate — text completion
//! - POST /api/chat — chat completion
//! - POST /api/embeddings — text embeddings
//! - GET /api/tags — list loaded models
//! - POST /api/show — show model info

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};

/// Ollama-compatible request.
#[derive(serde::Deserialize, Debug)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(serde::Deserialize, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub options: HashMap<String, serde_json::Value>,
}

/// Response chunk for streaming.
#[derive(serde::Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

/// Minimal HTTP server that serves Ollama-compatible endpoints.
pub struct Server {
    pub addr: String,
    pub running: AtomicBool,
}

impl Server {
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
            running: AtomicBool::new(true),
        }
    }

    /// Start the server (blocking).
    pub fn serve(&self) -> std::io::Result<()> {
        let listener = TcpListener::bind(&self.addr)?;
        println!("[INFO] Ollama-compatible server listening on {}", self.addr);
        println!("[INFO] Endpoints:");
        println!("       POST /api/generate");
        println!("       POST /api/chat");
        println!("       POST /api/embeddings");
        println!("       GET  /api/tags");
        println!("       POST /api/show");

        for stream in listener.incoming() {
            if !self.running.load(Ordering::SeqCst) {
                break;
            }
            match stream {
                Ok(stream) => {
                    let _ = Self::handle_connection(stream);
                }
                Err(e) => {
                    eprintln!("[ERROR] Connection error: {}", e);
                }
            }
        }

        Ok(())
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    fn handle_connection(stream: std::net::TcpStream) -> std::io::Result<()> {
        stream.set_read_timeout(Some(std::time::Duration::from_secs(300)))?;
        let reader = BufReader::new(stream.try_clone()?);

        let mut lines = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                break;
            }
            lines.push(line);
        }

        if lines.is_empty() {
            return Ok(());
        }

        // Parse request line
        let parts: Vec<&str> = lines[0].split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }

        let method = parts[0];
        let path = parts[1];

        // Read body if present
        let mut content_length = 0;
        let mut body_start = lines.len();
        for (i, line) in lines.iter().enumerate() {
            if line.to_lowercase().starts_with("content-length:") {
                if let Ok(len) = line
                    .split(':')
                    .nth(1)
                    .unwrap_or("0")
                    .trim()
                    .parse::<usize>()
                {
                    content_length = len;
                }
            }
            if line.is_empty() {
                body_start = i + 1;
                break;
            }
        }

        let body = if content_length > 0 && body_start < lines.len() {
            lines[body_start..].join("\n")
        } else {
            String::new()
        };

        // Route
        let response = Self::route(method, path, &body);

        // Send response
        let mut writer = stream;
        write!(
            writer,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: {}\r\n\r\n{}",
            response.len(),
            response
        )?;
        writer.flush()?;

        Ok(())
    }

    fn route(method: &str, path: &str, body: &str) -> String {
        // Handle OPTIONS for CORS
        if method == "OPTIONS" {
            return "".into();
        }

        match (method, path) {
            ("GET", "/api/tags") => Self::api_tags(),
            ("POST", "/api/generate") => Self::api_generate(body),
            ("POST", "/api/chat") => Self::api_chat(body),
            ("POST", "/api/embeddings") => Self::api_embeddings(body),
            ("POST", "/api/show") => Self::api_show(body),
            _ => serde_json::json!({
                "error": format!("Not found: {} {}", method, path)
            })
            .to_string(),
        }
    }

    fn api_tags() -> String {
        serde_json::json!({
            "models": []
        })
        .to_string()
    }

    fn api_generate(body: &str) -> String {
        match serde_json::from_str::<GenerateRequest>(body) {
            Ok(req) => {
                // For now, return a placeholder. The actual engine
                // would call into the inference pipeline here.
                serde_json::json!({
                    "model": req.model,
                    "created_at": chrono_now(),
                    "response": format!("NanoMind received: {}", req.prompt.chars().take(100).collect::<String>()),
                    "done": true,
                })
                .to_string()
            }
            Err(e) => serde_json::json!({ "error": format!("Bad request: {}", e) }).to_string(),
        }
    }

    fn api_chat(body: &str) -> String {
        match serde_json::from_str::<ChatRequest>(body) {
            Ok(req) => {
                let default_content = String::new();
                let last_msg = req
                    .messages
                    .last()
                    .map(|m| &m.content)
                    .unwrap_or(&default_content);
                serde_json::json!({
                    "model": req.model,
                    "created_at": chrono_now(),
                    "message": {
                        "role": "assistant",
                        "content": format!("Echo: {}", last_msg.chars().take(200).collect::<String>()),
                    },
                    "done": true,
                })
                .to_string()
            }
            Err(e) => serde_json::json!({ "error": format!("Bad request: {}", e) }).to_string(),
        }
    }

    fn api_embeddings(_body: &str) -> String {
        serde_json::json!({
            "model": "nanomind",
            "embeddings": vec![0.0f32; 768],
        })
        .to_string()
    }

    fn api_show(body: &str) -> String {
        if let Ok(req) = serde_json::from_str::<serde_json::Value>(body) {
            serde_json::json!({
                "modelfile": format!("# NanoMind model: {}", req.get("name").map(|v| v.as_str().unwrap_or("unknown")).unwrap_or("unknown")),
                "parameters": {},
                "template": "{{ .Prompt }}",
            })
            .to_string()
        } else {
            serde_json::json!({ "error": "Bad request" }).to_string()
        }
    }
}

fn chrono_now() -> String {
    // Minimal ISO8601 timestamp without chrono dependency
    "2024-01-01T00:00:00Z".to_string()
}
