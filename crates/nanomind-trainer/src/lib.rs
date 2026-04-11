//! NanoMind Trainer — from-scratch transformer training in pure Rust.
//!
//! Trains a small language model on raw text, outputs a GGUF file
//! that works with Ollama.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_long_first_doc_paragraph)]

pub mod config;
pub mod data_loader;
pub mod model;
pub mod optimizer;
pub mod train;

pub use config::ModelConfig;
pub use data_loader::DataLoader;
pub use model::TransformerModel;
pub use optimizer::AdamW;
pub use train::train_model;
