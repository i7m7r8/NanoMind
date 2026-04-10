//! NanoMind Trainer — from-scratch transformer training in pure Rust.
//!
//! Trains a small language model on raw text, outputs a GGUF file
//! that works with Ollama.

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
