//! NanoMind Trainer — from-scratch transformer training in pure Rust.
//!
//! Trains a small language model on raw text, outputs a GGUF file
//! that works with Ollama.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_long_first_doc_paragraph)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::new_without_default)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

pub mod autodiff;
pub mod config;
pub mod data_loader;
pub mod model;
pub mod optimizer;
pub mod train;

pub use autodiff::Tape;
pub use config::ModelConfig;
pub use data_loader::DataLoader;
pub use model::TransformerModel;
pub use optimizer::AdamW;
pub use train::train_model;
