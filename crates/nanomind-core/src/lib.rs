//! NanoMind Core — llama.cpp-compatible quantization, tensor ops, kernels.
//!
//! Supports: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//!           Q2_K, Q3_K, Q4_K (Q4_K_M), Q5_K, Q6_K, IQ4_NL

pub mod attention;
pub mod ggml;
pub mod gguf_writer;
pub mod ops;
pub mod rope;

pub use ggml::*;
pub use gguf_writer::*;
pub use ops::*;
pub use rope::*;

// Re-export half::f16 for downstream crates
pub use attention::*;
pub use half::f16;
