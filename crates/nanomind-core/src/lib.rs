//! NanoMind Core — Quantization, tensor ops, SIMD, RoPE.
//!
//! Pure Rust, minimal `unsafe` (only for SIMD intrinsics).
//! No_std-compatible (use `alloc` only).

#![no_std]

extern crate alloc;

pub mod ops;
pub mod quantization;
pub mod rope;

// Compression targets as constants
pub const PARAMS_NANO: usize = 135_000_000;
pub const PARAMS_MINI: usize = 500_000_000;
pub const PARAMS_SMALL: usize = 1_000_000_000;

// RAM budget (bytes) — refuse to load if estimate exceeds this
pub const MAX_RAM_BUDGET: usize = 480 * 1024 * 1024; // 480 MB
