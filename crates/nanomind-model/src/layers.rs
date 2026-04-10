//! Transformer layer weights loaded from GGUF.

use std::vec::Vec;

/// Single transformer layer weights (mmap-backed references).
pub struct LayerWeights {
    // Attention
    pub attn_q: LayerTensor,
    pub attn_k: LayerTensor,
    pub attn_v: LayerTensor,
    pub attn_o: LayerTensor,

    // FFN
    pub ffn_gate: LayerTensor, // SwiGLU gate
    pub ffn_up: LayerTensor,
    pub ffn_down: LayerTensor,

    // Norms
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,

    // MoE (if applicable)
    pub ffn_gate_experts: Option<Vec<LayerTensor>>,
    pub ffn_up_experts: Option<Vec<LayerTensor>>,
    pub ffn_down_experts: Option<Vec<LayerTensor>>,
}

/// A tensor that may be quantized or f32.
/// Stores a reference to the raw GGUF data.
#[derive(Clone)]
pub struct LayerTensor {
    /// Raw bytes from GGUF (memory-mapped).
    pub data: Vec<u8>,
    /// GGML quantization type.
    pub ggml_type: nanomind_core::GgmlType,
    /// Tensor shape.
    pub shape: Vec<usize>,
}

impl LayerTensor {
    /// Create from raw GGUF data.
    pub fn from_bytes(
        data: Vec<u8>,
        ggml_type: nanomind_core::GgmlType,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            data,
            ggml_type,
            shape,
        }
    }

    /// Perform matrix-vector multiply: `out = W * x`.
    /// Dequantizes on-the-fly without inflating the full weight matrix.
    pub fn matmul(&self, x: &[f32], out: &mut [f32]) {
        // W is [output_dim, input_dim] stored as GGUF dims
        // GGUF stores dims as [output_dim, input_dim] for 2D tensors
        let output_dim = self.shape[0];
        let input_dim = if self.shape.len() > 1 {
            self.shape[1]
        } else {
            self.data.len() / 4
        };

        debug_assert_eq!(
            x.len(),
            input_dim,
            "Input dimension mismatch: expected {}, got {}",
            input_dim,
            x.len()
        );
        debug_assert_eq!(out.len(), output_dim);

        let blck = self.ggml_type.blck_size();
        let type_size = self.ggml_type.type_size();

        // Each row is a quantized row
        for i in 0..output_dim {
            let row_start = i * input_dim;
            let row_data = &self.data[(row_start / blck * type_size)..];
            out[i] = nanomind_core::dot_q4_f32(row_data, self.ggml_type, x);
        }
    }
}

impl LayerWeights {
    /// Create an empty layer placeholder.
    pub fn empty() -> Self {
        Self {
            attn_q: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            attn_k: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            attn_v: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            attn_o: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            ffn_gate: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            ffn_up: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            ffn_down: LayerTensor {
                data: vec![],
                ggml_type: nanomind_core::GgmlType::F32,
                shape: vec![0],
            },
            attn_norm: vec![],
            ffn_norm: vec![],
            ffn_gate_experts: None,
            ffn_up_experts: None,
            ffn_down_experts: None,
        }
    }
}
