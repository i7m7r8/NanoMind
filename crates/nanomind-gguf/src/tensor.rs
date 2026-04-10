//! Tensor information from GGUF header.

use std::io;

use crate::reader::ReadLeExt;
use nanomind_core::GgmlType;

/// Information about a single tensor in a GGUF file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ty: GgmlType,
    pub n_dims: u32,
    pub dims: [u64; 4],
    /// Offset in bytes from the tensor data start.
    pub offset: u64,
}

impl TensorInfo {
    pub fn parse<R: io::Read>(cursor: &mut R, _version: u32) -> io::Result<Self> {
        let n_dims = cursor.read_u32_le()?;
        if n_dims > 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Tensor has {} dimensions (max 4)", n_dims),
            ));
        }

        let mut dims = [1u64; 4];
        for i in 0..n_dims as usize {
            dims[i] = cursor.read_u64_le()?;
        }

        let ty_id = cursor.read_u32_le()?;
        let ty = GgmlType::from_u32(ty_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown tensor type: {}", ty_id),
            )
        })?;

        let offset = cursor.read_u64_le()?;
        let name = cursor.read_string()?;

        Ok(Self {
            name,
            ty,
            n_dims,
            dims,
            offset,
        })
    }

    /// Number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.dims[..self.n_dims as usize].iter().product()
    }

    /// Number of bytes this tensor occupies in the file.
    pub fn n_bytes(&self) -> usize {
        let el = self.n_elements();
        let blck = self.ty.blck_size();
        let type_size = self.ty.type_size();
        if blck == 0 {
            return 0;
        }
        // For F32/F16, each element has its own size
        if blck == 1 {
            return (el as usize) * type_size;
        }
        // For quantized types: (n_elements / block_size) * type_size
        ((el as usize) / blck) * type_size
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[u64] {
        &self.dims[..self.n_dims as usize]
    }
}
