use crate::{tensor::Tensor, error::{MiniInferError, Result}};

pub fn validate_matmul_shape(a: &Tensor, b: &Tensor) -> Result<(usize, usize, usize)> {
    let left = a.shape();
    let right = b.shape();

    if left.len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: left.len() });
    }

    if right.len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: right.len() });
    }

    let m = left[0];
    let k = left[1];
    let n = right[1];

    if k != right[0] {
        return Err(MiniInferError::MatMulShapeMismatch { left: left.to_vec(), right: right.to_vec() });
    }

    Ok((m, k, n))
}