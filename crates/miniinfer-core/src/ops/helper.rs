use crate::tensor::Tensor;

pub fn validate_matmul_shape(a: &Tensor, b: &Tensor) -> Result<(usize, usize, usize), crate::error::MiniInferError> {
    let left = a.shape();
    let right = b.shape();

    if left.len() != 2 {
        return Err(crate::error::MiniInferError::WrongRank { expected: 2, actual: left.len() });
    }

    if right.len() != 2 {
        return Err(crate::error::MiniInferError::WrongRank { expected: 2, actual: right.len() });
    }

    let m = left[0];
    let k = left[1];
    let n = right[1];

    if k != right[0] {
        return Err(crate::error::MiniInferError::MatMulShapeMismatch { left: left.to_vec(), right: right.to_vec() });
    }

    Ok((m, k, n))
}