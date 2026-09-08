pub mod block;
pub mod weights;

pub use block::Gpt2BlockWeights;
pub use weights::{Gpt2Weights, LMHead};

use crate::{
    error::{MiniInferError, Result},
    tensor::{Tensor, WeightTensor},
};

pub(crate) fn validate_shape(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
    if tensor.shape() != expected_shape {
        return Err(MiniInferError::InvalidTensorShape {
            expected: expected_shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

pub(crate) fn validate_weight_shape(weight: &WeightTensor, expected_shape: &[usize]) -> Result<()> {
    if weight.shape() != expected_shape {
        return Err(MiniInferError::InvalidTensorShape {
            expected: expected_shape.to_vec(),
            actual: weight.shape().to_vec(),
        });
    }
    Ok(())
}