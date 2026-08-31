use crate::{error::Result, tensor::Tensor};
pub trait Sampler {
    fn sample(&mut self, logits: &Tensor) -> Result<usize>;
}

