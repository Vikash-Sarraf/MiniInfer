use crate::{error::{MiniInferError, Result}, sampling::Sampler, tensor::Tensor};

pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &Tensor) -> Result<usize> {
        if logits.shape().len() != 2 {
            return Err(MiniInferError::WrongRank { expected: 2, actual: logits.shape().len() });
        }

        let last_row = logits.shape()[0] - 1;
        let vocab_size = logits.shape()[1];
        
        let mut best_id = 0;
        let mut best_value = logits.get_2d(last_row, 0)?;

        for token_id in 1..vocab_size {
            let value = logits.get_2d(last_row, token_id)?;
            if value > best_value {
                best_value = value;
                best_id = token_id;
            }
        }
        Ok(best_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampler_selects_largest_logit_from_last_row() {
        let logits = Tensor::new(
            vec![2, 4],
            vec![
                10.0, 9.0, 8.0, 7.0,
                1.0, 5.0, 3.0, 2.0,
            ],
        )
        .expect("valid logits");
        let mut sampler = GreedySampler;

        let token_id = sampler.sample(&logits).expect("sample should succeed");

        assert_eq!(token_id, 1);
    }

    #[test]
    fn greedy_sampler_rejects_non_2d_logits() {
        let logits = Tensor::new(vec![4], vec![1.0, 5.0, 3.0, 2.0]).expect("valid logits");
        let mut sampler = GreedySampler;

        let err = sampler
            .sample(&logits)
            .expect_err("1D logits should fail");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }
}