use crate::{error::{Result, MiniInferError}, tensor::Tensor};

pub fn embedding_lookup(embedding: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
    if token_ids.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    if embedding.shape().len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: embedding.shape().len() });
    }

    let vocab_size = embedding.shape()[0];
    let hidden_size = embedding.shape()[1];

    for &i in token_ids {
        if i >= vocab_size {
            return Err(MiniInferError::IndexOutOfBounds { index: i, len: vocab_size });
        }
    }

    let mut output = Vec::with_capacity(token_ids.len() * hidden_size);
    for &token_id in token_ids {
        for col in 0..hidden_size {
            let val = embedding.get_2d(token_id, col)?;
            output.push(val);
        }
    }
    Tensor::new(vec![token_ids.len(), hidden_size], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_up_embedding_rows() {
        let embedding = Tensor::new(vec![3, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("valid embedding tensor");

        let output = embedding_lookup(&embedding, &[2, 0]).expect("embedding lookup should succeed");

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.data(), &[0.5, 0.6, 0.1, 0.2]);
    }

    #[test]
    fn rejects_empty_token_ids() {
        let embedding = Tensor::new(vec![3, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("valid embedding tensor");

        let err = embedding_lookup(&embedding, &[]).expect_err("empty token ids should fail");

        assert_eq!(err, MiniInferError::EmptyInput);
    }

    #[test]
    fn rejects_non_2d_embedding() {
        let embedding = Tensor::new(vec![6], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("valid tensor");

        let err = embedding_lookup(&embedding, &[0]).expect_err("non-2D embedding should fail");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_token_id_out_of_bounds() {
        let embedding = Tensor::new(vec![3, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("valid embedding tensor");

        let err = embedding_lookup(&embedding, &[3]).expect_err("token id should be out of bounds");

        assert_eq!(
            err,
            MiniInferError::IndexOutOfBounds { index: 3, len: 3 }
        );
    }
}