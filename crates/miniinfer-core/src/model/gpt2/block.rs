use crate::{
    error::{MiniInferError, Result},
    ops::{layer_norm, matmul},
    tensor::Tensor,
};

use super::validate_shape;

pub struct Gpt2BlockWeights {
    pub ln_1_weight: Tensor,
    pub ln_1_bias: Tensor,

    pub c_attn_weight: Tensor,
    pub c_attn_bias: Tensor,

    pub attn_c_proj_weight: Tensor,
    pub attn_c_proj_bias: Tensor,

    pub ln_2_weight: Tensor,
    pub ln_2_bias: Tensor,

    pub c_fc_weight: Tensor,
    pub c_fc_bias: Tensor,

    pub mlp_c_proj_weight: Tensor,
    pub mlp_c_proj_bias: Tensor,
}

impl Gpt2BlockWeights {
    pub fn apply_ln_1(&self, hidden: &Tensor, epsilon: f32) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: hidden.shape().len(),
            });
        }
        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        validate_shape(&self.ln_1_weight, &[hidden_size])?;
        validate_shape(&self.ln_1_bias, &[hidden_size])?;

        let mut output = Vec::with_capacity(seq_len * hidden_size);

        for row in 0..seq_len {
            let mut row_values = Vec::with_capacity(hidden_size);
            for col in 0..hidden_size {
                row_values.push(hidden.get_2d(row, col)?);
            }

            let normalized = layer_norm::layer_norm(
                &row_values,
                self.ln_1_weight.data(),
                self.ln_1_bias.data(),
                epsilon,
            )?;
            output.extend(normalized);
        }
        Tensor::new(vec![seq_len, hidden_size], output)
    }

    pub fn project_qkv(&self, hidden: &Tensor) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: hidden.shape().len(),
            });
        }

        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        validate_shape(&self.c_attn_weight, &[hidden_size, 3 * hidden_size])?;
        validate_shape(&self.c_attn_bias, &[3 * hidden_size])?;

        let projected = matmul::matmul(hidden, &self.c_attn_weight)?;

        let mut output = Vec::with_capacity(seq_len * 3 * hidden_size);

        for row in 0..seq_len {
            for col in 0..(3 * hidden_size) {
                let value = projected.get_2d(row, col)? + self.c_attn_bias.get_1d(col)?;
                output.push(value);
            }
        }

        Tensor::new(vec![seq_len, 3 * hidden_size], output)
    }

    pub fn attention_weights(&self, hidden: &Tensor, head_dim: usize) -> Result<Tensor> {
        let qkv = self.project_qkv(hidden)?;
        let (query, key, _value) = split_qkv(&qkv)?;

        attention_scores(&query, &key, head_dim)
    }
}

#[allow(dead_code)]
pub(crate) fn split_qkv(qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    if qkv.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: qkv.shape().len(),
        });
    }

    if qkv.shape()[1] % 3 != 0 {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![qkv.shape()[0], 3 * (qkv.shape()[1] / 3)],
            actual: qkv.shape().to_vec(),
        });
    }

    let hidden_size = qkv.shape()[1] / 3;
    let seq_len = qkv.shape()[0];
    let mut query = Vec::with_capacity(seq_len * hidden_size);
    let mut key = Vec::with_capacity(seq_len * hidden_size);
    let mut value = Vec::with_capacity(seq_len * hidden_size);

    for row in 0..seq_len {
        for col in 0..hidden_size {
            query.push(qkv.get_2d(row, col)?);
            key.push(qkv.get_2d(row, col + hidden_size)?);
            value.push(qkv.get_2d(row, col + 2 * hidden_size)?);
        }
    }

    Ok((
        Tensor::new(vec![seq_len, hidden_size], query)?,
        Tensor::new(vec![seq_len, hidden_size], key)?,
        Tensor::new(vec![seq_len, hidden_size], value)?,
    ))
}

#[allow(dead_code)]
pub(crate) fn attention_scores(query: &Tensor, key: &Tensor, head_dim: usize) -> Result<Tensor> {
    if query.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: query.shape().len(),
        });
    }

    if key.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: key.shape().len(),
        });
    }

    if head_dim == 0 {
        return Err(MiniInferError::InvalidConfig {
            message: "head_dim must be greater than zero".to_string(),
        });
    }

    let query_seq_len = query.shape()[0];
    let query_hidden_size = query.shape()[1];
    let key_seq_len = key.shape()[0];
    let key_hidden_size = key.shape()[1];

    if query_hidden_size != key_hidden_size {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![key_seq_len, query_hidden_size],
            actual: key.shape().to_vec(),
        });
    }

    let scale = (head_dim as f32).sqrt();
    let mut output = Vec::with_capacity(query_seq_len * key_seq_len);

    for query_row in 0..query_seq_len {
        for key_row in 0..key_seq_len {
            let mut score = 0.0;

            for col in 0..query_hidden_size {
                score += query.get_2d(query_row, col)? * key.get_2d(key_row, col)?;
            }

            output.push(score / scale);
        }
    }

    Tensor::new(vec![query_seq_len, key_seq_len], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize]) -> Tensor {
        let len = shape.iter().product();
        Tensor::new(shape.to_vec(), vec![0.0; len]).expect("test tensor shape should be valid")
    }

    fn tiny_block_weights() -> Gpt2BlockWeights {
        Gpt2BlockWeights {
            ln_1_weight: tensor(&[4]),
            ln_1_bias: tensor(&[4]),
            c_attn_weight: tensor(&[4, 12]),
            c_attn_bias: tensor(&[12]),
            attn_c_proj_weight: tensor(&[4, 4]),
            attn_c_proj_bias: tensor(&[4]),
            ln_2_weight: tensor(&[4]),
            ln_2_bias: tensor(&[4]),
            c_fc_weight: tensor(&[4, 16]),
            c_fc_bias: tensor(&[16]),
            mlp_c_proj_weight: tensor(&[16, 4]),
            mlp_c_proj_bias: tensor(&[4]),
        }
    }

    #[test]
    fn apply_ln_1_normalizes_hidden_rows() {
        let block = Gpt2BlockWeights {
            ln_1_weight: Tensor::new(vec![3], vec![1.0, 1.0, 1.0]).expect("valid ln_1 weight"),
            ln_1_bias: Tensor::new(vec![3], vec![0.0, 0.0, 0.0]).expect("valid ln_1 bias"),
            c_attn_weight: tensor(&[3, 9]),
            c_attn_bias: tensor(&[9]),
            attn_c_proj_weight: tensor(&[3, 3]),
            attn_c_proj_bias: tensor(&[3]),
            ln_2_weight: tensor(&[3]),
            ln_2_bias: tensor(&[3]),
            c_fc_weight: tensor(&[3, 12]),
            c_fc_bias: tensor(&[12]),
            mlp_c_proj_weight: tensor(&[12, 3]),
            mlp_c_proj_bias: tensor(&[3]),
        };
        let hidden = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).expect("valid hidden");

        let output = block
            .apply_ln_1(&hidden, 1e-5)
            .expect("ln_1 should succeed");

        let expected = [-1.2247356, 0.0, 1.2247356];
        assert_eq!(output.shape(), &[1, 3]);
        assert!(
            output
                .data()
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1e-5)
        );
    }

    #[test]
    fn apply_ln_1_rejects_non_2d_hidden() {
        let block = tiny_block_weights();
        let hidden = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("valid hidden");

        let err = block
            .apply_ln_1(&hidden, 1e-5)
            .expect_err("1D hidden tensor should fail");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn project_qkv_applies_weight_and_bias() {
        let block = Gpt2BlockWeights {
            ln_1_weight: tensor(&[2]),
            ln_1_bias: tensor(&[2]),
            c_attn_weight: Tensor::new(
                vec![2, 6],
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0],
            )
            .expect("valid c_attn weight"),
            c_attn_bias: Tensor::new(vec![6], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("valid c_attn bias"),
            attn_c_proj_weight: tensor(&[2, 2]),
            attn_c_proj_bias: tensor(&[2]),
            ln_2_weight: tensor(&[2]),
            ln_2_bias: tensor(&[2]),
            c_fc_weight: tensor(&[2, 8]),
            c_fc_bias: tensor(&[8]),
            mlp_c_proj_weight: tensor(&[8, 2]),
            mlp_c_proj_bias: tensor(&[2]),
        };
        let hidden = Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("valid hidden");

        let output = block
            .project_qkv(&hidden)
            .expect("QKV projection should succeed");

        assert_eq!(output.shape(), &[1, 6]);
        assert_eq!(output.data(), &[1.0, 2.0, 2.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn split_qkv_splits_combined_tensor() {
        let qkv = Tensor::new(
            vec![2, 6],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .expect("valid qkv tensor");

        let (query, key, value) = split_qkv(&qkv).expect("QKV split should succeed");

        assert_eq!(query.shape(), &[2, 2]);
        assert_eq!(query.data(), &[1.0, 2.0, 7.0, 8.0]);

        assert_eq!(key.shape(), &[2, 2]);
        assert_eq!(key.data(), &[3.0, 4.0, 9.0, 10.0]);

        assert_eq!(value.shape(), &[2, 2]);
        assert_eq!(value.data(), &[5.0, 6.0, 11.0, 12.0]);
    }

    #[test]
    fn split_qkv_rejects_non_2d_tensor() {
        let qkv = Tensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let err = split_qkv(&qkv).expect_err("1D QKV tensor should fail");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn split_qkv_rejects_width_not_divisible_by_three() {
        let qkv = Tensor::new(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("valid tensor");

        let err = split_qkv(&qkv).expect_err("QKV width must be divisible by 3");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![1, 3],
                actual: vec![1, 5],
            }
        );
    }

    #[test]
    fn attention_scores_computes_scaled_dot_products() {
        let query = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
            .expect("valid query");
        let key = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
            .expect("valid key");

        let scores = attention_scores(&query, &key, 2).expect("attention scores should succeed");

        let expected = [std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2];
        assert_eq!(scores.shape(), &[2, 2]);
        assert!(
            scores
                .data()
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1e-6)
        );
    }

    #[test]
    fn attention_scores_rejects_hidden_size_mismatch() {
        let query = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
            .expect("valid query");
        let key = Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .expect("valid key");

        let err = attention_scores(&query, &key, 2).expect_err("hidden mismatch should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![2, 2],
                actual: vec![2, 3],
            }
        );
    }

    #[test]
    fn attention_scores_rejects_zero_head_dim() {
        let query = Tensor::new(vec![1, 2], vec![1.0, 0.0]).expect("valid query");
        let key = Tensor::new(vec![1, 2], vec![1.0, 0.0]).expect("valid key");

        let err = attention_scores(&query, &key, 0).expect_err("zero head dim should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "head_dim must be greater than zero".to_string(),
            }
        );
    }
}