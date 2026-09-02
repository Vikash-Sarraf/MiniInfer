use crate::{
    error::{MiniInferError, Result}, ops::{backend::{OpsBackend, ReferenceBackend}, gelu, layer_norm, softmax, vector_add}, tensor::Tensor,
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

    pub fn apply_ln_2(&self, hidden: &Tensor, epsilon: f32) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: hidden.shape().len(),
            });
        }
        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        validate_shape(&self.ln_2_weight, &[hidden_size])?;
        validate_shape(&self.ln_2_bias, &[hidden_size])?;

        let mut output = Vec::with_capacity(seq_len * hidden_size);

        for row in 0..seq_len {
            let mut row_values = Vec::with_capacity(hidden_size);
            for col in 0..hidden_size {
                row_values.push(hidden.get_2d(row, col)?);
            }

            let normalized = layer_norm::layer_norm(
                &row_values,
                self.ln_2_weight.data(),
                self.ln_2_bias.data(),
                epsilon,
            )?;
            output.extend(normalized);
        }
        Tensor::new(vec![seq_len, hidden_size], output)
    }


    pub fn project_qkv(&self, hidden: &Tensor) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.project_qkv_with_backend(hidden, &backend)
    }

    pub fn project_qkv_with_backend(&self, hidden: &Tensor, backend: &dyn OpsBackend) -> Result<Tensor> {
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

        let projected = backend.matmul(hidden, &self.c_attn_weight)?;

        let mut output = Vec::with_capacity(seq_len * 3 * hidden_size);

        for row in 0..seq_len {
            for col in 0..(3 * hidden_size) {
                let value = projected.get_2d(row, col)? + self.c_attn_bias.get_1d(col)?;
                output.push(value);
            }
        }

        Tensor::new(vec![seq_len, 3 * hidden_size], output)
    }

    #[cfg(test)]
    fn attention_context(&self, hidden: &Tensor, head_dim: usize) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.attention_context_with_backend(hidden, head_dim, &backend)
    }

    fn attention_context_with_backend(
        &self,
        hidden: &Tensor,
        head_dim: usize,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        let qkv = self.project_qkv_with_backend(hidden, backend)?;
        let (query, key, value) = split_qkv(&qkv)?;

        let hidden_size = query.shape()[1];

        if head_dim == 0 {
            return Err(MiniInferError::InvalidConfig {
                message: "head_dim must be greater than zero".to_string(),
            });
        }

        if hidden_size % head_dim != 0 {
            return Err(MiniInferError::InvalidConfig {
                message: format!(
                    "hidden_size {hidden_size} must be divisible by head_dim {head_dim}"
                ),
            });
        }

        let num_heads = hidden_size / head_dim;

        let query_heads = split_heads(&query, num_heads)?;
        let key_heads = split_heads(&key, num_heads)?;
        let value_heads = split_heads(&value, num_heads)?;

        let mut context_heads = Vec::with_capacity(num_heads);
        for head_index in 0..num_heads {
            let scores =
                attention_scores(&query_heads[head_index], &key_heads[head_index], head_dim)?;
            let probabilities = causal_softmax(&scores)?;
            let context = attention_output_with_backend(&probabilities, &value_heads[head_index], backend)?;
            context_heads.push(context);
        }

        merge_heads(&context_heads)
    }

    pub fn apply_attention_sublayer(
        &self,
        hidden: &Tensor,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.apply_attention_sublayer_with_backend(hidden, head_dim, epsilon, &backend)
    }

    pub fn apply_attention_sublayer_with_backend(
        &self,
        hidden: &Tensor,
        head_dim: usize,
        epsilon: f32,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        let normalized = self.apply_ln_1(hidden, epsilon)?;
        let context = self.attention_context_with_backend(&normalized, head_dim, backend)?;

        if context.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: context.shape().len(),
            });
        }

        let seq_len = context.shape()[0];
        let hidden_size = context.shape()[1];

        validate_shape(&self.attn_c_proj_weight, &[hidden_size, hidden_size])?;
        validate_shape(&self.attn_c_proj_bias, &[hidden_size])?;

        let projected = backend.matmul(&context, &self.attn_c_proj_weight)?;
        let mut projected_with_bias = Vec::with_capacity(seq_len * hidden_size);

        for row in 0..seq_len {
            for col in 0..hidden_size {
                let value = projected.get_2d(row, col)? + self.attn_c_proj_bias.get_1d(col)?;
                projected_with_bias.push(value);
            }
        }

        let projected = Tensor::new(vec![seq_len, hidden_size], projected_with_bias)?;
        let output = vector_add::add(hidden.data(), projected.data())?;

        Tensor::new(hidden.shape().to_vec(), output)
    }

    pub fn apply_mlp(&self, hidden: &Tensor) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.apply_mlp_with_backend(hidden, &backend)
    }

    pub fn apply_mlp_with_backend(&self, hidden: &Tensor, backend: &dyn OpsBackend) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: hidden.shape().len(),
            });
        }

        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        validate_shape(&self.c_fc_weight, &[hidden_size, self.c_fc_bias.shape()[0]])?;

        let intermediate_size = self.c_fc_bias.shape()[0];

        validate_shape(&self.c_fc_bias, &[intermediate_size])?;
        validate_shape(&self.mlp_c_proj_weight, &[intermediate_size, hidden_size])?;
        validate_shape(&self.mlp_c_proj_bias, &[hidden_size])?;

        let expanded = backend.matmul(hidden, &self.c_fc_weight)?;
        let expanded_data = add_bias_rows(&expanded, &self.c_fc_bias)?;
        let expanded = Tensor::new(vec![seq_len, intermediate_size], expanded_data)?;

        let activated_data = gelu::gelu(expanded.data())?;
        let activated = Tensor::new(vec![seq_len, intermediate_size], activated_data)?;

        let projected = backend.matmul(&activated, &self.mlp_c_proj_weight)?;
        let projected_data = add_bias_rows(&projected, &self.mlp_c_proj_bias)?;

        Tensor::new(vec![seq_len, hidden_size], projected_data)
    }

    pub fn apply_mlp_sublayer(&self, hidden: &Tensor, epsilon: f32) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.apply_mlp_sublayer_with_backend(hidden, epsilon, &backend)
    }

    pub fn apply_mlp_sublayer_with_backend(
        &self,
        hidden: &Tensor,
        epsilon: f32,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        let normalized = self.apply_ln_2(hidden, epsilon)?;

        let mlp_output = self.apply_mlp_with_backend(&normalized, backend)?;

        let output = vector_add::add(hidden.data(), mlp_output.data())?;

        Tensor::new(hidden.shape().to_vec(), output)
    }

    pub fn forward(&self, hidden: &Tensor, head_dim: usize, epsilon: f32) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.forward_with_backend(hidden, head_dim, epsilon, &backend)
    }

    pub fn forward_with_backend(
        &self,
        hidden: &Tensor,
        head_dim: usize,
        epsilon: f32,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        let x = self.apply_attention_sublayer_with_backend(hidden, head_dim, epsilon, backend)?;
        self.apply_mlp_sublayer_with_backend(&x, epsilon, backend)
    }

}

fn add_bias_rows(matrix: &Tensor, bias: &Tensor) -> Result<Vec<f32>> {
    if matrix.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: matrix.shape().len(),
        });
    }

    let rows = matrix.shape()[0];
    let cols = matrix.shape()[1];

    validate_shape(bias, &[cols])?;

    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            output.push(matrix.get_2d(row, col)? + bias.get_1d(col)?);
        }
    }

    Ok(output)
}

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

fn causal_softmax(scores: &Tensor) -> Result<Tensor> {
    if scores.shape().len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: scores.shape().len() });
    }
    let rows = scores.shape()[0];
    let cols = scores.shape()[1];

    if rows != cols {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![rows, rows],
            actual: scores.shape().to_vec(),
        });
    }

    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let mut visible = Vec::with_capacity(rows + 1);
        for col in 0..=row {
            visible.push(scores.get_2d(row, col)?);
        }
        let softmax_data = softmax::softmax(&visible)?;

        for prob in softmax_data {
            output.push(prob);
        }

        for _ in (row + 1)..cols {
            output.push(0.0);
        }

    }
    Tensor::new(vec![rows, cols], output)
}

fn attention_output_with_backend(
    probabilities: &Tensor,
    value: &Tensor,
    backend: &dyn OpsBackend,
) -> Result<Tensor> {
    backend.matmul(probabilities, value)
}

fn split_heads(hidden: &Tensor, num_heads: usize) -> Result<Vec<Tensor>> {
    if hidden.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: hidden.shape().len(),
        });
    }

    if num_heads == 0 {
        return Err(MiniInferError::InvalidConfig {
            message: "num_heads must be greater than zero".to_string(),
        });
    }

    let seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];

    if hidden_size % num_heads != 0 {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            ),
        });
    }

    let head_dim = hidden_size / num_heads;
    let mut heads = Vec::with_capacity(num_heads);

    for head in 0..num_heads {
        let mut head_data = Vec::with_capacity(seq_len * head_dim);
        let start_col = head * head_dim;

        for row in 0..seq_len {
            for offset in 0..head_dim {
                head_data.push(hidden.get_2d(row, start_col + offset)?);
            }
        }

        heads.push(Tensor::new(vec![seq_len, head_dim], head_data)?);
    }

    Ok(heads)
}

fn merge_heads(heads: &[Tensor]) -> Result<Tensor> {
    if heads.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

        // Read shape from first head.
    if heads[0].shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: heads[0].shape().len(),
        });
    }

    let num_heads = heads.len();
    let seq_len = heads[0].shape()[0];
    let head_dim = heads[0].shape()[1];
    let hidden_size = num_heads * head_dim;

    for head in heads {
        if head.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: head.shape().len(),
            });
        }

        if head.shape() != [seq_len, head_dim] {
            return Err(MiniInferError::InvalidTensorShape {
                expected: vec![seq_len, head_dim],
                actual: head.shape().to_vec(),
            });
        }
    }

    let mut output = Vec::with_capacity(seq_len * hidden_size);

    for row in 0..seq_len {
        for head_index in 0..num_heads {
            for col in 0..head_dim {
                output.push(heads[head_index].get_2d(row, col)?);
            }
        }
    }

    Tensor::new(vec![seq_len, hidden_size], output)
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
        fn attention_context_computes_attention_per_head() {
            let block = Gpt2BlockWeights {
                c_attn_weight: Tensor::new(
                    vec![4, 12],
                    vec![
                        0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 1.0, 2.0, 3.0,
                        4.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 5.0, 6.0,
                        7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    ],
                )
                .expect("valid c_attn weight"),
                c_attn_bias: Tensor::new(vec![12], vec![0.0; 12]).expect("valid c_attn bias"),
                ..tiny_block_weights()
            };
            let hidden = Tensor::new(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .expect("valid hidden");

            let context = block
                .attention_context(&hidden, 2)
                .expect("attention context should succeed");

            let expected = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 7.0, 8.0];
            assert_eq!(context.shape(), &[2, 4]);
            assert!(
                context
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

    #[test]
    fn causal_softmax_masks_future_positions() {
        let scores = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .expect("valid scores");

        let probabilities = causal_softmax(&scores).expect("causal softmax should succeed");

        let expected = [1.0, 0.0, 0.26894143, 0.7310586];
        assert_eq!(probabilities.shape(), &[2, 2]);
        assert!(
            probabilities
                .data()
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1e-6)
        );
    }

    #[test]
    fn causal_softmax_rejects_non_2d_scores() {
        let scores = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("valid scores");

        let err = causal_softmax(&scores).expect_err("1D scores should fail");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn causal_softmax_rejects_non_square_scores() {
        let scores = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid scores");

        let err = causal_softmax(&scores).expect_err("non-square scores should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![2, 2],
                actual: vec![2, 3],
            }
        );
    }

    #[test]
    fn apply_attention_sublayer_projects_context_and_adds_residual() {
        let block = Gpt2BlockWeights {
            ln_1_weight: Tensor::new(vec![2], vec![0.0, 0.0]).expect("valid ln_1 weight"),
            ln_1_bias: Tensor::new(vec![2], vec![1.0, 0.0]).expect("valid ln_1 bias"),
            c_attn_weight: Tensor::new(
                vec![2, 6],
                vec![
                    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                ],
            )
            .expect("valid c_attn weight"),
            c_attn_bias: Tensor::new(vec![6], vec![0.0; 6]).expect("valid c_attn bias"),
            attn_c_proj_weight: Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("valid attention projection weight"),
            attn_c_proj_bias: Tensor::new(vec![2], vec![0.0, 0.0])
                .expect("valid attention projection bias"),
            ln_2_weight: tensor(&[2]),
            ln_2_bias: tensor(&[2]),
            c_fc_weight: tensor(&[2, 8]),
            c_fc_bias: tensor(&[8]),
            mlp_c_proj_weight: tensor(&[8, 2]),
            mlp_c_proj_bias: tensor(&[2]),
        };
        let hidden = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).expect("valid hidden");

        let output = block
            .apply_attention_sublayer(&hidden, 2, 1e-5)
            .expect("attention sublayer should succeed");

        let expected = [11.0, 20.0, 31.0, 40.0];
        assert_eq!(output.shape(), &[2, 2]);
        assert!(
            output
                .data()
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1e-6)
        );
    }

    #[test]
    fn apply_attention_sublayer_rejects_bad_projection_weight_shape() {
        let block = Gpt2BlockWeights {
            ln_1_weight: tensor(&[2]),
            ln_1_bias: tensor(&[2]),
            c_attn_weight: Tensor::new(
                vec![2, 6],
                vec![
                    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                ],
            )
            .expect("valid c_attn weight"),
            c_attn_bias: Tensor::new(vec![6], vec![0.0; 6]).expect("valid c_attn bias"),
            attn_c_proj_weight: tensor(&[2, 3]),
            attn_c_proj_bias: tensor(&[2]),
            ln_2_weight: tensor(&[2]),
            ln_2_bias: tensor(&[2]),
            c_fc_weight: tensor(&[2, 8]),
            c_fc_bias: tensor(&[8]),
            mlp_c_proj_weight: tensor(&[8, 2]),
            mlp_c_proj_bias: tensor(&[2]),
        };
        let hidden = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("valid hidden");

        let err = block
            .apply_attention_sublayer(&hidden, 2, 1e-5)
            .expect_err("bad attention projection shape should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![2, 2],
                actual: vec![2, 3],
            }
        );
    }

    #[test]
    fn add_bias_rows_adds_bias_to_each_row() {
        let matrix = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix");
        let bias = Tensor::new(vec![3], vec![0.1, 0.2, 0.3]).expect("valid bias");

        let output = add_bias_rows(&matrix, &bias).expect("bias add should succeed");

        assert_eq!(output, vec![1.1, 2.2, 3.3, 4.1, 5.2, 6.3]);
    }

    #[test]
    fn add_bias_rows_rejects_bad_bias_shape() {
        let matrix = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid matrix");
        let bias = Tensor::new(vec![2], vec![0.1, 0.2]).expect("valid bias");

        let err = add_bias_rows(&matrix, &bias).expect_err("bad bias shape should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![3],
                actual: vec![2],
            }
        );
    }

    #[test]
    fn apply_mlp_projects_activated_hidden() {
        let block = Gpt2BlockWeights {
            ln_1_weight: tensor(&[2]),
            ln_1_bias: tensor(&[2]),
            c_attn_weight: tensor(&[2, 6]),
            c_attn_bias: tensor(&[6]),
            attn_c_proj_weight: tensor(&[2, 2]),
            attn_c_proj_bias: tensor(&[2]),
            ln_2_weight: tensor(&[2]),
            ln_2_bias: tensor(&[2]),
            c_fc_weight: Tensor::new(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0])
                .expect("valid mlp expansion weight"),
            c_fc_bias: Tensor::new(vec![2], vec![0.0, 0.0]).expect("valid mlp expansion bias"),
            mlp_c_proj_weight: Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("valid mlp projection weight"),
            mlp_c_proj_bias: Tensor::new(vec![2], vec![0.5, -0.5])
                .expect("valid mlp projection bias"),
        };
        let hidden = Tensor::new(vec![1, 2], vec![3.0, 4.0]).expect("valid hidden");

        let output = block.apply_mlp(&hidden).expect("MLP should succeed");

        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.data(), &[0.5, -0.5]);
    }

    #[test]
    fn apply_mlp_sublayer_normalizes_mlp_input_and_adds_residual() {
        let block = Gpt2BlockWeights {
            ln_1_weight: tensor(&[2]),
            ln_1_bias: tensor(&[2]),
            c_attn_weight: tensor(&[2, 6]),
            c_attn_bias: tensor(&[6]),
            attn_c_proj_weight: tensor(&[2, 2]),
            attn_c_proj_bias: tensor(&[2]),
            ln_2_weight: Tensor::new(vec![2], vec![0.0, 0.0]).expect("valid ln_2 weight"),
            ln_2_bias: Tensor::new(vec![2], vec![0.0, 0.0]).expect("valid ln_2 bias"),
            c_fc_weight: Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("valid mlp expansion weight"),
            c_fc_bias: Tensor::new(vec![2], vec![0.0, 0.0]).expect("valid mlp expansion bias"),
            mlp_c_proj_weight: Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("valid mlp projection weight"),
            mlp_c_proj_bias: Tensor::new(vec![2], vec![0.0, 0.0])
                .expect("valid mlp projection bias"),
        };
        let hidden = Tensor::new(vec![1, 2], vec![3.0, 4.0]).expect("valid hidden");

        let output = block
            .apply_mlp_sublayer(&hidden, 1e-5)
            .expect("MLP sublayer should succeed");

        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.data(), &[3.0, 4.0]);
    }

    #[test]
    fn forward_applies_attention_then_mlp_residual_blocks() {
        let block = Gpt2BlockWeights {
            ln_1_weight: tensor(&[2]),
            ln_1_bias: tensor(&[2]),
            c_attn_weight: tensor(&[2, 6]),
            c_attn_bias: tensor(&[6]),
            attn_c_proj_weight: tensor(&[2, 2]),
            attn_c_proj_bias: tensor(&[2]),
            ln_2_weight: tensor(&[2]),
            ln_2_bias: tensor(&[2]),
            c_fc_weight: tensor(&[2, 2]),
            c_fc_bias: tensor(&[2]),
            mlp_c_proj_weight: tensor(&[2, 2]),
            mlp_c_proj_bias: tensor(&[2]),
        };
        let hidden = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid hidden");

        let output = block
            .forward(&hidden, 2, 1e-5)
            .expect("block forward should succeed");

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
fn split_heads_splits_hidden_dimension_into_heads() {
    let hidden = Tensor::new(
        vec![2, 4],
        vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ],
    )
    .expect("valid hidden");

    let heads = split_heads(&hidden, 2).expect("split heads should succeed");

    assert_eq!(heads.len(), 2);

    assert_eq!(heads[0].shape(), &[2, 2]);
    assert_eq!(heads[0].data(), &[1.0, 2.0, 5.0, 6.0]);

    assert_eq!(heads[1].shape(), &[2, 2]);
    assert_eq!(heads[1].data(), &[3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn merge_heads_merges_multiple_heads_into_hidden_dimension() {
    let head1 = Tensor::new(
        vec![2, 2],
        vec![
            1.0, 2.0,
            3.0, 4.0,
        ],
    )
    .expect("valid head1");
    let head2 = Tensor::new(
        vec![2, 2],
        vec![
            5.0, 6.0,
            7.0, 8.0,
        ],
    )
    .expect("valid head2");

    let merged = merge_heads(&[head1, head2]).expect("merge heads should succeed");

    assert_eq!(merged.shape(), &[2, 4]);
    assert_eq!(merged.data(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}
}