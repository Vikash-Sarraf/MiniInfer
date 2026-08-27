use crate::{error::{MiniInferError, Result}, model::config::ModelConfig, ops::{embedding, layer_norm, vector_add}, tensor::Tensor};
pub struct Gpt2Weights {
    pub wte: Tensor,
    pub wpe: Tensor,
    pub blocks: Vec<Gpt2BlockWeights>,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
    pub lm_head_weight: Tensor,
}

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

impl Gpt2Weights {
    pub fn validate_shapes(&self, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        validate_shape(&self.wte, &[config.vocab_size, config.hidden_size])?;
        validate_shape(&self.wpe, &[config.max_position_embeddings, config.hidden_size])?;

        if self.blocks.len() != config.num_layers {
            return Err(MiniInferError::InvalidConfig {
                message: "number of GPT-2 blocks must match num_layers".to_string(),
            });
        }

        validate_shape(&self.ln_f_weight, &[config.hidden_size])?;
        validate_shape(&self.ln_f_bias, &[config.hidden_size])?;
        validate_shape(&self.lm_head_weight, &[config.hidden_size, config.vocab_size])?;

        for block in &self.blocks {
            validate_shape(&block.ln_1_weight, &[config.hidden_size])?;
            validate_shape(&block.ln_1_bias, &[config.hidden_size])?;

            validate_shape(&block.c_attn_weight, &[config.hidden_size, 3 * config.hidden_size])?;
            validate_shape(&block.c_attn_bias, &[3 * config.hidden_size])?;

            validate_shape(&block.attn_c_proj_weight, &[config.hidden_size, config.hidden_size])?;
            validate_shape(&block.attn_c_proj_bias, &[config.hidden_size])?;

            validate_shape(&block.ln_2_weight, &[config.hidden_size])?;
            validate_shape(&block.ln_2_bias, &[config.hidden_size])?;

            validate_shape(&block.c_fc_weight, &[config.hidden_size, config.intermediate_size])?;
            validate_shape(&block.c_fc_bias, &[config.intermediate_size])?;

            validate_shape(&block.mlp_c_proj_weight, &[config.intermediate_size, config.hidden_size])?;
            validate_shape(&block.mlp_c_proj_bias, &[config.hidden_size])?;
        }
        Ok(())
    }

    pub fn embed_tokens(&self, token_ids: &[usize]) -> Result<Tensor> {
        let token_embeddings = embedding::embedding_lookup(&self.wte, token_ids)?;
        let position_ids: Vec<usize> = (0..token_ids.len()).collect();
        let position_embeddings = embedding::embedding_lookup(&self.wpe, &position_ids)?;

        let hidden_data = vector_add::add(token_embeddings.data(), position_embeddings.data())?;
        Tensor::new(token_embeddings.shape().to_vec(), hidden_data)
    }
}

fn validate_shape(tensor: &Tensor, expected_shape: &[usize]) -> Result<()> {
    if tensor.shape() != expected_shape {
        return Err(MiniInferError::InvalidTensorShape {
            expected: expected_shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

impl Gpt2BlockWeights {
    pub fn apply_ln_1(&self, hidden: &Tensor, epsilon: f32) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank { expected: 2, actual: hidden.shape().len() });
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

            let normalized = layer_norm::layer_norm(&row_values, self.ln_1_weight.data(), self.ln_1_bias.data(), epsilon)?;
            output.extend(normalized);
        }
        Tensor::new(vec![seq_len, hidden_size], output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::Architecture;

    fn tensor(shape: &[usize]) -> Tensor {
        let len = shape.iter().product();
        Tensor::new(shape.to_vec(), vec![0.0; len]).expect("test tensor shape should be valid")
    }

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 8,
            max_position_embeddings: 8,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 16,
            layer_norm_epsilon: 1e-5,
        }
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

    fn tiny_weights() -> Gpt2Weights {
        Gpt2Weights {
            wte: tensor(&[8, 4]),
            wpe: tensor(&[8, 4]),
            blocks: vec![tiny_block_weights()],
            ln_f_weight: tensor(&[4]),
            ln_f_bias: tensor(&[4]),
            lm_head_weight: tensor(&[4, 8]),
        }
    }

    #[test]
    fn validates_correct_shapes() {
        let config = tiny_config();
        let weights = tiny_weights();

        assert!(weights.validate_shapes(&config).is_ok());
    }

    #[test]
    fn rejects_wrong_block_count() {
        let config = tiny_config();
        let weights = Gpt2Weights {
            blocks: vec![],
            ..tiny_weights()
        };

        let err = weights
            .validate_shapes(&config)
            .expect_err("wrong block count should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "number of GPT-2 blocks must match num_layers".to_string(),
            }
        );
    }

    #[test]
    fn rejects_wrong_token_embedding_shape() {
        let config = tiny_config();
        let weights = Gpt2Weights {
            wte: tensor(&[7, 4]),
            ..tiny_weights()
        };

        let err = weights
            .validate_shapes(&config)
            .expect_err("wrong token embedding shape should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![8, 4],
                actual: vec![7, 4],
            }
        );
    }

    #[test]
    fn embed_tokens() {
        let weights = Gpt2Weights {
            wte: Tensor::new(
                vec![2, 2],
                vec![
                    1.0, 1.0,
                    2.0, 2.0,
                ],
            )
            .expect("valid token embedding"),
            wpe: Tensor::new(
                vec![2, 2],
                vec![
                    0.1, 0.2,
                    0.3, 0.4,
                ],
            )
            .expect("valid position embedding"),
            blocks: vec![tiny_block_weights()],
            ln_f_weight: tensor(&[2]),
            ln_f_bias: tensor(&[2]),
            lm_head_weight: tensor(&[2, 2]),
        };
        let token_ids = vec![1, 0];

        let result = weights.embed_tokens(&token_ids).expect("embedding should succeed");

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[2.1, 2.2, 1.3, 1.4]);
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
}