use crate::{error::{MiniInferError, Result}, model::config::ModelConfig, tensor::Tensor};
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
}