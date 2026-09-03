use crate::{
    error::{MiniInferError, Result}, model::config::ModelConfig, ops::{backend::{OpsBackend, ReferenceBackend}, embedding, vector_add, layer_norm}, runtime::kv_cache::KvCache, tensor::Tensor,
};

use super::{validate_shape, Gpt2BlockWeights};

pub struct Gpt2Weights {
    pub wte: Tensor,
    pub wpe: Tensor,
    pub blocks: Vec<Gpt2BlockWeights>,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
    pub lm_head_weight: LMHead,
}

#[derive(Debug)]
pub enum LMHead {
    Tied,
    Untied(Tensor),
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
        match &self.lm_head_weight {
            LMHead::Tied => {}
            LMHead::Untied(tensor) => {
                validate_shape(tensor, &[config.hidden_size, config.vocab_size])?;
            }
        }

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
        let position_ids: Vec<usize> = (0..token_ids.len()).collect();
        self.embed_tokens_at_positions(token_ids, &position_ids)
    }

    pub fn embed_tokens_at_positions(
        &self,
        token_ids: &[usize],
        position_ids: &[usize],
    ) -> Result<Tensor> {
        if token_ids.len() != position_ids.len() {
            return Err(MiniInferError::LengthMismatch {
                expected: token_ids.len(),
                actual: position_ids.len(),
            });
        }

        let token_embeddings = embedding::embedding_lookup(&self.wte, token_ids)?;
        let position_embeddings = embedding::embedding_lookup(&self.wpe, position_ids)?;

        let hidden_data = vector_add::add(token_embeddings.data(), position_embeddings.data())?;
        Tensor::new(token_embeddings.shape().to_vec(), hidden_data)
    }

    pub fn apply_f_ln(&self, hidden: &Tensor, epsilon: f32) -> Result<Tensor> {
        if hidden.shape().len() != 2 {
            return Err(MiniInferError::WrongRank {
                expected: 2,
                actual: hidden.shape().len(),
            });
        }
        let seq_len = hidden.shape()[0];
        let hidden_size = hidden.shape()[1];

        validate_shape(&self.ln_f_weight, &[hidden_size])?;
        validate_shape(&self.ln_f_bias, &[hidden_size])?;

        let mut output = Vec::with_capacity(seq_len * hidden_size);

        for row in 0..seq_len {
            let mut row_values = Vec::with_capacity(hidden_size);
            for col in 0..hidden_size {
                row_values.push(hidden.get_2d(row, col)?);
            }

            let normalized = layer_norm::layer_norm(
                &row_values,
                self.ln_f_weight.data(),
                self.ln_f_bias.data(),
                epsilon,
            )?;
            output.extend(normalized);
        }
        Tensor::new(vec![seq_len, hidden_size], output)
    }

    pub fn forward(&self, config: &ModelConfig, token_ids: &[usize]) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.forward_with_backend(config, token_ids, &backend)
    }

    pub fn forward_with_backend(
        &self,
        config: &ModelConfig,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        self.validate_shapes(config)?;
        let mut hidden = self.embed_tokens(token_ids)?;

        for block in &self.blocks {
            hidden = block.forward_with_backend(&hidden, config.head_dim(), config.layer_norm_epsilon, backend)?;
        }

        hidden = self.apply_f_ln(&hidden, config.layer_norm_epsilon)?;

        hidden = match &self.lm_head_weight {
            LMHead::Tied => project_tied_lm_head(&hidden, &self.wte)?,
            LMHead::Untied(weight) => backend.matmul(&hidden, weight)?,
        };

        Ok(hidden)
    }

    pub fn forward_last_logits(&self, config: &ModelConfig, token_ids: &[usize]) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.forward_last_logits_with_backend(config, token_ids, &backend)
    }

    pub fn forward_last_logits_with_backend(
        &self,
        config: &ModelConfig,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        self.validate_shapes(config)?;

        let mut hidden = self.embed_tokens(token_ids)?;

        for block in &self.blocks {
            hidden = block.forward_with_backend(&hidden, config.head_dim(), config.layer_norm_epsilon, backend)?;
        }

        hidden = self.apply_f_ln(&hidden, config.layer_norm_epsilon)?;

        let last_hidden = last_hidden_row(&hidden)?;

        match &self.lm_head_weight {
            LMHead::Tied => project_tied_lm_head(&last_hidden, &self.wte),
            LMHead::Untied(weight) => backend.matmul(&last_hidden, weight),
        }
    }

    pub fn forward_next_token_with_cache(
        &self,
        config: &ModelConfig,
        token_id: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.forward_next_token_with_cache_and_backend(config, token_id, kv_cache, &backend)
    }

    pub fn forward_next_token_with_cache_and_backend(
        &self,
        config: &ModelConfig,
        token_id: usize,
        kv_cache: &mut KvCache,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        self.validate_shapes(config)?;

        let position_id = kv_cache.current_position()?;
        let mut hidden = self.embed_tokens_at_positions(&[token_id], &[position_id])?;

        for (layer_index, block) in self.blocks.iter().enumerate() {
            let layer_cache = kv_cache.layer_mut(layer_index)?;
            hidden = block.forward_with_kv_cache_and_backend(
                &hidden,
                config.head_dim(),
                config.layer_norm_epsilon,
                layer_cache,
                backend,
            )?;
        }

        hidden = self.apply_f_ln(&hidden, config.layer_norm_epsilon)?;

        match &self.lm_head_weight {
            LMHead::Tied => project_tied_lm_head(&hidden, &self.wte),
            LMHead::Untied(weight) => backend.matmul(&hidden, weight),
        }
    }
}

fn last_hidden_row(hidden: &Tensor) -> Result<Tensor> {
    if hidden.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: hidden.shape().len(),
        });
    }

    let seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let last_row = seq_len - 1;

    let mut data = Vec::with_capacity(hidden_size);

    for col in 0..hidden_size {
        data.push(hidden.get_2d(last_row, col)?);
    }

    Tensor::new(vec![1, hidden_size], data)
}

fn project_tied_lm_head(hidden: &Tensor, wte: &Tensor) -> Result<Tensor> {
    if hidden.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: hidden.shape().len(),
        });
    }

    validate_shape(wte, &[wte.shape()[0], hidden.shape()[1]])?;

    let seq_len = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let vocab_size = wte.shape()[0];
    let mut output = Vec::with_capacity(seq_len * vocab_size);

    for row in 0..seq_len {
        for token_id in 0..vocab_size {
            let mut sum = 0.0;
            for hidden_col in 0..hidden_size {
                sum += hidden.get_2d(row, hidden_col)? * wte.get_2d(token_id, hidden_col)?;
            }
            output.push(sum);
        }
    }

    Tensor::new(vec![seq_len, vocab_size], output)
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
            eos_token_id: Some(6),
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
            lm_head_weight: LMHead::Untied(tensor(&[4, 8])),
        }
    }

    #[test]
    fn validates_correct_shapes() {
        let config = tiny_config();
        let weights = tiny_weights();

        assert!(weights.validate_shapes(&config).is_ok());
    }

    #[test]
    fn validates_tied_lm_head_without_explicit_weight() {
        let config = tiny_config();
        let weights = Gpt2Weights {
            lm_head_weight: LMHead::Tied,
            ..tiny_weights()
        };

        assert!(weights.validate_shapes(&config).is_ok());
    }

    #[test]
    fn project_tied_lm_head_uses_token_embedding_rows() {
        let hidden = Tensor::new(vec![1, 2], vec![2.0, 3.0]).expect("valid hidden tensor");
        let wte = Tensor::new(
            vec![3, 2],
            vec![
                1.0, 0.0,
                0.0, 1.0,
                1.0, 1.0,
            ],
        )
        .expect("valid token embeddings");

        let logits = project_tied_lm_head(&hidden, &wte).expect("projection should succeed");

        assert_eq!(logits.shape(), &[1, 3]);
        assert_eq!(logits.data(), &[2.0, 3.0, 5.0]);
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
            wte: Tensor::new(vec![2, 2], vec![1.0, 1.0, 2.0, 2.0])
                .expect("valid token embedding"),
            wpe: Tensor::new(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
                .expect("valid position embedding"),
            blocks: vec![tiny_block_weights()],
            ln_f_weight: tensor(&[2]),
            ln_f_bias: tensor(&[2]),
            lm_head_weight: LMHead::Untied(tensor(&[2, 2])),
        };
        let token_ids = vec![1, 0];

        let result = weights.embed_tokens(&token_ids).expect("embedding should succeed");

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[2.1, 2.2, 1.3, 1.4]);
    }

    #[test]
    fn embed_tokens_at_positions_uses_given_position_ids() {
        let weights = Gpt2Weights {
            wte: Tensor::new(vec![2, 2], vec![1.0, 1.0, 10.0, 20.0])
                .expect("valid token embedding"),
            wpe: Tensor::new(vec![4, 2], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0])
                .expect("valid position embedding"),
            blocks: vec![tiny_block_weights()],
            ln_f_weight: tensor(&[2]),
            ln_f_bias: tensor(&[2]),
            lm_head_weight: LMHead::Untied(tensor(&[2, 2])),
        };

        let result = weights
            .embed_tokens_at_positions(&[1], &[3])
            .expect("embedding should succeed");

        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result.data(), &[11.0, 22.0]);
    }

    #[test]
    fn embed_tokens_at_positions_rejects_length_mismatch() {
        let weights = tiny_weights();

        let err = weights
            .embed_tokens_at_positions(&[0, 1], &[0])
            .expect_err("token and position lengths should match");

        assert_eq!(err, MiniInferError::LengthMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn forward_next_token_with_cache_returns_logits_and_updates_cache() {
        let config = tiny_config();
        let weights = tiny_weights();
        let mut cache = KvCache::new(
            config.num_layers,
            config.num_heads,
            config.head_dim(),
            config.max_position_embeddings,
        )
        .expect("cache should be valid");

        let logits = weights
            .forward_next_token_with_cache(&config, 0, &mut cache)
            .expect("cached token forward should succeed");

        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(cache.current_position().expect("position should exist"), 1);
    }

    #[test]
    fn apply_f_ln_normalizes_hidden_rows() {
        let weights = Gpt2Weights {
            ln_f_weight: Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0])
                .expect("valid final layer norm weight"),
            ln_f_bias: Tensor::new(vec![4], vec![0.0, 0.0, 0.0, 0.0])
                .expect("valid final layer norm bias"),
            ..tiny_weights()
        };
        let hidden = Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("valid hidden");

        let output = weights
            .apply_f_ln(&hidden, 1e-5)
            .expect("final layer norm should succeed");

        let expected = [-1.3416355, -0.44721183, 0.44721183, 1.3416355];
        assert_eq!(output.shape(), &[1, 4]);
        assert!(
            output
                .data()
                .iter()
                .zip(expected.iter())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1e-5)
        );
    }

    #[test]
    fn forward_runs_blocks_final_layer_norm_and_lm_head() {
        let config = tiny_config();
        let weights = Gpt2Weights {
            ln_f_weight: Tensor::new(vec![4], vec![0.0, 0.0, 0.0, 0.0])
                .expect("valid final layer norm weight"),
            ln_f_bias: Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0])
                .expect("valid final layer norm bias"),
            lm_head_weight: LMHead::Untied(Tensor::new(
                vec![4, 8],
                vec![
                    1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                ],
            )
            .expect("valid LM head weight")),
            ..tiny_weights()
        };
        let token_ids = vec![2, 1];

        let logits = weights
            .forward(&config, &token_ids)
            .expect("model forward should succeed");

        assert_eq!(logits.shape(), &[2, 8]);
        assert_eq!(
            logits.data(),
            &[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0,
              1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0]
        );
    }

    #[test]
    fn forward_rejects_invalid_lm_head_shape() {
        let config = tiny_config();
        let weights = Gpt2Weights {
            lm_head_weight: LMHead::Untied(tensor(&[4, 7])),
            ..tiny_weights()
        };

        let err = weights
            .forward(&config, &[0])
            .expect_err("bad LM head shape should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![4, 8],
                actual: vec![4, 7],
            }
        );
    }
}