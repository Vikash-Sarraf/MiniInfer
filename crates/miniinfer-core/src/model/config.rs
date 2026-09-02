use crate::error::{MiniInferError, Result};
#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    Gpt2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub eos_token_id: Option<usize>,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid vocab size".to_string() });
        }

        if self.max_position_embeddings == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid max position embeddings".to_string() });
        }

        if self.hidden_size == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid hidden size".to_string() });
        }

        if self.num_heads == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid number of heads".to_string() });
        }

        if self.num_layers == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid number of layers".to_string() });
        }

        if self.intermediate_size == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid intermediate size".to_string() });
        }

        if self.layer_norm_epsilon <= 0.0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid layer norm epsilon".to_string() });
        }

        if self.hidden_size % self.num_heads != 0 {
            return Err(MiniInferError::InvalidConfig { message: "Hidden size must be divisible by number of heads".to_string() });
        }

        if let Some(eos_token_id) = self.eos_token_id {
            if eos_token_id >= self.vocab_size {
                return Err(MiniInferError::InvalidConfig { message: "eos_token_id must be less than vocab_size".to_string() });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_validation() {
        let config = ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            layer_norm_epsilon: 1e-5,
            eos_token_id: Some(50256),
        };

        assert!(config.validate().is_ok());

        let invalid_config = ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 0, 
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            layer_norm_epsilon: 1e-5,
            eos_token_id: Some(50256),
        };

        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            layer_norm_epsilon: 1e-5,
            eos_token_id: Some(50256),
        };

        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn rejects_hidden_size_not_divisible_by_num_heads() {
        let config = ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 770, 
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            layer_norm_epsilon: 1e-5,
            eos_token_id: Some(50256),
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_invalid_layer_norm_epsilon() {
        let config = ModelConfig {
            architecture: Architecture::Gpt2,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            layer_norm_epsilon: -1e-5,
            eos_token_id: Some(50256),
        };

        assert!(config.validate().is_err());
    }
}