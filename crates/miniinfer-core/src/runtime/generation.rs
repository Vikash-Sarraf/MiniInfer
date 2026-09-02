use crate::error::{MiniInferError, Result};
use crate::model::loader::LoadedModel;
use crate::ops::backend::{OpsBackend, ReferenceBackend};
use crate::sampling::greedy::GreedySampler;
use crate::sampling::Sampler;

pub fn generate_greedy(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
) -> Result<String> {
    let backend = ReferenceBackend::new();
    generate_greedy_with_backend(model, token_ids, max_new_tokens, &backend)
}

pub fn generate_greedy_with_backend(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
    backend: &dyn OpsBackend,
) -> Result<String> {
    let max_positions = model.config().max_position_embeddings;
    let requested_length = token_ids.len() + max_new_tokens;

    if requested_length > max_positions {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "requested sequence length {requested_length} exceeds max_position_embeddings {max_positions}"
            ),
        });
    }

    let mut sampler = GreedySampler;
    let mut token_ids = token_ids.to_vec();

    for _ in 0..max_new_tokens {
        let logits = model.forward_last_logits_with_backend(&token_ids, backend)?;
        let next_token_id = sampler.sample(&logits)?;
        token_ids.push(next_token_id);
    }
    model.decode_tokens(&token_ids)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::model::loader::load_model;

    #[test]
    fn generate_greedy_appends_one_token() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");

        let text = generate_greedy(&model, &[0, 1], 1).expect("generation should succeed");

        assert_eq!(text, "hello world <unused_6>");
    }

    #[test]
    fn generate_greedy_rejects_sequence_longer_than_context_window() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");

        let err = generate_greedy(&model, &[0, 1], 7)
            .expect_err("generation past max_position_embeddings should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message:
                    "requested sequence length 9 exceeds max_position_embeddings 8".to_string(),
            }
        );
    }
}