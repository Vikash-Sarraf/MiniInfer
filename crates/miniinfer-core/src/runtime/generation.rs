use crate::error::{MiniInferError, Result};
use crate::model::loader::LoadedModel;
use crate::ops::backend::{OpsBackend, ReferenceBackend};
use crate::sampling::greedy::GreedySampler;
use crate::sampling::temperature::TemperatureSampler;
use crate::sampling::Sampler;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>
}

impl GenerationOptions {
    pub fn new(max_new_tokens: usize, temperature: Option<f32>, seed: Option<u64>, top_k: Option<usize>, top_p: Option<f32>) -> Result<Self> {
        let options = GenerationOptions {
            max_new_tokens,
            temperature,
            seed,
            top_k,
            top_p
        };
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(temp) = self.temperature {
            if !temp.is_finite() || temp <= 0.0 {
                return Err(MiniInferError::InvalidTemperature { temperature: temp });
            }
        }

        if let Some(top_k) = self.top_k {
            if top_k == 0 {
                return Err(MiniInferError::InvalidTopK { top_k });
            }
        }

        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(MiniInferError::InvalidTopP { top_p });
            }
        }

        Ok(())
    }

    pub fn generate(&self, model: &LoadedModel, token_ids: &[usize]) -> Result<String> {
        let backend = ReferenceBackend::new();
        self.generate_with_backend(model, token_ids, &backend)
    }

    pub fn generate_with_backend(
        &self,
        model: &LoadedModel,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
    ) -> Result<String> {
        self.generate_with_token_observer_and_backend(model, token_ids, backend, |_, _| {})
    }

    pub fn generate_with_token_observer_and_backend<F>(
        &self,
        model: &LoadedModel,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(usize, usize),
    {
        self.validate()?;
        let mut sampler = self.create_sampler()?;
        generate_with_sampler_and_observer_and_backend(
            model,
            token_ids,
            self.max_new_tokens,
            backend,
            &mut *sampler,
            &mut on_token,
        )
    }

    pub fn generate_streaming_with_backend<F>(
        &self,
        model: &LoadedModel,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.validate()?;
        let mut sampler = self.create_sampler()?;
        generate_streaming_with_sampler_and_backend(
            model,
            token_ids,
            self.max_new_tokens,
            backend,
            &mut *sampler,
            on_text,
        )
    }

    fn create_sampler(&self) -> Result<Box<dyn Sampler>> {
        match self.temperature {
            Some(temperature) => {
                let sampler = TemperatureSampler::with_options(temperature, self.seed, self.top_k, self.top_p)?;
                Ok(Box::new(sampler))
            }
            None => Ok(Box::new(GreedySampler)),
        }
    }
}

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
    let mut sampler = GreedySampler;
    generate_with_sampler_and_backend(model, token_ids, max_new_tokens, backend, &mut sampler)
}

pub fn generate_with_sampler_and_backend(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
    backend: &dyn OpsBackend,
    sampler: &mut dyn Sampler,
) -> Result<String> {
    generate_with_sampler_and_observer_and_backend(
        model,
        token_ids,
        max_new_tokens,
        backend,
        sampler,
        &mut |_, _| {},
    )
}

fn generate_with_sampler_and_observer_and_backend<F>(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
    backend: &dyn OpsBackend,
    sampler: &mut dyn Sampler,
    on_token: &mut F,
) -> Result<String>
where
    F: FnMut(usize, usize),
{
    validate_requested_length(model, token_ids.len(), max_new_tokens)?;

    let mut token_ids = token_ids.to_vec();

    for generated_index in 0..max_new_tokens {
        let logits = model.forward_last_logits_with_backend(&token_ids, backend)?;
        let next_token_id = sampler.sample(&logits)?;

        if model.config().eos_token_id == Some(next_token_id) {
            break;
        }

        token_ids.push(next_token_id);
        on_token(generated_index, next_token_id);
    }
    model.decode_tokens(&token_ids)
}

fn generate_streaming_with_sampler_and_backend<F>(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
    backend: &dyn OpsBackend,
    sampler: &mut dyn Sampler,
    mut on_text: F,
) -> Result<String>
where
    F: FnMut(&str),
{
    validate_requested_length(model, token_ids.len(), max_new_tokens)?;

    let mut token_ids = token_ids.to_vec();
    let mut decoded_text = model.decode_tokens(&token_ids)?;
    on_text(&decoded_text);

    for _ in 0..max_new_tokens {
        let logits = model.forward_last_logits_with_backend(&token_ids, backend)?;
        let next_token_id = sampler.sample(&logits)?;

        if model.config().eos_token_id == Some(next_token_id) {
            break;
        }

        token_ids.push(next_token_id);

        let next_decoded_text = model.decode_tokens(&token_ids)?;
        let suffix = next_decoded_text
            .strip_prefix(&decoded_text)
            .unwrap_or(&next_decoded_text);
        on_text(suffix);
        decoded_text = next_decoded_text;
    }

    Ok(decoded_text)
}

fn validate_requested_length(
    model: &LoadedModel,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let max_positions = model.config().max_position_embeddings;
    let requested_length = prompt_tokens + max_new_tokens;

    if requested_length > max_positions {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "requested sequence length {requested_length} exceeds max_position_embeddings {max_positions}"
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::model::loader::load_model;

    #[test]
    fn generate_greedy_stops_before_decoding_eos_token() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");

        let text = generate_greedy(&model, &[0, 1], 1).expect("generation should succeed");

        assert_eq!(text, "hello world");
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

    #[test]
    fn generation_options_reject_invalid_temperature() {
        assert_eq!(
            GenerationOptions::new(1, Some(0.0), None, None, None),
            Err(MiniInferError::InvalidTemperature { temperature: 0.0 })
        );
    }

    #[test]
    fn generation_options_reject_zero_top_k() {
        assert_eq!(
            GenerationOptions::new(1, Some(1.0), None, Some(0), None),
            Err(MiniInferError::InvalidTopK { top_k: 0 })
        );
    }

    #[test]
    fn generation_options_reject_invalid_top_p() {
        assert_eq!(
            GenerationOptions::new(1, Some(1.0), None, None, Some(1.1)),
            Err(MiniInferError::InvalidTopP { top_p: 1.1 })
        );
    }

    #[test]
    fn generation_options_forwards_top_k_to_temperature_sampler() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");
        let options = GenerationOptions::new(1, Some(1.0), Some(42), Some(1), Some(0.5))
            .expect("valid options");

        let text = options
            .generate(&model, &[0, 1])
            .expect("generation should succeed");

        assert_eq!(text, "hello world");
    }

    #[test]
    fn generation_options_streaming_emits_prompt_without_eos_token() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");
        let options = GenerationOptions::new(1, None, None, None, None).expect("valid options");
        let backend = ReferenceBackend::new();
        let mut chunks = Vec::new();

        let text = options
            .generate_streaming_with_backend(&model, &[0, 1], &backend, |chunk| {
                chunks.push(chunk.to_string());
            })
            .expect("streaming generation should succeed");

        assert_eq!(text, "hello world");
        assert_eq!(chunks.concat(), "hello world");
    }
}