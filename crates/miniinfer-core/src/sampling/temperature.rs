use crate::{
    error::{MiniInferError, Result},
    ops::softmax::softmax,
    sampling::Sampler,
};

use rand::{rngs::StdRng, RngExt, SeedableRng};

pub struct TemperatureSampler {
    temperature: f32,
    rng: StdRng,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

impl TemperatureSampler {
    pub fn new(temperature: f32) -> Result<Self> {
        Self::with_options(temperature, None, None, None)
    }

    pub fn with_seed(temperature: f32, seed: u64) -> Result<Self> {
        Self::with_options(temperature, Some(seed), None, None)
    }

    pub fn with_options(temperature: f32, seed: Option<u64>, top_k: Option<usize>, top_p: Option<f32>) -> Result<Self> {
        Self::validate_temperature(temperature)?;

        if let Some(top_k) = top_k {
            Self::validate_top_k(top_k)?;
        }

        if let Some(top_p) = top_p {
            Self::validate_top_p(top_p)?;
        }

        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut rng = rand::rng();
                StdRng::from_rng(&mut rng)
            }
        };

        Ok(TemperatureSampler {
            temperature,
            rng,
            top_k,
            top_p,
        })
    }

    fn validate_top_k(top_k: usize) -> Result<()> {
        if top_k == 0 {
            return Err(MiniInferError::InvalidTopK { top_k });
        }
        Ok(())
    }

    fn validate_top_p(top_p: f32) -> Result<()> {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(MiniInferError::InvalidTopP { top_p });
        }
        Ok(())
    }

    fn validate_temperature(temperature: f32) -> Result<()> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(MiniInferError::InvalidTemperature { temperature });
        }

        Ok(())
    }
}

impl Sampler for TemperatureSampler {
    fn sample(&mut self, logits: &crate::tensor::Tensor) -> Result<usize> {
        if logits.shape().len() != 2 {
            return Err(crate::error::MiniInferError::WrongRank {
                expected: 2,
                actual: logits.shape().len(),
            });
        }

        let last_row = logits.shape()[0] - 1;
        let vocab_size = logits.shape()[1];
        let mut effective_top_k: usize = vocab_size;

        Self::validate_temperature(self.temperature)?;
        if let Some(top_k) = self.top_k {
            Self::validate_top_k(top_k)?;
            effective_top_k = top_k.min(vocab_size);
        }

        let mut candidates = Vec::with_capacity(vocab_size);
        for token_id in 0..vocab_size {
            let logit = logits.get_2d(last_row, token_id)?;
            let scaled = logit / self.temperature;
            candidates.push((token_id, scaled));
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(effective_top_k);
        let candidate_logits: Vec<f32> = candidates.iter().map(|&(_, logit)| logit).collect();

        let mut prob = softmax(&candidate_logits)?;
        if let Some(top_p) = self.top_p {
            Self::validate_top_p(top_p)?;

            let mut cumulative_prob = 0.0;
            let mut keep_len = prob.len();
            for (index, probability) in prob.iter().enumerate() {
                cumulative_prob += *probability;
                if cumulative_prob >= top_p {
                    keep_len = index + 1;
                    break;
                }
            }

            candidates.truncate(keep_len);
            prob.truncate(keep_len);

            let kept_sum: f32 = prob.iter().sum();
            for probability in &mut prob {
                *probability /= kept_sum;
            }
        }

        let random_val = self.rng.random_range(0.0..1.0);

        let mut cumulative_prob = 0.0;

        for candidate_index in 0..candidates.len() {
            cumulative_prob += prob[candidate_index];

            if random_val < cumulative_prob {
                return Ok(candidates[candidate_index].0);
            }
        }

        Ok(candidates[candidates.len() - 1].0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_invalid_temperature(result: Result<TemperatureSampler>, expected: f32) {
        match result {
            Err(MiniInferError::InvalidTemperature { temperature }) if expected.is_nan() => {
                assert!(temperature.is_nan());
            }
            Err(MiniInferError::InvalidTemperature { temperature }) => {
                assert_eq!(temperature, expected);
            }
            Err(err) => panic!("unexpected error: {err:?}"),
            Ok(_) => panic!("expected invalid temperature error"),
        }
    }

    fn expect_invalid_top_k(result: Result<TemperatureSampler>, expected: usize) {
        match result {
            Err(MiniInferError::InvalidTopK { top_k }) => assert_eq!(top_k, expected),
            Err(err) => panic!("unexpected error: {err:?}"),
            Ok(_) => panic!("expected invalid top-k error"),
        }
    }

    fn expect_invalid_top_p(result: Result<TemperatureSampler>, expected: f32) {
        match result {
            Err(MiniInferError::InvalidTopP { top_p }) if expected.is_nan() => {
                assert!(top_p.is_nan());
            }
            Err(MiniInferError::InvalidTopP { top_p }) => {
                assert_eq!(top_p, expected);
            }
            Err(err) => panic!("unexpected error: {err:?}"),
            Ok(_) => panic!("expected invalid top-p error"),
        }
    }

    #[test]
    fn temperature_sampler_rejects_non_positive_temperature() {
        expect_invalid_temperature(TemperatureSampler::new(0.0), 0.0);

        expect_invalid_temperature(TemperatureSampler::new(-1.0), -1.0);

        expect_invalid_temperature(TemperatureSampler::new(f32::NAN), f32::NAN);
    }

    #[test]
    fn temperature_sampler_rejects_non_finite_temperature() {
        expect_invalid_temperature(TemperatureSampler::new(f32::INFINITY), f32::INFINITY);

        expect_invalid_temperature(TemperatureSampler::new(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    #[test]
    fn sample_selects_token_based_on_temperature() {
        let logits = crate::tensor::Tensor::new(
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ],
        )
        .expect("valid logits");

        let mut sampler = TemperatureSampler::with_seed(1.0, 42).expect("valid temperature");
        let token_id = sampler.sample(&logits).expect("sample should succeed");

        assert!(token_id < 3);
    }

    #[test]
    fn temperature_sampler_rejects_zero_top_k() {
        expect_invalid_top_k(TemperatureSampler::with_options(1.0, None, Some(0), Some(0.5)), 0);
    }

    #[test]
    fn temperature_sampler_rejects_invalid_top_p() {
        expect_invalid_top_p(TemperatureSampler::with_options(1.0, None, None, Some(0.0)), 0.0);

        expect_invalid_top_p(TemperatureSampler::with_options(1.0, None, None, Some(1.1)), 1.1);

        expect_invalid_top_p(
            TemperatureSampler::with_options(1.0, None, None, Some(f32::NAN)),
            f32::NAN,
        );
    }

    #[test]
    fn top_k_one_selects_highest_logit_token() {
        let logits = crate::tensor::Tensor::new(vec![1, 3], vec![1.0, 10.0, 2.0])
            .expect("valid logits");

        let mut sampler =
            TemperatureSampler::with_options(1.0, Some(42), Some(1), Some(0.5)).expect("valid sampler");
        let token_id = sampler.sample(&logits).expect("sample should succeed");

        assert_eq!(token_id, 1);
    }

    #[test]
    fn top_k_larger_than_vocab_uses_full_vocab() {
        let logits = crate::tensor::Tensor::new(vec![1, 3], vec![1.0, 10.0, 2.0])
            .expect("valid logits");

        let mut sampler =
            TemperatureSampler::with_options(1.0, Some(42), Some(10), Some(0.5)).expect("valid sampler");
        let token_id = sampler.sample(&logits).expect("sample should succeed");

        assert!(token_id < 3);
    }

    #[test]
    fn top_p_keeps_cutoff_token() {
        let logits = crate::tensor::Tensor::new(vec![1, 3], vec![10.0, 1.0, 0.0])
            .expect("valid logits");

        let mut sampler =
            TemperatureSampler::with_options(1.0, Some(42), None, Some(0.5)).expect("valid sampler");
        let token_id = sampler.sample(&logits).expect("sample should succeed");

        assert_eq!(token_id, 0);
    }
}