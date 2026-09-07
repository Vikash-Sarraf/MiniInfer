use crate::error::{MiniInferError, Result};

pub fn softmax(value: &[f32]) -> Result<Vec<f32>> {
    if value.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    let mut probabilities = value.to_vec();
    softmax_in_place(&mut probabilities)?;

    Ok(probabilities)
}

pub fn softmax_in_place(value: &mut [f32]) -> Result<()> {
    if value.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    let max = value
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0;
    for value in value.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    for value in value.iter_mut() {
        *value /= sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_outputs_sum_to_one() {
        let probs = softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_empty_input() {
        let err = softmax(&[]).expect_err("empty softmax input should fail");

        assert_eq!(err, MiniInferError::EmptyInput);
    }

    #[test]
    fn softmax_in_place_matches_softmax() {
        let mut values = vec![2.0, 1.0, 0.0];

        softmax_in_place(&mut values).expect("in-place softmax should succeed");

        let expected = softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");
        assert_eq!(values, expected);
    }

    #[test]
    fn larger_input_gets_larger_probability() {
        let probs = softmax(&[0.0, 1.0, 2.0]).expect("softmax should succeed");

        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn handles_large_values_without_overflow() {
        let probs = softmax(&[1000.0, 1001.0, 1002.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!(probs.iter().all(|value| value.is_finite()));
        assert!((sum - 1.0).abs() < 1e-6);
    }
}