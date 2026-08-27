use crate::error::{MiniInferError, Result};
pub fn layer_norm(
    values: &[f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>> {
    if values.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

     if gamma.len() != values.len() {
        return Err(MiniInferError::LengthMismatch {
            expected: values.len(),
            actual: gamma.len(),
        });
    }

    if beta.len() != values.len() {
        return Err(MiniInferError::LengthMismatch {
            expected: values.len(),
            actual: beta.len(),
        });
    }

    if epsilon <= 0.0 {
        return Err(MiniInferError::InvalidEpsilon);
    }
    
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    
    let var = values.iter()
        .map(|x|{ 
            let centered = *x - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    
    let denom = (var + epsilon).sqrt();
    let output = values.iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((value, gamma), beta)| {
            let normalized = (*value - mean) / denom;
            normalized * *gamma + *beta
        }).collect();

    Ok(output)
}

#[cfg(test)]
    mod tests {
    use super::*;

    #[test]
    fn layer_norm_check() {
        let values = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let epsilon = 1e-5;

        let output = layer_norm(&values, &gamma, &beta, epsilon).expect("layer_norm should succeed");
        assert_eq!(output.len(), values.len());
        let expected = vec![-1.2247356, 0.0, 1.2247356];
        assert!(output.iter().zip(expected.iter()).all(|(&x, &y)| (x - y).abs() < 1e-5));
    }

    #[test]
    fn rejects_empty_input() {
        let values: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let epsilon = 1e-5;

        let err = layer_norm(&values, &gamma, &beta, epsilon).expect_err("empty input should fail");
        assert_eq!(err, MiniInferError::EmptyInput);
    }

    #[test]
    fn rejects_gamma_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let epsilon = 1e-5;

        let err = layer_norm(&values, &gamma, &beta, epsilon).expect_err("gamma length mismatch should fail");
        assert_eq!(err, MiniInferError::LengthMismatch { expected: 3, actual: 2 });
    }

    #[test]
    fn rejects_beta_mismatch() {
        let values = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0]; 
        let epsilon = 1e-5;

        let err = layer_norm(&values, &gamma, &beta, epsilon).expect_err("beta length mismatch should fail");
        assert_eq!(err, MiniInferError::LengthMismatch { expected: 3, actual: 2 });
    }
    
    #[test]
    fn rejects_invalid_epsilon() {
        let values = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let epsilon = -1e-5;

        let err = layer_norm(&values, &gamma, &beta, epsilon).expect_err("invalid epsilon should fail");
        assert_eq!(err, MiniInferError::InvalidEpsilon);
    }
}