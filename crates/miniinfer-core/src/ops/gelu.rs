use crate::error::{MiniInferError, Result};
pub fn gelu(values: &[f32]) -> Result<Vec<f32>> {
    if values.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    let output = values
        .iter()
        .map(|x| {
            let x = *x;
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * (x * x * x))).tanh())
        })
        .collect();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_works() {
        let input = vec![0.0, 1.0, -1.0];
        let output = gelu(&input).expect("gelu should succeed");
        assert_eq!(output.len(), input.len());

        let expected = vec![0.0, 0.841192, -0.158808];
        assert!(output.iter().zip(expected.iter()).all(|(o, e)| (o - e).abs() < 1e-5));
    }

    #[test]
    fn rejects_empty_input() {
        let err = gelu(&[]).expect_err("empty gelu input should fail");
        assert_eq!(err, MiniInferError::EmptyInput);
    }

}