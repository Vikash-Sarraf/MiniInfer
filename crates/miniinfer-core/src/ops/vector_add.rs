use crate::error::{MiniInferError, Result};

pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.is_empty() || b.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(MiniInferError::LengthMismatch { expected: a.len(), actual: b.len() });
    }

    let output = a.iter().zip(b).map(|(x,y)| {
        *x + *y
    }).collect();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_vectors_elements() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let expected = vec![5.0, 7.0, 9.0];
        let actual = add(&a, &b).expect("should add the two vectors");
        assert_eq!(expected, actual);
    }

    #[test]
    fn rejects_empty_input() {
        let a = vec![];
        let b = vec![1.0];
        let err = add(&a, &b).expect_err("empty input should be rejected");
        assert_eq!(err, MiniInferError::EmptyInput);
    }

    #[test]
    fn rejects_length_mismatch() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        let err = add(&a, &b).expect_err("length mismatch should be rejected");
        assert_eq!(err, MiniInferError::LengthMismatch { expected: 1, actual: 2 });
    }
}