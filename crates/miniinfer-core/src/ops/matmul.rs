use crate::{error::{MiniInferError, Result}, tensor::Tensor};

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let left = a.shape();
    let right = b.shape();

    if left.len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: left.len() });
    }

    if right.len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: right.len() });
    }

    let m = left[0];
    let k = left[1];
    let n = right[1];

    if k != right[0] {
        return Err(MiniInferError::MatMulShapeMismatch { left: left.to_vec(), right: right.to_vec() });
    }

    let mut output = Vec::with_capacity(m * n);

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            
            for i in 0..k {
                let a_val = a.get_2d(row, i)?;
                let b_val = b.get_2d(i, col)?;
                sum += a_val * b_val;
            }

            output.push(sum);
        }
    }
    Tensor::new(vec![m, n], output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multiplies_2x3_by_3x2() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let c = matmul(&a, &b).expect("matmul should succeed");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn rejects_left_tensor_with_wrong_rank() {
        let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).expect("valid tensor");
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let err = matmul(&a, &b).expect_err("left tensor rank should be rejected");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_right_tensor_with_wrong_rank() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");
        let b = Tensor::new(vec![3], vec![7.0, 8.0, 9.0]).expect("valid tensor");

        let err = matmul(&a, &b).expect_err("right tensor rank should be rejected");

        assert_eq!(
            err,
            MiniInferError::WrongRank {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_inner_dimension_mismatch() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");
        let b = Tensor::new(
            vec![4, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        )
        .expect("valid tensor");

        let err = matmul(&a, &b).expect_err("inner dimension mismatch should be rejected");

        assert_eq!(
            err,
            MiniInferError::MatMulShapeMismatch {
                left: vec![2, 3],
                right: vec![4, 2],
            }
        );
    }
}
