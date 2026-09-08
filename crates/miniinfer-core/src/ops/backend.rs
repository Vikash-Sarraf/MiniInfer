use crate::{error::{MiniInferError, Result}, ops::{helper, matmul, softmax}, tensor::{Tensor, WeightTensor}};
use ndarray::ArrayView2;

pub trait OpsBackend {
    fn name(&self) -> &'static str;

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    fn matmul_row_by_matrix(
        &self,
        row: &[f32],
        matrix: &[f32],
        matrix_rows: usize,
        matrix_cols: usize,
    ) -> Result<Vec<f32>>;

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>>;

    fn matmul_weight(&self, a: &Tensor, b: &WeightTensor) -> Result<Tensor>;
}

fn validate_row_matrix_shape(
    row: &[f32],
    matrix: &[f32],
    matrix_rows: usize,
    matrix_cols: usize,
) -> Result<()> {
    if matrix_rows == 0 || matrix_cols == 0 {
        return Err(MiniInferError::InvalidConfig {
            message: "matrix dimensions must be greater than zero".to_string(),
        });
    }

    if row.len() != matrix_rows {
        return Err(MiniInferError::LengthMismatch { expected: matrix_rows, actual: row.len() });
    }

    let expected_values = matrix_rows * matrix_cols;
    if matrix.len() != expected_values {
        return Err(MiniInferError::ShapeDataLengthMismatch { expected: expected_values, actual: matrix.len() });
    }

    Ok(())
}

pub struct ReferenceBackend;

impl ReferenceBackend {
    pub fn new() -> Self {
        ReferenceBackend
    }
}

impl Default for ReferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpsBackend for ReferenceBackend {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        matmul::matmul(a, b)
    }

    fn matmul_row_by_matrix(
        &self,
        row: &[f32],
        matrix: &[f32],
        matrix_rows: usize,
        matrix_cols: usize,
    ) -> Result<Vec<f32>> {
        validate_row_matrix_shape(row, matrix, matrix_rows, matrix_cols)?;

        let mut output = vec![0.0; matrix_cols];
        for matrix_row in 0..matrix_rows {
            let coefficient = row[matrix_row];
            let value_row = &matrix[(matrix_row * matrix_cols)..((matrix_row + 1) * matrix_cols)];
            for col in 0..matrix_cols {
                output[col] += coefficient * value_row[col];
            }
        }

        Ok(output)
    }

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>> {
        softmax::softmax(value)
    }

    fn matmul_weight(&self, a: &Tensor, b: &WeightTensor) -> Result<Tensor> {
        self.matmul(a, &b.dequantize()?)
    }
}

pub struct NdArrayBackend;
impl NdArrayBackend {
    pub fn new() -> Self {
        NdArrayBackend
    }
}

impl Default for NdArrayBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpsBackend for NdArrayBackend {
    fn name(&self) -> &'static str {
        "ndarray"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
       let (m, k, n) = helper::validate_matmul_shape(a, b)?;

        let a_view = ArrayView2::from_shape((m, k), a.data()).expect("shape should be validated");
        let b_view = ArrayView2::from_shape((k, n), b.data()).expect("shape should be validated");

        let output = a_view.dot(&b_view);
        let data = output.iter().copied().collect();

        Tensor::new(vec![m, n], data)

    }

    fn matmul_row_by_matrix(
        &self,
        row: &[f32],
        matrix: &[f32],
        matrix_rows: usize,
        matrix_cols: usize,
    ) -> Result<Vec<f32>> {
        validate_row_matrix_shape(row, matrix, matrix_rows, matrix_cols)?;

        let row_view = ArrayView2::from_shape((1, matrix_rows), row).expect("shape should be validated");
        let matrix_view = ArrayView2::from_shape((matrix_rows, matrix_cols), matrix).expect("shape should be validated");

        let output = row_view.dot(&matrix_view);
        Ok(output.iter().copied().collect())
    }

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>> { 
        softmax::softmax(value)    
    }
    
    fn matmul_weight(&self, a: &Tensor, b: &WeightTensor) -> Result<Tensor> {
        self.matmul(a, &b.dequantize()?)
    }
}

#[cfg(test)]
    mod tests {
    use super::*;

    #[test]
    fn reference_backend_works_through_trait() {
        let backend = ReferenceBackend::new();
        let backend: &dyn OpsBackend = &backend;

        assert_eq!(backend.name(), "reference");
    }

    #[test]
    fn reference_backend_name() {
        let backend = ReferenceBackend::new();
        assert_eq!(backend.name(), "reference");
    }

    #[test]
    fn reference_backend_matmul() {
        let backend = ReferenceBackend::new();

        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let c = backend.matmul(&a, &b).expect("matmul should succeed");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn reference_backend_matmul_row_by_matrix() {
        let backend = ReferenceBackend::new();

        let output = backend.matmul_row_by_matrix(
            &[0.25, 0.75],
            &[2.0, 4.0, 6.0, 8.0],
            2,
            2,
        )
        .expect("row-matrix multiply should succeed");

        assert_eq!(output, &[5.0, 7.0]);
    }

    #[test]
    fn reference_backend_softmax() {
        let backend = ReferenceBackend::new();

        let probs = backend.softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndarray_backend_works_through_trait() {
        let backend = NdArrayBackend::new();
        let backend: &dyn OpsBackend = &backend;

        assert_eq!(backend.name(), "ndarray");
    }

    #[test]
    fn ndarray_backend_name() {
        let backend = NdArrayBackend::new();
        assert_eq!(backend.name(), "ndarray");
    }

    #[test]
    fn ndarray_backend_matmul() {
        let backend = NdArrayBackend::new();

        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let c = backend.matmul(&a, &b).expect("matmul should succeed");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn ndarray_backend_matmul_row_by_matrix() {
        let backend = NdArrayBackend::new();

        let output = backend.matmul_row_by_matrix(
            &[0.25, 0.75],
            &[2.0, 4.0, 6.0, 8.0],
            2,
            2,
        )
        .expect("row-matrix multiply should succeed");

        assert_eq!(output, &[5.0, 7.0]);
    }

    #[test]
    fn ndarray_backend_softmax() {
        let backend = NdArrayBackend::new();

        let probs = backend.softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }
}