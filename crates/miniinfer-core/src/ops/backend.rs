use crate::{
    error::{MiniInferError, Result},
    ops::{helper, matmul, softmax},
    tensor::{QuantizationScale, QuantizedTensor, Tensor, WeightTensor, PackedI8Weight},
};
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
        match b {
            WeightTensor::F32(weight) => self.matmul(a, weight),
            WeightTensor::I8Symmetric(weight) => matmul_i8_symmetric_weight(a, weight),
            WeightTensor::PackedI8Symmetric(weight) => {
                if a.shape().len() == 2 && a.shape()[0] == 1 {
                    matmul_packed_i8_weight(a, weight)
                } else {
                    self.matmul(a, &weight.dequantize()?)
                }
            }
        }
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
        match b {
            WeightTensor::F32(weight) => self.matmul(a, weight),
            WeightTensor::I8Symmetric(weight) => self.matmul(a, &weight.dequantize()?),
            WeightTensor::PackedI8Symmetric(weight) => {
                if a.shape().len() == 2 && a.shape()[0] == 1 {
                    matmul_packed_i8_weight(a, weight)
                } else {
                    self.matmul(a, &weight.dequantize()?)
                }
            }
        }
    }
}

fn quantized_weight_value(
    data: &[i8],
    cols: usize,
    row: usize,
    col: usize,
    scale: &QuantizationScale,
) -> f32 {
    let value = data[row * cols + col] as f32;
    let scale = match scale {
        QuantizationScale::PerTensor(scale) => *scale,
        QuantizationScale::PerAxis { axis, scales } => match *axis {
            0 => scales[row],
            1 => scales[col],
            _ => unreachable!("quantized tensor validation rejects invalid scale axes"),
        },
    };

    value * scale
}

fn matmul_i8_symmetric_weight(
    a: &Tensor,
    b: &QuantizedTensor,
) -> Result<Tensor> {
    if a.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: a.shape().len(),
        });
    }

    if b.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: b.shape().len(),
        });
    }

    let a_rows = a.shape()[0];
    let a_cols = a.shape()[1];
    let b_rows = b.shape()[0];
    let b_cols = b.shape()[1];

    if a_cols != b_rows {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![a_cols],
            actual: vec![b_rows],
        });
    }

    let mut output = vec![0.0; a_rows * b_cols];

    for row in 0..a_rows {
        for col in 0..b_cols {
            let mut sum = 0.0;

            for inner in 0..a_cols {
                let lhs = a.data()[row * a_cols + inner];
                let rhs = quantized_weight_value(
                    b.data(),
                    b_cols,
                    inner,
                    col,
                    b.scale(),
                );

                sum += lhs * rhs;
            }

            output[row * b_cols + col] = sum;
        }
    }

    Tensor::new(vec![a_rows, b_cols], output)
}

fn validate_packed_i8_matmul_shape(a: &Tensor, b: &PackedI8Weight) -> Result<(usize, usize, usize)> {
    if a.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: a.shape().len(),
        });
    }

    if b.shape().len() != 2 {
        return Err(MiniInferError::WrongRank {
            expected: 2,
            actual: b.shape().len(),
        });
    }

    let a_rows = a.shape()[0];
    let a_cols = a.shape()[1];
    let b_rows = b.shape()[0];
    let b_cols = b.shape()[1];

    if a_cols != b_rows {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![a_cols],
            actual: vec![b_rows],
        });
    }

    Ok((a_rows, a_cols, b_cols))
}

fn matmul_packed_i8_weight(a: &Tensor, b: &PackedI8Weight) -> Result<Tensor> {
    let (a_rows, a_cols, b_cols) = validate_packed_i8_matmul_shape(a, b)?;
    let mut output = vec![0.0; a_rows * b_cols];

    match b.scale() {
        QuantizationScale::PerTensor(scale) => {
            for row in 0..a_rows {
                let input_row = &a.data()[row * a_cols..(row + 1) * a_cols];

                for col in 0..b_cols {
                    let weight_col = &b.data_by_col()[col * a_cols..(col + 1) * a_cols];
                    let mut sum = 0.0;

                    for inner in 0..a_cols {
                        sum += input_row[inner] * weight_col[inner] as f32;
                    }

                    output[row * b_cols + col] = sum * *scale;
                }
            }
        }
        QuantizationScale::PerAxis { axis: 1, scales } => {
            for row in 0..a_rows {
                let input_data = &a.data()[row * a_cols..(row + 1) * a_cols];

                for col in 0..b_cols {
                    let weight_col = &b.data_by_col()[col * a_cols..(col + 1) * a_cols];
                    let mut sum = 0.0;

                    for inner in 0..a_cols {
                        sum += input_data[inner] * weight_col[inner] as f32;
                    }

                    output[row * b_cols + col] = sum * scales[col];
                }
            }
        }
        _ => {
            for row in 0..a_rows {
                let input_row = &a.data()[row * a_cols..(row + 1) * a_cols];

                for col in 0..b_cols {
                    let weight_col = &b.data_by_col()[col * a_cols..(col + 1) * a_cols];
                    let mut sum = 0.0;

                    for inner in 0..a_cols {
                        let scale = match b.scale() {
                            QuantizationScale::PerAxis { axis: 0, scales } => scales[inner],
                            _ => unreachable!("handled above"),
                        };
                        sum += input_row[inner] * weight_col[inner] as f32 * scale;
                    }

                    output[row * b_cols + col] = sum;
                }
            }
        }
    }
    Tensor::new(vec![a_rows, b_cols], output)
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
    fn reference_backend_matmul_weight_i8_symmetric_per_tensor() {
        let backend = ReferenceBackend::new();
        let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
        let weight = QuantizedTensor::new(
            vec![2, 2],
            vec![2, -1, 0, 3],
            QuantizationScale::PerTensor(0.5),
        )
        .expect("valid quantized weight");

        let output = backend
            .matmul_weight(&input, &WeightTensor::I8Symmetric(weight))
            .expect("quantized matmul should succeed");

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.data(), &[1.0, 2.5, 3.0, 4.5]);
    }

    #[test]
    fn reference_backend_matmul_weight_i8_symmetric_per_column() {
        let backend = ReferenceBackend::new();
        let input = Tensor::new(vec![1, 2], vec![2.0, 3.0]).expect("valid input");
        let weight = QuantizedTensor::new(
            vec![2, 2],
            vec![1, 2, 3, 4],
            QuantizationScale::PerAxis {
                axis: 1,
                scales: vec![0.5, 0.25],
            },
        )
        .expect("valid quantized weight");

        let output = backend
            .matmul_weight(&input, &WeightTensor::I8Symmetric(weight))
            .expect("quantized matmul should succeed");

        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.data(), &[5.5, 4.0]);
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

    #[test]
    fn reference_backend_matmul_weight_packed_i8_symmetric_matches_dequantized() {
        let backend = &ReferenceBackend::new() as &dyn OpsBackend;
        let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
        let quantized = QuantizedTensor::new(
            vec![2, 3],
            vec![
                1, 2, 3,
                4, 5, 6,
            ],
            QuantizationScale::PerAxis {
                axis: 1,
                scales: vec![0.5, 0.25, 0.125],
            },
        )
        .expect("valid quantized weight");

        let expected = backend
            .matmul(&input, &quantized.dequantize().expect("dequantize should succeed"))
            .expect("fp32 matmul should succeed");

        let packed = PackedI8Weight::new(quantized).expect("packing should succeed");
        let actual = backend
            .matmul_weight(&input, &WeightTensor::PackedI8Symmetric(packed))
            .expect("packed matmul should succeed");

        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn ndarray_backend_matmul_weight_packed_i8_symmetric_matches_dequantized() {
        let backend = &NdArrayBackend::new() as &dyn OpsBackend;
        let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
        let quantized = QuantizedTensor::new(
            vec![2, 3],
            vec![1, 2, 3, 4, 5, 6],
            QuantizationScale::PerAxis {
                axis: 1,
                scales: vec![0.5, 0.25, 0.125],
            },
        )
        .expect("valid quantized weight");

        let expected = backend
            .matmul(&input, &quantized.dequantize().expect("dequantize should succeed"))
            .expect("fp32 matmul should succeed");

        let packed = PackedI8Weight::new(quantized).expect("packing should succeed");
        let actual = backend
            .matmul_weight(&input, &WeightTensor::PackedI8Symmetric(packed))
            .expect("packed matmul should succeed");

        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.data().iter().zip(expected.data()) {
            assert_close(*actual, *expected);
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "actual {actual} expected {expected}");
    }
}