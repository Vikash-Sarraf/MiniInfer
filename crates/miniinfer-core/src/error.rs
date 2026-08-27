#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiniInferError {
    EmptyShape,
    ZeroDimension,
    ShapeDataLengthMismatch {
        expected: usize,
        actual: usize,
    },
    WrongRank {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        index: usize,
        len: usize,
    },
    MatMulShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    EmptyInput,
}

pub type Result<T> = std::result::Result<T, MiniInferError>;