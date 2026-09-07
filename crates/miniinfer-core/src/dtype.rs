use crate::error::{MiniInferError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I8Symmetric,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I8Symmetric => 1,
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(DType::F32),
            "i8_symmetric" => Ok(DType::I8Symmetric),
            other => Err(MiniInferError::InvalidConfig {
                message: format!("unsupported tensor dtype {other}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_reports_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::I8Symmetric.size_in_bytes(), 1);
    }

    #[test]
    fn parses_dtype_strings() {
        assert_eq!(DType::parse("f32").expect("f32 dtype should parse"), DType::F32);
        assert_eq!(
            DType::parse("i8_symmetric").expect("i8_symmetric dtype should parse"),
            DType::I8Symmetric
        );
    }

    #[test]
    fn rejects_unknown_dtype_string() {
        let err = DType::parse("int8").expect_err("unsupported dtype should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "unsupported tensor dtype int8".to_string(),
            }
        );
    }
}
