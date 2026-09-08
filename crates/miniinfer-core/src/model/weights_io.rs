use std::{collections::HashMap, fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use serde::Deserialize;

use crate::{
    dtype::DType,
    error::{MiniInferError, Result},
    model::{
        config::ModelConfig,
        gpt2::{Gpt2BlockWeights, Gpt2Weights, LMHead},
    },
    tensor::{QuantizationScale, QuantizedTensor, Tensor},
};

#[derive(Deserialize)]
pub(super) struct TensorFile {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f32>,
}

#[derive(Deserialize)]
struct Gpt2BlockWeightsFile {
    ln_1_weight: TensorFile,
    ln_1_bias: TensorFile,
    c_attn_weight: TensorFile,
    c_attn_bias: TensorFile,
    attn_c_proj_weight: TensorFile,
    attn_c_proj_bias: TensorFile,
    ln_2_weight: TensorFile,
    ln_2_bias: TensorFile,
    c_fc_weight: TensorFile,
    c_fc_bias: TensorFile,
    mlp_c_proj_weight: TensorFile,
    mlp_c_proj_bias: TensorFile,
}

#[derive(Deserialize)]
struct Gpt2WeightsFile {
    wte: TensorFile,
    wpe: TensorFile,
    blocks: Vec<Gpt2BlockWeightsFile>,
    ln_f_weight: TensorFile,
    ln_f_bias: TensorFile,
    lm_head: Option<LmHeadFile>,
    lm_head_weight: Option<TensorFile>,
}

#[derive(Deserialize)]
pub(super) struct LmHeadFile {
    #[serde(rename = "type")]
    pub(super) head_type: String,
    pub(super) weight: Option<TensorFile>,
}

#[derive(Deserialize)]
struct BinaryWeightsIndexFile {
    format_version: u32,
    dtype: Option<String>,
    endianness: String,
    tensors: HashMap<String, BinaryTensorIndexFile>,
    lm_head: Option<LmHeadFile>,
}

#[derive(Deserialize)]
struct BinaryTensorIndexFile {
    shape: Vec<usize>,
    dtype: Option<String>,
    offset_bytes: u64,
    len: usize,
    scale: Option<f32>,
}

pub(super) fn load_gpt2_weights_from_model_dir(
    model_dir: &Path,
    config: &ModelConfig,
) -> Result<Gpt2Weights> {
    let binary_index_path = model_dir.join("weights.index.json");
    let binary_data_path = model_dir.join("weights.bin");

    match (binary_index_path.exists(), binary_data_path.exists()) {
        (true, true) => load_gpt2_binary_weights(binary_index_path, binary_data_path, config),
        (false, false) => load_gpt2_weights(model_dir.join("weights.json")),
        _ => Err(MiniInferError::InvalidConfig {
            message: "binary weights require both weights.index.json and weights.bin".to_string(),
        }),
    }
}

pub fn load_gpt2_weights(path: impl AsRef<Path>) -> Result<Gpt2Weights> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights {}: {error}", path.display()),
    })?;

    let file_weights: Gpt2WeightsFile =
        serde_json::from_reader(file).map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to parse weights {}: {error}", path.display()),
        })?;

    let mut blocks = Vec::with_capacity(file_weights.blocks.len());
    for block in file_weights.blocks {
        blocks.push(Gpt2BlockWeights {
            ln_1_weight: tensor_from_file(block.ln_1_weight)?,
            ln_1_bias: tensor_from_file(block.ln_1_bias)?,
            c_attn_weight: tensor_from_file(block.c_attn_weight)?,
            c_attn_bias: tensor_from_file(block.c_attn_bias)?,
            attn_c_proj_weight: tensor_from_file(block.attn_c_proj_weight)?,
            attn_c_proj_bias: tensor_from_file(block.attn_c_proj_bias)?,
            ln_2_weight: tensor_from_file(block.ln_2_weight)?,
            ln_2_bias: tensor_from_file(block.ln_2_bias)?,
            c_fc_weight: tensor_from_file(block.c_fc_weight)?,
            c_fc_bias: tensor_from_file(block.c_fc_bias)?,
            mlp_c_proj_weight: tensor_from_file(block.mlp_c_proj_weight)?,
            mlp_c_proj_bias: tensor_from_file(block.mlp_c_proj_bias)?,
        });
    }

    Ok(Gpt2Weights {
        wte: tensor_from_file(file_weights.wte)?,
        wpe: tensor_from_file(file_weights.wpe)?,
        blocks,
        ln_f_weight: tensor_from_file(file_weights.ln_f_weight)?,
        ln_f_bias: tensor_from_file(file_weights.ln_f_bias)?,
        lm_head_weight: load_lm_head(file_weights.lm_head, file_weights.lm_head_weight)?,
    })
}

pub fn load_gpt2_binary_weights(
    index_path: impl AsRef<Path>,
    data_path: impl AsRef<Path>,
    config: &ModelConfig,
) -> Result<Gpt2Weights> {
    let index_path = index_path.as_ref();
    let data_path = data_path.as_ref();

    let index_file = File::open(index_path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights index {}: {error}", index_path.display()),
    })?;
    let index: BinaryWeightsIndexFile = serde_json::from_reader(index_file).map_err(|error| {
        MiniInferError::InvalidConfig {
            message: format!("failed to parse weights index {}: {error}", index_path.display()),
        }
    })?;

    validate_binary_weight_index(&index)?;

    let mut data_file = File::open(data_path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights data {}: {error}", data_path.display()),
    })?;

    let mut blocks = Vec::with_capacity(config.num_layers);
    for layer_index in 0..config.num_layers {
        let prefix = format!("blocks.{layer_index}");
        blocks.push(Gpt2BlockWeights {
            ln_1_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_1_weight"))?,
            ln_1_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_1_bias"))?,
            c_attn_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_attn_weight"))?,
            c_attn_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_attn_bias"))?,
            attn_c_proj_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.attn_c_proj_weight"))?,
            attn_c_proj_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.attn_c_proj_bias"))?,
            ln_2_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_2_weight"))?,
            ln_2_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_2_bias"))?,
            c_fc_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_fc_weight"))?,
            c_fc_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_fc_bias"))?,
            mlp_c_proj_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.mlp_c_proj_weight"))?,
            mlp_c_proj_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.mlp_c_proj_bias"))?,
        });
    }

    let lm_head_weight = load_binary_lm_head(&mut data_file, &index)?;

    Ok(Gpt2Weights {
        wte: read_binary_tensor(&mut data_file, &index, "wte")?,
        wpe: read_binary_tensor(&mut data_file, &index, "wpe")?,
        blocks,
        ln_f_weight: read_binary_tensor(&mut data_file, &index, "ln_f_weight")?,
        ln_f_bias: read_binary_tensor(&mut data_file, &index, "ln_f_bias")?,
        lm_head_weight,
    })
}

fn validate_binary_weight_index(index: &BinaryWeightsIndexFile) -> Result<()> {
    if index.endianness != "little" {
        return Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights endianness {}", index.endianness),
        });
    }

    match index.format_version {
        1 => {
            let dtype = index.dtype.as_deref().ok_or_else(|| MiniInferError::InvalidConfig {
                message: "v1 weights index must include dtype".to_string(),
            })?;

            if dtype != "f32" {
                return Err(MiniInferError::InvalidConfig {
                    message: format!("unsupported v1 weights dtype {dtype}"),
                });
            }

            Ok(())
        }
        2 => Ok(()),
        other => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights index format version {other}"),
        }),
    }
}

fn load_binary_lm_head(data_file: &mut File, index: &BinaryWeightsIndexFile) -> Result<LMHead> {
    match index.lm_head.as_ref().map(|lm_head| lm_head.head_type.as_str()) {
        Some("tied") => Ok(LMHead::Tied),
        Some("untied") => Ok(LMHead::Untied(read_binary_tensor(data_file, index, "lm_head_weight")?)),
        Some(other) => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported LM head type {other}"),
        }),
        None if index.tensors.contains_key("lm_head_weight") => {
            Ok(LMHead::Untied(read_binary_tensor(data_file, index, "lm_head_weight")?))
        }
        None => Err(MiniInferError::InvalidConfig {
            message: "weights index must specify lm_head or lm_head_weight".to_string(),
        }),
    }
}

fn read_binary_tensor(
    data_file: &mut (impl Read + Seek),
    index: &BinaryWeightsIndexFile,
    name: &str,
) -> Result<Tensor> {
    let tensor_index = index.tensors.get(name).ok_or_else(|| MiniInferError::InvalidConfig {
        message: format!("weights index is missing tensor {name}"),
    })?;
    let expected_len: usize = tensor_index.shape.iter().product();
    if tensor_index.len != expected_len {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "tensor {name} index length {} does not match shape product {}",
                tensor_index.len, expected_len
            ),
        });
    }

    let dtype = tensor_dtype(index, tensor_index)?;

    let byte_len = tensor_index
        .len
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| MiniInferError::InvalidConfig {
            message: format!("tensor {name} byte length overflow"),
        })?;

    let mut bytes = vec![0u8; byte_len];
    data_file
        .seek(SeekFrom::Start(tensor_index.offset_bytes))
        .map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to seek tensor {name}: {error}"),
        })?;
    data_file.read_exact(&mut bytes).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to read tensor {name}: {error}"),
    })?;

    match dtype {
        DType::F32 => {
            let mut data = Vec::with_capacity(tensor_index.len);
            for chunk in bytes.chunks_exact(4) {
                let mut value_bytes = [0u8; 4];
                value_bytes.copy_from_slice(chunk);
                data.push(f32::from_le_bytes(value_bytes));
            }

            Tensor::new(tensor_index.shape.clone(), data)
        }
        DType::I8Symmetric => {
            let scale = tensor_index.scale.ok_or_else(|| MiniInferError::InvalidConfig {
                message: format!("tensor {name} i8_symmetric dtype requires scale"),
            })?;

            let data = bytes.iter().map(|byte| *byte as i8).collect::<Vec<i8>>();

            QuantizedTensor::new(
                tensor_index.shape.clone(),
                data,
                QuantizationScale::PerTensor(scale),
            )?
            .dequantize()
        }
    }
}

pub(super) fn load_lm_head(
    lm_head: Option<LmHeadFile>,
    legacy_lm_head_weight: Option<TensorFile>,
) -> Result<LMHead> {
    match (lm_head, legacy_lm_head_weight) {
        (Some(lm_head), None) if lm_head.head_type == "tied" => Ok(LMHead::Tied),
        (Some(lm_head), None) if lm_head.head_type == "untied" => {
            let weight = lm_head.weight.ok_or_else(|| MiniInferError::InvalidConfig {
                message: "untied LM head must include a weight tensor".to_string(),
            })?;
            Ok(LMHead::Untied(tensor_from_file(weight)?))
        }
        (Some(lm_head), None) => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported LM head type {}", lm_head.head_type),
        }),
        (None, Some(weight)) => Ok(LMHead::Untied(tensor_from_file(weight)?)),
        (Some(_), Some(_)) => Err(MiniInferError::InvalidConfig {
            message: "weights must specify either lm_head or lm_head_weight, not both".to_string(),
        }),
        (None, None) => Err(MiniInferError::InvalidConfig {
            message: "weights must specify lm_head or lm_head_weight".to_string(),
        }),
    }
}

fn tensor_from_file(tensor: TensorFile) -> Result<Tensor> {
    Tensor::new(tensor.shape, tensor.data)
}

fn tensor_dtype(index: &BinaryWeightsIndexFile, tensor: &BinaryTensorIndexFile) -> Result<DType> {
    match index.format_version {
        1 => {
            let dtype = index.dtype.as_deref().unwrap_or("f32");
            let dtype = DType::parse(dtype)?;
            if dtype != DType::F32 {
                return Err(MiniInferError::InvalidConfig {
                    message: format!("unsupported v1 weights dtype {}", index.dtype.as_deref().unwrap_or("missing")),
                });
            }
            Ok(dtype)
        }
        2 => {
            let dtype = tensor.dtype.as_deref().ok_or_else(|| MiniInferError::InvalidConfig {
                message: "v2 tensor index must include dtype".to_string(),
            })?;
            DType::parse(dtype)
        }
        other => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights index format version {other}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, io::{Cursor, Write}};

    #[test]
    fn tensor_dtype_uses_v1_top_level_f32_dtype() {
        let index = binary_index(1, Some("f32"));
        let tensor = binary_tensor(vec![2], None, 0, 2, None);

        let dtype = tensor_dtype(&index, &tensor).expect("v1 f32 dtype should resolve");

        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn tensor_dtype_rejects_v2_tensor_without_dtype() {
        let index = binary_index(2, None);
        let tensor = binary_tensor(vec![2], None, 0, 2, None);

        let err = tensor_dtype(&index, &tensor).expect_err("v2 tensor dtype should be required");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "v2 tensor index must include dtype".to_string(),
            }
        );
    }

    #[test]
    fn read_binary_tensor_reads_v2_f32_tensor() {
        let mut data_file = binary_data_file(&[1.0, -2.0]);
        let index = binary_index_with_tensor(
            2,
            None,
            "weight",
            binary_tensor(vec![2], Some("f32"), 0, 2, None),
        );

        let tensor = read_binary_tensor(&mut data_file, &index, "weight")
            .expect("v2 f32 tensor should load");

        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.data(), &[1.0, -2.0]);
    }

    #[test]
    fn read_binary_tensor_reads_v2_i8_symmetric_tensor_and_dequantizes() {
        let mut data_file = Cursor::new(vec![-2i8 as u8, 0i8 as u8, 4i8 as u8]);
        let index = binary_index_with_tensor(
            2,
            None,
            "weight",
            binary_tensor(vec![3], Some("i8_symmetric"), 0, 3, Some(0.5)),
        );

        let tensor = read_binary_tensor(&mut data_file, &index, "weight")
            .expect("v2 i8 tensor should load and dequantize");

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.data(), &[-1.0, 0.0, 2.0]);
    }

    #[test]
    fn read_binary_tensor_rejects_i8_without_scale() {
        let mut data_file = Cursor::new(vec![1u8, 2u8]);
        let index = binary_index_with_tensor(
            2,
            None,
            "weight",
            binary_tensor(vec![2], Some("i8_symmetric"), 0, 2, None),
        );

        let err = read_binary_tensor(&mut data_file, &index, "weight")
            .expect_err("i8 tensor without scale should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "tensor weight i8_symmetric dtype requires scale".to_string(),
            }
        );
    }

    fn binary_index(format_version: u32, dtype: Option<&str>) -> BinaryWeightsIndexFile {
        BinaryWeightsIndexFile {
            format_version,
            dtype: dtype.map(str::to_string),
            endianness: "little".to_string(),
            tensors: HashMap::new(),
            lm_head: Some(LmHeadFile { head_type: "tied".to_string(), weight: None }),
        }
    }

    fn binary_index_with_tensor(
        format_version: u32,
        dtype: Option<&str>,
        name: &str,
        tensor: BinaryTensorIndexFile,
    ) -> BinaryWeightsIndexFile {
        let mut index = binary_index(format_version, dtype);
        index.tensors.insert(name.to_string(), tensor);
        index
    }

    fn binary_tensor(
        shape: Vec<usize>,
        dtype: Option<&str>,
        offset_bytes: u64,
        len: usize,
        scale: Option<f32>,
    ) -> BinaryTensorIndexFile {
        BinaryTensorIndexFile {
            shape,
            dtype: dtype.map(str::to_string),
            offset_bytes,
            len,
            scale,
        }
    }

    fn binary_data_file(values: &[f32]) -> Cursor<Vec<u8>> {
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            data.write_all(&value.to_le_bytes()).expect("test data write should succeed");
        }
        Cursor::new(data)
    }
}