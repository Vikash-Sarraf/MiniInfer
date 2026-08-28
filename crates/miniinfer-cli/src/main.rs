use std::path::Path;

use miniinfer_core::{
    ops::backend::{NdArrayBackend, OpsBackend, ReferenceBackend},
    tensor::Tensor,
};
fn main() -> miniinfer_core::error::Result<()> {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("run") => println!("miniinfer run: not implemented yet"),
        Some("inspect") => inspect_model(args)?,
        Some("bench") => println!("miniinfer bench: not implemented yet"),
        Some("bench-matmul") => bench_matmul(),
        Some("--help") | Some("-h") | None => print_help(),
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            print_help();
            std::process::exit(2);
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "Usage: miniinfer <command> [options]
Commands:
    run       Run inference on a model (not implemented yet)
    inspect   Inspect a model (not implemented yet)
    bench     Benchmark a model (not implemented yet)
    bench-matmul Benchmark reference vs ndarray matmul
Options:
    -h, --help    Show this help message"
    );
}

fn bench_matmul() {
    let m = 64;
    let n = 64;
    let k = 64;

    let mut a_data: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            a_data.push(((i + j) % 13) as f32 * 0.01);
        }
    }

    let mut b_data: Vec<f32> = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            b_data.push(((i + j) % 17) as f32 * 0.01);
        }
    }

    let a = Tensor::new(vec![m, k], a_data).expect("valid tensor");
    let b = Tensor::new(vec![k, n], b_data).expect("valid tensor");

    println!("Matrix size: {m}x{k} * {k}x{n}");

    let reference_backend = ReferenceBackend::new();
    let start = std::time::Instant::now();
    let output_ref = reference_backend.matmul(&a, &b).expect("matmul should succeed");
    let reference_elapsed = start.elapsed();

    let nd_backend = NdArrayBackend::new();
    let start = std::time::Instant::now();
    let output_nd = nd_backend.matmul(&a, &b).expect("matmul should succeed");
    let ndarray_elapsed = start.elapsed();

    let outputs_match = tensors_close(&output_ref, &output_nd, 1e-4);

    println!("ReferenceBackend: {reference_elapsed:?}");
    println!("NdArrayBackend:   {ndarray_elapsed:?}");
    println!("Outputs match:    {outputs_match}");
}

fn tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    if a.data().len() != b.data().len() {
        return false;
    }

    a.data()
        .iter()
        .zip(b.data().iter())
        .all(|(left, right)| (*left - *right).abs() <= tolerance)
}

fn inspect_model(mut args: impl Iterator<Item = String>) -> miniinfer_core::error::Result<()> {
    let flag = match args.next() {
        Some(flag) => flag,
        None => {
            eprintln!("Usage: miniinfer inspect --model <path>");
            std::process::exit(1);
        }
    };

    if flag != "--model" {
        eprintln!("Unknown inspect option: {flag}");
        eprintln!("Usage: miniinfer inspect --model <path>");
        std::process::exit(2);
    }

    let model_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("Error: --model requires a path argument");
            std::process::exit(1);
        }
    };

    println!("Inspecting model at path: {model_path}");

    let config_path = Path::new(&model_path).join("config.json");
    let config = miniinfer_core::model::loader::load_config(config_path)?;
    print_config(&config);

    Ok(())
}

fn print_config(config: &miniinfer_core::model::config::ModelConfig) {
    let rows = [
        ("Architecture", format!("{:?}", config.architecture)),
        ("Vocab size", config.vocab_size.to_string()),
        ("Hidden size", config.hidden_size.to_string()),
        ("Layers", config.num_layers.to_string()),
        ("Heads", config.num_heads.to_string()),
        ("Head dim", config.head_dim().to_string()),
        ("Intermediate size", config.intermediate_size.to_string()),
        ("Max positions", config.max_position_embeddings.to_string()),
        ("LayerNorm epsilon", config.layer_norm_epsilon.to_string()),
    ];

    for (label, value) in rows {
        println!("{label}: {value}");
    }
}