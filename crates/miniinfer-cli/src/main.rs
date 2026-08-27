use miniinfer_core::{
    ops::backend::{NdArrayBackend, OpsBackend, ReferenceBackend},
    tensor::Tensor,
};
fn main() {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("run") => println!("miniinfer run: not implemented yet"),
        Some("inspect") => println!("miniinfer inspect: not implemented yet"),
        Some("bench") => println!("miniinfer bench: not implemented yet"),
        Some("bench-matmul") => bench_matmul(),
        Some("--help") | Some("-h") | None => print_help(),
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            print_help();
            std::process::exit(2);
        }
    }
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