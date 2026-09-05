use miniinfer_core::{
    error::{MiniInferError, Result},
    model::loader::{load_model, LoadedModel},
    ops::backend::{NdArrayBackend, OpsBackend, ReferenceBackend},
    runtime::generation::{GenerationOptions, KvCacheMemoryReport},
    tensor::Tensor,
};

use crate::{args::BenchGenerateArgs, encode_prompt_input, with_backend};

pub(crate) fn bench_matmul() {
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

pub(crate) fn bench_generate(args: BenchGenerateArgs) -> Result<()> {
    let total_start = std::time::Instant::now();
    let load_start = std::time::Instant::now();
    let model = load_model(args.model)?;
    model.validate()?;
    let load_elapsed = load_start.elapsed();

    let encode_start = std::time::Instant::now();
    let token_ids = encode_prompt_input(&model, args.input)?;
    let encode_elapsed = encode_start.elapsed();
    let prompt_tokens = token_ids.len();

    let requested_length = prompt_tokens + args.max_new_tokens;
    if requested_length > model.config().max_position_embeddings {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "requested sequence length {requested_length} exceeds max_position_embeddings {}",
                model.config().max_position_embeddings
            ),
        });
    }

    if args.compare_cache {
        let (no_cache_result, kv_cache_result) = with_backend(args.backend, |backend| {
            let no_cache_result = run_generation_benchmark(
                &model,
                &token_ids,
                args.max_new_tokens,
                backend,
                false,
            )?;
            let kv_cache_result = run_generation_benchmark(
                &model,
                &token_ids,
                args.max_new_tokens,
                backend,
                true,
            )?;

            Ok((no_cache_result, kv_cache_result))
        })?;

        let total_elapsed = total_start.elapsed();

        println!("Backend: {}", args.backend);
        println!("Prompt tokens: {prompt_tokens}");
        println!("Requested tokens: {}", args.max_new_tokens);
        println!("Load time: {:.3}s", load_elapsed.as_secs_f64());
        println!("Encode time: {:.3}s", encode_elapsed.as_secs_f64());
        println!();
        print_generation_benchmark_result("No cache", prompt_tokens, &no_cache_result);
        println!();
        print_generation_benchmark_result("KV cache", prompt_tokens, &kv_cache_result);
        println!();
        println!("Speedup:");
        println!(
            "Generation time: {:.3}x",
            speedup_ratio(
                no_cache_result.generation_elapsed.as_secs_f64(),
                kv_cache_result.generation_elapsed.as_secs_f64(),
            )
        );
        println!(
            "Tokens/sec: {:.3}x",
            speedup_ratio(
                kv_cache_result.tokens_per_second(),
                no_cache_result.tokens_per_second(),
            )
        );
        println!("Outputs match: {}", no_cache_result.decoded_text == kv_cache_result.decoded_text);
        println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
        println!("Result: {}", kv_cache_result.decoded_text);
    } else {
        let result = with_backend(args.backend, |backend| {
            run_generation_benchmark(&model, &token_ids, args.max_new_tokens, backend, args.kv_cache)
        })?;
        let total_elapsed = total_start.elapsed();

        println!("Backend: {}", args.backend);
        println!("Cache: {}", if args.kv_cache { "kv" } else { "none" });
        println!("Prompt tokens: {prompt_tokens}");
        println!("Generated tokens: {}", result.generated_tokens);
        println!("Final tokens: {}", prompt_tokens + result.generated_tokens);
        println!("Load time: {:.3}s", load_elapsed.as_secs_f64());
        println!("Encode time: {:.3}s", encode_elapsed.as_secs_f64());
        match result.first_token_elapsed {
            Some(elapsed) => println!("Time to first token: {:.3}s", elapsed.as_secs_f64()),
            None => println!("Time to first token: n/a"),
        }
        println!("Generation time: {:.3}s", result.generation_elapsed.as_secs_f64());
        println!("Tokens/sec: {:.3}", result.tokens_per_second());
        if let Some(memory) = &result.kv_cache_memory {
            print_kv_cache_memory_report(memory);
        }
        println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
        println!("Result: {}", result.decoded_text);
    }
    Ok(())
}

struct GenerationBenchmarkResult {
    decoded_text: String,
    generated_tokens: usize,
    first_token_elapsed: Option<std::time::Duration>,
    generation_elapsed: std::time::Duration,
    kv_cache_memory: Option<KvCacheMemoryReport>,
}

impl GenerationBenchmarkResult {
    fn tokens_per_second(&self) -> f64 {
        if self.generated_tokens == 0 {
            0.0
        } else {
            self.generated_tokens as f64 / self.generation_elapsed.as_secs_f64()
        }
    }
}

fn run_generation_benchmark(
    model: &LoadedModel,
    token_ids: &[usize],
    max_new_tokens: usize,
    backend: &dyn OpsBackend,
    use_kv_cache: bool,
) -> Result<GenerationBenchmarkResult> {
    let mut first_token_elapsed = None;
    let mut generated_tokens = 0;
    let generation_options = GenerationOptions::new(max_new_tokens, None, None, None, None)?;
    let generation_start = std::time::Instant::now();
    let decoded_text = if use_kv_cache {
        let report = generation_options.generate_with_kv_cache_report_and_token_observer_and_backend(
            model,
            token_ids,
            backend,
            |generated_index, _| {
                generated_tokens += 1;
                if generated_index == 0 {
                    first_token_elapsed = Some(generation_start.elapsed());
                }
            },
        )?;

        let generation_elapsed = generation_start.elapsed();
        return Ok(GenerationBenchmarkResult {
            decoded_text: report.decoded_text,
            generated_tokens,
            first_token_elapsed,
            generation_elapsed,
            kv_cache_memory: Some(report.kv_cache_memory),
        });
    } else {
        generation_options.generate_with_token_observer_and_backend(
            model,
            token_ids,
            backend,
            |generated_index, _| {
                generated_tokens += 1;
                if generated_index == 0 {
                    first_token_elapsed = Some(generation_start.elapsed());
                }
            },
        )?
    };
    let generation_elapsed = generation_start.elapsed();

    Ok(GenerationBenchmarkResult {
        decoded_text,
        generated_tokens,
        first_token_elapsed,
        generation_elapsed,
        kv_cache_memory: None,
    })
}

fn print_generation_benchmark_result(
    label: &str,
    prompt_tokens: usize,
    result: &GenerationBenchmarkResult,
) {
    println!("{label}:");
    println!("Generated tokens: {}", result.generated_tokens);
    println!("Final tokens: {}", prompt_tokens + result.generated_tokens);
    match result.first_token_elapsed {
        Some(elapsed) => println!("Time to first token: {:.3}s", elapsed.as_secs_f64()),
        None => println!("Time to first token: n/a"),
    }
    println!("Generation time: {:.3}s", result.generation_elapsed.as_secs_f64());
    println!("Tokens/sec: {:.3}", result.tokens_per_second());
    if let Some(memory) = &result.kv_cache_memory {
        print_kv_cache_memory_report(memory);
    }
}

fn print_kv_cache_memory_report(memory: &KvCacheMemoryReport) {
    println!("KV cache memory:");
    println!("  Active: {}", format_bytes(memory.active_bytes));
    println!("  Allocated: {}", format_bytes(memory.allocated_bytes));
    println!("  Capacity: {}", format_bytes(memory.capacity_bytes));
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;

    if (bytes as f64) >= MIB {
        format!("{} bytes ({:.3} MiB)", bytes, bytes as f64 / MIB)
    } else if (bytes as f64) >= KIB {
        format!("{} bytes ({:.3} KiB)", bytes, bytes as f64 / KIB)
    } else {
        format!("{bytes} bytes")
    }
}

fn speedup_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}