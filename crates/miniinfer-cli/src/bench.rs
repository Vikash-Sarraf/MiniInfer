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
    if args.runs == 0 {
        return Err(MiniInferError::InvalidConfig {
            message: "runs must be greater than zero".to_string(),
        });
    }

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
        let (no_cache_results, kv_cache_results) = with_backend(args.backend, |backend| {
            let mut no_cache_results = Vec::with_capacity(args.runs);
            let mut kv_cache_results = Vec::with_capacity(args.runs);

            for _ in 0..args.runs {
                no_cache_results.push(run_generation_benchmark(
                    &model,
                    &token_ids,
                    args.max_new_tokens,
                    backend,
                    false,
                )?);
                kv_cache_results.push(run_generation_benchmark(
                    &model,
                    &token_ids,
                    args.max_new_tokens,
                    backend,
                    true,
                )?);
            }

            Ok((no_cache_results, kv_cache_results))
        })?;

        let total_elapsed = total_start.elapsed();
        let no_cache_summary = summarize_generation_benchmark_results(&no_cache_results)?;
        let kv_cache_summary = summarize_generation_benchmark_results(&kv_cache_results)?;

        println!("Backend: {}", args.backend);
        println!("Runs: {}", args.runs);
        println!("Prompt tokens: {prompt_tokens}");
        println!("Requested tokens: {}", args.max_new_tokens);
        println!("Load time: {:.3}s", load_elapsed.as_secs_f64());
        println!("Encode time: {:.3}s", encode_elapsed.as_secs_f64());
        println!();
        print_generation_benchmark_runs("No cache", prompt_tokens, &no_cache_results, &no_cache_summary);
        println!();
        print_generation_benchmark_runs("KV cache", prompt_tokens, &kv_cache_results, &kv_cache_summary);
        println!();
        println!("Speedup:");
        println!(
            "Generation time: {:.3}x",
            speedup_ratio(
                no_cache_summary.generation_elapsed.median.as_secs_f64(),
                kv_cache_summary.generation_elapsed.median.as_secs_f64(),
            )
        );
        println!(
            "Tokens/sec: {:.3}x",
            speedup_ratio(
                kv_cache_summary.tokens_per_second.median,
                no_cache_summary.tokens_per_second.median,
            )
        );
        println!("Outputs match: {}", comparison_outputs_match(&no_cache_results, &kv_cache_results));
        println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
        println!("Result: {}", kv_cache_summary.decoded_text);
    } else {
        let results = with_backend(args.backend, |backend| {
            let mut results = Vec::with_capacity(args.runs);
            for _ in 0..args.runs {
                results.push(run_generation_benchmark(
                    &model,
                    &token_ids,
                    args.max_new_tokens,
                    backend,
                    args.kv_cache,
                )?);
            }
            Ok(results)
        })?;
        let summary = summarize_generation_benchmark_results(&results)?;
        let total_elapsed = total_start.elapsed();

        println!("Backend: {}", args.backend);
        println!("Cache: {}", if args.kv_cache { "kv" } else { "none" });
        println!("Runs: {}", args.runs);
        println!("Prompt tokens: {prompt_tokens}");
        println!("Load time: {:.3}s", load_elapsed.as_secs_f64());
        println!("Encode time: {:.3}s", encode_elapsed.as_secs_f64());
        print_generation_benchmark_runs("Generation", prompt_tokens, &results, &summary);
        println!("Total time: {:.3}s", total_elapsed.as_secs_f64());
        println!("Result: {}", summary.decoded_text);
    }
    Ok(())
}

struct GenerationBenchmarkSummary {
    decoded_text: String,
    generated_tokens: usize,
    first_token_elapsed: Option<DurationStats>,
    prefill_elapsed: Option<DurationStats>,
    decode_elapsed: Option<DurationStats>,
    generation_elapsed: DurationStats,
    tokens_per_second: FloatStats,
    decode_tokens_per_second: Option<FloatStats>,
    kv_cache_memory: Option<KvCacheMemoryReport>,
}

#[derive(Clone, Copy)]
struct DurationStats {
    min: std::time::Duration,
    median: std::time::Duration,
    max: std::time::Duration,
}

#[derive(Clone, Copy)]
struct FloatStats {
    min: f64,
    median: f64,
    max: f64,
}

struct GenerationBenchmarkResult {
    decoded_text: String,
    generated_tokens: usize,
    first_token_elapsed: Option<std::time::Duration>,
    prefill_elapsed: Option<std::time::Duration>,
    decode_elapsed: Option<std::time::Duration>,
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

    fn decode_tokens_per_second(&self) -> f64 {
        match self.decode_elapsed {
            Some(elapsed) if self.generated_tokens > 0 && !elapsed.is_zero() => {
                self.generated_tokens as f64 / elapsed.as_secs_f64()
            }
            _ => 0.0,
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
            prefill_elapsed: Some(report.prefill_elapsed),
            decode_elapsed: Some(report.decode_elapsed),
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
        prefill_elapsed: None,
        decode_elapsed: None,
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
    if let Some(elapsed) = result.prefill_elapsed {
        println!("Prompt prefill time: {:.3}s", elapsed.as_secs_f64());
    }
    if let Some(elapsed) = result.decode_elapsed {
        println!("Decode time: {:.3}s", elapsed.as_secs_f64());
        println!("Decode tokens/sec: {:.3}", result.decode_tokens_per_second());
    }
    println!("Generation time: {:.3}s", result.generation_elapsed.as_secs_f64());
    println!("Tokens/sec: {:.3}", result.tokens_per_second());
    if let Some(memory) = &result.kv_cache_memory {
        print_kv_cache_memory_report(memory);
    }
}

fn print_generation_benchmark_runs(
    label: &str,
    prompt_tokens: usize,
    results: &[GenerationBenchmarkResult],
    summary: &GenerationBenchmarkSummary,
) {
    if results.len() == 1 {
        print_generation_benchmark_result(label, prompt_tokens, &results[0]);
        return;
    }

    println!("{label}:");
    println!("Generated tokens: {}", summary.generated_tokens);
    println!("Final tokens: {}", prompt_tokens + summary.generated_tokens);
    match summary.first_token_elapsed {
        Some(stats) => println!("Time to first token: {}", format_duration_stats(stats)),
        None => println!("Time to first token: n/a"),
    }
    if let Some(stats) = summary.prefill_elapsed {
        println!("Prompt prefill time: {}", format_duration_stats(stats));
    }
    if let Some(stats) = summary.decode_elapsed {
        println!("Decode time: {}", format_duration_stats(stats));
    }
    if let Some(stats) = summary.decode_tokens_per_second {
        println!("Decode tokens/sec: {}", format_float_stats(stats));
    }
    println!("Generation time: {}", format_duration_stats(summary.generation_elapsed));
    println!("Tokens/sec: {}", format_float_stats(summary.tokens_per_second));
    if let Some(memory) = &summary.kv_cache_memory {
        print_kv_cache_memory_report(memory);
    }
}

fn summarize_generation_benchmark_results(
    results: &[GenerationBenchmarkResult],
) -> Result<GenerationBenchmarkSummary> {
    let Some(first_result) = results.first() else {
        return Err(MiniInferError::EmptyInput);
    };

    Ok(GenerationBenchmarkSummary {
        decoded_text: first_result.decoded_text.clone(),
        generated_tokens: first_result.generated_tokens,
        first_token_elapsed: optional_duration_stats(results.iter().filter_map(|result| result.first_token_elapsed)),
        prefill_elapsed: optional_duration_stats(results.iter().filter_map(|result| result.prefill_elapsed)),
        decode_elapsed: optional_duration_stats(results.iter().filter_map(|result| result.decode_elapsed)),
        generation_elapsed: duration_stats(results.iter().map(|result| result.generation_elapsed))?,
        tokens_per_second: float_stats(results.iter().map(GenerationBenchmarkResult::tokens_per_second))?,
        decode_tokens_per_second: optional_float_stats(
            results
                .iter()
                .filter(|result| result.decode_elapsed.is_some())
                .map(GenerationBenchmarkResult::decode_tokens_per_second),
        ),
        kv_cache_memory: first_result.kv_cache_memory,
    })
}

fn comparison_outputs_match(
    no_cache_results: &[GenerationBenchmarkResult],
    kv_cache_results: &[GenerationBenchmarkResult],
) -> bool {
    no_cache_results.len() == kv_cache_results.len()
        && no_cache_results
            .iter()
            .zip(kv_cache_results)
            .all(|(no_cache, kv_cache)| no_cache.decoded_text == kv_cache.decoded_text)
}

fn optional_duration_stats(
    values: impl Iterator<Item = std::time::Duration>,
) -> Option<DurationStats> {
    duration_stats(values).ok()
}

fn duration_stats(values: impl Iterator<Item = std::time::Duration>) -> Result<DurationStats> {
    let mut values: Vec<std::time::Duration> = values.collect();
    if values.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    values.sort();
    Ok(DurationStats {
        min: values[0],
        median: median_duration(&values),
        max: values[values.len() - 1],
    })
}

fn optional_float_stats(values: impl Iterator<Item = f64>) -> Option<FloatStats> {
    float_stats(values).ok()
}

fn float_stats(values: impl Iterator<Item = f64>) -> Result<FloatStats> {
    let mut values: Vec<f64> = values.collect();
    if values.is_empty() {
        return Err(MiniInferError::EmptyInput);
    }

    values.sort_by(f64::total_cmp);
    Ok(FloatStats {
        min: values[0],
        median: median_float(&values),
        max: values[values.len() - 1],
    })
}

fn median_duration(values: &[std::time::Duration]) -> std::time::Duration {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        std::time::Duration::from_secs_f64(
            (values[middle - 1].as_secs_f64() + values[middle].as_secs_f64()) / 2.0,
        )
    } else {
        values[middle]
    }
}

fn median_float(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn format_duration_stats(stats: DurationStats) -> String {
    format!(
        "min {:.3}s / median {:.3}s / max {:.3}s",
        stats.min.as_secs_f64(),
        stats.median.as_secs_f64(),
        stats.max.as_secs_f64()
    )
}

fn format_float_stats(stats: FloatStats) -> String {
    format!("min {:.3} / median {:.3} / max {:.3}", stats.min, stats.median, stats.max)
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