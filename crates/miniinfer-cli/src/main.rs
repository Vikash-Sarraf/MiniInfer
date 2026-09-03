use std::io::{Write, stdout};

use miniinfer_core::{
    error::{MiniInferError, Result}, model::{config::ModelConfig, loader::{LoadedModel, load_model}}, ops::backend::{NdArrayBackend, OpsBackend, ReferenceBackend}, runtime::{generation::GenerationOptions}, tensor::Tensor,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);

    match args.next().as_deref() {
        Some("run") => run_model(args)?,
        Some("inspect") => inspect_model(args)?,
        Some("logits") => print_logits(args)?,
        Some("bench") => println!("miniinfer bench: not implemented yet"),
        Some("bench-generate") => bench_generate(args)?,
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
    run       Run inference on a model
    inspect   Inspect a model 
    logits    Print selected logits for debugging/parity checks
    bench     Benchmark a model (not implemented yet)
    bench-generate Benchmark greedy generation timings
    bench-matmul Benchmark reference vs ndarray matmul
Options:
    --backend ndarray|reference    Select execution backend for run/logits (default: ndarray)
    --stream                       Print generated text token-by-token for run
    --kv-cache                     Use KV-cache decoding for run/bench-generate
    --compare-cache                Benchmark both no-cache and KV-cache generation
    --temperature <number>         Enable temperature sampling for run
    --top-k <integer>              Limit temperature sampling to the highest-k logits for run
    --top-p <number>               Limit temperature sampling to cumulative probability mass for run
    --seed <integer>               Seed temperature sampling for reproducible run output
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

fn inspect_model(mut args: impl Iterator<Item = String>) -> Result<()> {
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

    let model = load_model(model_path)?;
    model.validate()?;
    print_config(model.config());
    Ok(())
}

fn print_config(config: &ModelConfig) {
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

fn run_model(mut args: impl Iterator<Item = String>) -> Result<()> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut backend_name: Option<String> = None;
    let mut max_new_tokens = 1;
    let mut stream = false;
    let mut kv_cache = false;
    let mut temperature: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut top_k: Option<usize> = None;
    let mut top_p: Option<f32> = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model_path = args.next();
            }
            "--prompt" => {
                prompt = args.next();
            }
            "--tokens" => {
                tokens = args.next();
            }
            "--backend" => {
                backend_name = args.next();
            }
            "--max-new-tokens" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --max-new-tokens requires a number");
                        std::process::exit(1);
                    }
                };

                max_new_tokens = match value.parse::<usize>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --max-new-tokens must be a positive integer");
                        std::process::exit(1);
                    }
                };
            }
            "--stream" => {
                stream = true;
            }
            "--temperature" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --temperature requires a number");
                        std::process::exit(1);
                    }
                };

                temperature = Some(match value.parse::<f32>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --temperature must be a number");
                        std::process::exit(1);
                    }
                });
            }
            "--seed" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --seed requires an integer");
                        std::process::exit(1);
                    }
                };

                seed = Some(match value.parse::<u64>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --seed must be an unsigned integer");
                        std::process::exit(1);
                    }
                });
            }
            "--top-k" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --top-k requires an integer");
                        std::process::exit(1);
                    }
                };

                top_k = Some(match value.parse::<usize>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --top-k must be a positive integer");
                        std::process::exit(1);
                    }
                });
            }
            "--top-p" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --top-p requires a number");
                        std::process::exit(1);
                    }
                };

                top_p = Some(match value.parse::<f32>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --top-p must be a number");
                        std::process::exit(1);
                    }
                });
            }
            "--kv-cache" => {
                kv_cache = true;
            }
            other => {
                eprintln!("Unknown run option: {other}");
                std::process::exit(2);
            }
        }
    }
    if seed.is_some() && temperature.is_none() {
        eprintln!("Error: --seed requires --temperature");
        std::process::exit(1);
    }
    if top_k.is_some() && temperature.is_none() {
        eprintln!("Error: --top-k requires --temperature");
        std::process::exit(1);
    }
    if top_p.is_some() && temperature.is_none() {
        eprintln!("Error: --top-p requires --temperature");
        std::process::exit(1);
    }

    let model_path = match model_path {
        Some(path) => path,
        None => {
            eprintln!("Usage: miniinfer run --model <path> (--prompt <text> | --tokens <ids>)");
            std::process::exit(1);
        }
    };

    let model = load_model(model_path)?;
    model.validate()?;
    let token_ids = match (prompt, tokens) {
        (Some(prompt), None) => model.encode_prompt(&prompt)?,
        (None, Some(tokens)) => parse_token_ids(&tokens)?,
        (Some(_), Some(_)) => {
            eprintln!("Use either --prompt or --tokens, not both");
            std::process::exit(1);
        }
        (None, None) => {
            eprintln!("Usage: miniinfer run --model <path> (--prompt <text> | --tokens <ids>)");
            std::process::exit(1);
        }
    };

    let backend_name = backend_name.as_deref().unwrap_or("ndarray");
    let start = std::time::Instant::now();
    let generation_options = GenerationOptions::new(max_new_tokens, temperature, seed, top_k, top_p)?;
    if stream {
        let mut stdout = stdout();
        print!("Result: ");
        stdout.flush().expect("failed to flush stdout");

        with_backend(backend_name, |backend| {
            if kv_cache {
                generation_options.generate_streaming_with_kv_cache_and_backend(
                    &model,
                    &token_ids,
                    backend,
                    |chunk| {
                        print!("{chunk}");
                        stdout.flush().expect("failed to flush stdout");
                    },
                )
            } else {
                generation_options.generate_streaming_with_backend(&model, &token_ids, backend, |chunk| {
                    print!("{chunk}");
                    stdout.flush().expect("failed to flush stdout");
                })
            }
        })?;
        println!();
    } else {
        let decoded_text = if kv_cache {
            with_backend(backend_name, |backend| {
                generation_options.generate_with_kv_cache_and_backend(&model, &token_ids, backend)
            })?
        } else {
            with_backend(backend_name, |backend| {
                generation_options.generate_with_backend(&model, &token_ids, backend)
            })?
        };
        println!("Result: {}", decoded_text);
    }
    let elapsed = start.elapsed();
    println!("Elapsed: {:.3}s", elapsed.as_secs_f64());
    Ok(())
}

fn print_logits(mut args: impl Iterator<Item = String>) -> Result<()> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut selected_ids: Option<String> = None;
    let mut backend_name: Option<String> = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model_path = args.next();
            }
            "--prompt" => {
                prompt = args.next();
            }
            "--tokens" => {
                tokens = args.next();
            }
            "--ids" => {
                selected_ids = args.next();
            }
            "--backend" => {
                backend_name = args.next();
            }
            other => {
                eprintln!("Unknown logits option: {other}");
                std::process::exit(2);
            }
        }
    }

    let model_path = match model_path {
        Some(path) => path,
        None => {
            eprintln!("Usage: miniinfer logits --model <path> (--prompt <text> | --tokens <ids>) --ids <ids>");
            std::process::exit(1);
        }
    };
    let selected_ids = match selected_ids {
        Some(ids) => parse_token_ids(&ids)?,
        None => {
            eprintln!("Usage: miniinfer logits --model <path> (--prompt <text> | --tokens <ids>) --ids <ids>");
            std::process::exit(1);
        }
    };

    let model = load_model(model_path)?;
    model.validate()?;
    let token_ids = match (prompt, tokens) {
        (Some(prompt), None) => model.encode_prompt(&prompt)?,
        (None, Some(tokens)) => parse_token_ids(&tokens)?,
        (Some(_), Some(_)) => {
            eprintln!("Use either --prompt or --tokens, not both");
            std::process::exit(1);
        }
        (None, None) => {
            eprintln!("Usage: miniinfer logits --model <path> (--prompt <text> | --tokens <ids>) --ids <ids>");
            std::process::exit(1);
        }
    };

    let backend_name = backend_name.as_deref().unwrap_or("ndarray");
    let logits = with_backend(backend_name, |backend| model.forward_with_backend(&token_ids, backend))?;
    if logits.shape().len() != 2 {
        return Err(MiniInferError::WrongRank { expected: 2, actual: logits.shape().len() });
    }

    let row = logits.shape()[0] - 1;
    let vocab_size = logits.shape()[1];
    for token_id in selected_ids {
        if token_id >= vocab_size {
            return Err(MiniInferError::IndexOutOfBounds { index: token_id, len: vocab_size });
        }
        println!("{token_id}\t{:.8}", logits.get_2d(row, token_id)?);
    }

    Ok(())
}

fn with_backend<T>(backend_name: &str, run: impl FnOnce(&dyn OpsBackend) -> Result<T>) -> Result<T> {
    match backend_name {
        "ndarray" => {
            let backend = NdArrayBackend::new();
            run(&backend)
        }
        "reference" => {
            let backend = ReferenceBackend::new();
            run(&backend)
        }
        other => {
            eprintln!("Unknown backend: {other}");
            eprintln!("Use --backend ndarray or --backend reference");
            std::process::exit(2);
        }
    }
}

fn parse_token_ids(tokens: &str) -> Result<Vec<usize>> {
    tokens
        .split(',')
        .map(|token| token.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| MiniInferError::InvalidInput)
}

fn bench_generate(mut args: impl Iterator<Item = String>) -> Result<()> {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut backend_name: Option<String> = None;
    let mut max_new_tokens = 1;
    let mut kv_cache = false;
    let mut compare_cache = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model_path = args.next();
            }
            "--prompt" => {
                prompt = args.next();
            }
            "--tokens" => {
                tokens = args.next();
            }
            "--backend" => {
                backend_name = args.next();
            }
            "--max-new-tokens" => {
                let value = match args.next() {
                    Some(value) => value,
                    None => {
                        eprintln!("Error: --max-new-tokens requires a number");
                        std::process::exit(1);
                    }
                };

                max_new_tokens = match value.parse::<usize>() {
                    Ok(value) => value,
                    Err(_) => {
                        eprintln!("Error: --max-new-tokens must be a positive integer");
                        std::process::exit(1);
                    }
                };
            }
            "--kv-cache" => {
                kv_cache = true;
            }
            "--compare-cache" => {
                compare_cache = true;
            }
            other => {
                eprintln!("Unknown bench-generate option: {other}");
                std::process::exit(2);
            }
        }
    }
    if kv_cache && compare_cache {
        eprintln!("Use either --kv-cache or --compare-cache, not both");
        std::process::exit(1);
    }

    let model_path = match model_path {
        Some(path) => path,
        None => {
            eprintln!("Usage: miniinfer bench-generate --model <path> (--prompt <text> | --tokens <ids>)");
            std::process::exit(1);
        }
    };

    let total_start = std::time::Instant::now();
    let load_start = std::time::Instant::now();
    let model = load_model(model_path)?;
    model.validate()?;
    let load_elapsed = load_start.elapsed();

    let encode_start = std::time::Instant::now();
    let token_ids = match (prompt, tokens) {
        (Some(prompt), None) => model.encode_prompt(&prompt)?,
        (None, Some(tokens)) => parse_token_ids(&tokens)?,
        (Some(_), Some(_)) => {
            eprintln!("Use either --prompt or --tokens, not both");
            std::process::exit(1);
        }
        (None, None) => {
            eprintln!("Usage: miniinfer bench-generate --model <path> (--prompt <text> | --tokens <ids>)");
            std::process::exit(1);
        }
    };
    let encode_elapsed = encode_start.elapsed();
    let prompt_tokens = token_ids.len();

    let requested_length = prompt_tokens + max_new_tokens;
    if requested_length > model.config().max_position_embeddings {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "requested sequence length {requested_length} exceeds max_position_embeddings {}",
                model.config().max_position_embeddings
            ),
        });
    }

    let backend_name = backend_name.as_deref().unwrap_or("ndarray");
    if compare_cache {
        let (no_cache_result, kv_cache_result) = with_backend(backend_name, |backend| {
            let no_cache_result = run_generation_benchmark(
                &model,
                &token_ids,
                max_new_tokens,
                backend,
                false,
            )?;
            let kv_cache_result = run_generation_benchmark(
                &model,
                &token_ids,
                max_new_tokens,
                backend,
                true,
            )?;

            Ok((no_cache_result, kv_cache_result))
        })?;

        let total_elapsed = total_start.elapsed();

        println!("Backend: {backend_name}");
        println!("Prompt tokens: {prompt_tokens}");
        println!("Requested tokens: {max_new_tokens}");
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
        let result = with_backend(backend_name, |backend| {
            run_generation_benchmark(&model, &token_ids, max_new_tokens, backend, kv_cache)
        })?;
        let total_elapsed = total_start.elapsed();

        println!("Backend: {backend_name}");
        println!("Cache: {}", if kv_cache { "kv" } else { "none" });
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
        generation_options.generate_with_kv_cache_and_token_observer_and_backend(
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
}

fn speedup_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
