use std::io::{Write, stdout};

use clap::Parser;
use miniinfer_core::{
    error::{MiniInferError, Result}, model::{config::ModelConfig, loader::{LoadedModel, load_model, load_model_with_runtime}}, ops::backend::{NdArrayBackend, OpsBackend, ReferenceBackend}, runtime::generation::GenerationOptions,
};

mod args;
mod bench;

use args::{BackendName, Cli, Command, InspectArgs, LogitsArgs, PromptInputArgs, RunArgs};
use bench::{bench_generate, bench_matmul};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run(args) => run_model(args)?,
        Command::Inspect(args) => inspect_model(args)?,
        Command::Logits(args) => print_logits(args)?,
        Command::Bench => println!("miniinfer bench: not implemented yet"),
        Command::BenchGenerate(args) => bench_generate(args)?,
        Command::BenchMatmul => bench_matmul(),
    }
    Ok(())
}

fn inspect_model(args: InspectArgs) -> Result<()> {
    println!("Inspecting model at path: {}", args.model);

    let model = load_model(args.model)?;
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

fn run_model(args: RunArgs) -> Result<()> {
    let use_kv_cache = !args.no_kv_cache;
    let model = load_model_with_runtime(args.model, args.weight_runtime.into())?;
    model.validate()?;
    let token_ids = encode_prompt_input(&model, args.input)?;

    let start = std::time::Instant::now();
    let generation_options = GenerationOptions::new(args.max_new_tokens, args.temperature, args.seed, args.top_k, args.top_p)?;
    if args.stream {
        let mut stdout = stdout();
        print!("Result: ");
        stdout.flush().expect("failed to flush stdout");

        with_backend(args.backend, |backend| {
            if use_kv_cache {
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
        let decoded_text = if use_kv_cache {
            with_backend(args.backend, |backend| {
                generation_options.generate_with_kv_cache_and_backend(&model, &token_ids, backend)
            })?
        } else {
            with_backend(args.backend, |backend| {
                generation_options.generate_with_backend(&model, &token_ids, backend)
            })?
        };
        println!("Result: {}", decoded_text);
    }
    let elapsed = start.elapsed();
    println!("Elapsed: {:.3}s", elapsed.as_secs_f64());
    Ok(())
}

fn print_logits(args: LogitsArgs) -> Result<()> {
    let selected_ids = parse_token_ids(&args.ids)?;
    let model = load_model(args.model)?;
    model.validate()?;
    let token_ids = encode_prompt_input(&model, args.input)?;

    let logits = with_backend(args.backend, |backend| model.forward_with_backend(&token_ids, backend))?;
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

pub(crate) fn encode_prompt_input(model: &LoadedModel, input: PromptInputArgs) -> Result<Vec<usize>> {
    match (input.prompt, input.tokens) {
        (Some(prompt), None) => model.encode_prompt(&prompt),
        (None, Some(tokens)) => parse_token_ids(&tokens),
        _ => Err(MiniInferError::InvalidInput),
    }
}

pub(crate) fn with_backend<T>(backend_name: BackendName, run: impl FnOnce(&dyn OpsBackend) -> Result<T>) -> Result<T> {
    match backend_name {
        BackendName::Ndarray => {
            let backend = NdArrayBackend::new();
            run(&backend)
        }
        BackendName::Reference => {
            let backend = ReferenceBackend::new();
            run(&backend)
        }
    }
}

pub(crate) fn parse_token_ids(tokens: &str) -> Result<Vec<usize>> {
    tokens
        .split(',')
        .map(|token| token.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| MiniInferError::InvalidInput)
}
