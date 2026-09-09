use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "miniinfer")]
#[command(bin_name = "miniinfer")]
#[command(about = "CPU-first LLM inference runtime experiments")]
#[command(arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    #[command(about = "Run inference on a model")]
    Run(RunArgs),
    #[command(about = "Inspect a model")]
    Inspect(InspectArgs),
    #[command(about = "Print selected logits for debugging/parity checks")]
    Logits(LogitsArgs),
    #[command(about = "Benchmark a model")]
    Bench,
    #[command(name = "bench-generate", about = "Benchmark greedy generation timings")]
    BenchGenerate(BenchGenerateArgs),
    #[command(name = "bench-matmul", about = "Benchmark reference vs ndarray matmul")]
    BenchMatmul,
}

#[derive(Args)]
pub(crate) struct InspectArgs {
    #[arg(long)]
    pub model: String,
}

#[derive(Args)]
pub(crate) struct RunArgs {
    #[arg(long)]
    pub model: String,
    #[command(flatten)]
    pub input: PromptInputArgs,
    #[arg(long, default_value_t = BackendName::Ndarray)]
    pub backend: BackendName,
    #[arg(long, default_value_t = 1)]
    pub max_new_tokens: usize,
    #[arg(long)]
    pub stream: bool,
    #[arg(long, hide = true, conflicts_with = "no_kv_cache")]
    pub kv_cache: bool,
    #[arg(long)]
    pub no_kv_cache: bool,
    #[arg(long, value_enum, default_value_t = WeightRuntimeArg::PackedInt8)]
    pub weight_runtime: WeightRuntimeArg,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long, requires = "temperature")]
    pub seed: Option<u64>,
    #[arg(long, requires = "temperature")]
    pub top_k: Option<usize>,
    #[arg(long, requires = "temperature")]
    pub top_p: Option<f32>,
}

#[derive(Args)]
pub(crate) struct LogitsArgs {
    #[arg(long)]
    pub model: String,
    #[command(flatten)]
    pub input: PromptInputArgs,
    #[arg(long)]
    pub ids: String,
    #[arg(long, default_value_t = BackendName::Ndarray)]
    pub backend: BackendName,
}

#[derive(Args)]
pub(crate) struct BenchGenerateArgs {
    #[arg(long)]
    pub model: String,
    #[command(flatten)]
    pub input: PromptInputArgs,
    #[arg(long, default_value_t = BackendName::Ndarray)]
    pub backend: BackendName,
    #[arg(long, default_value_t = 1)]
    pub max_new_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pub runs: usize,
    #[arg(long, hide = true, conflicts_with_all = ["compare_cache", "no_kv_cache"])]
    pub kv_cache: bool,
    #[arg(long, conflicts_with_all = ["compare_cache", "kv_cache"])]
    pub no_kv_cache: bool,
    #[arg(long)]
    pub compare_cache: bool,
    #[arg(long, value_enum, default_value_t = WeightRuntimeArg::PackedInt8)]
    pub weight_runtime: WeightRuntimeArg,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
pub(crate) struct PromptInputArgs {
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long)]
    pub tokens: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum BackendName {
    Ndarray,
    Reference,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum WeightRuntimeArg {
    F32,
    #[value(alias = "packedint8")]
    PackedInt8,
}

impl From<WeightRuntimeArg> for miniinfer_core::model::loader::WeightRuntime {
    fn from(value: WeightRuntimeArg) -> Self {
        match value {
            WeightRuntimeArg::F32 => Self::F32,
            WeightRuntimeArg::PackedInt8 => Self::PackedInt8,
        }
    }
}

impl std::fmt::Display for WeightRuntimeArg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(formatter, "f32"),
            Self::PackedInt8 => write!(formatter, "packed-int8"),
        }
    }
}

impl std::fmt::Display for BackendName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ndarray => write!(formatter, "ndarray"),
            Self::Reference => write!(formatter, "reference"),
        }
    }
}