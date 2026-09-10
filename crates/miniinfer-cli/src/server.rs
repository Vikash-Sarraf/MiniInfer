use std::{convert::Infallible, net::SocketAddr, sync::Arc, time::{Duration, SystemTime, UNIX_EPOCH}};

use axum::{extract::State, http::StatusCode, response::{IntoResponse, Response, Sse}, routing::post, Json, Router};
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;
use axum::response::sse::{Event, KeepAlive};
use miniinfer_core::{
    error::{MiniInferError, Result},
    model::loader::{load_model_with_runtime, LoadedModel},
    runtime::generation::GenerationOptions,
};
use serde::{Deserialize, Serialize};

use crate::{args::{BackendName, ServeArgs, WeightRuntimeArg}, with_backend};

#[derive(Clone)]
struct ServerState {
    model: Arc<LoadedModel>,
    model_name: String,
    backend: BackendName,
    weight_runtime: WeightRuntimeArg,
    use_kv_cache: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: CompletionUsage,
    elapsed_ms: u128,
    backend: String,
    weight_runtime: String,
    kv_cache: bool,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct CompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct CompletionStreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionStreamChoice>,
}

#[derive(Serialize)]
struct CompletionStreamChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

type HttpError = (StatusCode, Json<ErrorResponse>);
type HttpResult<T> = std::result::Result<T, HttpError>;

enum CompletionOutput {
    Json(Json<CompletionResponse>),
    Sse(Response),
}

impl IntoResponse for CompletionOutput {
    fn into_response(self) -> Response {
        match self {
            CompletionOutput::Json(json) => json.into_response(),
            CompletionOutput::Sse(response) => response,
        }
    }
}

pub(crate) async fn serve(args: ServeArgs) -> Result<()> {
    let use_kv_cache = !args.no_kv_cache;
    let model = load_model_with_runtime(&args.model, args.weight_runtime.into())?;
    model.validate()?;

    let state = Arc::new(ServerState {
        model: Arc::new(model),
        model_name: args.model.clone(),
        backend: args.backend,
        weight_runtime: args.weight_runtime,
        use_kv_cache,
    });

    let app = Router::new()
        .route("/v1/completions", post(create_completion))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|error| MiniInferError::InvalidConfig { message: format!("invalid server address: {error}") })?;
    println!("Serving MiniInfer at http://{}", addr);
    println!("POST http://{}/v1/completions", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|error| MiniInferError::InvalidConfig { message: format!("failed to bind server to {addr}: {error}") })?;

    axum::serve(listener, app).await
        .map_err(|error| MiniInferError::InvalidConfig { message: format!("server failed: {error}") })
}

async fn create_completion(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<CompletionRequest>,
) -> HttpResult<CompletionOutput> {
    if request.stream.unwrap_or(false) {
        create_completion_stream(state, request)
    } else {
        run_completion(&state, request)
            .map(|response| CompletionOutput::Json(Json(response)))
            .map_err(server_error)
    }
}

fn create_completion_stream(
    state: Arc<ServerState>,
    request: CompletionRequest,
) -> HttpResult<CompletionOutput> {
    validate_completion_request(&request).map_err(server_error)?;

    let (sender, receiver) = mpsc::channel::<std::result::Result<Event, Infallible>>(16);

    tokio::task::spawn_blocking(move || {
        let result = run_completion_streaming(state, request, sender.clone());

        if let Err(err) = result {
            let _ = sender.blocking_send(Ok(
                Event::default().event("error").data(format!("{err:?}")),
            ));
        }
        let _ = sender.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(receiver);
    let sse = Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("keep-alive"),
    );

    Ok(CompletionOutput::Sse(
        sse.into_response()
    ))
}

fn run_completion_streaming(
    state: Arc<ServerState>,
    request: CompletionRequest,
    sender: mpsc::Sender<std::result::Result<Event, Infallible>>,
) -> Result<()> {
    let max_tokens = request.max_tokens.unwrap_or(1);
    let prompt_tokens = state.model.encode_prompt(&request.prompt)?;
    let options = GenerationOptions::new(
        max_tokens,
        request.temperature,
        request.seed,
        request.top_k,
        request.top_p,
    )?;

    let (created, created_nanos) = current_unix_time_parts();

    let id = format!("cmpl-{created}-{created_nanos}");
    let response_model = request.model.unwrap_or_else(|| state.model_name.clone());
    let mut saw_prompt = false;

    with_backend(state.backend, |backend| {
        if state.use_kv_cache {
            options.generate_streaming_with_kv_cache_and_backend(
                &state.model,
                &prompt_tokens,
                backend,
                |chunk| {
                    if !saw_prompt {
                        saw_prompt = true;
                        return;
                    }
                    send_completion_chunk(
                        &sender,
                        &id,
                        created,
                        &response_model,
                        chunk,
                        None,
                    );
                },
            )
        } else {
            options.generate_streaming_with_backend(
                &state.model,
                &prompt_tokens,
                backend,
                |chunk| {
                    if !saw_prompt {
                        saw_prompt = true;
                        return;
                    }
                    send_completion_chunk(
                        &sender,
                        &id,
                        created,
                        &response_model,
                        chunk,
                        None,
                    );
                },
            )
        }
    })?;

    send_completion_chunk(
        &sender,
        &id,
        created,
        &response_model,
        "",
        Some("length"),
    );
    Ok(())
}

fn send_completion_chunk(
    sender: &mpsc::Sender<std::result::Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: Option<&str>,
) {
    let chunk = CompletionStreamChunk {
        id: id.to_string(),
        object: "text_completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![CompletionStreamChoice {
            text: text.to_string(),
            index: 0,
            finish_reason: finish_reason.map(|s| s.to_string()),
        }],
    };

    if let Ok(data) = serde_json::to_string(&chunk) {
        let _ = sender.blocking_send(Ok(Event::default().data(data)));
    }
}

fn run_completion(state: &ServerState, request: CompletionRequest) -> Result<CompletionResponse> {
    if request.stream.unwrap_or(false) {
        return Err(MiniInferError::InvalidConfig {
            message: "streaming completions are not implemented yet".to_string(),
        });
    }

    validate_completion_request(&request)?;

    let max_tokens = request.max_tokens.unwrap_or(1);
    let prompt_tokens = state.model.encode_prompt(&request.prompt)?;
    let options = GenerationOptions::new(
        max_tokens,
        request.temperature,
        request.seed,
        request.top_k,
        request.top_p,
    )?;

    let start = std::time::Instant::now();
    let mut completion_tokens = 0;
    let decoded_text = with_backend(state.backend, |backend| {
        if state.use_kv_cache {
            options.generate_with_kv_cache_and_token_observer_and_backend(
                &state.model,
                &prompt_tokens,
                backend,
                |_, _| completion_tokens += 1,
            )
        } else {
            options.generate_with_token_observer_and_backend(
                &state.model,
                &prompt_tokens,
                backend,
                |_, _| completion_tokens += 1,
            )
        }
    })?;
    let elapsed_ms = start.elapsed().as_millis();

    let completion_text = decoded_text
        .strip_prefix(&request.prompt)
        .unwrap_or(&decoded_text)
        .to_string();
    let (created, created_nanos) = current_unix_time_parts();
    let finish_reason = if completion_tokens >= max_tokens { "length" } else { "stop" };
    let response_model = request.model.unwrap_or_else(|| state.model_name.clone());

    Ok(CompletionResponse {
        id: format!("cmpl-{created}-{created_nanos}"),
        object: "text_completion",
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            text: completion_text,
            index: 0,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: CompletionUsage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
        elapsed_ms,
        backend: state.backend.to_string(),
        weight_runtime: state.weight_runtime.to_string(),
        kv_cache: state.use_kv_cache,
    })
}

fn validate_completion_request(request: &CompletionRequest) -> Result<()> {
    if request.temperature.is_none()
        && (request.seed.is_some() || request.top_k.is_some() || request.top_p.is_some())
    {
        return Err(MiniInferError::InvalidConfig {
            message: "seed, top_k, and top_p require temperature".to_string(),
        });
    }

    Ok(())
}

fn current_unix_time_parts() -> (u64, u32) {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => (duration.as_secs(), duration.subsec_nanos()),
        Err(_) => (0, 0),
    }
}

fn server_error(error: MiniInferError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: format!("{error:?}") }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_accepts_supported_sampling_fields() {
        let request: CompletionRequest = serde_json::from_str(
            r#"{
                "prompt": "Hello",
                "max_tokens": 4,
                "temperature": 0.9,
                "top_k": 40,
                "top_p": 0.95,
                "seed": 42
            }"#,
        )
        .expect("valid completion request should parse");

        assert_eq!(request.prompt, "Hello");
        assert_eq!(request.max_tokens, Some(4));
        assert_eq!(request.temperature, Some(0.9));
        assert_eq!(request.top_k, Some(40));
        assert_eq!(request.top_p, Some(0.95));
        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn completion_request_rejects_unknown_fields() {
        let err = serde_json::from_str::<CompletionRequest>(
            r#"{
                "prompt": "Hello",
                "max_tokens": 4,
                "top-k": 40
            }"#,
        )
        .expect_err("unknown completion request fields should be rejected");

        assert!(
            err.to_string().contains("unknown field"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn completion_request_sampling_fields_require_temperature() {
        let request: CompletionRequest = serde_json::from_str(
            r#"{
                "prompt": "Hello",
                "max_tokens": 4,
                "top_k": 40
            }"#,
        )
        .expect("request should parse before semantic validation");

        let err = validate_completion_request(&request)
            .expect_err("top_k without temperature should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "seed, top_k, and top_p require temperature".to_string(),
            }
        );
    }
}
