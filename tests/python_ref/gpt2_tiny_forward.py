import numpy as np


EPSILON = 1e-5
EXPECTED_SHAPE = [2, 8]
EXPECTED_LOGITS = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0],
    ],
    dtype=np.float32,
)


def layer_norm(hidden, weight, bias, epsilon):
    mean = hidden.mean(axis=-1, keepdims=True)
    variance = ((hidden - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((hidden - mean) / np.sqrt(variance + epsilon)) * weight + bias


def gelu(hidden):
    return 0.5 * hidden * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden**3))
    )


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def attention_context(hidden, block, head_dim):
    qkv = hidden @ block["c_attn_weight"] + block["c_attn_bias"]
    query, key, value = np.split(qkv, 3, axis=-1)

    scores = (query @ key.T) / np.sqrt(head_dim)
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    masked_scores = np.where(mask, -np.inf, scores)
    probabilities = softmax(masked_scores)

    return probabilities @ value


def attention_sublayer(hidden, block, head_dim, epsilon):
    normalized = layer_norm(hidden, block["ln_1_weight"], block["ln_1_bias"], epsilon)
    context = attention_context(normalized, block, head_dim)
    projected = context @ block["attn_c_proj_weight"] + block["attn_c_proj_bias"]
    return hidden + projected


def mlp(hidden, block):
    expanded = hidden @ block["c_fc_weight"] + block["c_fc_bias"]
    activated = gelu(expanded)
    return activated @ block["mlp_c_proj_weight"] + block["mlp_c_proj_bias"]


def block_forward(hidden, block, head_dim, epsilon):
    hidden = attention_sublayer(hidden, block, head_dim, epsilon)
    normalized = layer_norm(hidden, block["ln_2_weight"], block["ln_2_bias"], epsilon)
    return hidden + mlp(normalized, block)


def embed_tokens(token_ids, weights):
    positions = np.arange(len(token_ids))
    return weights["wte"][token_ids] + weights["wpe"][positions]


def model_forward(token_ids, weights, config):
    hidden = embed_tokens(token_ids, weights)
    head_dim = config["hidden_size"] // config["num_heads"]

    for block in weights["blocks"]:
        hidden = block_forward(hidden, block, head_dim, config["layer_norm_epsilon"])

    hidden = layer_norm(
        hidden,
        weights["ln_f_weight"],
        weights["ln_f_bias"],
        config["layer_norm_epsilon"],
    )
    return hidden @ weights["lm_head_weight"]


def tiny_fixture():
    block = {
        "ln_1_weight": np.zeros(4, dtype=np.float32),
        "ln_1_bias": np.zeros(4, dtype=np.float32),
        "c_attn_weight": np.zeros((4, 12), dtype=np.float32),
        "c_attn_bias": np.zeros(12, dtype=np.float32),
        "attn_c_proj_weight": np.zeros((4, 4), dtype=np.float32),
        "attn_c_proj_bias": np.zeros(4, dtype=np.float32),
        "ln_2_weight": np.zeros(4, dtype=np.float32),
        "ln_2_bias": np.zeros(4, dtype=np.float32),
        "c_fc_weight": np.zeros((4, 16), dtype=np.float32),
        "c_fc_bias": np.zeros(16, dtype=np.float32),
        "mlp_c_proj_weight": np.zeros((16, 4), dtype=np.float32),
        "mlp_c_proj_bias": np.zeros(4, dtype=np.float32),
    }

    weights = {
        "wte": np.zeros((8, 4), dtype=np.float32),
        "wpe": np.zeros((8, 4), dtype=np.float32),
        "blocks": [block],
        "ln_f_weight": np.zeros(4, dtype=np.float32),
        "ln_f_bias": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "lm_head_weight": np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    config = {
        "vocab_size": 8,
        "max_position_embeddings": 8,
        "hidden_size": 4,
        "num_layers": 1,
        "num_heads": 2,
        "intermediate_size": 16,
        "layer_norm_epsilon": EPSILON,
    }

    return weights, config


def main():
    weights, config = tiny_fixture()
    token_ids = np.array([2, 1], dtype=np.int64)
    logits = model_forward(token_ids, weights, config)

    assert list(logits.shape) == EXPECTED_SHAPE
    np.testing.assert_allclose(logits, EXPECTED_LOGITS, rtol=0.0, atol=1e-6)

    print("shape:", list(logits.shape))
    print("logits:", logits.reshape(-1).tolist())
    print("rust expected:")
    print("[" + ", ".join(f"{value:.1f}" for value in logits.reshape(-1)) + "]")


if __name__ == "__main__":
    main()