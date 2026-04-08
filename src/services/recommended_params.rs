use crate::core::ProviderConfig;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RecommendedParams {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub enable_thinking: Option<bool>,
}

pub fn get_recommended_params(model_path: &str, quantization: &str) -> RecommendedParams {
    if let Some(hf_params) = fetch_from_huggingface(model_path) {
        return hf_params;
    }

    get_quantization_defaults(quantization)
}

fn fetch_from_huggingface(model_path: &str) -> Option<RecommendedParams> {
    let path_lower = model_path.to_lowercase();

    if path_lower.contains("huggingface.co") || path_lower.contains("/models--") {
        let model_id = extract_hf_model_id(model_path)?;
        return fetch_from_hf_api(&model_id);
    }

    if path_lower.contains("gguf") {
        if let Some(model_name) = extract_model_name_from_path(model_path) {
            if let Some(id) = find_huggingface_model_by_name(&model_name) {
                return fetch_from_hf_api(&id);
            }
        }
    }

    None
}

fn extract_hf_model_id(model_path: &str) -> Option<String> {
    let path = model_path;
    if path.contains("models--") {
        let parts: Vec<&str> = path.split("models--").collect::<Vec<_>>();
        if parts.len() > 1 {
            let rest = parts[1];
            let model_id = rest.replace("--", "/");
            return Some(model_id);
        }
    }

    if path.contains("huggingface.co/") {
        let parts: Vec<&str> = path.split("huggingface.co/").collect::<Vec<_>>();
        if parts.len() > 1 {
            return Some(parts[1].to_string());
        }
    }

    None
}

fn extract_model_name_from_path(model_path: &str) -> Option<String> {
    let path = std::path::Path::new(model_path);
    let filename = path.file_name()?.to_string_lossy().to_string();

    let name_without_ext = filename.rsplit('.').next().unwrap_or(&filename);

    let known_prefixes = [
        "llama",
        "mistral",
        "qwen",
        "phi",
        "gemma",
        "mixtral",
        " Yi-",
        "baichuan",
        "chatglm",
        "falcon",
        "stablelm",
        "redpajama",
        "m2",
        "tinyllama",
        "TinyLlama",
        "deepseek",
        "指令模型",
        "sft",
        "instruct",
    ];

    for prefix in known_prefixes.iter() {
        if name_without_ext
            .to_lowercase()
            .contains(&prefix.to_lowercase())
        {
            return Some(name_without_ext.to_string());
        }
    }

    None
}

fn find_huggingface_model_by_name(name: &str) -> Option<String> {
    let search_name = name.split_whitespace().next()?.to_lowercase();

    let model_aliases = [
        ("llama", "meta-llama/Llama-2-7b-hf"),
        ("llama3", "meta-llama/Meta-Llama-3-8B"),
        ("llama3.1", "meta-llama/Llama-3.1-8B"),
        ("mistral", "mistralai/Mistral-7B-v0.1"),
        ("mixtral", "mistralai/Mixtral-8x7B-v0.1"),
        ("qwen", "Qwen/Qwen2-7B"),
        ("qwen2", "Qwen/Qwen2-7B"),
        ("phi", "microsoft/Phi-3-mini-4k-instruct"),
        ("phi3", "microsoft/Phi-3-mini-4k-instruct"),
        ("gemma", "google/gemma-7b"),
        ("deepseek", "deepseek-ai/DeepSeek-Coder-V2"),
        ("Yi", "01-ai/Yi-6B"),
        ("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("chatglm", "THUDM/chatglm3-6b"),
    ];

    for (keyword, model_id) in model_aliases.iter() {
        if search_name.contains(keyword) {
            return Some(model_id.to_string());
        }
    }

    None
}

fn fetch_from_hf_api(model_id: &str) -> Option<RecommendedParams> {
    let url = format!("https://huggingface.co/api/models/{}", model_id);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;

    let response = client.get(&url).send().ok()?;

    if !response.status().is_success() {
        return None;
    }

    let json: serde_json::Value = response.json().ok()?;

    if let Some(card_data) = json.get("card_data") {
        if let Some(config) = card_data.get("config") {
            return parse_recommended_from_config(config);
        }
    }

    None
}

fn parse_recommended_from_config(config: &serde_json::Value) -> Option<RecommendedParams> {
    let mut params = RecommendedParams {
        temperature: None,
        top_k: None,
        top_p: None,
        min_p: None,
        presence_penalty: None,
        repetition_penalty: None,
        enable_thinking: None,
    };

    if let Some(val) = config.get("temperature").and_then(|v| v.as_f64()) {
        params.temperature = Some(val as f32);
    }
    if let Some(val) = config.get("top_k").and_then(|v| v.as_i64()) {
        params.top_k = Some(val as i32);
    }
    if let Some(val) = config.get("top_p").and_then(|v| v.as_f64()) {
        params.top_p = Some(val as f32);
    }
    if let Some(val) = config.get("min_p").and_then(|v| v.as_f64()) {
        params.min_p = Some(val as f32);
    }
    if let Some(val) = config.get("presence_penalty").and_then(|v| v.as_f64()) {
        params.presence_penalty = Some(val as f32);
    }
    if let Some(val) = config.get("repetition_penalty").and_then(|v| v.as_f64()) {
        params.repetition_penalty = Some(val as f32);
    }

    Some(params)
}

pub fn get_quantization_defaults(quantization: &str) -> RecommendedParams {
    let q_lower = quantization.to_lowercase();

    if q_lower.contains("q2") || q_lower.contains("q3") {
        RecommendedParams {
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: Some(0.05),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.1),
            enable_thinking: None,
        }
    } else if q_lower.contains("q4") || q_lower.contains("q5") {
        RecommendedParams {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.05),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.1),
            enable_thinking: None,
        }
    } else if q_lower.contains("q6") || q_lower.contains("q8") || q_lower == "f16" {
        RecommendedParams {
            temperature: Some(0.6),
            top_k: Some(40),
            top_p: Some(0.85),
            min_p: Some(0.1),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.15),
            enable_thinking: None,
        }
    } else {
        RecommendedParams {
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.9),
            min_p: Some(0.05),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.1),
            enable_thinking: None,
        }
    }
}

pub fn apply_recommended_params(config: &mut ProviderConfig, params: &RecommendedParams) {
    if let Some(temp) = params.temperature {
        config.temperature = Some(temp);
    }
    if let Some(top_k) = params.top_k {
        config.top_k = Some(top_k);
    }
    if let Some(top_p) = params.top_p {
        config.top_p = Some(top_p);
    }
    if let Some(min_p) = params.min_p {
        config.min_p = Some(min_p);
    }
    if let Some(presence_pen) = params.presence_penalty {
        config.presence_penalty = Some(presence_pen);
    }
    if let Some(repeat_pen) = params.repetition_penalty {
        config.repetition_penalty = Some(repeat_pen);
    }
    if let Some(thinking) = params.enable_thinking {
        config.enable_thinking = Some(thinking);
    }
}
