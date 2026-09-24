use std::process::{Command, Stdio};
use once_cell::sync::Lazy;

use crate::core::{
    CpuOffloadMode, DetectedServer, LlmProvider, ModelInfo, OptionValueType, ProviderConfig,
    ProviderError, ProviderOption, ProviderSettings, Result,
};
use crate::models::ModelType;

pub struct LlamaCppProvider {
    id: &'static str,
    name: &'static str,
}

impl LlamaCppProvider {
    pub fn new() -> Self {
        Self {
            id: "llama.cpp",
            name: "llama.cpp (GGUF)",
        }
    }
}

static LLAMA_SERVER_PATH: Lazy<String> = Lazy::new(|| {
    std::process::Command::new("which")
        .arg("llama-server")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "llama-server".to_string())
});

impl Default for LlamaCppProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the corresponding mmproj file for a given model path.
/// Returns the mmproj path if found, None otherwise.
///
/// mmproj files are typically named like "mmproj-BF16.gguf", "mmproj-f16.gguf", etc.
/// and are located in the same directory as the main model.
pub fn find_mmproj_for_model(model_path: &str) -> Option<String> {
    let path = std::path::Path::new(model_path);
    let parent = path.parent()?;
    let model_file_name = path.file_name()?.to_string_lossy();

    // Check if model file name suggests it's a vision/multimodal model
    let model_stem = model_file_name.rsplit_once('.').map(|x| x.0)
        .unwrap_or(&model_file_name);

    // Try to find mmproj file in the same directory
    if let Ok(entries) = std::fs::read_dir(parent) {
        let mut best_match: Option<String> = None;

        for entry in entries.flatten() {
            let entry_path = entry.path();
            if let Some(ext) = entry_path.extension() {
                if ext.to_string_lossy().to_lowercase() == "gguf" {
                    if let Some(file_name) = entry_path.file_name() {
                        let name_lower = file_name.to_string_lossy().to_lowercase();

                        // Check if it's an mmproj file
                        if name_lower.starts_with("mmproj") {
                            let mmproj_path = entry_path.to_string_lossy().to_string();

                            // Try to match model type hint in the filename
                            // e.g., if model has "qwen" in name, prefer mmproj with "qwen" in name
                            let model_lower = model_stem.to_lowercase();
                            if name_lower.contains(&model_lower)
                                || name_lower.contains("vision")
                                || name_lower.contains("mm")
                            {
                                // Prefer this match
                                best_match = Some(mmproj_path);
                            } else if best_match.is_none() {
                                // Use as fallback
                                best_match = Some(mmproj_path);
                            }
                        }
                    }
                }
            }
        }

        return best_match;
    }

    None
}

impl LlmProvider for LlamaCppProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn get_config_template(&self) -> ProviderConfig {
        ProviderConfig::default()
    }

    fn validate_config(&self, config: &ProviderConfig) -> Result<()> {
        let hf_id_trimmed = config.huggingface_id.trim();
        if config.model_path.is_empty() && hf_id_trimmed.is_empty() {
            return Err(ProviderError::InvalidConfig(
                "Model path or HuggingFace ID is required".into(),
            ));
        }
        if !hf_id_trimmed.is_empty() && !hf_id_trimmed.contains('/') {
            return Err(ProviderError::InvalidConfig(format!(
                "Invalid HuggingFace ID format (expected 'user/repo'): {}",
                config.huggingface_id
            )));
        }

        if !config.model_path.is_empty() {
            let path = std::path::Path::new(&config.model_path);
            if !path.exists() {
                return Err(ProviderError::InvalidConfig(format!(
                    "Model file not found: {}",
                    config.model_path
                )));
            }

            if !path.is_file() {
                return Err(ProviderError::InvalidConfig(format!(
                    "Model path is not a file: {}",
                    config.model_path
                )));
            }
        }

        Ok(())
    }

    fn default_settings(&self) -> ProviderSettings {
        ProviderSettings {
            binary_path: LLAMA_SERVER_PATH.clone(),
            env_script: String::new(),
            additional_args: String::new(),
            health_endpoint: "/health".to_string(),
            heartbeat_interval_secs: 6,
            venv_path: String::new(),
        }
    }

    fn start_server(
        &self,
        config: &ProviderConfig,
        settings: &ProviderSettings,
    ) -> Result<std::process::Child> {
        let command_line = self.build_command_line(config, settings);

        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(&command_line);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.envs(crate::core::parse_env_pairs(&config.environment));

        cmd.spawn().map_err(ProviderError::from)
    }

    fn supported_quantizations(&self) -> Vec<&'static str> {
        vec![
            "f16", "q8_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "q3_1", "q3_0", "q2_1", "q2_0",
        ]
    }

    fn supports_gguf(&self) -> bool {
        true
    }

    fn build_command_line(&self, config: &ProviderConfig, settings: &ProviderSettings) -> String {
        let effective_gpu_layers = match config.cpu_offload {
            CpuOffloadMode::FullOffload => 0,
            CpuOffloadMode::Disabled => -1,
            CpuOffloadMode::Offload => {
                if config.gpu_layers < 0 {
                    let model_size_gb = if !config.model_path.is_empty() {
                        std::fs::metadata(&config.model_path)
                            .map(|m| m.len() as f32 / (1024.0 * 1024.0 * 1024.0))
                            .unwrap_or(7.0)
                    } else {
                        7.0
                    };
                    let total_layers = read_gguf_n_layer(&config.model_path).unwrap_or(0) as i32;
                    let recommended =
                        crate::services::recommend_gpu_layers(model_size_gb, total_layers);
                    recommended.max(0)
                } else {
                    config.gpu_layers
                }
            }
            CpuOffloadMode::Auto => config.gpu_layers,
        };

        let binary = if settings.binary_path.is_empty() {
            "llama-server"
        } else {
            &settings.binary_path
        };

        let effective_hf_id = if !config.huggingface_id.is_empty() {
            Some(config.huggingface_id.clone())
        } else if !config.model_path.is_empty() {
            extract_hf_repo_id_from_path(&config.model_path)
        } else {
            None
        };

        let quant_suffix = if !config.model_path.is_empty() {
            let filename = std::path::Path::new(&config.model_path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let quant = extract_quantization(&filename);
            if quant != "unknown" {
                format!(":{}", quant)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let mut args: Vec<String> = Vec::new();
        args.push(binary.to_string());

        if let Some(hf_id) = effective_hf_id {
            args.push(format!("-hf \"{}\"", hf_id + &quant_suffix));
        } else {
            args.push(format!("-m \"{}\"", config.model_path));
        }
        if config.context_size > 0 { args.push(format!("-c {}", config.context_size)); }
        if config.batch_size > 0 { args.push(format!("-b {}", config.batch_size)); }
        args.push(format!("-ngl {}", effective_gpu_layers));
        if config.threads > 0 { args.push(format!("-t {}", config.threads)); }
        args.push(format!("--port {}", config.port));
        args.push(format!("--host {}", config.host));
        if config.num_prompt_tracking > 0 { args.push(format!("-np {}", config.num_prompt_tracking)); }
        if !config.cache_type_k.is_empty() { args.push(format!("--cache-type-k \"{}\"", config.cache_type_k)); }
        if !config.cache_type_v.is_empty() { args.push(format!("--cache-type-v \"{}\"", config.cache_type_v)); }
        if let Some(temp) = config.temperature { args.push(format!("--temperature {}", temp)); }
        if let Some(top_k) = config.top_k { args.push(format!("--top-k {}", top_k)); }
        if let Some(top_p) = config.top_p { args.push(format!("--top-p {}", top_p)); }
        if let Some(min_p) = config.min_p { args.push(format!("--min-p {}", min_p)); }
        if let Some(presence_pen) = config.presence_penalty { args.push(format!("--presence-penalty {}", presence_pen)); }
        if let Some(repeat_pen) = config.repetition_penalty { args.push(format!("--repeat-penalty {}", repeat_pen)); }
        if config.enable_thinking == Some(true) { args.push("--reasoning-format deepseek".to_string()); }
        if !config.mmproj_path.is_empty() { args.push(format!("--mmproj \"{}\"", config.mmproj_path)); }
        for arg in config.additional_args.split_whitespace() { if !arg.is_empty() { args.push(arg.to_string()); } }
        for arg in settings.additional_args.split_whitespace() { if !arg.is_empty() { args.push(arg.to_string()); } }

        let inner = args.join(" ");
        if settings.env_script.is_empty() {
            inner
        } else {
            format!("bash -c 'source \"{}\" && {}'", settings.env_script, inner)
        }
    }

    fn scan_models(&self, path: &str) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let path_obj = std::path::Path::new(path);
        if path_obj.exists() {
            scan_recursive(path_obj, &mut models);
        }
        models
    }

    fn add_model(&self, path: &str) -> Result<ModelInfo> {
        let path_obj = std::path::Path::new(path);

        if !path_obj.exists() {
            return Err(ProviderError::InvalidConfig("File does not exist".into()));
        }

        parse_gguf_file(path_obj)
            .ok_or_else(|| ProviderError::InvalidConfig("Failed to parse GGUF file".into()))
    }

    fn default_model_directories(&self) -> Vec<String> {
        let mut dirs = Vec::new();

        if let Some(home) = dirs::cache_dir() {
            let llama_cpp_dir = home.join("llama.cpp");
            if llama_cpp_dir.exists() {
                dirs.push(llama_cpp_dir.to_string_lossy().to_string());
            }
        }

        if let Some(home) = dirs::cache_dir() {
            let hf_cache = home.join("huggingface").join("hub");
            if hf_cache.exists() {
                dirs.push(hf_cache.to_string_lossy().to_string());
            }
        }

        if let Some(home) = dirs::home_dir() {
            let models_dir = home.join("models");
            if models_dir.exists() {
                dirs.push(models_dir.to_string_lossy().to_string());
            }
        }

        dirs
    }

    fn get_options(&self) -> Vec<ProviderOption> {
        vec![
            ProviderOption {
                id: "temperature".to_string(),
                name: "Temperature".to_string(),
                value_type: OptionValueType::Number,
                default_value: "0.7".to_string(),
                description: "Sampling temperature (0.0-2.0)".to_string(),
            },
            ProviderOption {
                id: "top_k".to_string(),
                name: "Top-K".to_string(),
                value_type: OptionValueType::Number,
                default_value: "40".to_string(),
                description: "Top-K sampling (0-100)".to_string(),
            },
            ProviderOption {
                id: "top_p".to_string(),
                name: "Top-P".to_string(),
                value_type: OptionValueType::Number,
                default_value: "0.95".to_string(),
                description: "Top-P (nucleus) sampling (0.0-1.0)".to_string(),
            },
            ProviderOption {
                id: "min_p".to_string(),
                name: "Min-P".to_string(),
                value_type: OptionValueType::Number,
                default_value: "0.05".to_string(),
                description: "Min-P sampling (0.0-1.0)".to_string(),
            },
            ProviderOption {
                id: "presence_penalty".to_string(),
                name: "Presence Penalty".to_string(),
                value_type: OptionValueType::Number,
                default_value: "0.0".to_string(),
                description: "Presence penalty (-2.0-2.0)".to_string(),
            },
            ProviderOption {
                id: "repetition_penalty".to_string(),
                name: "Repetition Penalty".to_string(),
                value_type: OptionValueType::Number,
                default_value: "1.1".to_string(),
                description: "Repetition penalty (0.0-5.0)".to_string(),
            },
            ProviderOption {
                id: "enable_thinking".to_string(),
                name: "Enable Thinking".to_string(),
                value_type: OptionValueType::Bool,
                default_value: "false".to_string(),
                description: "Enable thinking/deepseek reasoning".to_string(),
            },
            ProviderOption {
                id: "additional_args".to_string(),
                name: "Additional CLI Args".to_string(),
                value_type: OptionValueType::String,
                default_value: String::new(),
                description: "Additional command-line arguments (space-separated)".to_string(),
            },
        ]
    }

    fn detect_running_servers(&self) -> Vec<DetectedServer> {
        let mut servers = Vec::new();

        if let Ok(output) = Command::new("pgrep").args(["-a", "llama-server"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Some(space_pos) = line.find(' ') {
                    let pid_str = &line[..space_pos];
                    let cmdline = &line[space_pos + 1..];
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        servers.push(DetectedServer {
                            pid,
                            binary: "llama-server".to_string(),
                            command_line: cmdline.to_string(),
                        });
                    }
                }
            }
        }

        servers
    }

    fn parse_server_config(&self, cmd_line: &str) -> ProviderConfig {
        let mut config = ProviderConfig::default();
        let mut additional_args = Vec::new();

        let args: Vec<&str> = cmd_line.split_whitespace().collect();
        let mut i = 0;

        while i < args.len() {
            let arg = args[i];

            match arg {
                "-m" | "--model" => {
                    if i + 1 < args.len() {
                        config.model_path = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "-hf" => {
                    if i + 1 < args.len() {
                        config.huggingface_id = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "-c" | "--ctx-size" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.context_size = val;
                        }
                        i += 1;
                    }
                }
                "-b" | "--batch-size" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.batch_size = val;
                        }
                        i += 1;
                    }
                }
                "-ngl" | "--n-gpu-layers" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<i32>() {
                            config.gpu_layers = val;
                        }
                        i += 1;
                    }
                }
                "-t" | "--threads" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.threads = val;
                        }
                        i += 1;
                    }
                }
                "--port" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.port = val;
                        }
                        i += 1;
                    }
                }
                "--host" => {
                    if i + 1 < args.len() {
                        config.host = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "-np" | "--parallel" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.num_prompt_tracking = val;
                        }
                        i += 1;
                    }
                }
                "--cache-type-k" => {
                    if i + 1 < args.len() {
                        config.cache_type_k = args[i + 1].to_string();
                        i += 1;
                    }
                }
        "--cache-type-v" => {
            if i + 1 < args.len() {
                config.cache_type_v = args[i + 1].to_string();
                i += 1;
            }
        }
        "--mmproj" => {
            if i + 1 < args.len() {
                config.mmproj_path = args[i + 1].to_string();
                i += 1;
            }
        }
        "--temperature" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.temperature = Some(val);
                        }
                        i += 1;
                    }
                }
                "--top-k" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<i32>() {
                            config.top_k = Some(val);
                        }
                        i += 1;
                    }
                }
                "--top-p" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.top_p = Some(val);
                        }
                        i += 1;
                    }
                }
                "--min-p" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.min_p = Some(val);
                        }
                        i += 1;
                    }
                }
                "--presence-penalty" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.presence_penalty = Some(val);
                        }
                        i += 1;
                    }
                }
                "--repeat-penalty" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.repetition_penalty = Some(val);
                        }
                        i += 1;
                    }
                }
                "--reasoning-format" => {
                    if i + 1 < args.len() {
                        let format = args[i + 1];
                        config.enable_thinking = Some(format == "deepseek");
                        i += 1;
                    }
                }
                _ => {
                    // Collect unknown arguments
                    if arg.starts_with('-') {
                        if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                            additional_args.push(arg);
                            additional_args.push(args[i + 1]);
                            i += 1;
                        } else {
                            additional_args.push(arg);
                        }
                    }
                }
            }
            i += 1;
        }

        if !additional_args.is_empty() {
            config.additional_args = additional_args.join(" ");
        }

        config
    }
}

fn scan_recursive(dir: &std::path::Path, models: &mut Vec<ModelInfo>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_recursive(&path, models);
            } else if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "gguf" {
                    if let Some(file_name) = path.file_name() {
                        let name_lower = file_name.to_string_lossy().to_lowercase();
                        if name_lower.starts_with("mmproj") {
                            continue;
                        }
                    }
                    if let Some(model) = parse_gguf_file(&path) {
                        models.push(model);
                    }
                }
            }
        }
    }
}

fn parse_gguf_file(path: &std::path::Path) -> Option<ModelInfo> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let size_gb = metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0);

    let quantization = extract_quantization(&filename);

    let name = extract_model_name_from_path(path, &filename);

    Some(ModelInfo {
        path: path.to_string_lossy().to_string(),
        name,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
        is_moe: filename.to_lowercase().contains("moe"),
    })
}

fn extract_model_name_from_path(path: &std::path::Path, filename: &str) -> String {
    let mut current = path;
    for _ in 0..4 {
        if let Some(parent) = current.parent() {
            if let Some(dir_name) = parent.file_name() {
                let dir_name = dir_name.to_string_lossy();
                if dir_name.starts_with("models--") {
                    let repo_id = dir_name.strip_prefix("models--").unwrap_or(&dir_name);
                    return repo_id.replace("--", "/").to_string();
                }
            }
            current = parent;
        } else {
            break;
        }
    }

    // Fallback: use filename without extension
    if let Some(stem) = filename.rsplit_once('.').map(|x| x.0) {
        stem.to_string()
    } else {
        filename.to_string()
    }
}

fn extract_quantization(filename: &str) -> String {
    let lower = filename.to_lowercase();

    // Weight quantization patterns (from filename)
    // Order matters: check more specific patterns first
    let quantizations = [
        // Standard Q4_K_M style
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k",
        "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
        "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0",
        "f16", "f32", "bf16",
        // Unsloth variants with prefixes
        "ud-q5_k_xl", "ud-q4_k_m", "ud-q5_k_m", "ud-q6_k",
        "mxfp4_moe", "mxfp4", "mxfp6", "mxfp8",
        // Other variants
        "iq3_m", "iq4_xs", "iq4_nl", "q3_k_s", "q3_k_m", "q3_k_l",
        "q4_k_l", "q5_k_l", "q6_k_l", "q8_k", "q8_k_m",
    ];

    for q in &quantizations {
        if lower.contains(q) {
            return q.to_string();
        }
    }

    // Try to extract any Q/X pattern like Q4, Q5, etc.
    if let Some(pos) = lower.find('q') {
        let after = &lower[pos..];
        if after.len() >= 2 && after.chars().nth(1).unwrap().is_ascii_digit() {
            let end = after
                .char_indices()
                .find(|(_, c)| !c.is_alphanumeric() && *c != '_')
                .map(|(i, _)| i)
                .unwrap_or(after.len());
            return after[..end].to_string();
        }
    }

    "unknown".to_string()
}

/// Read n_layer from GGUF file header (uses cached metadata)
pub fn read_gguf_n_layer(path: &str) -> Option<u32> {
    if let Some(meta) = crate::services::get_model_metadata(path) {
        return meta.n_layer;
    }
    None
}

/// Extract HuggingFace repo ID from local cache path
/// Returns Some("user/repo") if path is in HF cache format
pub fn extract_hf_repo_id_from_path(path: &str) -> Option<String> {
    let path_obj = std::path::Path::new(path);
    let mut current = path_obj;

    // Walk up to 5 levels up
    for _ in 0..5 {
        if let Some(parent) = current.parent() {
            if let Some(dir_name) = parent.file_name() {
                let dir_name = dir_name.to_string_lossy();
                if dir_name.starts_with("models--") {
                    let repo_id = dir_name.strip_prefix("models--")?;
                    return Some(repo_id.replace("--", "/").to_string());
                }
            }
            current = parent;
        } else {
            break;
        }
    }
    None
}
