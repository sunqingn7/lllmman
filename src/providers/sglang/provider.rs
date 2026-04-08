use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::core::{
    DetectedServer, LlmProvider, ModelInfo, OptionValueType, ProviderConfig, ProviderError,
    ProviderOption, ProviderSettings, Result,
};
use crate::models::ModelType;

use dirs;

pub struct SglangProvider {
    id: &'static str,
    name: &'static str,
}

impl SglangProvider {
    pub fn new() -> Self {
        Self {
            id: "sglang",
            name: "SGLang",
        }
    }
}

impl Default for SglangProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for SglangProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn get_config_template(&self) -> ProviderConfig {
        ProviderConfig {
            gpu_layers: 90,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            num_prompt_tracking: 0,
            ..ProviderConfig::default()
        }
    }

    fn validate_config(&self, config: &ProviderConfig) -> Result<()> {
        let hf_id_trimmed = config.huggingface_id.trim();
        if config.model_path.is_empty() && hf_id_trimmed.is_empty() {
            return Err(ProviderError::InvalidConfig(
                "Model path or HuggingFace model ID is required".into(),
            ));
        }
        if !hf_id_trimmed.is_empty() && !hf_id_trimmed.contains('/') {
            return Err(ProviderError::InvalidConfig(format!(
                "Invalid HuggingFace ID format (expected 'user/repo'): {}",
                config.huggingface_id
            )));
        }

        if !config.model_path.is_empty() && config.model_path.contains('/') {
            let path = Path::new(&config.model_path);
            if path.exists() && !path.is_dir() {
                return Err(ProviderError::InvalidConfig(
                    "Model path must be a directory for SGLang".into(),
                ));
            }
        }

        Ok(())
    }

    fn default_settings(&self) -> ProviderSettings {
        let binary_path = std::process::Command::new("which")
            .arg("sglang")
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "sglang".to_string());

        ProviderSettings {
            binary_path,
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

        cmd.spawn().map_err(ProviderError::from)
    }

    fn supported_quantizations(&self) -> Vec<&'static str> {
        vec![
            "fp16",
            "fp8",
            "int8",
            "int4",
            "awq",
            "gptq",
            "marlin",
            "squeezellm",
        ]
    }

    fn supports_gguf(&self) -> bool {
        false
    }

    fn build_command_line(&self, config: &ProviderConfig, settings: &ProviderSettings) -> String {
        let mut cmd = String::new();

        if !settings.venv_path.is_empty() {
            cmd.push_str(&format!(
                "source \"{}/bin/activate\" && ",
                settings.venv_path
            ));
        }

        if let Some(gpu) = config.selected_gpu {
            cmd.push_str(&format!("CUDA_VISIBLE_DEVICES={} ", gpu));
        }

        let binary = if settings.binary_path.is_empty() {
            "sglang"
        } else {
            &settings.binary_path
        };

        cmd.push_str(&format!("{} serve ", binary));

        // For SGLang, try to find the model in HF cache first
        let model_arg = if !config.model_path.is_empty() {
            if Path::new(&config.model_path).exists() {
                config.model_path.clone()
            } else {
                if let Some(hf_path) = find_huggingface_model_path(&config.model_path) {
                    hf_path
                } else if !config.huggingface_id.is_empty() {
                    if let Some(hf_path) = find_huggingface_model_path(&config.huggingface_id) {
                        hf_path
                    } else {
                        config.huggingface_id.clone()
                    }
                } else {
                    config.model_path.clone()
                }
            }
        } else if !config.huggingface_id.is_empty() {
            if let Some(hf_path) = find_huggingface_model_path(&config.huggingface_id) {
                hf_path
            } else {
                config.huggingface_id.clone()
            }
        } else {
            String::new()
        };

        if !model_arg.is_empty() {
            if model_arg.starts_with('/') || model_arg.contains('/') {
                cmd.push_str(&format!("--model-path \"{}\" ", model_arg));
            } else if model_arg.contains('/')
                && !model_arg.starts_with('/')
                && !model_arg.starts_with('.')
            {
                cmd.push_str(&format!("--model \"{}\" ", model_arg));
            } else {
                cmd.push_str(&format!("--model-path {} ", model_arg));
            }
        }

        if config.context_size > 0 {
            cmd.push_str(&format!("--context-length {} ", config.context_size));
        }
        cmd.push_str(&format!("--port {} ", config.port));
        cmd.push_str(&format!("--host {} ", config.host));

        if config.gpu_layers > 0 {
            cmd.push_str(&format!(
                "--mem-fraction-static {:.2} ",
                config.gpu_layers as f32 / 100.0
            ));
        }

        if !config.additional_args.is_empty() {
            for arg in config.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    cmd.push_str(&format!("{} ", arg));
                }
            }
        }

        if !settings.additional_args.is_empty() {
            for arg in settings.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    cmd.push_str(&format!("{} ", arg));
                }
            }
        }

        cmd
    }

    fn scan_models(&self, path: &str) -> Vec<ModelInfo> {
        let mut models = Vec::new();

        let path_obj = Path::new(path);
        if !path_obj.exists() || !path_obj.is_dir() {
            return models;
        }

        if let Ok(entries) = std::fs::read_dir(path_obj) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    if let Some(model) = parse_hf_model_dir(&entry_path) {
                        models.push(model);
                    }
                }
            }
        }

        models
    }

    fn add_model(&self, path: &str) -> Result<ModelInfo> {
        if path.is_empty() {
            return Err(ProviderError::InvalidConfig("Model ID is required".into()));
        }

        let name = path.split('/').last().unwrap_or(path).to_string();
        let size_gb = if Path::new(path).exists() {
            calculate_dir_size(path)
        } else {
            0.0
        };

        Ok(ModelInfo {
            path: path.to_string(),
            name,
            size_gb,
            quantization: "unknown".to_string(),
            model_type: ModelType::TextOnly,
            is_moe: false,
        })
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
                id: "additional_args".to_string(),
                name: "Additional CLI Args".to_string(),
                value_type: OptionValueType::String,
                default_value: String::new(),
                description: "Additional command-line arguments (space-separated)".to_string(),
            },
        ]
    }

    fn default_model_directories(&self) -> Vec<String> {
        let mut dirs = Vec::new();

        if let Some(cache) = dirs::cache_dir() {
            let hf_hub = cache.join("huggingface").join("hub");
            if hf_hub.exists() {
                dirs.push(hf_hub.to_string_lossy().to_string());
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

    fn detect_running_servers(&self) -> Vec<DetectedServer> {
        let mut servers = Vec::new();

        if let Ok(output) = Command::new("pgrep").args(["-a", "sglang"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let lower = line.to_lowercase();
                if !lower.contains("serve") && !lower.contains("launch_server") {
                    continue;
                }
                if let Some(space_pos) = line.find(' ') {
                    let pid_str = &line[..space_pos];
                    let cmdline = &line[space_pos + 1..];
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        servers.push(DetectedServer {
                            pid,
                            binary: "sglang".to_string(),
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
                "--model-path" | "--model" => {
                    if i + 1 < args.len() {
                        config.model_path = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "--context-length" | "--max-model-len" | "--context-len" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.context_size = val;
                        }
                        i += 1;
                    }
                }
                "--mem-fraction-static" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.gpu_layers = (val * 100.0) as i32;
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
                "--port" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<u16>() {
                            config.port = val;
                        }
                        i += 1;
                    }
                }
                "--tp-size" | "--tensor-parallel-size" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.threads = val;
                        }
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
                "--repetition-penalty" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.repetition_penalty = Some(val);
                        }
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

fn parse_hf_model_dir(path: &Path) -> Option<ModelInfo> {
    let dir_name = path.file_name()?.to_string_lossy().to_string();

    if !dir_name.contains("models--") {
        return None;
    }

    let repo_id = dir_name.strip_prefix("models--")?;
    let name = repo_id.replace("--", "/");

    let snapshot_dir = path.join("snapshots");
    if !snapshot_dir.exists() {
        return None;
    }

    let mut has_config_json = false;
    if let Ok(snapshot_entries) = std::fs::read_dir(&snapshot_dir) {
        for snapshot in snapshot_entries.flatten() {
            if snapshot.path().is_dir() {
                if let Ok(sub_entries) = std::fs::read_dir(snapshot.path()) {
                    for sub in sub_entries.flatten() {
                        if let Some(name_str) = sub.file_name().to_str() {
                            if name_str == "config.json" {
                                has_config_json = true;
                            }
                        }
                    }
                }
            }
        }
    }

    if !has_config_json {
        return None;
    }

    let size_gb = calculate_dir_size(&path.to_string_lossy());

    let quantization = detect_quantization(path);

    Some(ModelInfo {
        path: name.clone(),
        name,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
        is_moe: false,
    })
}

fn calculate_dir_size(path: &str) -> f32 {
    let mut total_size = 0u64;

    fn walk_dir(path: &Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    walk_dir(&entry_path, total);
                } else if let Ok(meta) = entry.metadata() {
                    *total += meta.len();
                }
            }
        }
    }

    walk_dir(Path::new(path), &mut total_size);
    total_size as f32 / (1024.0 * 1024.0 * 1024.0)
}

fn detect_quantization(model_dir: &Path) -> String {
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                for sub_entry in std::fs::read_dir(&entry_path)
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    if let Some(name) = sub_entry.file_name().to_str() {
                        if name == "config.json" {
                            if let Ok(content) = fs::read_to_string(sub_entry.path()) {
                                if let Ok(json) =
                                    serde_json::from_str::<serde_json::Value>(&content)
                                {
                                    if let Some(quant) = json.get("quantization_config") {
                                        if let Some(method) = quant.get("quant_method") {
                                            if let Some(method_str) = method.as_str() {
                                                return method_str.to_lowercase();
                                            }
                                        }
                                    }

                                    if let Some(torch_dtype) = json.get("torch_dtype") {
                                        if let Some(dtype_str) = torch_dtype.as_str() {
                                            return match dtype_str {
                                                "float16" | "torch.float16" => "fp16".to_string(),
                                                "float32" | "torch.float32" => "fp32".to_string(),
                                                "bfloat16" | "torch.bfloat16" => "bf16".to_string(),
                                                _ => dtype_str.to_lowercase(),
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    "unknown".to_string()
}

fn find_huggingface_model_path(model_id: &str) -> Option<String> {
    let possible_paths = [
        dirs::cache_dir().map(|p| {
            p.join("huggingface")
                .join("hub")
                .join(format!("models--{}", model_id.replace('/', "--")))
        }),
        dirs::home_dir().map(|p| {
            p.join(".cache")
                .join("huggingface")
                .join("hub")
                .join(format!("models--{}", model_id.replace('/', "--")))
        }),
    ];

    for path_opt in possible_paths.iter().flatten() {
        let path = path_opt;
        if !path.exists() {
            continue;
        }

        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let snapshot_path = entry.path();
                if !snapshot_path.is_dir() {
                    continue;
                }

                if let Ok(sub_entries) = std::fs::read_dir(&snapshot_path) {
                    for sub in sub_entries.flatten() {
                        let sub_path = sub.path();
                        if sub_path.is_dir() {
                            if let Ok(gguf_entries) = std::fs::read_dir(&sub_path) {
                                for gguf in gguf_entries.flatten() {
                                    if gguf.path().to_string_lossy().ends_with(".gguf") {
                                        return Some(sub_path.to_string_lossy().to_string());
                                    }
                                }
                            }
                        } else if sub_path.to_string_lossy().ends_with(".gguf") {
                            return Some(snapshot_path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    None
}
