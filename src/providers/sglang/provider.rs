use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::core::{
    DetectedServer, LlmProvider, ModelInfo, ProviderConfig, ProviderError, ProviderSettings, Result,
};
use crate::models::ModelType;

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
        ProviderSettings {
            binary_path: "python3 -m sglang.launch_server".to_string(),
            env_script: String::new(),
            additional_args: String::new(),
            health_endpoint: "/health".to_string(),
            heartbeat_interval_secs: 6,
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

    fn build_command_line(&self, config: &ProviderConfig, settings: &ProviderSettings) -> String {
        let mut cmd = String::new();

        if let Some(gpu) = config.selected_gpu {
            cmd.push_str(&format!("CUDA_VISIBLE_DEVICES={} ", gpu));
        }

        let binary = if settings.binary_path.is_empty() {
            "python3 -m sglang.launch_server"
        } else {
            &settings.binary_path
        };

        if !settings.env_script.is_empty() {
            cmd.push_str("bash -c ");
            cmd.push_str(&format!("source \"{}\" exec ", settings.env_script));
        }

        let parts: Vec<&str> = binary.split_whitespace().collect();
        if parts.len() > 1 {
            cmd.push_str(&format!("{} ", parts[0]));
            for part in &parts[1..] {
                cmd.push_str(&format!("{} ", part));
            }
        } else {
            cmd.push_str(&format!("\"{}\" ", binary));
        }

        if !config.huggingface_id.is_empty() {
            cmd.push_str(&format!("--model-path \"{}\" ", config.huggingface_id));
        } else if config.model_path.contains('/') && Path::new(&config.model_path).exists() {
            cmd.push_str(&format!("--model-path \"{}\" ", config.model_path));
        } else {
            cmd.push_str(&format!("--model-path {} ", config.model_path));
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

        if config.threads > 0 {
            cmd.push_str(&format!("--tp-size {} ", config.threads));
        }

        if config.batch_size > 0 {
            cmd.push_str(&format!(
                "--schedule-conservativeness {} ",
                config.batch_size
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

        scan_gguf_files(path_obj, &mut models);

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
                if !lower.contains("launch_server") {
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

fn scan_gguf_files(dir: &Path, models: &mut Vec<ModelInfo>) {
    fn scan_recursive(dir: &Path, models: &mut Vec<ModelInfo>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    scan_recursive(&path, models);
                } else if let Some(ext) = path.extension() {
                    if ext.to_string_lossy().to_lowercase() == "gguf" {
                        if let Some(model) = parse_gguf_file(&path) {
                            models.push(model);
                        }
                    }
                }
            }
        }
    }

    scan_recursive(dir, models);
}

fn parse_gguf_file(path: &Path) -> Option<ModelInfo> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let size_gb = metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0);

    let quantization = extract_quantization(&filename);

    Some(ModelInfo {
        path: path.to_string_lossy().to_string(),
        name: filename.clone(),
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
        is_moe: filename.to_lowercase().contains("moe"),
    })
}

fn extract_quantization(filename: &str) -> String {
    let lower = filename.to_lowercase();
    let quantizations = [
        "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0", "f16", "q2_k", "q3_k", "q4_k", "q5_k",
        "q6_k",
    ];

    for q in &quantizations {
        if lower.contains(*q) {
            return q.to_string();
        }
    }
    "unknown".to_string()
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
