use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::core::{
    DetectedServer, LlmProvider, ModelInfo, ProviderConfig, ProviderError, ProviderSettings, Result,
};
use crate::models::ModelType;

pub struct VllmProvider {
    id: &'static str,
    name: &'static str,
}

impl VllmProvider {
    pub fn new() -> Self {
        Self {
            id: "vllm",
            name: "vLLM",
        }
    }
}

impl Default for VllmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for VllmProvider {
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
                    "Model path must be a directory for vLLM".into(),
                ));
            }
        }

        Ok(())
    }

    fn default_settings(&self) -> ProviderSettings {
        ProviderSettings {
            binary_path: "vllm".to_string(),
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
            "fp8",
            "fp8_e4m3",
            "fp8_e5m2",
            "int8",
            "int4",
            "int4_awq",
            "int4_gptq",
            "gptq",
            "awq",
            "squeezellm",
        ]
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
            "vllm"
        } else {
            &settings.binary_path
        };

        let parts: Vec<&str> = binary.split_whitespace().collect();
        if parts.len() > 1 {
            cmd.push_str(&format!("{} ", parts[0]));
            for part in &parts[1..] {
                cmd.push_str(&format!("{} ", part));
            }
        } else {
            cmd.push_str(&format!("\"{}\" ", binary));
        }

        cmd.push_str("serve ");

        if !config.huggingface_id.is_empty() {
            cmd.push_str(&format!("\"{}\" ", config.huggingface_id));
        } else if config.model_path.contains('/') && Path::new(&config.model_path).exists() {
            cmd.push_str(&format!("\"{}\" ", config.model_path));
        } else {
            cmd.push_str(&format!("{} ", config.model_path));
        }

        if config.context_size > 0 {
            cmd.push_str(&format!("--max-model-len {} ", config.context_size));
        }
        cmd.push_str(&format!("--port {} ", config.port));
        cmd.push_str(&format!("--host {} ", config.host));

        if config.gpu_layers > 0 {
            cmd.push_str(&format!(
                "--gpu-memory-utilization {:.2} ",
                config.gpu_layers as f32 / 100.0
            ));
        }

        if config.threads > 0 {
            cmd.push_str(&format!("--max-num-seqs {} ", config.threads));
        }

        if config.batch_size > 0 {
            cmd.push_str(&format!("--max-num-batched-tokens {} ", config.batch_size));
        }

        if !config.cache_type_k.is_empty() {
            cmd.push_str(&format!("--kv-cache-dtype {} ", config.cache_type_k));
        }

        if !config.additional_args.is_empty() {
            cmd.push_str(&config.additional_args);
            cmd.push(' ');
        }

        if !settings.additional_args.is_empty() {
            cmd.push_str(&settings.additional_args);
            cmd.push(' ');
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

        if let Ok(output) = Command::new("pgrep").args(["-a", "vllm"]).output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let lower = line.to_lowercase();
                if !lower.contains("serve") {
                    continue;
                }
                if let Some(space_pos) = line.find(' ') {
                    let pid_str = &line[..space_pos];
                    let cmdline = &line[space_pos + 1..];
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        servers.push(DetectedServer {
                            pid,
                            binary: "vllm".to_string(),
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
                "serve" => {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        config.model_path = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "--model" => {
                    if i + 1 < args.len() {
                        config.model_path = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "--max-model-len" | "--context-len" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.context_size = val;
                        }
                        i += 1;
                    }
                }
                "--max-num-batched-tokens" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.batch_size = val;
                        }
                        i += 1;
                    }
                }
                "--gpu-memory-utilization" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            config.gpu_layers = (val * 100.0) as i32;
                        }
                        i += 1;
                    }
                }
                "--max-num-seqs" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse() {
                            config.threads = val;
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
                        if let Ok(val) = args[i + 1].parse::<u32>() {
                            config.port = val as u16;
                        }
                        i += 1;
                    }
                }
                "--kv-cache-dtype" => {
                    if i + 1 < args.len() {
                        config.cache_type_k = args[i + 1].to_string();
                        config.cache_type_v = args[i + 1].to_string();
                        i += 1;
                    }
                }
                _ => {
                    // Collect unknown arguments (skip 'serve' command)
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

    let (quantization, is_moe) = detect_quantization_and_moe(path);

    Some(ModelInfo {
        path: name.clone(),
        name,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
        is_moe,
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

fn detect_quantization_and_moe(model_dir: &Path) -> (String, bool) {
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
                                    let quantization = if let Some(quant) =
                                        json.get("quantization_config")
                                    {
                                        if let Some(method) = quant.get("quant_method") {
                                            if let Some(method_str) = method.as_str() {
                                                method_str.to_lowercase()
                                            } else {
                                                "unknown".to_string()
                                            }
                                        } else {
                                            "unknown".to_string()
                                        }
                                    } else if let Some(torch_dtype) = json.get("torch_dtype") {
                                        if let Some(dtype_str) = torch_dtype.as_str() {
                                            match dtype_str {
                                                "float16" | "torch.float16" => "fp16".to_string(),
                                                "float32" | "torch.float32" => "fp32".to_string(),
                                                "bfloat16" | "torch.bfloat16" => "bf16".to_string(),
                                                _ => dtype_str.to_lowercase(),
                                            }
                                        } else {
                                            "unknown".to_string()
                                        }
                                    } else {
                                        "unknown".to_string()
                                    };

                                    let is_moe = json.get("num_local_experts").is_some()
                                        || json.get("num_experts").is_some()
                                        || json
                                            .get("architectures")
                                            .and_then(|a| a.as_array())
                                            .map(|archs| {
                                                archs.iter().any(|a| {
                                                    a.as_str()
                                                        .map(|s| s.contains("MoE"))
                                                        .unwrap_or(false)
                                                })
                                            })
                                            .unwrap_or(false);

                                    return (quantization, is_moe);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ("unknown".to_string(), false)
}

#[allow(dead_code)]
fn detect_quantization(model_dir: &Path) -> String {
    detect_quantization_and_moe(model_dir).0
}
