use std::process::{Command, Stdio};

use crate::core::{
    DetectedServer, LlmProvider, ModelInfo, ProviderConfig, ProviderError, ProviderSettings, Result,
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

impl Default for LlamaCppProvider {
    fn default() -> Self {
        Self::new()
    }
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
            binary_path: "llama-server".to_string(),
            env_script: String::new(),
            additional_args: String::new(),
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
            "f16", "q8_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "q3_1", "q3_0", "q2_1", "q2_0",
        ]
    }

    fn build_command_line(&self, config: &ProviderConfig, settings: &ProviderSettings) -> String {
        let mut cmd = String::new();

        let binary = if settings.binary_path.is_empty() {
            "llama-server"
        } else {
            &settings.binary_path
        };

        if settings.env_script.is_empty() {
            cmd.push_str(binary);
            if !config.huggingface_id.is_empty() {
                cmd.push_str(&format!(" -hf \"{}\"", config.huggingface_id));
            } else {
                cmd.push_str(&format!(" -m \"{}\"", config.model_path));
            }
            if config.context_size > 0 {
                cmd.push_str(&format!(" -c {}", config.context_size));
            }
            if config.batch_size > 0 {
                cmd.push_str(&format!(" -b {}", config.batch_size));
            }
            cmd.push_str(&format!(" -ngl {}", config.gpu_layers));
            if config.threads > 0 {
                cmd.push_str(&format!(" -t {}", config.threads));
            }
            cmd.push_str(&format!(" --port {}", config.port));
            cmd.push_str(&format!(" --host {}", config.host));
            if config.num_prompt_tracking > 0 {
                cmd.push_str(&format!(" -np {}", config.num_prompt_tracking));
            }

            if !config.cache_type_k.is_empty() {
                cmd.push_str(&format!(" --cache-type-k \"{}\"", config.cache_type_k));
            }
            if !config.cache_type_v.is_empty() {
                cmd.push_str(&format!(" --cache-type-v \"{}\"", config.cache_type_v));
            }

            for arg in config.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    cmd.push_str(&format!(" {}", arg));
                }
            }

            for arg in settings.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    cmd.push_str(&format!(" {}", arg));
                }
            }
        } else {
            cmd.push_str("bash -c ");
            cmd.push_str(&format!("source \"{}\" exec ", settings.env_script));
            cmd.push_str(&format!("\"{}\" ", binary));
            if !config.huggingface_id.is_empty() {
                cmd.push_str(&format!("-hf \"{}\" ", config.huggingface_id));
            } else {
                cmd.push_str(&format!("-m \"{}\" ", config.model_path));
            }
            if config.context_size > 0 {
                cmd.push_str(&format!("-c {} ", config.context_size));
            }
            if config.batch_size > 0 {
                cmd.push_str(&format!("-b {} ", config.batch_size));
            }
            cmd.push_str(&format!("-ngl {} ", config.gpu_layers));
            if config.threads > 0 {
                cmd.push_str(&format!("-t {} ", config.threads));
            }
            cmd.push_str(&format!("--port {} ", config.port));
            cmd.push_str(&format!("--host {} ", config.host));
            if config.num_prompt_tracking > 0 {
                cmd.push_str(&format!("-np {} ", config.num_prompt_tracking));
            }

            if !config.cache_type_k.is_empty() {
                cmd.push_str(&format!("--cache-type-k \"{}\" ", config.cache_type_k));
            }
            if !config.cache_type_v.is_empty() {
                cmd.push_str(&format!("--cache-type-v \"{}\" ", config.cache_type_v));
            }

            for arg in config.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    cmd.push_str(&format!("{} ", arg));
                }
            }

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
        let path_obj = std::path::Path::new(path);

        if !path_obj.exists() {
            return models;
        }

        fn scan_recursive(dir: &std::path::Path, models: &mut Vec<ModelInfo>) {
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

        scan_recursive(path_obj, &mut models);
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
                        config.model_path = args[i + 1].to_string();
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
                _ => {}
            }
            i += 1;
        }

        config
    }
}

fn parse_gguf_file(path: &std::path::Path) -> Option<ModelInfo> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let size_gb = metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0);

    let quantization = extract_quantization(&filename);

    Some(ModelInfo {
        path: path.to_string_lossy().to_string(),
        name: filename,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
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

/// Read n_layer from GGUF file header
pub fn read_gguf_n_layer(path: &str) -> Option<u32> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return None,
    };

    if data.len() < 24 {
        return None;
    }

    if &data[0..4] != b"GGUF" {
        return None;
    }

    let metadata_count = u64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]) as usize;

    if metadata_count > 10000 {
        return None;
    }

    let mut offset = 24usize;

    for _ in 0..metadata_count {
        if offset + 12 > data.len() {
            break;
        }

        let key_len = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        if key_len > 200 || offset + 12 + key_len > data.len() {
            break;
        }

        let key = match String::from_utf8(data[offset + 8..offset + 8 + key_len].to_vec()) {
            Ok(k) => k,
            Err(_) => {
                offset += 8 + key_len + 4;
                continue;
            }
        };
        offset += 8 + key_len;

        if offset + 4 > data.len() {
            break;
        }
        let val_type = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        if key.ends_with(".block_count") {
            if val_type == 4 {
                if offset + 4 <= data.len() {
                    let block_count = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    return Some(block_count);
                }
            }
        }

        match val_type {
            0 | 1 => {
                offset += 1;
            }
            2 | 3 => {
                offset += 2;
            }
            4 | 5 | 6 => {
                offset += 4;
            }
            7 => {
                offset += 8;
            }
            8 => {
                if offset + 8 <= data.len() {
                    let str_len = u64::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]) as usize;
                    offset += 8 + str_len.min(10000);
                }
            }
            9 => {
                if offset + 12 <= data.len() {
                    let arr_type = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    offset += 4;
                    let arr_count = u64::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]) as usize;
                    offset += 8;

                    match arr_type {
                        0 | 1 => {
                            offset += arr_count;
                        }
                        2 | 3 => {
                            offset += arr_count * 2;
                        }
                        4 | 5 | 6 => {
                            offset += arr_count * 4;
                        }
                        7 => {
                            offset += arr_count * 8;
                        }
                        8 => {
                            for _ in 0..arr_count.min(1000) {
                                if offset + 8 > data.len() {
                                    break;
                                }
                                let el_len = u64::from_le_bytes([
                                    data[offset],
                                    data[offset + 1],
                                    data[offset + 2],
                                    data[offset + 3],
                                    data[offset + 4],
                                    data[offset + 5],
                                    data[offset + 6],
                                    data[offset + 7],
                                ]) as usize;
                                offset += 8 + el_len.min(10000);
                            }
                        }
                        _ => {}
                    }
                }
            }
            10 | 11 | 12 => {
                offset += 8;
            }
            _ => {}
        }
    }

    None
}
