use std::io::Read;
use std::process::Command;

use crate::core::{LlmProvider, ModelInfo, ProviderConfig, ProviderError, Result};
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
        if config.model_path.is_empty() {
            return Err(ProviderError::InvalidConfig(
                "Model path is required".into(),
            ));
        }

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

        Ok(())
    }

    fn build_start_command(&self, config: &ProviderConfig) -> Command {
        let mut cmd = Command::new("llama-server");

        cmd.arg("-m").arg(&config.model_path);
        cmd.arg("-c").arg(config.context_size.to_string());
        cmd.arg("-b").arg(config.batch_size.to_string());
        cmd.arg("-ngl").arg(config.gpu_layers.to_string());
        cmd.arg("-t").arg(config.threads.to_string());
        cmd.arg("--port").arg(config.port.to_string());
        cmd.arg("--host").arg(&config.host);
        cmd.arg("-np").arg(config.num_prompt_tracking.to_string());

        if !config.cache_type_k.is_empty() {
            cmd.arg("--cache-type-k").arg(&config.cache_type_k);
        }
        if !config.cache_type_v.is_empty() {
            cmd.arg("--cache-type-v").arg(&config.cache_type_v);
        }

        for arg in config.additional_args.split_whitespace() {
            if !arg.is_empty() {
                cmd.arg(arg);
            }
        }

        cmd
    }

    fn supported_quantizations(&self) -> Vec<&'static str> {
        vec![
            "f16", "q8_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "q3_1", "q3_0", "q2_1", "q2_0",
        ]
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

        // ~/.cache/llama.cpp/ - common location for llama.cpp models
        if let Some(home) = dirs::cache_dir() {
            let llama_cpp_dir = home.join("llama.cpp");
            if llama_cpp_dir.exists() {
                dirs.push(llama_cpp_dir.to_string_lossy().to_string());
            }
        }

        // ~/.cache/huggingface/hub/ - HuggingFace cache (where GGUF models are often downloaded)
        if let Some(home) = dirs::cache_dir() {
            let hf_cache = home.join("huggingface").join("hub");
            if hf_cache.exists() {
                dirs.push(hf_cache.to_string_lossy().to_string());
            }
        }

        // ~/models/ - generic models directory
        if let Some(home) = dirs::home_dir() {
            let models_dir = home.join("models");
            if models_dir.exists() {
                dirs.push(models_dir.to_string_lossy().to_string());
            }
        }

        dirs
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

        // Check for block_count
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

        // Skip value
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
