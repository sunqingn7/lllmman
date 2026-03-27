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
