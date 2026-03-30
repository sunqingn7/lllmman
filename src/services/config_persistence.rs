use crate::core::ProviderConfig;
use crate::models::{AppSettings, ModelConfigEntry, ModelConfigs};
use std::fs;
use std::path::PathBuf;

fn get_config_path() -> PathBuf {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lllmman");

    fs::create_dir_all(&config_dir).ok();
    config_dir.join("config.json")
}

fn get_model_configs_path() -> PathBuf {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lllmman");

    fs::create_dir_all(&config_dir).ok();
    config_dir.join("model_configs.json")
}

pub fn load_settings() -> AppSettings {
    let path = get_config_path();

    if path.exists() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(settings) = serde_json::from_str(&content) {
                return settings;
            }
        }
    }

    AppSettings::default()
}

pub fn save_settings(settings: &AppSettings) -> Result<(), String> {
    let path = get_config_path();
    let content = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(path, content).map_err(|e| e.to_string())
}

pub fn load_model_configs() -> ModelConfigs {
    let path = get_model_configs_path();

    if path.exists() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(configs) = serde_json::from_str(&content) {
                return configs;
            }
        }
    }

    ModelConfigs::default()
}

pub fn save_model_configs(configs: &ModelConfigs) -> Result<(), String> {
    let path = get_model_configs_path();
    let content = serde_json::to_string_pretty(configs).map_err(|e| e.to_string())?;
    fs::write(path, content).map_err(|e| e.to_string())
}

pub fn save_model_config(model_path: &str, config: &ProviderConfig) -> Result<(), String> {
    let mut configs = load_model_configs();

    configs.configs.insert(
        model_path.to_string(),
        ModelConfigEntry {
            context_size: config.context_size,
            batch_size: config.batch_size,
            gpu_layers: config.gpu_layers,
            threads: config.threads,
            port: config.port,
            host: config.host.clone(),
            cache_type_k: config.cache_type_k.clone(),
            cache_type_v: config.cache_type_v.clone(),
            num_prompt_tracking: config.num_prompt_tracking,
        },
    );
    configs.last_model_path = Some(model_path.to_string());

    save_model_configs(&configs)
}

pub fn load_model_config(model_path: &str) -> Option<ProviderConfig> {
    let configs = load_model_configs();

    if let Some(entry) = configs.configs.get(model_path) {
        return Some(ProviderConfig {
            model_path: model_path.to_string(),
            context_size: entry.context_size,
            batch_size: entry.batch_size,
            gpu_layers: entry.gpu_layers,
            threads: entry.threads,
            port: entry.port,
            host: entry.host.clone(),
            cache_type_k: entry.cache_type_k.clone(),
            cache_type_v: entry.cache_type_v.clone(),
            num_prompt_tracking: entry.num_prompt_tracking,
            ..Default::default()
        });
    }

    None
}

pub fn get_fallback_config() -> Option<ProviderConfig> {
    let configs = load_model_configs();

    if let Some(last_path) = configs.last_model_path {
        if let Some(entry) = configs.configs.get(&last_path) {
            return Some(ProviderConfig {
                model_path: String::new(),
                context_size: entry.context_size,
                batch_size: entry.batch_size,
                gpu_layers: entry.gpu_layers,
                threads: entry.threads,
                port: entry.port,
                host: entry.host.clone(),
                cache_type_k: entry.cache_type_k.clone(),
                cache_type_v: entry.cache_type_v.clone(),
                num_prompt_tracking: entry.num_prompt_tracking,
                ..Default::default()
            });
        }
    }

    None
}
