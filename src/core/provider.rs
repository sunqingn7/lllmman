use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::models::ModelType;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum ProviderError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Server error: {0}")]
    ServerError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Download error: {0}")]
    DownloadError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
}

pub type Result<T> = std::result::Result<T, ProviderError>;

#[derive(Clone, Debug)]
pub struct DetectedServer {
    pub pid: u32,
    pub binary: String,
    pub command_line: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub size_gb: f32,
    pub quantization: String,
    pub model_type: ModelType,
    pub is_moe: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadableModel {
    pub id: String,
    pub name: String,
    pub size_gb: f32,
    pub downloads: u32,
    pub source: ModelSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFace {
        repo_id: String,
    },
    DirectUrl {
        url: String,
    },
    GitHubRelease {
        owner: String,
        repo: String,
        tag: String,
        asset_name: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadTask {
    pub id: String,
    pub source: ModelSource,
    pub file_name: String,
    pub dest_path: String,
    pub status: DownloadStatus,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderSettings {
    pub binary_path: String,
    pub env_script: String,
    pub additional_args: String,
    #[serde(default = "default_health_endpoint")]
    pub health_endpoint: String,
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,
}

fn default_health_endpoint() -> String {
    "/health".to_string()
}

fn default_heartbeat_interval() -> u64 {
    6
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            binary_path: String::new(),
            env_script: String::new(),
            additional_args: String::new(),
            health_endpoint: default_health_endpoint(),
            heartbeat_interval_secs: default_heartbeat_interval(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CpuOffloadMode {
    Auto,
    Offload,
    FullOffload,
    Disabled,
}

impl Default for CpuOffloadMode {
    fn default() -> Self {
        CpuOffloadMode::Auto
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub model_path: String,
    pub huggingface_id: String,
    pub context_size: u32,
    pub batch_size: u32,
    pub gpu_layers: i32,
    pub threads: u32,
    pub host: String,
    pub port: u16,
    pub additional_args: String,
    pub parsed_options: HashMap<String, String>,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub num_prompt_tracking: u32,
    pub gpu_allocation: crate::models::GpuAllocation,
    #[serde(default)]
    pub selected_gpu: Option<u32>,
    #[serde(default)]
    pub cpu_offload: CpuOffloadMode,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            huggingface_id: String::new(),
            context_size: 4096,
            batch_size: 512,
            gpu_layers: -1,
            threads: 8,
            host: "0.0.0.0".to_string(),
            port: 8080,
            additional_args: String::new(),
            parsed_options: HashMap::new(),
            cache_type_k: "q4_0".to_string(),
            cache_type_v: "q4_0".to_string(),
            num_prompt_tracking: 1,
            gpu_allocation: crate::models::GpuAllocation::All,
            selected_gpu: None,
            cpu_offload: CpuOffloadMode::Auto,
        }
    }
}

impl ProviderConfig {
    pub fn parse_additional_args(&mut self) {
        if self.additional_args.is_empty() {
            return;
        }

        let mut args = self.additional_args.split_whitespace();
        while let Some(key) = args.next() {
            let key = key.trim_start_matches('-');
            if let Some(value) = args.next() {
                if !value.starts_with('-') {
                    self.parsed_options
                        .insert(key.to_string(), value.to_string());
                }
            }
        }
    }

    pub fn serialize_options_to_args(&mut self) {
        if self.parsed_options.is_empty() {
            self.additional_args.clear();
            return;
        }

        let mut args = Vec::new();
        for (key, value) in &self.parsed_options {
            args.push(format!("--{} {}", key, value));
        }
        self.additional_args = args.join(" ");
    }

    pub fn get_option(&self, key: &str) -> Option<String> {
        self.parsed_options.get(key).cloned()
    }

    pub fn set_option(&mut self, key: &str, value: String) {
        if value.is_empty() {
            self.parsed_options.remove(key);
        } else {
            self.parsed_options.insert(key.to_string(), value);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderOption {
    pub id: String,
    pub name: String,
    pub value_type: OptionValueType,
    pub default_value: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptionValueType {
    String,
    Number,
    Bool,
    Select(Vec<String>),
}

pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn id(&self) -> &'static str;

    fn get_config_template(&self) -> ProviderConfig;
    fn validate_config(&self, config: &ProviderConfig) -> Result<()>;

    fn default_settings(&self) -> ProviderSettings;
    fn start_server(
        &self,
        config: &ProviderConfig,
        settings: &ProviderSettings,
    ) -> Result<std::process::Child>;

    fn build_command_line(&self, config: &ProviderConfig, settings: &ProviderSettings) -> String;

    fn get_metrics_endpoint(&self, config: &ProviderConfig) -> Option<String> {
        Some(format!("http://{}:{}/stats", config.host, config.port))
    }

    fn supported_quantizations(&self) -> Vec<&'static str>;

    fn scan_models(&self, path: &str) -> Vec<ModelInfo>;
    fn add_model(&self, path: &str) -> Result<ModelInfo>;

    fn get_options(&self) -> Vec<ProviderOption> {
        vec![ProviderOption {
            id: "additional_args".to_string(),
            name: "Additional CLI Args".to_string(),
            value_type: OptionValueType::String,
            default_value: String::new(),
            description: "Additional command-line arguments (space-separated)".to_string(),
        }]
    }

    fn default_model_directories(&self) -> Vec<String> {
        vec![]
    }

    fn detect_running_servers(&self) -> Vec<DetectedServer>;
    fn parse_server_config(&self, cmd_line: &str) -> ProviderConfig;
}

pub trait ModelDownloader: Send + Sync {
    fn search(
        &self,
        query: &str,
    ) -> impl std::future::Future<Output = Result<Vec<DownloadableModel>>> + Send;
    fn download(&self, model_id: &str, dest_dir: &str) -> Result<ModelInfo>;
}
