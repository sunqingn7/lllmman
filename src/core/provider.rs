use serde::{Deserialize, Serialize};
use std::process::Command;
use thiserror::Error;

use crate::models::ModelType;

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Server error: {0}")]
    ServerError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ProviderError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub size_gb: f32,
    pub quantization: String,
    pub model_type: ModelType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadableModel {
    pub id: String,
    pub name: String,
    pub size_gb: f32,
    pub downloads: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub model_path: String,
    pub context_size: u32,
    pub batch_size: u32,
    pub gpu_layers: u32,
    pub threads: u32,
    pub host: String,
    pub port: u16,
    pub additional_args: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub num_prompt_tracking: u32,
    pub gpu_allocation: crate::models::GpuAllocation,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_size: 4096,
            batch_size: 512,
            gpu_layers: 35,
            threads: 8,
            host: "0.0.0.0".to_string(),
            port: 8080,
            additional_args: String::new(),
            cache_type_k: "q4_0".to_string(),
            cache_type_v: "q4_0".to_string(),
            num_prompt_tracking: 1,
            gpu_allocation: crate::models::GpuAllocation::All,
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

    fn build_start_command(&self, config: &ProviderConfig) -> Command;

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
}

pub trait ModelDownloader: Send + Sync {
    fn search(
        &self,
        query: &str,
    ) -> impl std::future::Future<Output = Result<Vec<DownloadableModel>>> + Send;
    fn download(&self, model_id: &str, dest_dir: &str) -> Result<ModelInfo>;
}
