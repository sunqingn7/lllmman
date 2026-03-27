use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GpuAllocation {
    Single(u32),
    Multi(Vec<u32>),
    All,
    VramLimit {
        gpu: u32,
        max_vram_mb: u32,
        layers: u32,
    },
}

impl Default for GpuAllocation {
    fn default() -> Self {
        GpuAllocation::All
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MonitorStats {
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub ram_used_mb: u32,
    pub ram_total_mb: u32,
    pub cpu_percent: f32,
    pub tokens_per_second: f32,
    pub active_connections: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Error(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppSettings {
    pub scan_directories: Vec<String>,
    pub download_directory: String,
    pub default_port: u16,
    pub default_context_size: u32,
    pub default_batch_size: u32,
    pub default_gpu_layers: u32,
    pub default_threads: u32,
    pub default_cache_type_k: String,
    pub default_cache_type_v: String,
    pub gpu_strategy: GpuAllocation,
    pub selected_provider: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_default();
        Self {
            scan_directories: vec![],
            download_directory: home.join(".cache/lllmman/models").to_string_lossy().to_string(),
            default_port: 8080,
            default_context_size: 4096,
            default_batch_size: 512,
            default_gpu_layers: 35,
            default_threads: 8,
            default_cache_type_k: "q4_0".to_string(),
            default_cache_type_v: "q4_0".to_string(),
            gpu_strategy: GpuAllocation::All,
            selected_provider: "llama.cpp".to_string(),
        }
    }
}
