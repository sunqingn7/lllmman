use crate::models::{GpuTemperature, MonitorStats};
use crate::providers::llama_cpp::read_gguf_n_layer;
use crate::services::gpu_detector;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn get_sys() -> Arc<Mutex<sysinfo::System>> {
    static SYS: std::sync::OnceLock<Arc<Mutex<sysinfo::System>>> = std::sync::OnceLock::new();
    SYS.get_or_init(|| Arc::new(Mutex::new(sysinfo::System::new_all())))
        .clone()
}

pub fn get_system_stats() -> MonitorStats {
    let sys = get_sys();
    {
        let mut sys_guard = sys.lock().unwrap();
        sys_guard.refresh_all();
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    {
        let mut sys_guard = sys.lock().unwrap();
        sys_guard.refresh_all();
    }

    let sys_guard = sys.lock().unwrap();
    let cpu = sys_guard
        .cpus()
        .first()
        .map(|c| c.cpu_usage())
        .unwrap_or(0.0);
    let total_ram = sys_guard.total_memory() as u64;
    let used_ram = sys_guard.used_memory() as u64;
    drop(sys_guard);

    let gpus = gpu_detector::detect_gpus();
    let gpu_usage = gpu_detector::get_all_gpu_usage();

    // Build index-based lookup for usage data
    let usage_by_index: HashMap<u32, u32> = gpu_usage
        .iter()
        .map(|u| (u.index, u.used_vram_mb))
        .collect();

    let total_vram: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
    let used_vram: u32 = gpu_usage.iter().map(|u| u.used_vram_mb).sum();

    let gpu_vram_usage: Vec<(u32, u32, u32)> = gpus
        .iter()
        .map(|gpu| {
            let used = usage_by_index.get(&gpu.index).copied().unwrap_or(0);
            (gpu.index, used, gpu.total_vram_mb)
        })
        .collect();

    let gpu_temperatures: Vec<GpuTemperature> = gpus
        .iter()
        .map(|gpu| GpuTemperature {
            index: gpu.index,
            name: gpu.name.clone(),
            temperature_c: gpu_detector::get_gpu_temperature(gpu).map(|t| t as f32),
        })
        .collect();

    let cpu_temperature = gpu_detector::get_cpu_temperature();

    MonitorStats {
        vram_used_mb: used_vram,
        vram_total_mb: total_vram,
        ram_used_mb: (used_ram / (1024 * 1024)) as u32,
        ram_total_mb: (total_ram / (1024 * 1024)) as u32,
        cpu_percent: cpu,
        tokens_per_second: 0.0,
        active_connections: 0,
        gpu_temperatures,
        cpu_temperature,
        gpu_vram_usage,
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct ServerStats {
    pub queue_size: Option<u32>,
    pub queue_duration_ms: Option<u64>,
    pub requests_queue_size: Option<u32>,
    pub context_requests: Option<u32>,
    pub cache_misses: Option<u64>,
    pub cache_hits: Option<u64>,
    pub peak_kv_cache_block_usage: Option<f32>,
    pub generation_duration_ms: Option<u64>,
    pub prompt_processing_duration_ms: Option<u64>,
    pub time_per_token: Option<f64>,
    pub tokens_generated: Option<u64>,
}

fn http_get_with_timeout(url: &str, timeout_secs: u64) -> Option<reqwest::blocking::Response> {
    reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .ok()?
        .get(url)
        .send()
        .ok()
        .filter(|r| r.status().is_success())
}

pub fn fetch_server_stats(host: &str, port: u16, health_endpoint: &str) -> Option<ServerStats> {
    let check_url = format!("http://{}:{}{}", host, port, health_endpoint);
    if let Some(_resp) = http_get_with_timeout(&check_url, 5) {
        let url = format!("http://{}:{}/stats", host, port);
        http_get_with_timeout(&url, 5)?.json::<ServerStats>().ok()
    } else {
        None
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct ServerProps {
    pub model_path: Option<String>,
}

pub fn get_server_model_path(host: &str, port: u16) -> Option<String> {
    let url = format!("http://{}:{}/props", host, port);
    let response = http_get_with_timeout(&url, 5)?;
    let props: ServerProps = response.json().ok()?;
    props.model_path
}

pub fn get_actual_gpu_layers(host: &str, port: u16, requested_layers: i32) -> i32 {
    // If requested_layers != -1, just return it
    if requested_layers != -1 {
        return requested_layers;
    }

    // If -1, try to get actual layer count from GGUF
    if let Some(model_path) = get_server_model_path(host, port) {
        // Only try to read GGUF if path looks like a file
        if std::path::Path::new(&model_path).is_file() {
            if let Some(n_layer) = read_gguf_n_layer(&model_path) {
                return n_layer as i32;
            }
        }
    }

    // Fallback to -1
    -1
}