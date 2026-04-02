use crate::models::{GpuTemperature, MonitorStats};
use crate::providers::llama_cpp::read_gguf_n_layer;
use crate::services::gpu_detector;

pub fn get_system_stats() -> MonitorStats {
    let mut sys = sysinfo::System::new();
    sys.refresh_all();

    let cpu = sys.cpus().first().map(|c| c.cpu_usage()).unwrap_or(0.0);
    let total_ram = sys.total_memory() as u64;
    let used_ram = sys.used_memory() as u64;

    let gpus = gpu_detector::detect_gpus();
    let gpu_usage = gpu_detector::get_all_gpu_usage();
    let (vram_used, vram_total) = if !gpu_usage.is_empty() {
        let total: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
        let used: u32 = gpu_usage.iter().map(|u| u.used_vram_mb).sum();
        (used, total)
    } else {
        (0, 0)
    };

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
        vram_used_mb: vram_used,
        vram_total_mb: vram_total,
        ram_used_mb: (used_ram / (1024 * 1024)) as u32,
        ram_total_mb: (total_ram / (1024 * 1024)) as u32,
        cpu_percent: cpu,
        tokens_per_second: 0.0,
        active_connections: 0,
        gpu_temperatures,
        cpu_temperature,
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

pub fn fetch_server_stats(host: &str, port: u16) -> Option<ServerStats> {
    let url = format!("http://{}:{}/stats", host, port);

    // First check if server is ready with a simple GET
    let check_url = format!("http://{}:{}/", host, port);
    if let Ok(response) = reqwest::blocking::get(&check_url) {
        if !response.status().is_success() {
            return None;
        }
    } else {
        // Server not responding
        return None;
    }

    // Now fetch stats
    let response = reqwest::blocking::get(&url).ok()?;
    if !response.status().is_success() {
        return None;
    }

    response.json::<ServerStats>().ok()
}

#[derive(Debug, serde::Deserialize)]
pub struct ServerProps {
    pub model_path: Option<String>,
}

pub fn get_server_model_path(host: &str, port: u16) -> Option<String> {
    let url = format!("http://{}:{}/props", host, port);
    let response = reqwest::blocking::get(&url).ok()?;
    if !response.status().is_success() {
        return None;
    }
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
        if let Some(n_layer) = read_gguf_n_layer(&model_path) {
            return n_layer as i32;
        }
    }

    // Fallback to -1
    -1
}
