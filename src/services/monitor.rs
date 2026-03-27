use crate::models::MonitorStats;
use crate::services::gpu_detector;

pub fn get_system_stats() -> MonitorStats {
    let mut sys = sysinfo::System::new();
    sys.refresh_all();

    let cpu = sys.cpus().first().map(|c| c.cpu_usage()).unwrap_or(0.0);
    let total_ram = sys.total_memory() as u32;
    let used_ram = sys.used_memory() as u32;

    let gpu_usage = gpu_detector::get_all_gpu_usage();
    let (vram_used, vram_total) = if !gpu_usage.is_empty() {
        let gpus = gpu_detector::detect_gpus();
        let total: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
        let used: u32 = gpu_usage.iter().map(|u| u.used_vram_mb).sum();
        (used, total)
    } else {
        (0, 0)
    };

    MonitorStats {
        vram_used_mb: vram_used,
        vram_total_mb: vram_total,
        ram_used_mb: used_ram / (1024 * 1024),
        ram_total_mb: total_ram / (1024 * 1024),
        cpu_percent: cpu,
        tokens_per_second: 0.0,
        active_connections: 0,
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

    match reqwest::blocking::get(&url).ok()?.json::<ServerStats>() {
        Ok(stats) => Some(stats),
        Err(_) => None,
    }
}
