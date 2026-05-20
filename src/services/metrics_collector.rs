use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::core::{InstanceManager, InstanceMetrics, InstanceStatus};
use crate::services::gpu_detector::{get_all_gpu_usage, get_cpu_temperature};

/// Shared state for metrics that the GUI/TUI can read.
#[derive(Clone, Debug)]
pub struct MetricsSnapshot {
    pub instances: Vec<InstanceMetricsSnapshot>,
    pub router: Option<RouterMetricsSnapshot>,
    pub gpu_temps: Vec<GpuTempSnapshot>,
    pub cpu_temp: Option<f32>,
    pub timestamp: std::time::SystemTime,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            instances: Vec::new(),
            router: None,
            gpu_temps: Vec::new(),
            cpu_temp: None,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct InstanceMetricsSnapshot {
    pub id: u32,
    pub port: u16,
    pub status: InstanceStatus,
    pub metrics: InstanceMetrics,
}

#[derive(Clone, Debug, Default)]
pub struct RouterMetricsSnapshot {
    pub port: u16,
    pub status: InstanceStatus,
    pub connected_workers: u32,
    pub total_workers: u32,
    pub total_qps: f32,
}

#[derive(Clone, Debug, Default)]
pub struct GpuTempSnapshot {
    pub index: u32,
    pub temperature_c: f32,
    pub utilization: f32,
}

/// Background metrics collector that polls all instances and GPU metrics.
pub struct MetricsCollector {
    snapshot: Arc<Mutex<MetricsSnapshot>>,
    poll_interval: Duration,
    running: Arc<Mutex<bool>>,
}

impl MetricsCollector {
    pub fn new(poll_interval_secs: u64) -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(MetricsSnapshot::default())),
            poll_interval: Duration::from_secs(poll_interval_secs),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Get the latest metrics snapshot.
    pub fn get_snapshot(&self) -> MetricsSnapshot {
        self.snapshot.lock().unwrap().clone()
    }

    /// Start the background polling loop in a separate thread.
    pub fn start(&self, instance_manager: Arc<Mutex<InstanceManager>>) {
        *self.running.lock().unwrap() = true;

        let snapshot = self.snapshot.clone();
        let running = self.running.clone();
        let interval = self.poll_interval;

        std::thread::spawn(move || {
            while *running.lock().unwrap() {
                let manager = instance_manager.lock().unwrap();
                let mut snap = MetricsSnapshot {
                    timestamp: std::time::SystemTime::now(),
                    ..Default::default()
                };

                // Collect instance metrics
                for instance in manager.instances() {
                    let status = instance.status.lock().unwrap().clone();
                    let metrics = if matches!(status, InstanceStatus::Running) {
                        fetch_instance_metrics(instance.port).unwrap_or_default()
                    } else {
                        InstanceMetrics::default()
                    };

                    snap.instances.push(InstanceMetricsSnapshot {
                        id: instance.id,
                        port: instance.port,
                        status,
                        metrics,
                    });
                }

                // Collect router metrics
                if let Some(router) = manager.router() {
                    let status = router.status.lock().unwrap().clone();
                    snap.router = Some(RouterMetricsSnapshot {
                        port: router.config.router_port,
                        status,
                        connected_workers: router.connected_workers,
                        total_workers: router.total_workers,
                        total_qps: router.total_qps,
                    });
                }

                // Collect GPU metrics
                for usage in get_all_gpu_usage() {
                    snap.gpu_temps.push(GpuTempSnapshot {
                        index: usage.index,
                        temperature_c: usage.temperature_c.unwrap_or(0.0),
                        utilization: 0.0, // Would need nvidia-smi --query-gpu=utilization.gpu
                    });
                }

                snap.cpu_temp = get_cpu_temperature();

                drop(manager);
                *snapshot.lock().unwrap() = snap;
                std::thread::sleep(interval);
            }
        });
    }

    /// Stop the background polling loop.
    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }
}

/// Fetch metrics from a running instance's /stats endpoint.
fn fetch_instance_metrics(port: u16) -> Option<InstanceMetrics> {
    let url = format!("http://127.0.0.1:{}/stats", port);

    let response = reqwest::blocking::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .ok()?;

    if !response.status().is_success() {
        return None;
    }

    let json: serde_json::Value = response.json().ok()?;

    Some(InstanceMetrics {
        tokens_per_second: json.get("token_per_second")
            .or_else(|| json.get("tokens_per_second"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        queue_size: json.get("queue_size").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        cache_hit_rate: json.get("cache_hit_rate").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        vram_used_mb: json.get("gpu_cache_usage_gb")
            .and_then(|v| v.as_f64())
            .map(|gb| (gb * 1024.0) as u32)
            .unwrap_or(0),
        vram_total_mb: json.get("gpu_cache_capacity_gb")
            .and_then(|v| v.as_f64())
            .map(|gb| (gb * 1024.0) as u32)
            .unwrap_or(0),
        avg_latency_ms: json.get("avg_latency_ms").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        gpu_utilization: 0.0,
        temperature_c: 0.0,
        requests_total: json.get("num_requests_total").and_then(|v| v.as_u64()).unwrap_or(0),
        requests_failed: json.get("num_requests_failed").and_then(|v| v.as_u64()).unwrap_or(0),
    })
}
