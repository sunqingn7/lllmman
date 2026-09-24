use std::io::{BufRead, BufReader};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::LogBuffer;
use crate::models::deployment::{DeploymentProfile, InstanceConfig, RouterConfig};

/// Status of a managed instance.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum InstanceStatus {
    #[default]
    Stopped,
    Starting,
    Running,
    Error(String),
    Restarting,
}

/// Metrics for a single instance.
#[derive(Clone, Debug, Default)]
pub struct InstanceMetrics {
    pub tokens_per_second: f32,
    pub queue_size: u32,
    pub cache_hit_rate: f32,
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub avg_latency_ms: f32,
    pub gpu_utilization: f32,
    pub temperature_c: f32,
    pub requests_total: u64,
    pub requests_failed: u64,
}

/// A handle to a running instance.
pub struct InstanceHandle {
    pub id: u32,
    pub config: InstanceConfig,
    pub status: Arc<Mutex<InstanceStatus>>,
    pub metrics: Arc<Mutex<InstanceMetrics>>,
    pub process: Arc<Mutex<Option<Child>>>,
    pub log_buffer: LogBuffer,
    pub port: u16,
    pub stop_flag: Arc<AtomicBool>,
}

/// A handle to the router process.
pub struct RouterHandle {
    pub config: RouterConfig,
    pub status: Arc<Mutex<InstanceStatus>>,
    pub process: Arc<Mutex<Option<Child>>>,
    pub log_buffer: LogBuffer,
    pub connected_workers: u32,
    pub total_workers: u32,
    pub total_qps: f32,
}

/// Manages multiple server instances and an optional router.
pub struct InstanceManager {
    pub instances: Vec<InstanceHandle>,
    pub router: Option<RouterHandle>,
    pub deployment_profile: Option<DeploymentProfile>,
    next_id: u32,
}

impl InstanceManager {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            router: None,
            deployment_profile: None,
            next_id: 1,
        }
    }

    /// Check if a port is available.
    pub fn check_port_available(port: u16) -> bool {
        std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
    }

    /// Launch a new instance with the given config.
    pub fn launch_instance(
        &mut self,
        config: &InstanceConfig,
        provider: &dyn crate::core::LlmProvider,
        provider_settings: &crate::core::ProviderSettings,
    ) -> Result<u32, String> {
        if !Self::check_port_available(config.port) {
            return Err(format!("Port {} is already in use", config.port));
        }

        let id = self.next_id;
        self.next_id += 1;

        let status = Arc::new(Mutex::new(InstanceStatus::Starting));
        let metrics = Arc::new(Mutex::new(InstanceMetrics::default()));
        let log_buffer = LogBuffer::new();

        let command_line = provider.build_command_line_for_instance(config, provider_settings);
        log_buffer.push_info(format!("Starting instance #{} with command: {}", id, command_line));

        let mut child = Command::new("bash")
            .arg("-c")
            .arg(&command_line)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn process: {}", e))?;

        let log_buf = log_buffer.clone();
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let log = log_buf.clone();
            std::thread::spawn(move || {
                for line in reader.lines().map_while(|r| r.ok()) {
                    log.push_info(line);
                }
            });
        }

        let log_buf = log_buffer.clone();
        let status_clone = status.clone();
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            std::thread::spawn(move || {
                for line in reader.lines().map_while(|r| r.ok()) {
                    let lower = line.to_lowercase();
                    let is_real_error = lower.contains("error:")
                        || (lower.contains("failed") && lower.contains("abort"))
                        || (lower.contains("error") && lower.contains("cannot") && lower.contains("fatal"));

                    if is_real_error {
                        log_buf.push_error(line);
                    } else if lower.contains("warning") || lower.contains("failed") || lower.contains("error") {
                        log_buf.push_warn(line);
                    } else {
                        log_buf.push_info(line);
                    }
                }
                // If stderr thread ends, check if process is still running
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Ok(mut s) = status_clone.lock() {
                    if *s == InstanceStatus::Starting {
                        *s = InstanceStatus::Error("Process exited unexpectedly".to_string());
                    }
                }
            });
        }

        let stop_flag = Arc::new(AtomicBool::new(false));
        let handle = InstanceHandle {
            id,
            config: config.clone(),
            status,
            metrics,
            process: Arc::new(Mutex::new(Some(child))),
            log_buffer: log_buffer.clone(),
            port: config.port,
            stop_flag: stop_flag.clone(),
        };

        Self::spawn_metrics_updater(
            config.port,
            config.provider.clone(),
            handle.metrics.clone(),
            handle.status.clone(),
            handle.stop_flag.clone(),
        );

        let status_clone = handle.status.clone();
        let log_buf = handle.log_buffer.clone();
        let instance_id = id;
        let instance_port = config.port;

        self.instances.push(handle);

        // Spawn background thread for start health-check (non-blocking for UI)
        std::thread::spawn(move || {
            let start_time = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(15);
            while start_time.elapsed() < timeout {
                std::thread::sleep(std::time::Duration::from_millis(500));
                // Check HTTP endpoint as health signal
                let url = format!("http://127.0.0.1:{}/health", instance_port);
                if reqwest::blocking::Client::new()
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(1))
                    .send()
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
                {
                    *status_clone.lock().unwrap() = InstanceStatus::Running;
                    return;
                }
            }
            *status_clone.lock().unwrap() =
                InstanceStatus::Error("Instance failed to start within timeout".to_string());
            log_buf.push_error(format!("Instance #{} failed to start within timeout", instance_id));
        });

        Ok(id)
    }

    /// Launch the router.
    pub fn launch_router(&mut self, config: &RouterConfig) -> Result<(), String> {
        let status = Arc::new(Mutex::new(InstanceStatus::Starting));
        let log_buffer = LogBuffer::new();

        let worker_urls = config.worker_urls.join(" ");
        let command = format!(
            "python -m sglang_router.launch_router --worker-urls {} --port {} --host 0.0.0.0 --policy {:?}",
            worker_urls, config.router_port, config.policy
        );

        log_buffer.push_info(format!("Starting router: {}", command));

        let mut child = Command::new("bash")
            .arg("-c")
            .arg(&command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn router: {}", e))?;

        let log_buf = log_buffer.clone();
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let log = log_buf.clone();
            std::thread::spawn(move || {
                for line in reader.lines().map_while(|r| r.ok()) {
                    log.push_info(line);
                }
            });
        }

        let log_buf = log_buffer.clone();
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            std::thread::spawn(move || {
                for line in reader.lines().map_while(|r| r.ok()) {
                    let lower = line.to_lowercase();
                    if lower.contains("error") {
                        log_buf.push_error(line);
                    } else {
                        log_buf.push_info(line);
                    }
                }
            });
        }

        self.router = Some(RouterHandle {
            config: config.clone(),
            status,
            process: Arc::new(Mutex::new(Some(child))),
            log_buffer,
            connected_workers: 0,
            total_workers: config.worker_urls.len() as u32,
            total_qps: 0.0,
        });

        Ok(())
    }

    /// Stop a specific instance.
    pub fn stop_instance(&mut self, id: u32) -> Result<(), String> {
        let idx = self.instances.iter().position(|i| i.id == id).ok_or("Instance not found")?;

        let handle = &mut self.instances[idx];
        *handle.status.lock().unwrap() = InstanceStatus::Stopped;
        handle.stop_flag.store(true, Ordering::Relaxed);

        let mut process_guard = handle.process.lock().unwrap();
        if let Some(mut child) = process_guard.take() {
            let pid = child.id();
            let _ = Command::new("pkill").args(["-P", &pid.to_string()]).output();
            let _ = Command::new("kill").args(["-9", &format!("-{}", pid)]).output();
            let _ = child.kill();
            let _ = child.wait();
        }

        Ok(())
    }

    /// Stop all instances and the router.
    pub fn stop_all(&mut self) {
        // Stop router first
        if let Some(router) = self.router.take() {
            *router.status.lock().unwrap() = InstanceStatus::Stopped;
            let mut process_guard = router.process.lock().unwrap();
            if let Some(mut child) = process_guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        // Then stop instances
        for handle in &mut self.instances {
            *handle.status.lock().unwrap() = InstanceStatus::Stopped;
            handle.stop_flag.store(true, Ordering::Relaxed);
            let mut process_guard = handle.process.lock().unwrap();
            if let Some(mut child) = process_guard.take() {
                let pid = child.id();
                let _ = Command::new("pkill").args(["-P", &pid.to_string()]).output();
                let _ = Command::new("kill").args(["-9", &format!("-{}", pid)]).output();
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    /// Check if a specific instance is running.
    pub fn is_instance_running(&self, id: u32) -> bool {
        self.instances.iter().any(|i| {
            if i.id != id {
                return false;
            }
            let mut process_guard = i.process.lock().unwrap();
            if let Some(child) = process_guard.as_mut() {
                matches!(child.try_wait(), Ok(None))
            } else {
                false
            }
        })
    }

    /// Get the number of running instances.
    pub fn running_count(&self) -> u32 {
        self.instances.iter().filter(|i| self.is_instance_running(i.id)).count() as u32
    }

    /// Get instance by ID.
    pub fn get_instance(&self, id: u32) -> Option<&InstanceHandle> {
        self.instances.iter().find(|i| i.id == id)
    }

    /// Get mutable instance by ID.
    pub fn get_instance_mut(&mut self, id: u32) -> Option<&mut InstanceHandle> {
        self.instances.iter_mut().find(|i| i.id == id)
    }

    /// Get all instance handles.
    pub fn instances(&self) -> &[InstanceHandle] {
        &self.instances
    }

    /// Get router handle.
    pub fn router(&self) -> Option<&RouterHandle> {
        self.router.as_ref()
    }

    /// Check if any instance or router is running.
    pub fn is_any_running(&self) -> bool {
        let instances_running = self.instances.iter().any(|i| self.is_instance_running(i.id));

        let router_running = self.router.as_ref().is_some_and(|r| {
            let mut process_guard = r.process.lock().unwrap();
            if let Some(p) = process_guard.as_mut() {
                matches!(p.try_wait(), Ok(None))
            } else {
                false
            }
        });

        instances_running || router_running
    }

    /// Register an externally-started instance (e.g., started by ServerController).
    /// This creates a tracking entry so the instance appears in Instance Manager and Performance Monitor.
    /// Spawn a background thread that keeps `handle.metrics` fresh.
    fn spawn_metrics_updater(
        port: u16,
        provider: String,
        metrics: Arc<Mutex<InstanceMetrics>>,
        status: Arc<Mutex<InstanceStatus>>,
        stop_flag: Arc<AtomicBool>,
    ) {
        std::thread::spawn(move || {
            let mut prev: Option<(std::time::Instant, f64, f64, f64, f64, f64)> = None;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(2000));
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                let running = matches!(
                    status.lock().unwrap().clone(),
                    InstanceStatus::Running
                );
                if !running {
                    prev = None;
                    continue;
                }

                let mut m = InstanceMetrics::default();
                if provider == "llama.cpp" {
                    m = crate::services::metrics_collector::fetch_instance_metrics(port)
                        .unwrap_or_default();
                }

                if let Some(prom) = fetch_prometheus_metrics(port) {
                    let now = std::time::Instant::now();
                    let gen = prom
                        .get("vllm:generation_tokens_total")
                        .or_else(|| prom.get("sglang:generation_tokens_total"))
                        .copied();
                    let hits = prom
                        .get("vllm:prefix_cache_hits_total")
                        .or_else(|| prom.get("sglang:prefix_cache_hits_total"))
                        .copied()
                        .unwrap_or(0.0);
                    let queries = prom
                        .get("vllm:prefix_cache_queries_total")
                        .or_else(|| prom.get("sglang:prefix_cache_queries_total"))
                        .copied()
                        .unwrap_or(0.0);
                    let e2e_sum = prom
                        .get("vllm:e2e_request_latency_seconds_sum")
                        .or_else(|| prom.get("sglang:e2e_request_latency_seconds_sum"))
                        .copied()
                        .unwrap_or(0.0);
                    let e2e_count = prom
                        .get("vllm:e2e_request_latency_seconds_count")
                        .or_else(|| prom.get("sglang:e2e_request_latency_seconds_count"))
                        .copied()
                        .unwrap_or(0.0);

                    if let Some(gen_total) = gen {
                        if let Some((p_now, p_gen, p_hits, p_queries, p_sum, p_count)) = prev {
                            let dt = now.duration_since(p_now).as_secs_f64().max(0.001);
                            m.tokens_per_second = ((gen_total - p_gen) / dt).max(0.0) as f32;
                            let dq = queries - p_queries;
                            if dq > 0.0 {
                                m.cache_hit_rate = ((hits - p_hits) / dq).clamp(0.0, 1.0) as f32;
                            }
                            let dc = e2e_count - p_count;
                            if dc > 0.0 {
                                m.avg_latency_ms = ((e2e_sum - p_sum) / dc * 1000.0).max(0.0)
                                    as f32;
                            }
                        }
                        prev = Some((now, gen_total, hits, queries, e2e_sum, e2e_count));

                        m.queue_size = prom
                            .get("vllm:num_requests_waiting")
                            .or_else(|| prom.get("sglang:num_waiting_requests"))
                            .copied()
                            .unwrap_or(0.0)
                            .max(0.0) as u32;
                    }

                    if let Some(kv_perc) = prom
                        .get("vllm:kv_cache_usage_perc")
                        .or_else(|| prom.get("sglang:token_usage"))
                        .copied()
                    {
                        match nvidia_vram_util() {
                            Some((util, used, total)) => {
                                m.gpu_utilization = util;
                                m.vram_total_mb = total as u32;
                                m.vram_used_mb = used as u32;
                            }
                            // No nvidia-smi: report KV-cache fill percentage
                            None => {
                                m.vram_total_mb = 100;
                                m.vram_used_mb = (kv_perc * 100.0).clamp(0.0, 100.0) as u32;
                            }
                        }
                    }
                }

                if m.vram_total_mb == 0 {
                    if let Some((util, used, total)) = nvidia_vram_util() {
                        m.gpu_utilization = util;
                        m.vram_total_mb = total as u32;
                        m.vram_used_mb = used as u32;
                    }
                }

                if let Ok(mut guard) = metrics.lock() {
                    *guard = m;
                }
            }
        });
    }

    pub fn register_external_instance(
        &mut self,
        model_path: &str,
        provider: &str,
        port: u16,
        context_size: u32,
        log_buffer: &LogBuffer,
        status: Arc<std::sync::Mutex<crate::models::ServerStatus>>,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let instance_status = Arc::new(Mutex::new(InstanceStatus::Running));
        let metrics = Arc::new(Mutex::new(InstanceMetrics::default()));

        let config = InstanceConfig {
            gpu_indices: Vec::new(),
            port,
            memory_utilization: 0.9,
            max_num_seqs: None,
            model_path: model_path.to_string(),
            context_size,
            provider: provider.to_string(),
        };

        let stop_flag = Arc::new(AtomicBool::new(false));
        let handle = InstanceHandle {
            id,
            config,
            status: instance_status.clone(),
            metrics: metrics.clone(),
            process: Arc::new(Mutex::new(None)),
            log_buffer: log_buffer.clone(),
            port,
            stop_flag: stop_flag.clone(),
        };

        // Spawn a background thread to sync status from ServerController
        let status_clone = status.clone();
        let inst_status_clone = instance_status.clone();
        let stop_flag_for_updater = stop_flag.clone();
        std::thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                let server_status = status_clone.lock().unwrap().clone();
                let new_status = match server_status {
                    crate::models::ServerStatus::Running => InstanceStatus::Running,
                    crate::models::ServerStatus::Stopped => InstanceStatus::Stopped,
                    crate::models::ServerStatus::Starting => InstanceStatus::Starting,
                    crate::models::ServerStatus::Error(e) => InstanceStatus::Error(e),
                };
                *inst_status_clone.lock().unwrap() = new_status;
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });

        self.instances.push(handle);
        Self::spawn_metrics_updater(
            port,
            provider.to_string(),
            metrics,
            instance_status,
            stop_flag_for_updater,
        );
        id
    }

    /// Unregister an external instance when it's stopped.
    pub fn unregister_external_instance(&mut self, model_path: &str, port: u16) {
        self.instances.retain(|h| {
            let stopping = h.config.model_path == model_path && h.config.port == port;
            if stopping {
                h.stop_flag.store(true, Ordering::Relaxed);
            }
            !stopping
        });
    }
}

impl Default for InstanceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Fetch and aggregate a Prometheus-style /metrics endpoint into a map of
/// metric name -> summed value (across label sets).
fn fetch_prometheus_metrics(port: u16) -> Option<std::collections::HashMap<String, f64>> {
    let url = format!("http://127.0.0.1:{}/metrics", port);
    let text = reqwest::blocking::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .ok()?
        .text()
        .ok()?;

    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next()?;
        let value: f64 = parts.next()?.parse().ok()?;
        let name = key.split('{').next().unwrap_or(key);
        *out.entry(name.to_string()).or_insert(0.0) += value;
    }
    Some(out)
}

/// Query nvidia-smi for (avg utilization %, used MB, total MB) across GPUs.
fn nvidia_vram_util() -> Option<(f32, f64, f64)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut util_sum = 0.0f32;
    let mut used_sum = 0.0f64;
    let mut total_sum = 0.0f64;
    let mut count = 0usize;
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if cols.len() < 3 {
            continue;
        }
        let util: f32 = cols[0].parse().ok()?;
        let used: f64 = cols[1].parse().ok()?;
        let total: f64 = cols[2].parse().ok()?;
        util_sum += util;
        used_sum += used;
        total_sum += total;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    Some((util_sum / count as f32, used_sum, total_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_manager_new() {
        let manager = InstanceManager::new();
        assert!(manager.instances.is_empty());
        assert!(manager.router.is_none());
        assert!(!manager.is_any_running());
    }

    #[test]
    fn test_check_port_available() {
        // Port 0 should always be available (OS assigns dynamically)
        // We can't test a specific port reliably, but we can test the function doesn't panic
        let _ = InstanceManager::check_port_available(0);
    }

    #[test]
    fn test_running_count_empty() {
        let manager = InstanceManager::new();
        assert_eq!(manager.running_count(), 0);
    }
}
