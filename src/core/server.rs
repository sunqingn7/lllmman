use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::core::{LlmProvider, ProviderConfig, ProviderError, Result};
use crate::models::ServerStatus;

pub struct ServerController {
    process: Option<Child>,
    external_pid: Option<u32>, // PID of externally running server
    status: Arc<Mutex<ServerStatus>>,
    provider_id: &'static str,
}

impl ServerController {
    pub fn new(provider_id: &'static str) -> Self {
        let mut controller = Self {
            process: None,
            external_pid: None,
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            provider_id,
        };

        // Check for externally running server
        controller.detect_external_server();

        controller
    }

    fn detect_external_server(&mut self) {
        // Check for llama-server process
        if self.provider_id == "llama.cpp" {
            if let Ok(output) = Command::new("pgrep").arg("llama-server").output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if !output_str.trim().is_empty() {
                    if let Ok(pid) = output_str.trim().parse::<u32>() {
                        self.external_pid = Some(pid);
                        *self.status.lock().unwrap() = ServerStatus::Running;
                        log::info!("Detected externally running llama-server (PID {})", pid);
                    }
                }
            }
        }
    }

    pub fn is_external_running(&self) -> bool {
        if let Some(pid) = self.external_pid {
            // Check if process still exists
            Command::new("kill")
                .args(["-0", &pid.to_string()])
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
        } else {
            false
        }
    }

    pub fn start(
        &mut self,
        provider: &Arc<dyn LlmProvider>,
        config: &ProviderConfig,
    ) -> Result<()> {
        if self.is_running() {
            return Err(ProviderError::ServerError("Server already running".into()));
        }

        provider.validate_config(config)?;

        let mut cmd = provider.build_start_command(config);

        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            std::thread::spawn(move || {
                for line in reader.lines().flatten() {
                    log::info!("server: {}", line);
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            std::thread::spawn(move || {
                for line in reader.lines().flatten() {
                    // Log server output as info by default, ERROR for actual errors
                    let lower = line.to_lowercase();
                    // Only ERROR for actual critical errors (not warnings/info that contain "error" or "failed")
                    let is_real_error = lower.contains("error:")
                        || (lower.contains("failed") && lower.contains("abort"))
                        || (lower.contains("error")
                            && lower.contains("cannot")
                            && lower.contains("fatal"));

                    if is_real_error {
                        log::error!("server: {}", line);
                    } else if lower.contains("warning")
                        || lower.contains("failed")
                        || lower.contains("error")
                    {
                        log::warn!("server: {}", line);
                    } else {
                        log::info!("server: {}", line);
                    }
                }
            });
        }

        *self.status.lock().unwrap() = ServerStatus::Starting;
        self.process = Some(child);

        // Wait for server to start with timeout (up to 10 seconds, checking every 200ms)
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);

        while start_time.elapsed() < timeout {
            if self.is_running() {
                *self.status.lock().unwrap() = ServerStatus::Running;
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        // Server failed to start within timeout
        *self.status.lock().unwrap() =
            ServerStatus::Error("Server failed to start within timeout".into());
        Err(ProviderError::ServerError(
            "Server failed to start within timeout".into(),
        ))
    }

    pub fn stop(&mut self) -> Result<()> {
        // Kill internal process if exists
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            *self.status.lock().unwrap() = ServerStatus::Stopped;
        }

        // Kill external process if exists
        if let Some(pid) = self.external_pid {
            if Command::new("kill")
                .arg("-9")
                .arg(pid.to_string())
                .output()
                .is_ok()
            {
                log::info!("Killed external server with PID {}", pid);
                self.external_pid = None;
                *self.status.lock().unwrap() = ServerStatus::Stopped;
                return Ok(());
            }
        }

        if self.process.is_none() && self.external_pid.is_none() {
            return Err(ProviderError::ServerError("No server running".into()));
        }

        Ok(())
    }

    pub fn is_running(&mut self) -> bool {
        // Check internal process first
        if let Some(ref mut child) = self.process {
            if child.try_wait().ok().flatten().is_none() {
                return true;
            }
        }

        // Check external process
        self.is_external_running()
    }

    pub fn get_status(&mut self) -> ServerStatus {
        let stored_status = self.status.lock().unwrap().clone();

        // If status says Running, verify process is actually alive
        if matches!(stored_status, ServerStatus::Running) {
            if !self.is_running() {
                // Clear external PID if it's dead
                if self.external_pid.is_some() {
                    self.external_pid = None;
                }
                return ServerStatus::Error("Server crashed".into());
            }
        }

        stored_status
    }

    pub fn refresh_external_detection(&mut self) {
        // Re-detect external server (useful for periodic checks)
        if self.provider_id == "llama.cpp" {
            if let Ok(output) = Command::new("pgrep").arg("llama-server").output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if !output_str.trim().is_empty() {
                    if let Ok(pid) = output_str.trim().parse::<u32>() {
                        // Only update if not already tracking this PID
                        if self.external_pid != Some(pid) {
                            self.external_pid = Some(pid);
                            let current_status = self.status.lock().unwrap().clone();
                            if !matches!(current_status, ServerStatus::Running) {
                                *self.status.lock().unwrap() = ServerStatus::Running;
                                log::info!(
                                    "Detected externally running llama-server (PID {})",
                                    pid
                                );
                            }
                        }
                    }
                } else if self.external_pid.is_some() {
                    // Process is gone
                    self.external_pid = None;
                    *self.status.lock().unwrap() = ServerStatus::Stopped;
                }
            }
        }
    }
}
