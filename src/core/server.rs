use std::io::{BufRead, BufReader};
use std::process::Command;
use std::sync::{Arc, Mutex};

use crate::core::{
    LlmProvider, LogBuffer, ProviderConfig, ProviderError, ProviderSettings, Result,
};
use crate::models::ServerStatus;

pub struct ServerController {
    process: Option<std::process::Child>,
    external_pid: Option<u32>,
    external_cmd: Option<String>,
    status: Arc<Mutex<ServerStatus>>,
    provider: Option<Arc<dyn LlmProvider>>,
    provider_settings: ProviderSettings,
    log_buffer: LogBuffer,
}

impl ServerController {
    pub fn new() -> Self {
        Self {
            process: None,
            external_pid: None,
            external_cmd: None,
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            provider: None,
            provider_settings: ProviderSettings::default(),
            log_buffer: LogBuffer::new(),
        }
    }

    pub fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.provider_settings = provider.default_settings();
        self.provider = Some(provider);
        self.detect_external_server();
    }

    pub fn set_provider_settings(&mut self, settings: ProviderSettings) {
        self.provider_settings = settings;
    }

    pub fn get_provider_settings(&self) -> ProviderSettings {
        self.provider_settings.clone()
    }

    pub fn get_log_buffer(&self) -> LogBuffer {
        self.log_buffer.clone()
    }

    fn detect_external_server(&mut self) {
        let Some(provider) = &self.provider else {
            return;
        };

        let servers = provider.detect_running_servers();
        if let Some(server) = servers.into_iter().next() {
            self.external_pid = Some(server.pid);
            self.external_cmd = Some(server.command_line.clone());
            *self.status.lock().unwrap() = ServerStatus::Running;
            self.log_buffer.push_info(format!(
                "Detected externally running {} server (PID {})",
                provider.id(),
                server.pid
            ));
        }
    }

    pub fn is_external_running(&self) -> bool {
        if let Some(pid) = self.external_pid {
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

        // Build and log the full command line
        let command_line = provider.build_command_line(config, &self.provider_settings);
        self.log_buffer
            .push_info(format!("Starting server with command: {}", command_line));

        let mut child = provider.start_server(config, &self.provider_settings)?;

        let log_buffer = self.log_buffer.clone();
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let log_buf = log_buffer.clone();
            std::thread::spawn(move || {
                for line in reader.lines().flatten() {
                    log_buf.push_info(line);
                }
            });
        }

        let log_buffer = self.log_buffer.clone();
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            std::thread::spawn(move || {
                for line in reader.lines().flatten() {
                    let lower = line.to_lowercase();
                    let is_real_error = lower.contains("error:")
                        || (lower.contains("failed") && lower.contains("abort"))
                        || (lower.contains("error")
                            && lower.contains("cannot")
                            && lower.contains("fatal"));

                    if is_real_error {
                        log_buffer.push_error(line);
                    } else if lower.contains("warning")
                        || lower.contains("failed")
                        || lower.contains("error")
                    {
                        log_buffer.push_warn(line);
                    } else {
                        log_buffer.push_info(line);
                    }
                }
            });
        }

        *self.status.lock().unwrap() = ServerStatus::Starting;
        self.process = Some(child);

        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);

        while start_time.elapsed() < timeout {
            if self.is_running() {
                *self.status.lock().unwrap() = ServerStatus::Running;
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        *self.status.lock().unwrap() =
            ServerStatus::Error("Server failed to start within timeout".into());
        Err(ProviderError::ServerError(
            "Server failed to start within timeout".into(),
        ))
    }

    pub fn stop(&mut self) -> Result<()> {
        let mut killed = false;

        if let Some(mut child) = self.process.take() {
            // Kill the entire process group to ensure all child processes are terminated
            // vLLM and other providers spawn worker processes that survive parent kill
            let pid = child.id();
            let _ = Command::new("pkill")
                .args(["-P", &pid.to_string()])
                .output();
            let _ = Command::new("kill")
                .args(["-9", &format!("-{}", pid)])
                .output();
            let _ = child.kill();
            // Wait for process to terminate with timeout (max 5 seconds)
            let start = std::time::Instant::now();
            while child.try_wait().map(|r| r.is_none()).unwrap_or(true) {
                if start.elapsed().as_secs() > 5 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let _ = child.wait();
            *self.status.lock().unwrap() = ServerStatus::Stopped;
            killed = true;
        }

        if let Some(pid) = self.external_pid {
            // Kill all child processes first, then the parent
            // vLLM and other Python-based servers spawn worker processes that survive parent kill
            let _ = Command::new("pkill")
                .args(["-P", &pid.to_string()])
                .output();

            // Also try killing the process group (negative PID) as fallback
            let _ = Command::new("kill")
                .args(["-9", &format!("-{}", pid)])
                .output();

            // Finally kill the parent process
            if Command::new("kill")
                .arg("-9")
                .arg(pid.to_string())
                .output()
                .is_ok()
            {
                let _ = Command::new("wait").arg(pid.to_string()).output();
                self.log_buffer
                    .push_info(format!("Killed external server with PID {}", pid));
                self.external_pid = None;
                self.external_cmd = None;
                *self.status.lock().unwrap() = ServerStatus::Stopped;
                killed = true;
            }
        }

        if !killed {
            return Err(ProviderError::ServerError("No server running".into()));
        }

        Ok(())
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(ref mut child) = self.process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    self.process = None;
                    return false;
                }
                Ok(None) => return true,
                Err(_) => {
                    self.process = None;
                    return false;
                }
            }
        }

        self.is_external_running()
    }

    pub fn get_status(&mut self) -> ServerStatus {
        // Clone status first to release lock quickly
        let stored_status = {
            let guard = self.status.lock().unwrap();
            guard.clone()
        };

        if matches!(stored_status, ServerStatus::Running) {
            if !self.is_running() {
                if self.external_pid.is_some() {
                    self.external_pid = None;
                    self.external_cmd = None;
                }
                return ServerStatus::Error("Server crashed".into());
            }
        }

        stored_status
    }

    pub fn refresh_external_detection(&mut self) {
        let Some(provider) = &self.provider else {
            return;
        };

        let servers = provider.detect_running_servers();
        if let Some(server) = servers.into_iter().next() {
            if self.external_pid != Some(server.pid) {
                self.external_pid = Some(server.pid);
                self.external_cmd = Some(server.command_line.clone());
                let current_status = self.status.lock().unwrap().clone();
                if !matches!(current_status, ServerStatus::Running) {
                    *self.status.lock().unwrap() = ServerStatus::Running;
                    self.log_buffer.push_info(format!(
                        "Detected externally running {} server (PID {})",
                        provider.id(),
                        server.pid
                    ));
                }
            }
        } else if self.external_pid.is_some() {
            self.external_pid = None;
            self.external_cmd = None;
            *self.status.lock().unwrap() = ServerStatus::Stopped;
        }
    }

    pub fn get_external_config(&self) -> Option<ProviderConfig> {
        let provider = self.provider.as_ref()?;
        let cmd = self.external_cmd.as_ref()?;
        Some(provider.parse_server_config(cmd))
    }

    pub fn has_external_server(&self) -> bool {
        self.external_pid.is_some() && self.is_external_running()
    }

    pub fn get_log_buffer_ref(&self) -> &LogBuffer {
        &self.log_buffer
    }
}
