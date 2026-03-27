use std::io::{BufRead, BufReader};
use std::process::{Child, Stdio};
use std::sync::{Arc, Mutex};

use crate::core::{LlmProvider, ProviderConfig, ProviderError, Result};
use crate::models::ServerStatus;

pub struct ServerController {
    process: Option<Child>,
    status: Arc<Mutex<ServerStatus>>,
    provider_id: &'static str,
}

impl ServerController {
    pub fn new(provider_id: &'static str) -> Self {
        Self {
            process: None,
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            provider_id,
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

        *self.status.lock().unwrap() = ServerStatus::Starting;
        self.process = Some(child);

        std::thread::sleep(std::time::Duration::from_secs(2));

        if self.is_running() {
            *self.status.lock().unwrap() = ServerStatus::Running;
            Ok(())
        } else {
            *self.status.lock().unwrap() = ServerStatus::Error("Server failed to start".into());
            Err(ProviderError::ServerError("Server failed to start".into()))
        }
    }

    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.process.take() {
            child.kill()?;
            *self.status.lock().unwrap() = ServerStatus::Stopped;
            Ok(())
        } else {
            Err(ProviderError::ServerError("No server running".into()))
        }
    }

    pub fn is_running(&mut self) -> bool {
        self.process
            .as_mut()
            .and_then(|c| c.try_wait().ok())
            .flatten()
            .is_none()
    }

    pub fn get_status(&self) -> ServerStatus {
        self.status.lock().unwrap().clone()
    }
}
