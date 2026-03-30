//! Process detection for running LLM servers

use std::process::Command;

use crate::core::ProviderConfig;

/// Detected running server information
#[derive(Debug, Clone)]
pub struct DetectedServer {
    pub pid: u32,
    pub binary: String,
    pub command_line: String,
    pub provider_id: String,
}

/// Detect running LLM server processes
pub fn detect_running_servers() -> Vec<DetectedServer> {
    let mut servers = Vec::new();

    // Known server binaries and their provider IDs
    let known_servers = [
        ("llama-server", "llama.cpp"),
        ("llama.cpp", "llama.cpp"),
        ("vllm", "vllm"),
        ("text-generation-server", "tgi"),
        ("Ollama", "ollama"),
    ];

    for (binary, provider_id) in &known_servers {
        if let Some(server) = find_process_by_name(binary, provider_id) {
            servers.push(server);
        }
    }

    servers
}

/// Find a process by its binary name
fn find_process_by_name(binary: &str, provider_id: &str) -> Option<DetectedServer> {
    // Try different methods to find the process

    // Method 1: Use pgrep (Linux)
    if let Ok(output) = Command::new("pgrep").args(["-a", binary]).output() {
        let output_str = String::from_utf8_lossy(&output.stdout);
        if !output_str.is_empty() {
            let line = output_str.trim();
            // Format: "PID command args..."
            if let Some(space_pos) = line.find(' ') {
                let pid_str = &line[..space_pos];
                let cmdline = &line[space_pos + 1..];
                if let Ok(pid) = pid_str.parse::<u32>() {
                    log::debug!("Found process via pgrep: PID={} Cmd={}", pid, cmdline);
                    return Some(DetectedServer {
                        pid,
                        binary: binary.to_string(),
                        command_line: cmdline.to_string(),
                        provider_id: provider_id.to_string(),
                    });
                }
            }
        }
    }

    // Method 2: Use ps command (cross-platform fallback)
    if let Ok(output) = Command::new("ps")
        .args(["-A", "-o", "pid=,command="])
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.contains(binary) {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() >= 2 {
                    if let Ok(pid) = parts[0].parse::<u32>() {
                        return Some(DetectedServer {
                            pid,
                            binary: binary.to_string(),
                            command_line: parts[1].to_string(),
                            provider_id: provider_id.to_string(),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Parse llama-server command-line arguments into ProviderConfig
pub fn parse_llama_server_args(cmd_line: &str) -> ProviderConfig {
    let mut config = ProviderConfig::default();

    let args: Vec<&str> = cmd_line.split_whitespace().collect();
    let mut i = 0;

    while i < args.len() {
        let arg = args[i];

        match arg {
            "-m" | "--model" => {
                if i + 1 < args.len() {
                    config.model_path = args[i + 1].to_string();
                    i += 1;
                }
            }
            "-hf" => {
                // HuggingFace model load flag
                if i + 1 < args.len() {
                    config.model_path = args[i + 1].to_string();
                    i += 1;
                }
            }
            "-c" | "--ctx-size" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.context_size = val;
                    }
                    i += 1;
                }
            }
            "-b" | "--batch-size" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.batch_size = val;
                    }
                    i += 1;
                }
            }
            "-ngl" | "--n-gpu-layers" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<i32>() {
                        config.gpu_layers = val;
                    }
                    i += 1;
                }
            }
            "-t" | "--threads" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.threads = val;
                    }
                    i += 1;
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.port = val;
                    }
                    i += 1;
                }
            }
            "--host" => {
                if i + 1 < args.len() {
                    config.host = args[i + 1].to_string();
                    i += 1;
                }
            }
            "-np" | "--parallel" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.num_prompt_tracking = val;
                    }
                    i += 1;
                }
            }
            "--cache-type-k" => {
                if i + 1 < args.len() {
                    config.cache_type_k = args[i + 1].to_string();
                    i += 1;
                }
            }
            "--cache-type-v" => {
                if i + 1 < args.len() {
                    config.cache_type_v = args[i + 1].to_string();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    config
}

/// Parse vllm command-line arguments into ProviderConfig
pub fn parse_vllm_args(cmd_line: &str) -> ProviderConfig {
    let mut config = ProviderConfig::default();

    let args: Vec<&str> = cmd_line.split_whitespace().collect();
    let mut i = 0;

    while i < args.len() {
        let arg = args[i];

        match arg {
            "--model" => {
                if i + 1 < args.len() {
                    config.model_path = args[i + 1].to_string();
                    i += 1;
                }
            }
            "--dtype"
            | "--quantization"
            | "--tensor-parallel-size"
            | "--pipeline-parallel-size" => {
                i += 1;
            }
            "--gpu-memory-utilization" => {
                // vllm-specific, skip
                i += 1;
            }
            "--max-model-len" | "--context-len" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.context_size = val;
                    }
                    i += 1;
                }
            }
            "--host" => {
                if i + 1 < args.len() {
                    config.host = args[i + 1].to_string();
                    i += 1;
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<u32>() {
                        config.port = val as u16;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    config
}

/// Parse command-line arguments based on provider
pub fn parse_server_args(provider_id: &str, cmd_line: &str) -> ProviderConfig {
    match provider_id {
        "llama.cpp" => parse_llama_server_args(cmd_line),
        "vllm" => parse_vllm_args(cmd_line),
        _ => ProviderConfig::default(),
    }
}

/// Check if a process is still running
pub fn is_process_running(pid: u32) -> bool {
    // Try to send signal 0 to check if process exists
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llama_server_args() {
        let cmd = "llama-server -m /path/to/model.gguf -c 4096 -b 512 -ngl -1 -t 8 --port 8080";
        let config = parse_llama_server_args(cmd);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.gpu_layers, -1);
        assert_eq!(config.threads, 8);
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_parse_llama_server_args_with_long_flags() {
        let cmd =
            "llama-server --model /path/to/model.gguf --ctx-size 8192 --port 9000 --host 127.0.0.1";
        let config = parse_llama_server_args(cmd);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_size, 8192);
        assert_eq!(config.port, 9000);
        assert_eq!(config.host, "127.0.0.1");
    }
}
