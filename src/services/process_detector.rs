use crate::core::{ProviderConfig, ProviderRegistry};

pub struct DetectedServer {
    pub pid: u32,
    pub binary: String,
    pub command_line: String,
    pub provider_id: String,
}

pub fn detect_running_servers() -> Vec<DetectedServer> {
    let mut servers = Vec::new();

    for (id, _name) in ProviderRegistry::list() {
        if let Some(provider) = ProviderRegistry::get(id) {
            for server in provider.detect_running_servers() {
                servers.push(DetectedServer {
                    pid: server.pid,
                    binary: server.binary.clone(),
                    command_line: server.command_line.clone(),
                    provider_id: id.to_string(),
                });
            }
        }
    }

    servers
}

pub fn parse_server_args(provider_id: &str, cmd_line: &str) -> ProviderConfig {
    if let Some(provider) = ProviderRegistry::get(provider_id) {
        provider.parse_server_config(cmd_line)
    } else {
        ProviderConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_providers() {
        crate::providers::register_all_providers();
    }

    #[test]
    fn test_parse_llama_server_args() {
        setup_providers();
        let cmd = "llama-server -m /path/to/model.gguf -c 4096 -b 512 -ngl -1 -t 8 --port 8080";
        let config = parse_server_args("llama.cpp", cmd);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.gpu_layers, -1);
        assert_eq!(config.threads, 8);
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_parse_llama_server_args_with_long_flags() {
        setup_providers();
        let cmd =
            "llama-server --model /path/to/model.gguf --ctx-size 8192 --port 9000 --host 127.0.0.1";
        let config = parse_server_args("llama.cpp", cmd);

        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_size, 8192);
        assert_eq!(config.port, 9000);
        assert_eq!(config.host, "127.0.0.1");
    }
}
