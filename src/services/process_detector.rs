use std::collections::HashSet;

use crate::core::{ProviderConfig, ProviderRegistry};

pub struct DetectedServer {
    pub pid: u32,
    pub binary: String,
    pub command_line: String,
    pub provider_id: String,
}

pub fn detect_running_servers() -> Vec<DetectedServer> {
    let mut servers = Vec::new();
    let mut seen_pids = HashSet::new();

    for (id, _name) in ProviderRegistry::list() {
        if let Some(provider) = ProviderRegistry::get(id) {
            for server in provider.detect_running_servers() {
                if seen_pids.insert(server.pid) {
                    servers.push(DetectedServer {
                        pid: server.pid,
                        binary: server.binary.clone(),
                        command_line: server.command_line.clone(),
                        provider_id: id.to_string(),
                    });
                }
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
#[cfg(test)]
mod vllm_detect_tests {
    use super::*;

    #[test]
    fn parse_live_vllm_command() {
        crate::providers::register_all_providers();
        let cmd = r#"20584 /home/qing/Project/vllm/.venv/bin/python3 /home/qing/Project/vllm/.venv/bin/vllm serve RadixArk/Qwen3.8-Flash-Next-NVFP4 --served-model-name Qwen/Qwen3.8-Flash-Next --host 0.0.0.0 --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.96 --tensor-parallel-size 1 --distributed-executor-backend mp --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype auto --enable-prefix-caching --no-enable-flashinfer-autotune --speculative-config {"method":"mtp","num_speculative_tokens":3} --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"#;
        let c = parse_server_args("vllm", cmd);
        println!("model={} ctx={} batch={} port={} host={} gmulayer={} cache={} extra=[{}]",
            c.model_path, c.context_size, c.batch_size, c.port, c.host, c.gpu_layers, c.cache_type_k, c.additional_args);
        assert_eq!(c.model_path, "RadixArk/Qwen3.8-Flash-Next-NVFP4");
        assert_eq!(c.context_size, 262144);
        assert_eq!(c.batch_size, 8192);
        assert_eq!(c.port, 8000);
        assert_eq!(c.gpu_layers, 96);
        assert_eq!(c.cache_type_k, "auto");
        assert!(c.additional_args.contains("--max-num-seqs 2"));
        assert!(c.additional_args.contains("--tool-call-parser qwen3_coder"));
    }
}
