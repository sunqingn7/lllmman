use crate::core::ProviderConfig;

/// Parse a complete command line string and extract known parameters into ProviderConfig
pub fn parse_command_line(cmd_line: &str) -> ProviderConfig {
    let mut config = ProviderConfig::default();
    let args: Vec<&str> = cmd_line.split_whitespace().collect();
    let mut i = 0;

    while i < args.len() {
        let arg = args[i];

        match arg {
            "--model" | "--model-path" | "-m" => {
                if i + 1 < args.len() {
                    config.model_path = args[i + 1].to_string();
                    i += 1;
                }
            }
            "--ctx-size" | "-c" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.context_size = val;
                    }
                    i += 1;
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.batch_size = val;
                    }
                    i += 1;
                }
            }
            "--n-gpu-layers" | "-ngl" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.gpu_layers = val;
                    }
                    i += 1;
                }
            }
            "--threads" | "-t" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.threads = val;
                    }
                    i += 1;
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.port = val;
                    }
                    i += 1;
                }
            }
            "--host" | "--host-address" => {
                if i + 1 < args.len() {
                    config.host = args[i + 1].to_string();
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
            "--parallel" | "-np" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.num_prompt_tracking = val;
                    }
                    i += 1;
                }
            }
            // For SGLang
            "--mem-fraction-static" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.gpu_layers = (val * 100.0) as i32;
                    }
                    i += 1;
                }
            }
            "--context-length" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.context_size = val;
                    }
                    i += 1;
                }
            }
            // For vLLM
            "--max-num-batched-tokens" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.batch_size = val;
                    }
                    i += 1;
                }
            }
            // Sampling parameters (common to all providers)
            "--temperature" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.temperature = Some(val);
                    }
                    i += 1;
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<i32>() {
                        config.top_k = Some(val);
                    }
                    i += 1;
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.top_p = Some(val);
                    }
                    i += 1;
                }
            }
            "--min-p" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.min_p = Some(val);
                    }
                    i += 1;
                }
            }
            "--presence-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.presence_penalty = Some(val);
                    }
                    i += 1;
                }
            }
            "--repetition-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        config.repetition_penalty = Some(val);
                    }
                    i += 1;
                }
            }
            "--max-model-len" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        config.context_size = val;
                    }
                    i += 1;
                }
            }
            "--tokenizer" | "--tokenizer-path" => {
                if i + 1 < args.len() {
                    config.tokenizer = args[i + 1].to_string();
                    i += 1;
                }
            }
            // --max-num-seqs is vLLM's max concurrent sequences, no matching ProviderConfig field
            // Collect unrecognized args as additional_args
            _ => {
                // If it looks like a path and no model_path set yet, treat as model path
                if arg.contains('/') && config.model_path.is_empty() {
                    config.model_path = arg.to_string();
                } else if arg.starts_with('-') {
                    let mut additional = if config.additional_args.is_empty() {
                        String::new()
                    } else {
                        format!("{} ", config.additional_args)
                    };
                    additional.push_str(arg);
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        additional.push(' ');
                        additional.push_str(args[i + 1]);
                        i += 1;
                    }
                    config.additional_args = additional;
                }
            }
        }
        i += 1;
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llama_cpp_command() {
        let cmd = "llama-server -m /path/to/model.gguf -c 8192 -ngl 35 -t 4 --port 8080";
        let config = parse_command_line(cmd);
        assert_eq!(config.model_path, "/path/to/model.gguf");
        assert_eq!(config.context_size, 8192);
        assert_eq!(config.gpu_layers, 35);
        assert_eq!(config.threads, 4);
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_parse_sglang_command() {
        let cmd = "python3 -m sglang.launch_server --model-path /path/to/model --port 30000 --mem-fraction-static 0.8";
        let config = parse_command_line(cmd);
        assert_eq!(config.model_path, "/path/to/model");
        assert_eq!(config.port, 30000);
        assert_eq!(config.gpu_layers, 80);
    }

    #[test]
    fn test_parse_vllm_command() {
        let cmd = "vllm serve /path/to/model --port 8000 --gpu-memory-utilization 0.9 --max-model-len 4096";
        let config = parse_command_line(cmd);
        assert_eq!(config.model_path, "/path/to/model");
        assert_eq!(config.port, 8000);
        assert_eq!(config.gpu_layers, 90);
        assert_eq!(config.context_size, 4096);
    }
}
