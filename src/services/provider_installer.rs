use std::path::Path;
use std::process::Command;

use crate::services::config_persistence::load_provider_settings_for;

pub struct ProviderInstallInfo {
    pub provider_name: &'static str,
    pub simple_command: &'static str,
    pub simple_description: &'static str,
    pub advanced_command: &'static str,
    pub advanced_description: &'static str,
    pub check_command: &'static str,
    pub check_args: &'static [&'static str],
}

pub fn get_provider_install_info(provider_id: &str) -> Option<ProviderInstallInfo> {
    match provider_id {
        "llama.cpp" => Some(ProviderInstallInfo {
            provider_name: "llama.cpp",
            simple_command: "pip install llama-cpp-python",
            simple_description: "Install via pip (includes llama-server binary)",
            advanced_command: "git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && cmake -B build && cmake --build build --config Release",
            advanced_description: "Clone from GitHub and compile with cmake (full features, best performance)",
            check_command: "llama-server",
            check_args: &["--version"],
        }),
        "vllm" => Some(ProviderInstallInfo {
            provider_name: "vLLM",
            simple_command: "pip install vllm",
            simple_description: "Install via pip",
            advanced_command: "git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -e .",
            advanced_description: "Clone from GitHub and install in editable mode (development version)",
            check_command: "vllm",
            check_args: &["--version"],
        }),
        "sglang" => Some(ProviderInstallInfo {
            provider_name: "SGLang",
            simple_command: "pip install sglang",
            simple_description: "Install via pip",
            advanced_command: "git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e \"python[all]\"",
            advanced_description: "Clone from GitHub and install with all dependencies (latest features)",
            check_command: "sglang",
            check_args: &["serve", "--help"],
        }),
        _ => None,
    }
}

pub fn check_provider_installed(provider_id: &str) -> bool {
    let info = match get_provider_install_info(provider_id) {
        Some(i) => i,
        None => return false,
    };

    let settings = load_provider_settings_for(provider_id);

    let check_binary = if settings.binary_path.is_empty() {
        info.check_command.to_string()
    } else {
        settings.binary_path.clone()
    };

    let result = if !settings.binary_path.is_empty() && Path::new(&check_binary).exists() {
        let parts: Vec<&str> = check_binary.split_whitespace().collect();
        if parts.len() > 1 {
            Command::new(parts[0])
                .args(parts[1..].iter().chain(info.check_args.iter()))
                .output()
        } else {
            Command::new(&check_binary).args(info.check_args).output()
        }
    } else if !settings.env_script.is_empty() {
        Command::new("bash")
            .args([
                "-c",
                &format!(
                    "source \"{}\" && {} {}",
                    settings.env_script,
                    check_binary,
                    info.check_args.join(" ")
                ),
            ])
            .output()
    } else {
        let parts: Vec<&str> = check_binary.split_whitespace().collect();
        if parts.len() > 1 {
            Command::new(parts[0])
                .args(parts[1..].iter().chain(info.check_args.iter()))
                .output()
        } else {
            Command::new(&check_binary).args(info.check_args).output()
        }
    };

    match result {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}
