use std::process::Command;

#[allow(dead_code)]
pub enum InstallMethod {
    Simple,
    Advanced,
}

pub struct ProviderInstallInfo {
    pub provider_id: &'static str,
    pub provider_name: &'static str,
    pub binary_name: &'static str,
    pub simple_command: &'static str,
    pub simple_description: &'static str,
    pub advanced_command: &'static str,
    pub advanced_description: &'static str,
    pub check_command: &'static str,
    pub check_args: &'static [&'static str],
}

impl ProviderInstallInfo {
    #[allow(dead_code)]
    pub fn simple_description(&self) -> &str {
        self.simple_description
    }
    pub fn advanced_command(&self) -> &str {
        self.advanced_command
    }
    #[allow(dead_code)]
    pub fn advanced_description(&self) -> &str {
        self.advanced_description
    }
    pub fn check_command(&self) -> &str {
        self.check_command
    }
    pub fn check_args(&self) -> &[&str] {
        self.check_args
    }
}

pub fn get_provider_install_info(provider_id: &str) -> Option<ProviderInstallInfo> {
    match provider_id {
        "llama.cpp" => Some(ProviderInstallInfo {
            provider_id: "llama.cpp",
            provider_name: "llama.cpp",
            binary_name: "llama-server",
            simple_command: "pip install llama-cpp-python",
            simple_description: "Install via pip (includes llama-server binary)",
            advanced_command: "git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && cmake -B build && cmake --build build --config Release",
            advanced_description: "Clone from GitHub and compile with cmake (full features, best performance)",
            check_command: "llama-server",
            check_args: &["--version"],
        }),
        "vllm" => Some(ProviderInstallInfo {
            provider_id: "vllm",
            provider_name: "vLLM",
            binary_name: "vllm",
            simple_command: "pip install vllm",
            simple_description: "Install via pip",
            advanced_command: "git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -e .",
            advanced_description: "Clone from GitHub and install in editable mode (development version)",
            check_command: "vllm",
            check_args: &["--version"],
        }),
        "sglang" => Some(ProviderInstallInfo {
            provider_id: "sglang",
            provider_name: "SGLang",
            binary_name: "python3",
            simple_command: "pip install sglang",
            simple_description: "Install via pip",
            advanced_command: "git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e \"python[all]\"",
            advanced_description: "Clone from GitHub and install with all dependencies (latest features)",
            check_command: "python3",
            check_args: &["-m", "sglang.launch_server", "--help"],
        }),
        _ => None,
    }
}

pub fn check_provider_installed(provider_id: &str) -> bool {
    let info = match get_provider_install_info(provider_id) {
        Some(i) => i,
        None => return false,
    };

    let output = Command::new(info.check_command())
        .args(info.check_args())
        .output();

    match output {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}
