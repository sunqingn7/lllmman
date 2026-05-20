use std::path::Path;
use std::process::Command;

use crate::services::config_persistence::load_provider_settings_for;

/// Return type for provider installation plan: (mode, commands, env vars, warnings, notes, summary)
pub type InstallPlan = (InstallMode, Vec<SetupCommand>, Vec<(String, String)>, Vec<String>, Vec<String>, String);

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

// ============================================================
// Dynamic Setup Plan Generation (Phase 3)
// ============================================================

/// Installation mode for a provider.
#[derive(Clone, Debug, PartialEq)]
pub enum InstallMode {
    PrecompiledWheel,
    SourceWithPrecompiledKernels,
    SourceFullBuild,
    HeterogeneousBuild,
}

impl InstallMode {
    pub fn label(&self) -> &'static str {
        match self {
            InstallMode::PrecompiledWheel => "Quick Install (Precompiled Wheel)",
            InstallMode::SourceWithPrecompiledKernels => "Development Install (Precompiled Kernels)",
            InstallMode::SourceFullBuild => "Full Source Build",
            InstallMode::HeterogeneousBuild => "Heterogeneous Build (Multi-Arch)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            InstallMode::PrecompiledWheel => "Fastest (~30s). Uses pre-built binaries.",
            InstallMode::SourceWithPrecompiledKernels => "Editable source with pre-built CUDA kernels (~2min).",
            InstallMode::SourceFullBuild => "Compile everything from source (~10-30min).",
            InstallMode::HeterogeneousBuild => "Compiles for ALL detected GPU architectures (~15-45min).",
        }
    }
}

/// A single setup command with metadata.
#[derive(Clone, Debug)]
pub struct SetupCommand {
    pub description: String,
    pub command: String,
    pub is_critical: bool,
    pub estimated_seconds: u32,
}

/// A dynamically generated setup plan for installing a provider.
#[derive(Clone, Debug)]
pub struct SetupPlan {
    pub provider: String,
    pub install_mode: InstallMode,
    pub commands: Vec<SetupCommand>,
    pub env_vars: Vec<(String, String)>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
    pub estimated_time: String,
}

/// Generate a setup plan based on detected GPUs and system state.
pub fn generate_setup_plan(
    provider_id: &str,
    gpu_archs: &[crate::services::GpuArchInfo],
    is_heterogeneous: bool,
) -> Option<SetupPlan> {
    let base_info = get_provider_install_info(provider_id)?;

    let (install_mode, commands, env_vars, warnings, recommendations, estimated_time) =
        match provider_id {
            "vllm" => generate_vllm_plan(gpu_archs, is_heterogeneous),
            "sglang" => generate_sglang_plan(gpu_archs, is_heterogeneous),
            "llama.cpp" => generate_llamacpp_plan(),
            _ => return None,
        };

    Some(SetupPlan {
        provider: base_info.provider_name.to_string(),
        install_mode,
        commands,
        env_vars,
        warnings,
        recommendations,
        estimated_time,
    })
}

fn generate_vllm_plan(
    gpu_archs: &[crate::services::GpuArchInfo],
    is_heterogeneous: bool,
) -> InstallPlan {
    let has_uv = Command::new("uv").arg("--version").output().is_ok();
    let pip_cmd = if has_uv { "uv pip" } else { "pip" };

    if is_heterogeneous {
        let arch_list = crate::services::gpu_arch::get_combined_arch_list(gpu_archs);
        let env_vars = vec![
            ("TORCH_CUDA_ARCH_LIST".to_string(), arch_list.clone()),
            ("FORCE_CUDA".to_string(), "1".to_string()),
            ("VLLM_SKIP_P2P_CHECK".to_string(), "1".to_string()),
        ];

        let commands = vec![
            SetupCommand {
                description: "Set multi-arch CUDA environment".to_string(),
                command: format!("export TORCH_CUDA_ARCH_LIST=\"{}\"", arch_list),
                is_critical: true,
                estimated_seconds: 1,
            },
            SetupCommand {
                description: "Install vLLM from source with multi-arch support".to_string(),
                command: format!(
                    "cd /path/to/vllm && {} install --no-build-isolation -e .",
                    pip_cmd
                ),
                is_critical: true,
                estimated_seconds: 1800,
            },
        ];

        let warnings = vec![
            "Heterogeneous GPUs detected. Tensor Parallelism will NOT work efficiently.".to_string(),
            "Multi-arch build takes 15-45 minutes.".to_string(),
        ];

        let recommendations = vec![
            "Use multi-instance mode with SGLang Router for heterogeneous setups.".to_string(),
            format!("TORCH_CUDA_ARCH_LIST=\"{}\" will be set automatically.", arch_list),
        ];

        (
            InstallMode::HeterogeneousBuild,
            commands,
            env_vars,
            warnings,
            recommendations,
            "15-45 minutes".to_string(),
        )
    } else {
        let commands = vec![
            SetupCommand {
                description: "Install vLLM via pip".to_string(),
                command: format!("{} install vllm --torch-backend=auto", pip_cmd),
                is_critical: true,
                estimated_seconds: 30,
            },
        ];

        (
            InstallMode::PrecompiledWheel,
            commands,
            vec![],
            vec![],
            vec!["Use --torch-backend=auto for optimal CUDA detection.".to_string()],
            "~30 seconds".to_string(),
        )
    }
}

fn generate_sglang_plan(
    gpu_archs: &[crate::services::GpuArchInfo],
    is_heterogeneous: bool,
) -> InstallPlan {
    let has_uv = Command::new("uv").arg("--version").output().is_ok();
    let pip_cmd = if has_uv { "uv pip" } else { "pip" };

    if is_heterogeneous {
        let arch_list = crate::services::gpu_arch::get_combined_arch_list(gpu_archs);
        let env_vars = vec![
            ("TORCH_CUDA_ARCH_LIST".to_string(), arch_list.clone()),
        ];

        let commands = vec![
            SetupCommand {
                description: "Set multi-arch CUDA environment".to_string(),
                command: format!("export TORCH_CUDA_ARCH_LIST=\"{}\"", arch_list),
                is_critical: true,
                estimated_seconds: 1,
            },
            SetupCommand {
                description: "Build SGLang kernel with multi-arch support".to_string(),
                command: "cd sglang/sgl-kernel && make build".to_string(),
                is_critical: true,
                estimated_seconds: 600,
            },
            SetupCommand {
                description: "Install SGLang Python package".to_string(),
                command: format!("cd sglang/python && {} install -e \".[all]\"", pip_cmd),
                is_critical: true,
                estimated_seconds: 120,
            },
        ];

        let warnings = vec![
            "Heterogeneous GPUs detected. Use multi-instance mode with SGLang Router.".to_string(),
        ];

        let recommendations = vec![
            format!("TORCH_CUDA_ARCH_LIST=\"{}\" will be set automatically.", arch_list),
            "Use --disable-cuda-graph for heterogeneous setups.".to_string(),
        ];

        (
            InstallMode::HeterogeneousBuild,
            commands,
            env_vars,
            warnings,
            recommendations,
            "10-20 minutes".to_string(),
        )
    } else {
        let commands = vec![
            SetupCommand {
                description: "Install SGLang via pip".to_string(),
                command: format!("{} install sglang --torch-backend=auto", pip_cmd),
                is_critical: true,
                estimated_seconds: 30,
            },
        ];

        (
            InstallMode::PrecompiledWheel,
            commands,
            vec![],
            vec![],
            vec![],
            "~30 seconds".to_string(),
        )
    }
}

fn generate_llamacpp_plan() -> InstallPlan {
    let commands = vec![
        SetupCommand {
            description: "Install llama-cpp-python via pip".to_string(),
            command: "pip install llama-cpp-python".to_string(),
            is_critical: true,
            estimated_seconds: 60,
        },
    ];

    (
        InstallMode::PrecompiledWheel,
        commands,
        vec![],
        vec![],
        vec![],
        "~1 minute".to_string(),
    )
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::models::GpuTier;
    use crate::services::GpuArchInfo;

    fn make_test_gpus() -> Vec<GpuArchInfo> {
        vec![
            GpuArchInfo {
                index: 0,
                name: "RTX 3090".to_string(),
                compute_capability: Some((8, 6)),
                sm_string: "sm_86".to_string(),
                vram_mb: 24 * 1024,
                performance_tier: GpuTier::Mid,
            },
            GpuArchInfo {
                index: 1,
                name: "RTX PRO 6000".to_string(),
                compute_capability: Some((12, 0)),
                sm_string: "sm_120".to_string(),
                vram_mb: 96 * 1024,
                performance_tier: GpuTier::Ultra,
            },
        ]
    }

    fn make_single_gpu() -> Vec<GpuArchInfo> {
        vec![GpuArchInfo {
            index: 0,
            name: "RTX 3090".to_string(),
            compute_capability: Some((8, 6)),
            sm_string: "sm_86".to_string(),
            vram_mb: 24 * 1024,
            performance_tier: GpuTier::Mid,
        }]
    }

    #[test]
    fn test_generate_vllm_plan_heterogeneous() {
        let plan = generate_setup_plan("vllm", &make_test_gpus(), true).unwrap();
        assert_eq!(plan.install_mode, InstallMode::HeterogeneousBuild);
        assert!(!plan.env_vars.is_empty());
        assert!(plan.env_vars.iter().any(|(k, _)| k == "TORCH_CUDA_ARCH_LIST"));
        assert!(!plan.warnings.is_empty());
        assert!(!plan.recommendations.is_empty());
    }

    #[test]
    fn test_generate_vllm_plan_single() {
        let plan = generate_setup_plan("vllm", &make_single_gpu(), false).unwrap();
        assert_eq!(plan.install_mode, InstallMode::PrecompiledWheel);
        assert!(plan.env_vars.is_empty());
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn test_generate_sglang_plan_heterogeneous() {
        let plan = generate_setup_plan("sglang", &make_test_gpus(), true).unwrap();
        assert_eq!(plan.install_mode, InstallMode::HeterogeneousBuild);
        assert!(plan.commands.len() >= 2);
    }

    #[test]
    fn test_generate_sglang_plan_single() {
        let plan = generate_setup_plan("sglang", &make_single_gpu(), false).unwrap();
        assert_eq!(plan.install_mode, InstallMode::PrecompiledWheel);
        assert_eq!(plan.commands.len(), 1);
    }

    #[test]
    fn test_generate_llamacpp_plan() {
        let plan = generate_setup_plan("llama.cpp", &make_test_gpus(), true).unwrap();
        assert_eq!(plan.provider, "llama.cpp");
        assert_eq!(plan.commands.len(), 1);
    }

    #[test]
    fn test_generate_unknown_provider() {
        assert!(generate_setup_plan("unknown", &make_test_gpus(), true).is_none());
    }

    #[test]
    fn test_install_mode_labels() {
        assert_eq!(InstallMode::PrecompiledWheel.label(), "Quick Install (Precompiled Wheel)");
        assert_eq!(InstallMode::HeterogeneousBuild.label(), "Heterogeneous Build (Multi-Arch)");
    }
}
