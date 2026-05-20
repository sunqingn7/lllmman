use std::process::Command;

use crate::models::GpuInfo;
use crate::services::gpu_arch::{detect_gpu_architectures, get_cuda_driver_version};

/// Report on CUDA compatibility for the current system.
pub struct CudaCompatibilityReport {
    pub driver_version: String,
    pub runtime_version: Option<String>,
    pub torch_installed: bool,
    pub torch_cuda_version: Option<String>,
    pub torch_version: Option<String>,
    pub uv_available: bool,
    pub pip_available: bool,
    pub python_version: Option<String>,
    pub max_supported_sm: String,
    pub gpu_archs: Vec<String>,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Check CUDA compatibility for the detected GPUs.
pub fn check_cuda_compatibility(gpus: &[GpuInfo]) -> CudaCompatibilityReport {
    let mut report = CudaCompatibilityReport {
        driver_version: "unknown".to_string(),
        runtime_version: None,
        torch_installed: false,
        torch_cuda_version: None,
        torch_version: None,
        uv_available: false,
        pip_available: false,
        python_version: None,
        max_supported_sm: "unknown".to_string(),
        gpu_archs: Vec::new(),
        issues: Vec::new(),
        recommendations: Vec::new(),
    };

    // 1. Get CUDA driver version
    if let Some(driver_ver) = get_cuda_driver_version() {
        report.driver_version = driver_ver.clone();
    } else {
        report.issues.push(
            "Could not detect NVIDIA driver version. Is nvidia-smi available?".to_string(),
        );
    }

    // 2. Check Python availability
    if let Ok(output) = Command::new("python3").args(["--version"]).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            report.python_version = Some(stdout.trim().to_string());
        }
    }

    // 3. Check if torch is installed and what CUDA it was built with
    if let Ok(output) = Command::new("python3")
        .args([
            "-c",
            "import torch; print(torch.__version__); print(getattr(torch.version, 'cuda', 'None'))",
        ])
        .output()
    {
        if output.status.success() {
            report.torch_installed = true;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            if !lines.is_empty() {
                report.torch_version = Some(lines[0].to_string());
            }
            if lines.len() > 1 && lines[1] != "None" {
                report.torch_cuda_version = Some(lines[1].to_string());
            }
        }
    }

    // 4. Check uv availability
    report.uv_available = Command::new("uv").arg("--version").output().is_ok();

    // 5. Check pip availability
    report.pip_available = Command::new("pip3").arg("--version").output().is_ok();

    // 6. Analyze GPU architectures
    let arch_infos = detect_gpu_architectures(gpus);
    for arch in &arch_infos {
        report.gpu_archs.push(arch.sm_string.clone());
    }

    // 7. Determine max supported SM
    if let Some(max_arch) = arch_infos
        .iter()
        .filter_map(|a| a.compute_capability)
        .max()
    {
        report.max_supported_sm = format!("sm_{}{}", max_arch.0, max_arch.1);
    }

    // 8. Run compatibility checks
    check_driver_compatibility(&mut report, &arch_infos);
    check_torch_compatibility(&mut report);
    check_build_requirements(&mut report, &arch_infos);

    report
}

/// Check if CUDA driver is new enough for all detected GPUs.
fn check_driver_compatibility(report: &mut CudaCompatibilityReport, arch_infos: &[crate::services::GpuArchInfo]) {
    let driver_ver = &report.driver_version;
    if driver_ver == "unknown" {
        return;
    }

    for arch in arch_infos {
        if let Some((major, minor)) = arch.compute_capability {
            let required_driver = min_driver_for_sm(major, minor);
            if let Some(required) = required_driver {
                if compare_driver_versions(driver_ver, &required) < 0 {
                    report.issues.push(format!(
                        "CUDA driver {} is too old for {} (SM {}.{}). Upgrade to driver {}+.",
                        driver_ver, arch.name, major, minor, required
                    ));
                    report.recommendations.push(format!(
                        "Upgrade NVIDIA driver to {}+ for {} support",
                        required, arch.name
                    ));
                }
            }
        }
    }
}

/// Check PyTorch CUDA compatibility.
fn check_torch_compatibility(report: &mut CudaCompatibilityReport) {
    if !report.torch_installed {
        report.issues.push("PyTorch is not installed.".to_string());
        report.recommendations.push(
            "Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124".to_string(),
        );
        return;
    }

    if let Some(torch_cuda) = &report.torch_cuda_version {
        let torch_cuda_num: f32 = torch_cuda.replace("cu", "").parse().unwrap_or(0.0);

        for sm_str in &report.gpu_archs {
            if sm_str.starts_with("sm_12") && torch_cuda_num < 12.8 {
                report.issues.push(format!(
                    "PyTorch built with cu{}; {} requires cu128+.",
                    torch_cuda_num as u32, sm_str
                ));
                report.recommendations.push(
                    "Install torch with cu128: pip install torch --index-url https://download.pytorch.org/whl/cu128".to_string(),
                );
            }
        }
    }
}

/// Check build tool availability.
fn check_build_requirements(report: &mut CudaCompatibilityReport, arch_infos: &[crate::services::GpuArchInfo]) {
    if !report.uv_available && !report.pip_available {
        report.issues.push("Neither uv nor pip3 is available.".to_string());
        report.recommendations.push(
            "Install uv (recommended): pip install uv, or use pip3 directly".to_string(),
        );
    } else if !report.uv_available {
        report.recommendations.push(
            "Consider installing uv for faster package installation: pip install uv".to_string(),
        );
    }

    let has_heterogeneous = arch_infos.len() > 1
        && arch_infos
            .iter()
            .filter_map(|a| a.compute_capability)
            .collect::<std::collections::HashSet<_>>()
            .len() > 1;

    if has_heterogeneous {
        report.recommendations.push(
            "Heterogeneous GPUs detected. Use multi-arch build with TORCH_CUDA_ARCH_LIST.".to_string(),
        );
    }
}

/// Get minimum NVIDIA driver version required for a given SM version.
fn min_driver_for_sm(major: u32, minor: u32) -> Option<&'static str> {
    match (major, minor) {
        (12, 0) => Some("570"),
        (11, _) => Some("560"),
        (10, _) => Some("550"),
        (9, 0) => Some("525"),
        (8, 6..=9) => Some("470"),
        (8, 0) => Some("450"),
        (7, 5) => Some("410"),
        _ => None,
    }
}

/// Compare two driver version strings. Returns -1, 0, or 1.
fn compare_driver_versions(a: &str, b: &str) -> i32 {
    let parse = |v: &str| -> Vec<u32> {
        v.split('.')
            .filter_map(|s| s.parse().ok())
            .collect()
    };

    let a_parts = parse(a);
    let b_parts = parse(b);

    let max_len = a_parts.len().max(b_parts.len());

    for i in 0..max_len {
        let a_val = a_parts.get(i).copied().unwrap_or(0);
        let b_val = b_parts.get(i).copied().unwrap_or(0);

        if a_val < b_val {
            return -1;
        }
        if a_val > b_val {
            return 1;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_driver_for_sm() {
        assert_eq!(min_driver_for_sm(12, 0), Some("570"));
        assert_eq!(min_driver_for_sm(9, 0), Some("525"));
        assert_eq!(min_driver_for_sm(8, 6), Some("470"));
        assert_eq!(min_driver_for_sm(8, 9), Some("470"));
        assert_eq!(min_driver_for_sm(7, 5), Some("410"));
        assert_eq!(min_driver_for_sm(6, 1), None);
    }

    #[test]
    fn test_compare_driver_versions() {
        assert_eq!(compare_driver_versions("570.124", "570"), 1);
        assert_eq!(compare_driver_versions("570", "570"), 0);
        assert_eq!(compare_driver_versions("550", "570"), -1);
        assert_eq!(compare_driver_versions("570.100", "570.99"), 1);
        assert_eq!(compare_driver_versions("525.85", "525.85.05"), -1);
    }

    #[test]
    fn test_check_cuda_compatibility_basic() {
        let gpus = vec![
            GpuInfo {
                name: "NVIDIA GeForce RTX 3090".to_string(),
                total_vram_mb: 24 * 1024,
                index: 0,
                provider: crate::models::GpuProvider::Nvidia,
                temperature_c: Some(60.0),
                compute_capability: Some((8, 6)),
                performance_tier: crate::models::GpuTier::Mid,
            },
        ];

        let report = check_cuda_compatibility(&gpus);
        assert!(!report.driver_version.is_empty());
        assert!(!report.gpu_archs.is_empty());
        assert!(report.gpu_archs.contains(&"sm_86".to_string()));
    }

    #[test]
    fn test_check_cuda_compatibility_heterogeneous() {
        let gpus = vec![
            GpuInfo {
                name: "NVIDIA GeForce RTX 3090".to_string(),
                total_vram_mb: 24 * 1024,
                index: 0,
                provider: crate::models::GpuProvider::Nvidia,
                temperature_c: Some(60.0),
                compute_capability: Some((8, 6)),
                performance_tier: crate::models::GpuTier::Mid,
            },
            GpuInfo {
                name: "NVIDIA RTX PRO 6000".to_string(),
                total_vram_mb: 96 * 1024,
                index: 1,
                provider: crate::models::GpuProvider::Nvidia,
                temperature_c: Some(55.0),
                compute_capability: Some((12, 0)),
                performance_tier: crate::models::GpuTier::Ultra,
            },
        ];

        let report = check_cuda_compatibility(&gpus);
        assert!(report.gpu_archs.contains(&"sm_86".to_string()));
        assert!(report.gpu_archs.contains(&"sm_120".to_string()));
        assert_eq!(report.max_supported_sm, "sm_120");
    }
}
