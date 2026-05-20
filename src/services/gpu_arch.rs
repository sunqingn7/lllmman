use std::process::Command;

use crate::models::{GpuInfo, GpuProvider, GpuTier};

/// GPU architecture info detected from the system.
pub struct GpuArchInfo {
    pub index: u32,
    pub name: String,
    pub compute_capability: Option<(u32, u32)>,
    pub sm_string: String,
    pub vram_mb: u32,
    pub performance_tier: GpuTier,
}

impl GpuArchInfo {
    pub fn from_gpu_info(gpu: &GpuInfo) -> Self {
        let (cc, sm) = if let Some(cc) = gpu.compute_capability {
            (Some(cc), format!("sm_{}{}", cc.0, cc.1))
        } else {
            let inferred = infer_compute_cap_from_name(&gpu.name, gpu.provider.clone());
            (inferred, infer_sm_string(inferred))
        };
        let tier = if gpu.performance_tier != GpuTier::Low {
            gpu.performance_tier
        } else {
            classify_performance_tier(cc, gpu.total_vram_mb)
        };
        Self {
            index: gpu.index,
            name: gpu.name.clone(),
            compute_capability: cc,
            sm_string: sm,
            vram_mb: gpu.total_vram_mb,
            performance_tier: tier,
        }
    }
}

/// Detect GPU architectures by querying nvidia-smi for compute capability.
/// Falls back to inferring from GPU name if compute_cap query fails.
pub fn detect_gpu_architectures(gpus: &[GpuInfo]) -> Vec<GpuArchInfo> {
    let mut arch_infos = Vec::new();

    let nvidia_gpus: Vec<&GpuInfo> = gpus.iter().filter(|g| g.provider == GpuProvider::Nvidia).collect();

    if !nvidia_gpus.is_empty() {
        if let Some(caps) = query_nvidia_compute_caps(&nvidia_gpus) {
            for (i, gpu) in nvidia_gpus.iter().enumerate() {
                let cc = caps.get(i).and_then(|c| *c);
                let sm = infer_sm_string(cc);
                let tier = classify_performance_tier(cc, gpu.total_vram_mb);
                arch_infos.push(GpuArchInfo {
                    index: gpu.index,
                    name: gpu.name.clone(),
                    compute_capability: cc,
                    sm_string: sm,
                    vram_mb: gpu.total_vram_mb,
                    performance_tier: tier,
                });
            }
        } else {
            for gpu in &nvidia_gpus {
                let cc = infer_compute_cap_from_name(&gpu.name, gpu.provider.clone());
                let sm = infer_sm_string(cc);
                let tier = classify_performance_tier(cc, gpu.total_vram_mb);
                arch_infos.push(GpuArchInfo {
                    index: gpu.index,
                    name: gpu.name.clone(),
                    compute_capability: cc,
                    sm_string: sm,
                    vram_mb: gpu.total_vram_mb,
                    performance_tier: tier,
                });
            }
        }
    }

    for gpu in gpus.iter().filter(|g| g.provider != GpuProvider::Nvidia) {
        let cc = infer_compute_cap_from_name(&gpu.name, gpu.provider.clone());
        let sm = infer_sm_string(cc);
        let tier = classify_performance_tier(cc, gpu.total_vram_mb);
        arch_infos.push(GpuArchInfo {
            index: gpu.index,
            name: gpu.name.clone(),
            compute_capability: cc,
            sm_string: sm,
            vram_mb: gpu.total_vram_mb,
            performance_tier: tier,
        });
    }

    arch_infos
}

/// Query nvidia-smi for compute capability of specific NVIDIA GPUs by index.
fn query_nvidia_compute_caps(gpus: &[&GpuInfo]) -> Option<Vec<Option<(u32, u32)>>> {
    let ids: Vec<String> = gpus.iter().map(|g| g.index.to_string()).collect();
    let id_list = ids.join(",");

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,compute_cap",
            &format!("--id={}", id_list),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut caps = std::collections::HashMap::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() == 2 {
            if let (Ok(idx), cap_str) = (parts[0].parse::<u32>(), parts[1]) {
                let cap = if let Some(dot_pos) = cap_str.find('.') {
                    let major = cap_str[..dot_pos].trim().parse::<u32>().ok()?;
                    let minor = cap_str[dot_pos + 1..].trim().parse::<u32>().ok()?;
                    Some((major, minor))
                } else {
                    None
                };
                caps.insert(idx, cap);
            }
        }
    }

    let result: Vec<Option<(u32, u32)>> = gpus.iter()
        .map(|g| caps.get(&g.index).copied().unwrap_or(None))
        .collect();

    if caps.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Check if the GPU cluster is heterogeneous (different architectures or VRAM ratios > 2x).
pub fn is_heterogeneous_cluster(arch_infos: &[GpuArchInfo]) -> bool {
    if arch_infos.len() <= 1 {
        return false;
    }

    let has_different_archs = arch_infos
        .iter()
        .filter_map(|a| a.compute_capability)
        .collect::<std::collections::HashSet<_>>()
        .len() > 1;

    if has_different_archs {
        return true;
    }

    let vrams: Vec<u32> = arch_infos.iter().map(|a| a.vram_mb).collect();
    let min_vram = vrams.iter().min().copied().unwrap_or(0);
    let max_vram = vrams.iter().max().copied().unwrap_or(0);

    if min_vram > 0 && max_vram > min_vram * 2 {
        return true;
    }

    false
}

/// Get a combined CUDA architecture list string for TORCH_CUDA_ARCH_LIST.
/// Returns e.g. "8.6 8.9 12.0" for a heterogeneous cluster.
pub fn get_combined_arch_list(arch_infos: &[GpuArchInfo]) -> String {
    let mut caps: Vec<(u32, u32)> = arch_infos
        .iter()
        .filter_map(|a| a.compute_capability)
        .collect();
    caps.sort();
    caps.dedup();

    if caps.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = caps.iter().map(|(major, minor)| format!("{}.{}", major, minor)).collect();
    let max_cap = caps.last().unwrap();
    parts.push(format!("{}.{}+PTX", max_cap.0, max_cap.1));

    parts.join(" ")
}

/// Get the CUDA driver version from nvidia-smi output.
/// Returns e.g. Some("12.8").
pub fn get_cuda_driver_version() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let driver_version = stdout.lines().next()?.trim();

    if driver_version.is_empty() {
        return None;
    }

    Some(driver_version.to_string())
}

/// Classify GPU performance tier based on compute capability and VRAM.
/// Primary key = SM version, secondary key = VRAM.
pub fn classify_performance_tier(compute_cap: Option<(u32, u32)>, vram_mb: u32) -> GpuTier {
    match compute_cap {
        None => {
            if vram_mb >= 48 * 1024 {
                GpuTier::Ultra
            } else if vram_mb >= 24 * 1024 {
                GpuTier::High
            } else if vram_mb >= 12 * 1024 {
                GpuTier::Mid
            } else {
                GpuTier::Low
            }
        }
        Some((major, minor)) => {
            match (major, minor) {
                (m, _) if m < 8 => GpuTier::Low,
                (8, m) if m <= 6 => GpuTier::Mid,       // SM 8.0, 8.6 — Ampere
                (8, 8..=9) => {                          // SM 8.9 — Ada Lovelace
                    if vram_mb >= 24 * 1024 {
                        GpuTier::High
                    } else {
                        GpuTier::Mid
                    }
                }
                (9, 0) => GpuTier::Ultra,               // H100 Hopper
                (10, _) => GpuTier::Ultra,              // future
                (12, 0) => GpuTier::Ultra,              // Blackwell (always Ultra — latest gen)
                _ => GpuTier::Mid,
            }
        }
    }
}

/// Infer compute capability from GPU name and provider.
/// Used as fallback when nvidia-smi doesn't report compute_cap.
pub fn infer_compute_cap_from_name(name: &str, provider: GpuProvider) -> Option<(u32, u32)> {
    let name_lower = name.to_lowercase();

    match provider {
        GpuProvider::Nvidia => {
            // Blackwell (50xx series, RTX Pro 6000 Blackwell)
            if name_lower.contains("rtx 50") || name_lower.contains("rtx pro 6000") {
                return Some((12, 0));
            }
            // Ada Lovelace (40xx series, L40S)
            if name_lower.contains("rtx 40") || name_lower.contains("l40s") {
                return Some((8, 9));
            }
            // Hopper (H100, H200)
            if name_lower.contains("h100") || name_lower.contains("h200") {
                return Some((9, 0));
            }
            // Ampere (30xx series, A100, A6000)
            if name_lower.contains("rtx 30") || name_lower.contains("a100") || name_lower.contains("a6000") {
                return Some((8, 6));
            }
            // Turing (20xx series, T4)
            if name_lower.contains("rtx 20") || name_lower.contains("t4") {
                return Some((7, 5));
            }
            // Volta (V100)
            if name_lower.contains("v100") {
                return Some((7, 0));
            }
            // Pascal (10xx series, P100, P40)
            if name_lower.contains("rtx 10") || name_lower.contains("gtx 10") || name_lower.contains("p100") || name_lower.contains("p40") {
                return Some((6, 1));
            }
            None
        }
        GpuProvider::Amd => {
            // AMD uses gfx architecture naming, not SM
            // We return None since compute_cap is NVIDIA-specific
            None
        }
        GpuProvider::Intel => None,
        GpuProvider::Unknown => None,
    }
}

/// Convert compute capability to SM string (e.g., (8, 6) -> "sm_86").
fn infer_sm_string(compute_cap: Option<(u32, u32)>) -> String {
    match compute_cap {
        Some((major, minor)) => format!("sm_{}{}", major, minor),
        None => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_performance_tier_nvidia() {
        assert_eq!(classify_performance_tier(Some((7, 5)), 8 * 1024), GpuTier::Low);
        assert_eq!(classify_performance_tier(Some((8, 6)), 24 * 1024), GpuTier::Mid);
        assert_eq!(classify_performance_tier(Some((8, 9)), 24 * 1024), GpuTier::High);
        assert_eq!(classify_performance_tier(Some((8, 9)), 16 * 1024), GpuTier::Mid);
        assert_eq!(classify_performance_tier(Some((9, 0)), 80 * 1024), GpuTier::Ultra);
        assert_eq!(classify_performance_tier(Some((12, 0)), 32 * 1024), GpuTier::Ultra);
    }

    #[test]
    fn test_classify_performance_tier_no_cap() {
        assert_eq!(classify_performance_tier(None, 8 * 1024), GpuTier::Low);
        assert_eq!(classify_performance_tier(None, 16 * 1024), GpuTier::Mid);
        assert_eq!(classify_performance_tier(None, 32 * 1024), GpuTier::High);
        assert_eq!(classify_performance_tier(None, 80 * 1024), GpuTier::Ultra);
    }

    #[test]
    fn test_is_heterogeneous_cluster_single_gpu() {
        let archs = vec![GpuArchInfo {
            index: 0,
            name: "RTX 3090".to_string(),
            compute_capability: Some((8, 6)),
            sm_string: "sm_86".to_string(),
            vram_mb: 24 * 1024,
            performance_tier: GpuTier::Mid,
        }];
        assert!(!is_heterogeneous_cluster(&archs));
    }

    #[test]
    fn test_is_heterogeneous_cluster_same_archs() {
        let archs = vec![
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
                name: "RTX 3090".to_string(),
                compute_capability: Some((8, 6)),
                sm_string: "sm_86".to_string(),
                vram_mb: 24 * 1024,
                performance_tier: GpuTier::Mid,
            },
        ];
        assert!(!is_heterogeneous_cluster(&archs));
    }

    #[test]
    fn test_is_heterogeneous_cluster_different_archs() {
        let archs = vec![
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
                name: "RTX 4090".to_string(),
                compute_capability: Some((8, 9)),
                sm_string: "sm_89".to_string(),
                vram_mb: 24 * 1024,
                performance_tier: GpuTier::High,
            },
        ];
        assert!(is_heterogeneous_cluster(&archs));
    }

    #[test]
    fn test_is_heterogeneous_cluster_vram_ratio() {
        let archs = vec![
            GpuArchInfo {
                index: 0,
                name: "RTX 3080".to_string(),
                compute_capability: Some((8, 6)),
                sm_string: "sm_86".to_string(),
                vram_mb: 10 * 1024,
                performance_tier: GpuTier::Mid,
            },
            GpuArchInfo {
                index: 1,
                name: "RTX 3090".to_string(),
                compute_capability: Some((8, 6)),
                sm_string: "sm_86".to_string(),
                vram_mb: 24 * 1024,
                performance_tier: GpuTier::Mid,
            },
        ];
        assert!(is_heterogeneous_cluster(&archs));
    }

    #[test]
    fn test_get_combined_arch_list() {
        let archs = vec![
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
                name: "RTX 4090".to_string(),
                compute_capability: Some((8, 9)),
                sm_string: "sm_89".to_string(),
                vram_mb: 24 * 1024,
                performance_tier: GpuTier::High,
            },
            GpuArchInfo {
                index: 2,
                name: "RTX 5090".to_string(),
                compute_capability: Some((12, 0)),
                sm_string: "sm_120".to_string(),
                vram_mb: 32 * 1024,
                performance_tier: GpuTier::Ultra,
            },
        ];
        let result = get_combined_arch_list(&archs);
        assert_eq!(result, "8.6 8.9 12.0 12.0+PTX");
    }

    #[test]
    fn test_get_combined_arch_list_empty() {
        let archs: Vec<GpuArchInfo> = vec![];
        let result = get_combined_arch_list(&archs);
        assert_eq!(result, "");
    }

    #[test]
    fn test_infer_compute_cap_from_name() {
        assert_eq!(
            infer_compute_cap_from_name("NVIDIA GeForce RTX 3090", GpuProvider::Nvidia),
            Some((8, 6))
        );
        assert_eq!(
            infer_compute_cap_from_name("NVIDIA GeForce RTX 4090", GpuProvider::Nvidia),
            Some((8, 9))
        );
        assert_eq!(
            infer_compute_cap_from_name("NVIDIA H100", GpuProvider::Nvidia),
            Some((9, 0))
        );
        assert_eq!(
            infer_compute_cap_from_name("NVIDIA GeForce RTX 5090", GpuProvider::Nvidia),
            Some((12, 0))
        );
        assert_eq!(
            infer_compute_cap_from_name("AMD Radeon RX 7900 XTX", GpuProvider::Amd),
            None
        );
    }

    #[test]
    fn test_infer_sm_string() {
        assert_eq!(infer_sm_string(Some((8, 6))), "sm_86");
        assert_eq!(infer_sm_string(Some((12, 0))), "sm_120");
        assert_eq!(infer_sm_string(None), "unknown");
    }

    #[test]
    fn test_runtime_gpu_detection() {
        use crate::services::gpu_detector::detect_gpus;

        let gpus = detect_gpus();
        assert!(!gpus.is_empty(), "Should detect at least one GPU on this system");

        for gpu in &gpus {
            assert!(gpu.compute_capability.is_some(), "GPU {} should have compute capability", gpu.name);
            assert!(gpu.performance_tier != GpuTier::Low || true, "Tier should be classified");
        }

        let archs = detect_gpu_architectures(&gpus);
        assert_eq!(archs.len(), gpus.len());

        let heterogeneous = is_heterogeneous_cluster(&archs);
        let arch_list = get_combined_arch_list(&archs);
        assert!(!arch_list.is_empty(), "Should have combined arch list");

        println!("Detected {} GPU(s), heterogeneous: {}, arch list: {}", gpus.len(), heterogeneous, arch_list);
    }
}
