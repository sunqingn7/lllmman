use std::process::Command;

use crate::models::{GpuInfo, GpuProvider, GpuUsage};

pub fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    if let (Ok(index), Ok(vram)) =
                        (parts[0].parse::<u32>(), parts[2].parse::<u32>())
                    {
                        gpus.push(GpuInfo {
                            name: parts[1].to_string(),
                            total_vram_mb: vram,
                            index,
                            provider: GpuProvider::Nvidia,
                        });
                    }
                }
            }
        }
    }

    gpus
}

pub fn get_gpu_usage(index: u32) -> Result<GpuUsage, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            &format!("--id={}", index),
        ])
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err("Failed to query GPU usage".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let used = stdout
        .trim()
        .parse::<u32>()
        .map_err(|_| "Failed to parse VRAM")?;

    Ok(GpuUsage {
        index,
        used_vram_mb: used,
    })
}

pub fn get_all_gpu_usage() -> Vec<GpuUsage> {
    detect_gpus()
        .iter()
        .filter_map(|gpu| get_gpu_usage(gpu.index).ok())
        .collect()
}
