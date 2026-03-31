use std::fs;
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
                        let mut gpu_info = GpuInfo {
                            name: parts[1].to_string(),
                            total_vram_mb: vram,
                            index,
                            provider: GpuProvider::Nvidia,
                            temperature_c: None,
                        };
                        gpu_info.temperature_c = get_gpu_temperature(&gpu_info).map(|t| t as f32);
                        gpus.push(gpu_info);
                    }
                }
            }
        }
    }

    if gpus.is_empty() {
        if let Ok(output) = Command::new("rocm-smi")
            .args(["--showproductname", "--json"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if let Some(cards) = json.as_object() {
                        for (key, value) in cards {
                            if let Some(name) = value.get("Card series").and_then(|v| v.as_str()) {
                                if let Ok(index) = key.parse::<u32>() {
                                    let mut gpu_info = GpuInfo {
                                        name: name.to_string(),
                                        total_vram_mb: 0,
                                        index,
                                        provider: GpuProvider::Amd,
                                        temperature_c: None,
                                    };
                                    gpu_info.temperature_c =
                                        get_gpu_temperature(&gpu_info).map(|t| t as f32);
                                    gpus.push(gpu_info);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if gpus.is_empty() {
        if let Ok(entries) = fs::read_dir("/sys/class/drm") {
            let mut card_index = 0u32;
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("card") && !name_str.contains('-') {
                    if let Ok(device_path) = fs::read_link(entry.path()) {
                        let uevent_path = device_path.join("uevent");
                        if let Ok(uevent) = fs::read_to_string(&uevent_path) {
                            let mut driver = String::new();
                            let mut name_val = String::new();
                            for line in uevent.lines() {
                                if line.starts_with("DRIVER=") {
                                    driver = line.strip_prefix("DRIVER=").unwrap_or("").to_string();
                                }
                                if line.starts_with("MODALIAS=") {
                                    name_val =
                                        line.strip_prefix("MODALIAS=").unwrap_or("").to_string();
                                }
                            }
                            if driver.contains("i915") || driver.contains("xe") {
                                let mut gpu_info = GpuInfo {
                                    name: if name_val.is_empty() {
                                        "Intel GPU".to_string()
                                    } else {
                                        name_val
                                    },
                                    total_vram_mb: 0,
                                    index: card_index,
                                    provider: GpuProvider::Intel,
                                    temperature_c: None,
                                };
                                gpu_info.temperature_c =
                                    get_gpu_temperature(&gpu_info).map(|t| t as f32);
                                gpus.push(gpu_info);
                                card_index += 1;
                            }
                        }
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

    let gpu = GpuInfo {
        name: String::new(),
        total_vram_mb: 0,
        index,
        provider: GpuProvider::Nvidia,
        temperature_c: None,
    };
    let temp = get_gpu_temperature(&gpu).map(|t| t as f32);

    Ok(GpuUsage {
        index,
        used_vram_mb: used,
        temperature_c: temp,
    })
}

pub fn get_all_gpu_usage() -> Vec<GpuUsage> {
    detect_gpus()
        .iter()
        .filter_map(|gpu| {
            let mut usage = GpuUsage {
                index: gpu.index,
                used_vram_mb: 0,
                temperature_c: None,
            };
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    &format!("--id={}", gpu.index),
                ])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if let Ok(used) = stdout.trim().parse::<u32>() {
                        usage.used_vram_mb = used;
                    }
                }
            }
            usage.temperature_c = get_gpu_temperature(gpu).map(|t| t as f32);
            Some(usage)
        })
        .collect()
}

pub fn get_gpu_temperature_nvidia(index: u32) -> Option<u32> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
            &format!("--id={}", index),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<u32>().ok()
}

pub fn get_gpu_temperature_amd(index: u32) -> Option<u32> {
    if let Ok(output) = Command::new("rocm-smi")
        .args(["--showtemp", "--json"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                if let Some(card) = json.get(&index.to_string()).or_else(|| json.get("card0")) {
                    if let Some(temp) = card.get("temperature").and_then(|v| v.as_f64()) {
                        return Some(temp as u32);
                    }
                }
            }
        }
    }

    read_sysfs_gpu_temp(index)
}

pub fn get_gpu_temperature_intel(index: u32) -> Option<u32> {
    read_sysfs_gpu_temp(index)
}

pub fn read_sysfs_gpu_temp(_index: u32) -> Option<u32> {
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("card") && !name_str.contains('-') {
                let hwmon_path = entry.path().join("device").join("hwmon");
                if let Ok(hwmon_entries) = fs::read_dir(hwmon_path) {
                    for hwmon_entry in hwmon_entries.flatten() {
                        let temp_path = hwmon_entry.path().join("temp1_input");
                        if let Ok(content) = fs::read_to_string(temp_path) {
                            if let Ok(millidegrees) = content.trim().parse::<f32>() {
                                return Some((millidegrees / 1000.0).round() as u32);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn get_cpu_temperature() -> Option<f32> {
    let mut temps: Vec<f32> = Vec::new();

    if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("thermal_zone") {
                let temp_path = entry.path().join("temp");
                if let Ok(content) = fs::read_to_string(temp_path) {
                    if let Ok(millidegrees) = content.trim().parse::<f32>() {
                        temps.push(millidegrees / 1000.0);
                    }
                }
            }
        }
    }

    if temps.is_empty() {
        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            for entry in entries.flatten() {
                let temp_path = entry.path().join("temp1_input");
                if let Ok(content) = fs::read_to_string(temp_path) {
                    if let Ok(millidegrees) = content.trim().parse::<f32>() {
                        temps.push(millidegrees / 1000.0);
                    }
                }
            }
        }
    }

    if temps.is_empty() {
        None
    } else {
        Some(temps.iter().sum::<f32>() / temps.len() as f32)
    }
}

pub fn get_gpu_temperature(gpu: &GpuInfo) -> Option<u32> {
    match gpu.provider {
        GpuProvider::Nvidia => get_gpu_temperature_nvidia(gpu.index),
        GpuProvider::Amd => get_gpu_temperature_amd(gpu.index),
        GpuProvider::Intel => get_gpu_temperature_intel(gpu.index),
        GpuProvider::Unknown => read_sysfs_gpu_temp(gpu.index),
    }
}
