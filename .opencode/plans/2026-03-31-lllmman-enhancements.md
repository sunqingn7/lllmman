# LLLMMan Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add GPU/CPU temperature monitoring, SGLang provider, provider setup wizard, and smart auto-parameter calculation.

**Architecture:** Extend existing plugin-based architecture. Add temperature collection to gpu_detector, create new SGLang provider following llama.cpp/vllm patterns, add wizard modal windows to GUI, and add auto-calculation service for GPU layers.

**Tech Stack:** Rust, egui/eframe (GUI), sysinfo, nvidia-smi, rocm-smi (AMD), /sys/class/thermal (CPU temp)

---

### Task 1: Add GPU Temperature to GpuInfo and GpuUsage Models

**Files:**
- Modify: `src/models/gpu.rs`
- Modify: `src/models/config.rs` (MonitorStats)

**Step 1: Add temperature fields to models**

```rust
// src/models/gpu.rs - add to GpuInfo
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u32,
    pub index: u32,
    pub provider: GpuProvider,
    pub temperature_c: Option<u32>,  // NEW
}

// src/models/gpu.rs - add to GpuUsage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuUsage {
    pub index: u32,
    pub used_vram_mb: u32,
    pub temperature_c: Option<u32>,  // NEW
}

// src/models/config.rs - add to MonitorStats
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MonitorStats {
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub ram_used_mb: u32,
    pub ram_total_mb: u32,
    pub cpu_percent: f32,
    pub tokens_per_second: f32,
    pub active_connections: u32,
    pub gpu_temperatures: Vec<(u32, String, Option<u32>)>,  // NEW: (index, name, temp_c)
    pub cpu_temperature: Option<f32>,  // NEW
}
```

**Step 2: Commit**

```bash
git add src/models/gpu.rs src/models/config.rs
git commit -m "feat: add temperature fields to GPU and monitor models"
```

---

### Task 2: Implement Multi-Vendor GPU Temperature Detection

**Files:**
- Modify: `src/services/gpu_detector.rs`
- Modify: `src/services/mod.rs`

**Step 1: Add temperature detection functions**

```rust
// Add to src/services/gpu_detector.rs

pub fn get_gpu_temperature_nvidia(index: u32) -> Option<u32> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
            &format!("--id={}", index),
        ])
        .output()
        .ok()?;

    if output.status.success() {
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u32>()
            .ok()
    } else {
        None
    }
}

pub fn get_gpu_temperature_amd(index: u32) -> Option<u32> {
    // Try rocm-smi first
    let output = Command::new("rocm-smi")
        .args(["--showtemp", "--json"])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse JSON output to find temperature for the given index
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(card) = json.get(&format!("card{}", index)) {
                if let Some(temp) = card.get("Temperature (Sensor edge) (C)") {
                    return temp.as_f64().map(|t| t as u32);
                }
            }
        }
    }

    // Fallback: try reading from sysfs
    read_sysfs_gpu_temp(index)
}

pub fn get_gpu_temperature_intel(index: u32) -> Option<u32> {
    // Intel GPUs typically report temperature via sysfs or hwmon
    read_sysfs_gpu_temp(index)
}

fn read_sysfs_gpu_temp(_index: u32) -> Option<u32> {
    // Try /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name()?.to_string_lossy().starts_with("card") {
                let hwmon_path = path.join("device/hwmon");
                if let Ok(hwmon_entries) = std::fs::read_dir(hwmon_path) {
                    for hwmon in hwmon_entries.flatten() {
                        let temp_file = hwmon.path().join("temp1_input");
                        if let Ok(content) = std::fs::read_to_string(&temp_file) {
                            if let Ok(temp) = content.trim().parse::<u32>() {
                                return Some(temp / 1000); // Convert millidegrees to degrees
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
    // Try /sys/class/thermal/thermal_zone*/temp
    if let Ok(entries) = std::fs::read_dir("/sys/class/thermal") {
        let mut temps = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.file_name()?.to_string_lossy().starts_with("thermal_zone") {
                let temp_file = path.join("temp");
                if let Ok(content) = std::fs::read_to_string(&temp_file) {
                    if let Ok(temp) = content.trim().parse::<u32>() {
                        temps.push(temp as f32 / 1000.0);
                    }
                }
            }
        }
        if !temps.is_empty() {
            return Some(temps.iter().sum::<f32>() / temps.len() as f32);
        }
    }

    // Fallback: try hwmon
    if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
        for entry in entries.flatten() {
            let path = entry.path();
            let temp_file = path.join("temp1_input");
            if let Ok(content) = std::fs::read_to_string(&temp_file) {
                if let Ok(temp) = content.trim().parse::<u32>() {
                    return Some(temp as f32 / 1000.0);
                }
            }
        }
    }

    None
}

pub fn get_gpu_temperature(gpu: &GpuInfo) -> Option<u32> {
    match gpu.provider {
        GpuProvider::Nvidia => get_gpu_temperature_nvidia(gpu.index),
        GpuProvider::Amd => get_gpu_temperature_amd(gpu.index),
        GpuProvider::Intel => get_gpu_temperature_intel(gpu.index),
        GpuProvider::Unknown => read_sysfs_gpu_temp(gpu.index),
    }
}
```

**Step 2: Update detect_gpus to include temperature**

```rust
// In detect_gpus(), after creating each GpuInfo:
let mut gpu_info = GpuInfo { ... };
gpu_info.temperature_c = get_gpu_temperature(&gpu_info);
gpus.push(gpu_info);
```

**Step 3: Update get_all_gpu_usage to include temperature**

```rust
// In get_all_gpu_usage():
pub fn get_all_gpu_usage() -> Vec<GpuUsage> {
    detect_gpus()
        .iter()
        .filter_map(|gpu| {
            let usage = get_gpu_usage(gpu.index).ok()?;
            let temp = get_gpu_temperature(gpu);
            Some(GpuUsage {
                index: usage.index,
                used_vram_mb: usage.used_vram_mb,
                temperature_c: temp,
            })
        })
        .collect()
}
```

**Step 4: Export new functions in mod.rs**

```rust
// src/services/mod.rs - add to pub use:
pub use gpu_detector::{get_cpu_temperature, get_gpu_temperature};
```

**Step 5: Commit**

```bash
git add src/services/gpu_detector.rs src/services/mod.rs
git commit -m "feat: add multi-vendor GPU and CPU temperature detection"
```

---

### Task 3: Update Monitor Service to Include Temperatures

**Files:**
- Modify: `src/services/monitor.rs`
- Modify: `src/services/mod.rs`

**Step 1: Update get_system_stats to include temperatures**

```rust
// src/services/monitor.rs - update get_system_stats
pub fn get_system_stats() -> MonitorStats {
    let mut sys = sysinfo::System::new();
    sys.refresh_all();

    let cpu = sys.cpus().first().map(|c| c.cpu_usage()).unwrap_or(0.0);
    let total_ram = sys.total_memory() as u32;
    let used_ram = sys.used_memory() as u32;

    let gpu_usage = gpu_detector::get_all_gpu_usage();
    let gpus = gpu_detector::detect_gpus();
    
    let (vram_used, vram_total) = if !gpu_usage.is_empty() {
        let total: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
        let used: u32 = gpu_usage.iter().map(|u| u.used_vram_mb).sum();
        (used, total)
    } else {
        (0, 0)
    };

    // NEW: Collect GPU temperatures
    let gpu_temperatures: Vec<(u32, String, Option<u32>)> = gpus.iter().map(|gpu| {
        let temp = gpu_detector::get_gpu_temperature(gpu);
        (gpu.index, gpu.name.clone(), temp)
    }).collect();

    // NEW: Get CPU temperature
    let cpu_temperature = gpu_detector::get_cpu_temperature();

    MonitorStats {
        vram_used_mb: vram_used,
        vram_total_mb: vram_total,
        ram_used_mb: used_ram / (1024 * 1024),
        ram_total_mb: total_ram / (1024 * 1024),
        cpu_percent: cpu,
        tokens_per_second: 0.0,
        active_connections: 0,
        gpu_temperatures,  // NEW
        cpu_temperature,    // NEW
    }
}
```

**Step 2: Commit**

```bash
git add src/services/monitor.rs
git commit -m "feat: include temperatures in system stats"
```

---

### Task 4: Update GUI Monitor Toolbar with Temperatures

**Files:**
- Modify: `src/gui/app.rs` (render_bottom_panel method)

**Step 1: Update the monitor display in render_bottom_panel**

Replace the existing monitor section in `render_bottom_panel` (around lines 225-259):

```rust
if self.bottom_view == BottomView::Monitor {
    let stats = get_system_stats();
    
    // VRAM
    ui.label(format!(
        "VRAM: {}/{} MB",
        stats.vram_used_mb, stats.vram_total_mb
    ));
    ui.separator();
    
    // RAM
    ui.label(format!(
        "RAM: {}/{} MB",
        stats.ram_used_mb, stats.ram_total_mb
    ));
    ui.separator();
    
    // CPU with temperature
    let cpu_text = if let Some(cpu_temp) = stats.cpu_temperature {
        format!("CPU: {:.1}% ({}°C)", stats.cpu_percent, cpu_temp as u32)
    } else {
        format!("CPU: {:.1}%", stats.cpu_percent)
    };
    ui.label(&cpu_text);
    ui.separator();

    // GPU temperatures
    if !stats.gpu_temperatures.is_empty() {
        for (idx, name, temp) in &stats.gpu_temperatures {
            let gpu_text = if let Some(t) = temp {
                format!("GPU{}: {}°C", idx, t)
            } else {
                format!("GPU{}: N/A", idx)
            };
            ui.label(&gpu_text);
        }
        ui.separator();
    }

    // Server stats (TPS, Queue)
    if self.server_controller.get_status()
        == crate::models::ServerStatus::Running
        && self.frame_counter % 300 == 0
    {
        if let Some(server_stats) = crate::services::fetch_server_stats(
            &self.server_config.host,
            self.server_config.port,
        ) {
            let tps = server_stats
                .time_per_token
                .map(|t| if t > 0.0 { 1000.0 / t } else { 0.0 })
                .unwrap_or(0.0);
            ui.label(format!("TPS: {:.1}", tps));
            ui.separator();
            ui.label(format!(
                "Queue: {}",
                server_stats.queue_size.unwrap_or(0)
            ));
        }
    }
}
```

**Step 2: Commit**

```bash
git add src/gui/app.rs
git commit -m "feat: display GPU and CPU temperatures in monitor toolbar"
```

---

### Task 5: Update TUI Footer with Temperatures

**Files:**
- Modify: `src/tui/app.rs` (render_footer function)

**Step 1: Update render_footer to show temperatures**

```rust
// Replace render_footer function (around lines 585-611)
fn render_footer(f: &mut Frame, area: Rect, app: &TuiApp) {
    let stats = get_system_stats();

    let vram_text = format!("VRAM: {}/{} MB", stats.vram_used_mb, stats.vram_total_mb);
    
    let ram_text = if let Some(cpu_temp) = stats.cpu_temperature {
        format!("RAM: {}/{} MB | CPU: {:.1}% ({}°C)", 
            stats.ram_used_mb, stats.ram_total_mb, 
            stats.cpu_percent, cpu_temp as u32)
    } else {
        format!("RAM: {}/{} MB | CPU: {:.1}%", 
            stats.ram_used_mb, stats.ram_total_mb, 
            stats.cpu_percent)
    };

    // Build GPU temperature string
    let gpu_text = if stats.gpu_temperatures.is_empty() {
        "No GPU".to_string()
    } else {
        stats.gpu_temperatures.iter()
            .map(|(idx, _name, temp)| {
                match temp {
                    Some(t) => format!("GPU{}: {}°C", idx, t),
                    None => format!("GPU{}: N/A", idx),
                }
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };

    let status_text = format!("{} | {}", gpu_text, app.status_message);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(35),
            Constraint::Percentage(20),
            Constraint::Min(0),
        ])
        .split(area);

    f.render_widget(Paragraph::new(vram_text), chunks[0]);
    f.render_widget(Paragraph::new(ram_text), chunks[1]);
    f.render_widget(Paragraph::new(gpu_text), chunks[2]);
    f.render_widget(
        Paragraph::new(app.status_message.as_str())
            .alignment(ratatui::layout::Alignment::Right)
            .style(Style::default().fg(Color::Yellow)),
        chunks[3],
    );
}
```

**Step 2: Commit**

```bash
git add src/tui/app.rs
git commit -m "feat: display GPU and CPU temperatures in TUI footer"
```

---

### Task 6: Create SGLang Provider

**Files:**
- Create: `src/providers/sglang/mod.rs`
- Create: `src/providers/sglang/provider.rs`
- Modify: `src/providers/mod.rs`

**Step 1: Create SGLang provider implementation**

```rust
// src/providers/sglang/provider.rs
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::core::{
    DetectedServer, LlmProvider, ModelInfo, ProviderConfig, ProviderError, ProviderSettings, Result,
};
use crate::models::ModelType;

pub struct SglangProvider {
    id: &'static str,
    name: &'static str,
}

impl SglangProvider {
    pub fn new() -> Self {
        Self {
            id: "sglang",
            name: "SGLang",
        }
    }
}

impl Default for SglangProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for SglangProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn get_config_template(&self) -> ProviderConfig {
        ProviderConfig {
            gpu_layers: 90,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            num_prompt_tracking: 0,
            ..ProviderConfig::default()
        }
    }

    fn validate_config(&self, config: &ProviderConfig) -> Result<()> {
        if config.model_path.is_empty() {
            return Err(ProviderError::InvalidConfig(
                "Model path or HuggingFace model ID is required".into(),
            ));
        }

        // If it's a local path, verify it exists
        if !config.model_path.contains('/') {
            // Assume it's a HF model ID
            return Ok(());
        }

        let path = Path::new(&config.model_path);
        if path.exists() && !path.is_dir() {
            return Err(ProviderError::InvalidConfig(
                "Model path must be a directory for SGLang".into(),
            ));
        }

        Ok(())
    }

    fn default_settings(&self) -> ProviderSettings {
        ProviderSettings {
            binary_path: "python3".to_string(),
            env_script: String::new(),
            additional_args: String::new(),
        }
    }

    fn start_server(
        &self,
        config: &ProviderConfig,
        settings: &ProviderSettings,
    ) -> Result<std::process::Child> {
        let python = if settings.binary_path.is_empty() {
            "python3"
        } else {
            &settings.binary_path
        };

        let mut cmd = Command::new("bash");
        cmd.arg("-c");

        let mut script = String::new();
        if !settings.env_script.is_empty() {
            script.push_str(&format!("source \"{}\"\n", settings.env_script));
        }
        script.push_str("exec ");
        script.push_str(&format!("\"{}\" -m sglang.launch_server ", python));

        // Model path
        if config.model_path.contains('/') && Path::new(&config.model_path).exists() {
            script.push_str(&format!("--model-path \"{}\" ", config.model_path));
        } else {
            script.push_str(&format!("--model-path {} ", config.model_path));
        }

        // Port and host
        script.push_str(&format!("--port {} ", config.port));
        script.push_str(&format!("--host {} ", config.host));

        // Memory fraction (maps from gpu_layers 0-100)
        if config.gpu_layers > 0 {
            script.push_str(&format!(
                "--mem-fraction-static {:.2} ",
                config.gpu_layers as f32 / 100.0
            ));
        }

        // Context size
        if config.context_size > 0 {
            script.push_str(&format!("--context-length {} ", config.context_size));
        }

        // Additional args
        if !config.additional_args.is_empty() {
            for arg in config.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    script.push_str(&format!("{} ", arg));
                }
            }
        }

        if !settings.additional_args.is_empty() {
            for arg in settings.additional_args.split_whitespace() {
                if !arg.is_empty() {
                    script.push_str(&format!("{} ", arg));
                }
            }
        }

        log::info!("Starting SGLang: {}", script);

        cmd.arg(script);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        cmd.spawn().map_err(ProviderError::from)
    }

    fn supported_quantizations(&self) -> Vec<&'static str> {
        vec![
            "fp16", "fp8", "int8", "int4", "awq", "gptq", "marlin", "squeezellm",
        ]
    }

    fn scan_models(&self, path: &str) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let path_obj = Path::new(path);

        if !path_obj.exists() || !path_obj.is_dir() {
            return models;
        }

        // Scan for HuggingFace cache directories
        if let Ok(entries) = std::fs::read_dir(path_obj) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    if let Some(model) = parse_hf_model_dir(&entry_path) {
                        models.push(model);
                    }
                }
            }
        }

        // Also scan for GGUF files (SGLang can load GGUF)
        fn scan_gguf_recursive(dir: &Path, models: &mut Vec<ModelInfo>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        scan_gguf_recursive(&path, models);
                    } else if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().to_lowercase() == "gguf" {
                            if let Some(model) = parse_gguf_file(&path) {
                                models.push(model);
                            }
                        }
                    }
                }
            }
        }
        scan_gguf_recursive(path_obj, &mut models);

        models
    }

    fn add_model(&self, path: &str) -> Result<ModelInfo> {
        if path.is_empty() {
            return Err(ProviderError::InvalidConfig("Model path is required".into()));
        }

        let name = path.split('/').last().unwrap_or(path).to_string();
        let size_gb = if Path::new(path).exists() {
            calculate_dir_size(path)
        } else {
            0.0
        };

        Ok(ModelInfo {
            path: path.to_string(),
            name,
            size_gb,
            quantization: "unknown".to_string(),
            model_type: ModelType::TextOnly,
        })
    }

    fn default_model_directories(&self) -> Vec<String> {
        let mut dirs = Vec::new();

        if let Some(cache) = dirs::cache_dir() {
            let hf_hub = cache.join("huggingface").join("hub");
            if hf_hub.exists() {
                dirs.push(hf_hub.to_string_lossy().to_string());
            }
        }

        if let Some(home) = dirs::home_dir() {
            let models_dir = home.join("models");
            if models_dir.exists() {
                dirs.push(models_dir.to_string_lossy().to_string());
            }
        }

        dirs
    }

    fn detect_running_servers(&self) -> Vec<DetectedServer> {
        let mut servers = Vec::new();

        // Detect via pgrep for sglang.launch_server
        if let Ok(output) = Command::new("pgrep")
            .args(["-a", "sglang"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let lower = line.to_lowercase();
                if !lower.contains("sglang") || !lower.contains("launch_server") {
                    continue;
                }
                if let Some(space_pos) = line.find(' ') {
                    let pid_str = &line[..space_pos];
                    let cmdline = &line[space_pos + 1..];
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        servers.push(DetectedServer {
                            pid,
                            binary: "sglang".to_string(),
                            command_line: cmdline.to_string(),
                        });
                    }
                }
            }
        }

        servers
    }

    fn parse_server_config(&self, cmd_line: &str) -> ProviderConfig {
        let mut config = ProviderConfig::default();
        let args: Vec<&str> = cmd_line.split_whitespace().collect();
        let mut i = 0;

        while i < args.len() {
            let arg = args[i];

            match arg {
                "--model-path" | "--model" => {
                    if i + 1 < args.len() {
                        config.model_path = args[i + 1].to_string();
                        i += 1;
                    }
                }
                "--port" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<u32>() {
                            config.port = val as u16;
                        }
                        i += 1;
                    }
                }
                "--host" => {
                    if i + 1 < args.len() {
                        config.host = args[i + 1].to_string();
                        i += 1;
                    }
                }
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
                _ => {}
            }
            i += 1;
        }

        config
    }
}

fn parse_hf_model_dir(path: &Path) -> Option<ModelInfo> {
    let dir_name = path.file_name()?.to_string_lossy().to_string();

    if !dir_name.contains("models--") {
        return None;
    }

    let repo_id = dir_name.strip_prefix("models--")?;
    let name = repo_id.replace("--", "/");
    let size_gb = calculate_dir_size(&path.to_string_lossy());
    let quantization = detect_quantization(path);

    Some(ModelInfo {
        path: name.clone(),
        name,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
    })
}

fn parse_gguf_file(path: &Path) -> Option<ModelInfo> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let size_gb = metadata.len() as f32 / (1024.0 * 1024.0 * 1024.0);

    let quantization = extract_quantization(&filename);

    Some(ModelInfo {
        path: path.to_string_lossy().to_string(),
        name: filename,
        size_gb: (size_gb * 100.0).round() / 100.0,
        quantization,
        model_type: ModelType::TextOnly,
    })
}

fn extract_quantization(filename: &str) -> String {
    let lower = filename.to_lowercase();
    let quantizations = [
        "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0", "f16", "q2_k", "q3_k", "q4_k", "q5_k",
        "q6_k",
    ];

    for q in &quantizations {
        if lower.contains(*q) {
            return q.to_string();
        }
    }
    "unknown".to_string()
}

fn calculate_dir_size(path: &str) -> f32 {
    let mut total_size = 0u64;

    fn walk_dir(path: &Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    walk_dir(&entry_path, total);
                } else if let Ok(meta) = entry.metadata() {
                    *total += meta.len();
                }
            }
        }
    }

    walk_dir(Path::new(path), &mut total_size);
    total_size as f32 / (1024.0 * 1024.0 * 1024.0)
}

fn detect_quantization(model_dir: &Path) -> String {
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                for sub_entry in std::fs::read_dir(&entry_path)
                    .into_iter()
                    .flatten()
                    .flatten()
                {
                    if let Some(name) = sub_entry.file_name().to_str() {
                        if name == "config.json" {
                            if let Ok(content) = fs::read_to_string(sub_entry.path()) {
                                if let Ok(json) =
                                    serde_json::from_str::<serde_json::Value>(&content)
                                {
                                    if let Some(quant) = json.get("quantization_config") {
                                        if let Some(method) = quant.get("quant_method") {
                                            if let Some(method_str) = method.as_str() {
                                                return method_str.to_lowercase();
                                            }
                                        }
                                    }

                                    if let Some(torch_dtype) = json.get("torch_dtype") {
                                        if let Some(dtype_str) = torch_dtype.as_str() {
                                            return match dtype_str {
                                                "float16" | "torch.float16" => "fp16".to_string(),
                                                "float32" | "torch.float32" => "fp32".to_string(),
                                                "bfloat16" | "torch.bfloat16" => "bf16".to_string(),
                                                _ => dtype_str.to_lowercase(),
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    "unknown".to_string()
}
```

**Step 2: Create SGLang module registration**

```rust
// src/providers/sglang/mod.rs
pub mod provider;

pub use provider::SglangProvider;

use crate::core::LlmProvider;
use crate::register_provider;
use once_cell::sync::Lazy;
use std::sync::Arc;

static SGLANG: Lazy<Arc<SglangProvider>> = Lazy::new(|| Arc::new(SglangProvider::new()));

pub fn register() {
    register_provider!(SGLANG.clone() as Arc<dyn LlmProvider>);
}
```

**Step 3: Register SGLang in providers/mod.rs**

```rust
// src/providers/mod.rs - update:
pub mod llama_cpp;
pub mod vllm;
pub mod sglang;  // NEW

pub use llama_cpp::register as register_llama_cpp;
pub use llama_cpp::LlamaCppProvider;
pub use vllm::register as register_vllm;
pub use sglang::register as register_sglang;  // NEW
pub use sglang::SglangProvider;  // NEW

pub fn register_all_providers() {
    register_llama_cpp();
    register_vllm();
    register_sglang();  // NEW
}
```

**Step 4: Commit**

```bash
git add src/providers/sglang/mod.rs src/providers/sglang/provider.rs src/providers/mod.rs
git commit -m "feat: add SGLang provider with full plugin integration"
```

---

### Task 7: Create Provider Setup Wizard (GUI)

**Files:**
- Create: `src/services/provider_installer.rs`
- Modify: `src/services/mod.rs`
- Modify: `src/gui/app.rs`

**Step 1: Create provider installer service**

```rust
// src/services/provider_installer.rs
use std::process::Command;

pub enum InstallMethod {
    Simple,
    Advanced,
}

pub enum InstallStatus {
    NotInstalled,
    Installed,
    Installing,
    Failed(String),
}

pub struct ProviderInstallInfo {
    pub provider_id: &'static str,
    pub provider_name: &'static str,
    pub binary_name: &'static str,
    pub simple_command: &'static str,
    simple_description: &'static str,
    advanced_command: &'static str,
    advanced_description: &'static str,
    check_command: &'static str,
    check_args: &'static [&'static str],
}

impl ProviderInstallInfo {
    pub fn simple_description(&self) -> &str {
        self.simple_description
    }
    pub fn advanced_command(&self) -> &str {
        self.advanced_command
    }
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

pub fn get_install_command(provider_id: &str, method: InstallMethod) -> Option<String> {
    let info = get_provider_install_info(provider_id)?;
    match method {
        InstallMethod::Simple => Some(info.simple_command.to_string()),
        InstallMethod::Advanced => Some(info.advanced_command().to_string()),
    }
}
```

**Step 2: Export in mod.rs**

```rust
// src/services/mod.rs - add:
pub mod provider_installer;
pub use provider_installer::{get_provider_install_info, check_provider_installed, get_install_command, InstallMethod, InstallStatus};
```

**Step 3: Add wizard UI to GUI app**

Add to `src/gui/app.rs` - new field in App struct:
```rust
show_provider_setup: bool,
provider_setup_provider: String,
```

Initialize in `App::new()`:
```rust
show_provider_setup: false,
provider_setup_provider: String::new(),
```

Add wizard window rendering in `update()` method (after existing windows):
```rust
if self.show_provider_setup {
    let provider_id = &self.provider_setup_provider;
    if let Some(info) = crate::services::get_provider_install_info(provider_id) {
        let installed = crate::services::check_provider_installed(provider_id);
        
        egui::Window::new(format!("Setup: {}", info.provider_name))
            .open(&mut self.show_provider_setup)
            .default_width(600.0)
            .default_height(400.0)
            .show(ctx, |ui| {
                ui.heading(format!("{} Setup", info.provider_name));
                ui.separator();
                
                if installed {
                    ui.colored_label(
                        egui::Color32::GREEN,
                        format!("✓ {} is installed", info.provider_name),
                    );
                } else {
                    ui.colored_label(
                        egui::Color32::RED,
                        format!("✗ {} is NOT installed", info.provider_name),
                    );
                }
                
                ui.separator();
                ui.heading("Quick Install");
                ui.label(info.simple_description());
                ui.horizontal(|ui| {
                    ui.code(info.simple_command);
                    if ui.button("Copy").clicked() {
                        ui.output_mut(|o| o.copied_text = info.simple_command.to_string());
                    }
                });
                
                ui.separator();
                ui.heading("Advanced Install");
                ui.label(info.advanced_description());
                ui.horizontal(|ui| {
                    ui.code(info.advanced_command());
                    if ui.button("Copy").clicked() {
                        ui.output_mut(|o| o.copied_text = info.advanced_command().to_string());
                    }
                });
                
                ui.separator();
                if ui.button("Refresh Status").clicked() {
                    // Force re-check
                }
            });
    }
}
```

Add a "Setup" button in the provider selection area (in `render_main_content`):
```rust
if ui.button("Setup").clicked() {
    self.show_provider_setup = true;
    self.provider_setup_provider = self.selected_provider.clone();
}
```

**Step 4: Commit**

```bash
git add src/services/provider_installer.rs src/services/mod.rs src/gui/app.rs
git commit -m "feat: add provider setup wizard with install instructions"
```

---

### Task 8: Create Smart Auto-Parameter Service (GPU Layers)

**Files:**
- Create: `src/services/auto_params.rs`
- Modify: `src/services/mod.rs`
- Modify: `src/gui/app.rs` (add auto button)
- Modify: `src/tui/app.rs` (add auto shortcut)

**Step 1: Create auto-parameter calculation service**

```rust
// src/services/auto_params.rs
use crate::models::GpuInfo;
use crate::services::gpu_detector;

/// Model size category with estimated VRAM requirements
pub struct ModelSizeEstimate {
    pub size_gb: f32,
    pub vram_needed_mb: u32,  // Full GPU offload
    pub recommended_layers: i32,
}

/// Estimate VRAM needed based on model size
/// Rule of thumb: FP16 needs ~2GB per billion params, Q4 needs ~0.5GB per billion
/// We use model file size as a proxy
fn estimate_full_vram(model_size_gb: f32) -> u32 {
    // Model file size (quantized) is roughly the VRAM needed for full offload
    // Add 20% overhead for KV cache and context
    (model_size_gb * 1024.0 * 1.2) as u32
}

/// Calculate optimal GPU layers based on available VRAM and model size
pub fn calculate_gpu_layers(
    model_size_gb: f32,
    gpus: &[GpuInfo],
    total_layers: i32,
) -> i32 {
    if gpus.is_empty() {
        return 0; // No GPU, all CPU
    }

    let total_vram: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
    let vram_needed = estimate_full_vram(model_size_gb);

    // If we have enough VRAM, offload everything
    if total_vram >= vram_needed {
        return if total_layers > 0 { total_layers } else { -1 };
    }

    // Calculate how many layers we can fit
    if total_layers <= 0 {
        // If we don't know total layers, estimate based on model size
        // Rough estimate: 7B model has ~32 layers, 13B has ~40, 70B has ~80
        let estimated_layers = (model_size_gb * 4.5) as i32;
        let vram_per_layer = vram_needed / estimated_layers.max(1) as u32;
        return (total_vram / vram_per_layer.max(1)) as i32;
    }

    let vram_per_layer = vram_needed / total_layers as u32;
    let layers_fit = (total_vram / vram_per_layer.max(1)) as i32;

    layers_fit.min(total_layers).max(0)
}

/// Get system info summary for display
pub fn get_system_info_summary() -> String {
    let mut sys = sysinfo::System::new();
    sys.refresh_all();

    let total_ram_gb = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let gpus = gpu_detector::detect_gpus();
    let total_vram_gb: f32 = gpus.iter().map(|g| g.total_vram_mb as f32).sum::<f32>() / 1024.0;

    let gpu_info = if gpus.is_empty() {
        "No GPUs detected".to_string()
    } else {
        gpus.iter()
            .map(|g| format!("GPU{}: {} ({:.1} GB)", g.index, g.name, g.total_vram_mb as f32 / 1024.0))
            .collect::<Vec<_>>()
            .join(", ")
    };

    format!(
        "RAM: {:.1} GB | VRAM: {:.1} GB | {}",
        total_ram_gb, total_vram_gb, gpu_info
    )
}

/// Auto-calculate and return recommended GPU layers
pub fn recommend_gpu_layers(
    model_size_gb: f32,
    total_layers: i32,
) -> i32 {
    let gpus = gpu_detector::detect_gpus();
    calculate_gpu_layers(model_size_gb, &gpus, total_layers)
}
```

**Step 2: Export in mod.rs**

```rust
// src/services/mod.rs - add:
pub mod auto_params;
pub use auto_params::{calculate_gpu_layers, get_system_info_summary, recommend_gpu_layers};
```

**Step 3: Add auto button to GUI**

In `src/gui/app.rs`, in the GPU Memory Utilization row in `render_main_content`:

```rust
ui.label("GPU Memory Utilization:");
ui.horizontal(|ui| {
    let mut gpu_mem_value = if self.server_config.gpu_layers <= 0 {
        90
    } else {
        self.server_config.gpu_layers
    };
    ui.add(
        egui::DragValue::new(&mut gpu_mem_value)
            .clamp_range(1..=100)
            .suffix("%"),
    );
    self.server_config.gpu_layers = gpu_mem_value;
    
    // NEW: Auto button
    if ui.button("Auto").clicked() {
        // Estimate model size from file
        let model_size_gb = if let Ok(meta) = std::fs::metadata(&self.server_config.model_path) {
            meta.len() as f32 / (1024.0 * 1024.0 * 1024.0)
        } else {
            7.0 // Default estimate for 7B model
        };
        
        // Try to get total layers from GGUF
        let total_layers = if self.selected_provider == "llama.cpp" {
            crate::providers::llama_cpp::read_gguf_n_layer(&self.server_config.model_path)
                .map(|l| l as i32)
                .unwrap_or(-1)
        } else {
            -1
        };
        
        let recommended = crate::services::recommend_gpu_layers(model_size_gb, total_layers);
        self.server_config.gpu_layers = recommended.max(1).min(100);
        
        self.status_message = format!("Auto-set GPU layers to {} (model: {:.1} GB)", recommended, model_size_gb);
    }
});
```

**Step 4: Add system info display to GUI**

In the GPU Settings window, add system info:
```rust
ui.label("System Info:");
ui.label(crate::services::get_system_info_summary());
```

**Step 5: Commit**

```bash
git add src/services/auto_params.rs src/services/mod.rs src/gui/app.rs
git commit -m "feat: add smart auto-parameter calculation for GPU layers"
```

---

### Task 9: Build and Test

**Step 1: Build with GUI feature**

```bash
cargo build --features gui
```

**Step 2: Build with TUI feature**

```bash
cargo build --features tui
```

**Step 3: Run tests**

```bash
cargo test
```

**Step 4: Commit**

```bash
git add .
git commit -m "chore: verify build and tests pass"
```

---

## Summary of Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `src/models/gpu.rs` | Modify | Add temperature_c to GpuInfo and GpuUsage |
| `src/models/config.rs` | Modify | Add gpu_temperatures and cpu_temperature to MonitorStats |
| `src/services/gpu_detector.rs` | Modify | Add multi-vendor GPU temp and CPU temp detection |
| `src/services/monitor.rs` | Modify | Include temperatures in system stats |
| `src/services/mod.rs` | Modify | Export new functions |
| `src/services/provider_installer.rs` | Create | Provider install info and check functions |
| `src/services/auto_params.rs` | Create | Smart GPU layer calculation |
| `src/providers/sglang/mod.rs` | Create | SGLang provider registration |
| `src/providers/sglang/provider.rs` | Create | SGLang provider implementation |
| `src/providers/mod.rs` | Modify | Register SGLang provider |
| `src/gui/app.rs` | Modify | Add temp display, setup wizard, auto button |
| `src/tui/app.rs` | Modify | Add temp display in footer |
