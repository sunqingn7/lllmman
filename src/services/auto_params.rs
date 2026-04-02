use crate::models::GpuInfo;
use crate::services::gpu_detector;

fn estimate_full_vram(model_size_gb: f32) -> u32 {
    if model_size_gb <= 0.0 {
        return 0;
    }
    // Model file size (quantized) is roughly the VRAM needed for full offload
    // Add 20% overhead for KV cache and context
    (model_size_gb * 1024.0 * 1.2) as u32
}

pub fn calculate_gpu_layers(model_size_gb: f32, gpus: &[GpuInfo], total_layers: i32) -> i32 {
    if gpus.is_empty() {
        return -1;
    }

    let total_vram: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
    let vram_needed = estimate_full_vram(model_size_gb);

    if total_vram >= vram_needed {
        return -1;
    }

    if total_layers <= 0 {
        let estimated_layers = (model_size_gb * 4.5) as i32;
        let vram_per_layer = vram_needed / estimated_layers.max(1) as u32;
        return (total_vram / vram_per_layer.max(1)) as i32;
    }

    let vram_per_layer = vram_needed / total_layers as u32;
    let layers_fit = (total_vram / vram_per_layer.max(1)) as i32;

    layers_fit.min(total_layers).max(-1)
}

pub fn get_system_info_summary() -> String {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();

    let total_ram_gb = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let gpus = gpu_detector::detect_gpus();
    let total_vram_gb: f32 = gpus.iter().map(|g| g.total_vram_mb as f32).sum::<f32>() / 1024.0;

    let gpu_info = if gpus.is_empty() {
        "No GPUs detected".to_string()
    } else {
        gpus.iter()
            .map(|g| {
                format!(
                    "GPU{}: {} ({:.1} GB)",
                    g.index,
                    g.name,
                    g.total_vram_mb as f32 / 1024.0
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    format!(
        "RAM: {:.1} GB | VRAM: {:.1} GB | {}",
        total_ram_gb, total_vram_gb, gpu_info
    )
}

pub fn recommend_gpu_layers(model_size_gb: f32, total_layers: i32) -> i32 {
    let gpus = gpu_detector::detect_gpus();
    calculate_gpu_layers(model_size_gb, &gpus, total_layers)
}
