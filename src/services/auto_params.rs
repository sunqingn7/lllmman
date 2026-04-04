use crate::core::CpuOffloadMode;
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

fn estimate_vram_with_context(model_size_gb: f32, context_size: u32) -> u32 {
    if model_size_gb <= 0.0 {
        return 0;
    }
    let base = (model_size_gb * 1024.0) as u32;
    // KV cache: ~2 bytes per token per layer per context, estimate ~1MB per 1024 context for 32-layer model
    let kv_cache_mb = (context_size as u32 * 2) / 1024;
    base + kv_cache_mb + (base / 5) // base + kv cache + 20% overhead
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

pub fn calculate_cpu_offload_mode(
    model_size_gb: f32,
    available_vram_mb: u32,
    total_layers: i32,
    context_size: u32,
) -> (CpuOffloadMode, i32) {
    if model_size_gb <= 0.0 {
        return (CpuOffloadMode::Auto, -1);
    }

    let vram_needed = estimate_vram_with_context(model_size_gb, context_size);

    if available_vram_mb >= vram_needed {
        return (CpuOffloadMode::Auto, -1); // VRAM sufficient, full GPU offload
    }

    // VRAM insufficient, calculate partial offload
    if total_layers <= 0 {
        let estimated_layers = (model_size_gb * 4.5) as i32;
        let vram_per_layer = vram_needed / estimated_layers.max(1) as u32;
        let layers_fit = (available_vram_mb / vram_per_layer.max(1)) as i32;
        return (
            CpuOffloadMode::Offload,
            layers_fit.min(estimated_layers).max(0),
        );
    }

    let vram_per_layer = vram_needed / total_layers as u32;
    let layers_fit = (available_vram_mb / vram_per_layer.max(1)) as i32;
    let gpu_layers = layers_fit.min(total_layers).max(0);

    (CpuOffloadMode::Offload, gpu_layers)
}

pub fn should_enable_cpu_offload(model_size_gb: f32, available_vram_mb: u32) -> bool {
    if model_size_gb <= 0.0 {
        return false;
    }
    let vram_needed = estimate_full_vram(model_size_gb);
    available_vram_mb < vram_needed
}

pub struct VllmSmartConfig {
    pub tensor_parallel_size: u32,
    pub pipeline_parallel_size: u32,
    pub gpu_memory_utilization: f32,
    pub max_num_seqs: u32,
    pub max_num_batched_tokens: u32,
    pub swap_space: u32,
    pub distributed_executor_backend: String,
    pub enable_expert_parallel: bool,
    pub data_parallel_size: u32,
    pub enable_eplb: bool,
    pub eplb_config: String,
}

impl Default for VllmSmartConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 0,
            gpu_memory_utilization: 0.9,
            max_num_seqs: 256,
            max_num_batched_tokens: 0,
            swap_space: 4,
            distributed_executor_backend: String::new(),
            enable_expert_parallel: false,
            data_parallel_size: 0,
            enable_eplb: false,
            eplb_config: String::new(),
        }
    }
}

impl VllmSmartConfig {
    pub fn to_additional_args(&self) -> String {
        let mut args = Vec::new();

        if self.swap_space > 0 {
            args.push(format!("--swap-space {}", self.swap_space));
        }

        if self.tensor_parallel_size > 1 {
            args.push(format!(
                "--tensor-parallel-size {}",
                self.tensor_parallel_size
            ));
        }

        if self.pipeline_parallel_size > 0 {
            args.push(format!(
                "--pipeline-parallel-size {}",
                self.pipeline_parallel_size
            ));
        }

        if !self.distributed_executor_backend.is_empty() {
            args.push(format!(
                "--distributed-executor-backend {}",
                self.distributed_executor_backend
            ));
        }

        if self.enable_expert_parallel {
            args.push("--enable-expert-parallel".to_string());

            if self.data_parallel_size > 0 {
                args.push(format!("--data-parallel-size {}", self.data_parallel_size));
            }

            if self.enable_eplb {
                args.push("--enable-eplb".to_string());
                if !self.eplb_config.is_empty() {
                    args.push(format!("--eplb-config {}", self.eplb_config));
                }
            }
        }

        args.join(" ")
    }
}

pub fn calculate_vllm_smart_args(
    model_size_gb: f32,
    context_size: u32,
    is_moe: bool,
    gpus: Option<&[GpuInfo]>,
) -> String {
    let config = calculate_vllm_smart_config(model_size_gb, context_size, is_moe, gpus);
    config.to_additional_args()
}

pub fn apply_vllm_smart_config(
    model_size_gb: f32,
    context_size: u32,
    is_moe: bool,
    gpus: Option<&[GpuInfo]>,
    server_config: &mut crate::core::ProviderConfig,
) {
    let smart = calculate_vllm_smart_config(model_size_gb, context_size, is_moe, gpus);

    if server_config.gpu_layers <= 0 {
        server_config.gpu_layers = (smart.gpu_memory_utilization * 100.0) as i32;
    }

    if server_config.threads == 0 || server_config.threads == 8 {
        server_config.threads = smart.max_num_seqs;
    }

    if server_config.batch_size == 0 || server_config.batch_size == 512 {
        if smart.max_num_batched_tokens > 0 {
            server_config.batch_size = smart.max_num_batched_tokens;
        }
    }

    let extra_args = smart.to_additional_args();
    if !extra_args.is_empty() && server_config.additional_args.is_empty() {
        server_config.additional_args = extra_args;
    }
}

pub fn calculate_vllm_smart_config(
    model_size_gb: f32,
    context_size: u32,
    is_moe: bool,
    gpus: Option<&[GpuInfo]>,
) -> VllmSmartConfig {
    let mut config = VllmSmartConfig::default();

    let gpus = match gpus {
        Some(g) => g.to_vec(),
        None => gpu_detector::detect_gpus(),
    };

    if gpus.is_empty() {
        return config;
    }

    let num_gpus = gpus.len() as u32;
    let total_vram_mb: u32 = gpus.iter().map(|g| g.total_vram_mb).sum();
    let min_vram_mb = gpus.iter().map(|g| g.total_vram_mb).min().unwrap_or(0);
    let total_ram_mb = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        (sys.total_memory() / (1024 * 1024)) as u32
    };

    let model_vram_mb = (model_size_gb * 1024.0) as u32;
    let kv_cache_estimate_mb = estimate_kv_cache_for_context(model_size_gb, context_size);
    let total_needed_mb = model_vram_mb + kv_cache_estimate_mb;

    if is_moe {
        calculate_moe_config(
            &mut config,
            model_size_gb,
            context_size,
            num_gpus,
            total_vram_mb,
            min_vram_mb,
            total_ram_mb,
            model_vram_mb,
            total_needed_mb,
        );
    } else {
        calculate_dense_config(
            &mut config,
            model_size_gb,
            context_size,
            num_gpus,
            total_vram_mb,
            min_vram_mb,
            total_ram_mb,
            model_vram_mb,
            total_needed_mb,
        );
    }

    config
}

fn estimate_kv_cache_for_context(model_size_gb: f32, context_size: u32) -> u32 {
    let num_layers = estimate_num_layers(model_size_gb);
    let hidden_size = estimate_hidden_size(model_size_gb);
    // KV cache: 2 (K and V) * num_layers * hidden_size * context_size * bytes_per_element
    // For fp16: 2 bytes per element
    // Simplified: ~2 bytes * num_layers * context_size * (hidden_size / 1024)
    let kv_cache_mb =
        (2 * num_layers as u64 * context_size as u64 * (hidden_size as u64 / 1024)) / (1024 * 1024);
    kv_cache_mb as u32
}

fn estimate_num_layers(model_size_gb: f32) -> u32 {
    if model_size_gb < 2.0 {
        24
    } else if model_size_gb < 5.0 {
        32
    } else if model_size_gb < 15.0 {
        40
    } else if model_size_gb < 30.0 {
        48
    } else if model_size_gb < 60.0 {
        64
    } else if model_size_gb < 100.0 {
        80
    } else {
        120
    }
}

fn estimate_hidden_size(model_size_gb: f32) -> u32 {
    if model_size_gb < 2.0 {
        2048
    } else if model_size_gb < 5.0 {
        3072
    } else if model_size_gb < 15.0 {
        4096
    } else if model_size_gb < 30.0 {
        5120
    } else if model_size_gb < 60.0 {
        6144
    } else if model_size_gb < 100.0 {
        8192
    } else {
        16384
    }
}

fn calculate_dense_config(
    config: &mut VllmSmartConfig,
    _model_size_gb: f32,
    _context_size: u32,
    num_gpus: u32,
    total_vram_mb: u32,
    min_vram_mb: u32,
    total_ram_mb: u32,
    model_vram_mb: u32,
    total_needed_mb: u32,
) {
    if model_vram_mb <= min_vram_mb {
        config.tensor_parallel_size = 1;
        let vram_headroom = min_vram_mb as f32 / model_vram_mb as f32;
        config.gpu_memory_utilization = if vram_headroom > 1.5 {
            0.95
        } else if vram_headroom > 1.2 {
            0.9
        } else {
            0.85
        };
    } else if total_needed_mb <= total_vram_mb {
        config.tensor_parallel_size = num_gpus;
        let per_gpu_needed = total_needed_mb / num_gpus;
        let utilization = (per_gpu_needed as f32 / min_vram_mb as f32).min(0.95);
        config.gpu_memory_utilization = utilization.max(0.7);

        if num_gpus > 4 {
            config.distributed_executor_backend = "mp".to_string();
        }
    } else {
        config.tensor_parallel_size = num_gpus;
        config.gpu_memory_utilization = 0.9;
    }

    if num_gpus > 1 && num_gpus % 2 != 0 {
        config.pipeline_parallel_size = num_gpus;
        config.tensor_parallel_size = 1;
    }

    config.max_num_seqs = calculate_max_num_seqs(min_vram_mb, model_vram_mb, num_gpus);
    config.max_num_batched_tokens = 0;
    config.swap_space = calculate_swap_space(total_ram_mb, total_vram_mb, total_needed_mb);
}

fn calculate_moe_config(
    config: &mut VllmSmartConfig,
    _model_size_gb: f32,
    context_size: u32,
    num_gpus: u32,
    total_vram_mb: u32,
    min_vram_mb: u32,
    total_ram_mb: u32,
    model_vram_mb: u32,
    total_needed_mb: u32,
) {
    let _kv_cache_mb = estimate_kv_cache_for_context(_model_size_gb, context_size);

    if num_gpus == 1 {
        config.tensor_parallel_size = 1;
        config.data_parallel_size = 0;
        config.enable_expert_parallel = false;

        let vram_headroom = min_vram_mb as f32 / model_vram_mb as f32;
        config.gpu_memory_utilization = if vram_headroom > 1.5 {
            0.95
        } else if vram_headroom > 1.2 {
            0.9
        } else {
            0.85
        };
    } else if total_needed_mb <= total_vram_mb && num_gpus >= 4 {
        config.tensor_parallel_size = 1;
        config.data_parallel_size = num_gpus;
        config.enable_expert_parallel = true;
        config.gpu_memory_utilization = 0.9;

        if total_ram_mb > 64 * 1024 {
            config.enable_eplb = true;
            let num_redundant = if num_gpus >= 8 { 32 } else { 2 };
            config.eplb_config = format!(
                r#"{{"window_size":1000,"step_interval":3000,"num_redundant_experts":{},"log_balancedness":true}}"#,
                num_redundant
            );
        }
    } else if total_needed_mb <= total_vram_mb {
        config.tensor_parallel_size = num_gpus;
        config.data_parallel_size = 0;
        config.enable_expert_parallel = false;

        let per_gpu_needed = total_needed_mb / num_gpus;
        let utilization = (per_gpu_needed as f32 / min_vram_mb as f32).min(0.95);
        config.gpu_memory_utilization = utilization.max(0.7);
    } else {
        config.tensor_parallel_size = num_gpus;
        config.data_parallel_size = 0;
        config.enable_expert_parallel = false;
        config.gpu_memory_utilization = 0.9;
    }

    config.max_num_seqs = calculate_max_num_seqs(min_vram_mb, model_vram_mb, num_gpus);
    config.max_num_batched_tokens = 0;
    config.swap_space = calculate_swap_space(total_ram_mb, total_vram_mb, total_needed_mb);
}

fn calculate_max_num_seqs(min_vram_mb: u32, model_vram_mb: u32, num_gpus: u32) -> u32 {
    let per_gpu_vram = min_vram_mb / num_gpus.max(1);
    let available_for_kv = if per_gpu_vram > model_vram_mb {
        per_gpu_vram - model_vram_mb
    } else {
        per_gpu_vram / 4
    };

    if available_for_kv > 8192 {
        512
    } else if available_for_kv > 4096 {
        256
    } else if available_for_kv > 2048 {
        128
    } else {
        64
    }
}

fn calculate_swap_space(total_ram_mb: u32, total_vram_mb: u32, total_needed_mb: u32) -> u32 {
    let ram_gb = total_ram_mb / 1024;

    if ram_gb > 128 {
        16
    } else if ram_gb > 64 {
        8
    } else if ram_gb > 32 {
        4
    } else if ram_gb > 16 {
        2
    } else {
        let deficit_mb = total_needed_mb.saturating_sub(total_vram_mb);
        (deficit_mb / 1024).max(2)
    }
}
