use crate::models::deployment::{DeploymentMode, RouterPolicy, RouterProvider};
use crate::models::gpu::{GpuInfo, GpuTier};

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub action: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecommendationCategory {
    Deployment,
    Provider,
    Router,
    Performance,
    Safety,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl RecommendationCategory {
    pub fn icon(&self) -> &'static str {
        match self {
            RecommendationCategory::Deployment => "🚀",
            RecommendationCategory::Provider => "⚙️",
            RecommendationCategory::Router => "🔀",
            RecommendationCategory::Performance => "⚡",
            RecommendationCategory::Safety => "🛡️",
        }
    }
}

impl RecommendationPriority {
    pub fn color_label(&self) -> &'static str {
        match self {
            RecommendationPriority::Critical => "CRITICAL",
            RecommendationPriority::High => "HIGH",
            RecommendationPriority::Medium => "MEDIUM",
            RecommendationPriority::Low => "LOW",
            RecommendationPriority::Info => "INFO",
        }
    }
}

pub struct RecommendationEngine;

impl RecommendationEngine {
    pub fn analyze(gpus: &[GpuInfo], model_size_gb: f32, model_name: &str) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        if gpus.is_empty() {
            recs.push(Recommendation {
                category: RecommendationCategory::Deployment,
                title: "No GPUs detected".to_string(),
                description: "Running in CPU-only mode. Performance will be significantly limited.".to_string(),
                priority: RecommendationPriority::High,
                action: Some("Install CUDA drivers and verify GPU detection.".to_string()),
            });
            return recs;
        }

        let gpu_count = gpus.len() as u32;
        let total_vram_gb: f32 = gpus.iter().map(|g| g.total_vram_mb as f32 / 1024.0).sum();
        let min_vram_gb = gpus.iter().map(|g| g.total_vram_mb as f32 / 1024.0).fold(f32::MAX, f32::min);
        let max_vram_gb = gpus.iter().map(|g| g.total_vram_mb as f32 / 1024.0).fold(0.0f32, f32::max);
        let is_heterogeneous = Self::is_heterogeneous(gpus);
        let has_low_tier = gpus.iter().any(|g| g.performance_tier == GpuTier::Low);
        let has_ultra_tier = gpus.iter().any(|g| g.performance_tier == GpuTier::Ultra);

        // Deployment mode recommendations
        if gpu_count == 1 {
            recs.push(Recommendation {
                category: RecommendationCategory::Deployment,
                title: "Single GPU mode recommended".to_string(),
                description: format!(
                    "With 1 GPU ({} GB VRAM), use Single GPU mode. Model {:.1} GB fits in VRAM.",
                    gpus[0].total_vram_mb as f32 / 1024.0, model_size_gb
                ),
                priority: RecommendationPriority::Info,
                action: Some("Use --deployment-mode single-gpu".to_string()),
            });
        } else if is_heterogeneous {
            recs.push(Recommendation {
                category: RecommendationCategory::Deployment,
                title: "Multi-Instance + Router required".to_string(),
                description: format!(
                    "Heterogeneous GPUs detected ({} GPUs, VRAM range: {:.0}-{:.0} GB). \
                     Tensor Parallelism will fail. Use separate instances per GPU with a router.",
                    gpu_count, min_vram_gb, max_vram_gb
                ),
                priority: RecommendationPriority::Critical,
                action: Some("Use multi-instance mode with SGLang Router.".to_string()),
            });
        } else if gpu_count >= 2 && !is_heterogeneous {
            if model_size_gb > min_vram_gb * 0.8 {
                recs.push(Recommendation {
                    category: RecommendationCategory::Deployment,
                    title: "Data Parallel recommended".to_string(),
                    description: format!(
                        "Model ({:.1} GB) is large relative to per-GPU VRAM ({:.0} GB). \
                         Use Data Parallel mode across {} identical GPUs for higher throughput.",
                        model_size_gb, min_vram_gb, gpu_count
                    ),
                    priority: RecommendationPriority::High,
                    action: Some(format!("Use --deployment-mode data-parallel with {} instances.", gpu_count)),
                });
            } else {
                recs.push(Recommendation {
                    category: RecommendationCategory::Deployment,
                    title: "Multi-Instance for throughput".to_string(),
                    description: format!(
                        "Model ({:.1} GB) fits in each GPU ({:.0} GB). \
                         Run multiple instances for higher concurrent request handling.",
                        model_size_gb, min_vram_gb
                    ),
                    priority: RecommendationPriority::Medium,
                    action: Some(format!("Deploy {} instances with round-robin routing.", gpu_count)),
                });
            }
        }

        // Provider recommendations
        let is_vision_model = model_name.to_lowercase().contains("vision")
            || model_name.to_lowercase().contains("llava")
            || model_name.to_lowercase().contains("qwen2-vl");

        if is_vision_model {
            recs.push(Recommendation {
                category: RecommendationCategory::Provider,
                title: "Use vLLM or SGLang for vision models".to_string(),
                description: "llama.cpp has limited vision model support. Use vLLM or SGLang for multimodal inference.".to_string(),
                priority: RecommendationPriority::High,
                action: Some("Select vLLM or SGLang as provider.".to_string()),
            });
        }

        if has_ultra_tier && model_size_gb > 20.0 {
            recs.push(Recommendation {
                category: RecommendationCategory::Provider,
                title: "SGLang recommended for large models on Hopper/Blackwell".to_string(),
                description: "SGLang has superior performance on SM 9.0+ GPUs with RadixAttention and CUDA graph optimizations.".to_string(),
                priority: RecommendationPriority::Medium,
                action: Some("Use SGLang provider for best performance.".to_string()),
            });
        }

        // Router recommendations
        if is_heterogeneous && gpu_count >= 2 {
            recs.push(Recommendation {
                category: RecommendationCategory::Router,
                title: "SGLang Router with Cache-Aware policy".to_string(),
                description: "For heterogeneous GPUs, SGLang Router with cache-aware routing maximizes KV cache utilization across different GPU tiers.".to_string(),
                priority: RecommendationPriority::High,
                action: Some("Set router_provider=sglang_router, policy=cache_aware".to_string()),
            });

            if has_low_tier {
                recs.push(Recommendation {
                    category: RecommendationCategory::Router,
                    title: "Power-of-Two routing for mixed-tier setups".to_string(),
                    description: "With mixed GPU tiers (Low + higher), Power-of-Two routing picks the best \
                         of 2 random workers, balancing load while avoiding the slowest GPU.".to_string(),
                    priority: RecommendationPriority::Medium,
                    action: Some("Set router_policy=power_of_two".to_string()),
                });
            }
        }

        // Performance tuning recommendations
        if model_size_gb > min_vram_gb * 0.9 {
            recs.push(Recommendation {
                category: RecommendationCategory::Performance,
                title: "Model nearly fills GPU VRAM".to_string(),
                description: format!(
                    "Model ({:.1} GB) uses >90% of smallest GPU's VRAM ({:.0} GB). \
                     Reduce context size or use lower quantization to avoid OOM.",
                    model_size_gb, min_vram_gb
                ),
                priority: RecommendationPriority::Critical,
                action: Some("Reduce --max-model-len or use a lower quantization (e.g., Q4 instead of Q8).".to_string()),
            });
        }

        if is_heterogeneous {
            recs.push(Recommendation {
                category: RecommendationCategory::Safety,
                title: "Enable safety flags for heterogeneous setup".to_string(),
                description: "Heterogeneous multi-instance setups require specific environment variables to avoid P2P and CUDA graph issues.".to_string(),
                priority: RecommendationPriority::High,
                action: Some("Set VLLM_SKIP_P2P_CHECK=1 and --enforce-eager (vLLM) or --disable-cuda-graph (SGLang).".to_string()),
            });
        }

        if total_vram_gb > 48.0 && model_size_gb < total_vram_gb * 0.3 {
            recs.push(Recommendation {
                category: RecommendationCategory::Performance,
                title: "Consider speculative decoding".to_string(),
                description: format!(
                    "Total VRAM ({:.0} GB) is much larger than model size ({:.1} GB). \
                     You have room for a draft model to accelerate generation.",
                    total_vram_gb, model_size_gb
                ),
                priority: RecommendationPriority::Medium,
                action: Some("Add --speculative-draft-model with a small draft model (e.g., EAGLE or AutoSpec).".to_string()),
            });
        }

        if gpus.iter().any(|g| g.compute_capability.map(|(maj, _)| maj >= 9).unwrap_or(false)) {
            recs.push(Recommendation {
                category: RecommendationCategory::Performance,
                title: "Enable FP8 attention on Hopper+".to_string(),
                description: "GPUs with SM 9.0+ support FP8 attention for 2x throughput improvement on attention-heavy workloads.".to_string(),
                priority: RecommendationPriority::Medium,
                action: Some("Add --attention-backend flashinfer --kv-cache-dtype fp8_e5m2".to_string()),
            });
        }

        // Info recommendations
        let tier_label = |t: &GpuTier| match t {
            GpuTier::Low => "Low",
            GpuTier::Mid => "Mid",
            GpuTier::High => "High",
            GpuTier::Ultra => "Ultra",
        };
        recs.push(Recommendation {
            category: RecommendationCategory::Deployment,
            title: format!("System summary: {} GPU(s), {:.0} GB total VRAM", gpu_count, total_vram_gb),
            description: gpus.iter().map(|g| format!("  GPU {}: {} ({:.0} GB, {})", g.index, g.name, g.total_vram_mb as f32 / 1024.0, tier_label(&g.performance_tier))).collect::<Vec<_>>().join("\n"),
            priority: RecommendationPriority::Info,
            action: None,
        });

        recs
    }

    pub fn is_heterogeneous(gpus: &[GpuInfo]) -> bool {
        if gpus.len() < 2 {
            return false;
        }
        let caps: Vec<_> = gpus.iter().filter_map(|g| g.compute_capability).collect();
        if caps.len() < 2 {
            return false;
        }
        !caps.iter().all(|c| c == &caps[0])
    }

    pub fn recommended_deployment_mode(gpus: &[GpuInfo], model_size_gb: f32) -> DeploymentMode {
        if gpus.len() <= 1 {
            return DeploymentMode::SingleGpu;
        }

        let is_heterogeneous = Self::is_heterogeneous(gpus);
        let min_vram_gb = gpus.iter().map(|g| g.total_vram_mb as f32 / 1024.0).fold(f32::MAX, f32::min);

        if is_heterogeneous {
            DeploymentMode::MultiInstance
        } else if model_size_gb > min_vram_gb * 0.8 {
            DeploymentMode::DataParallel
        } else {
            DeploymentMode::MultiInstance
        }
    }

    pub fn recommended_router_provider(_gpus: &[GpuInfo]) -> RouterProvider {
        RouterProvider::SglangRouter
    }

    pub fn recommended_router_policy(gpus: &[GpuInfo]) -> RouterPolicy {
        if gpus.iter().any(|g| g.performance_tier == GpuTier::Low) && gpus.iter().any(|g| g.performance_tier == GpuTier::High || g.performance_tier == GpuTier::Ultra) {
            RouterPolicy::PowerOfTwo
        } else {
            RouterPolicy::CacheAware
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gpu(name: &str, vram_mb: u32, cap: Option<(u32, u32)>, tier: GpuTier) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            total_vram_mb: vram_mb,
            index: 0,
            provider: crate::models::gpu::GpuProvider::Nvidia,
            temperature_c: None,
            compute_capability: cap,
            performance_tier: tier,
        }
    }

    #[test]
    fn test_single_gpu_recommendation() {
        let gpus = vec![make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High)];
        let recs = RecommendationEngine::analyze(&gpus, 8.0, "llama-3-8b");
        assert!(recs.iter().any(|r| r.title.contains("Single GPU")));
    }

    #[test]
    fn test_heterogeneous_recommendation() {
        let gpus = vec![
            make_gpu("RTX 3090", 24576, Some((8, 6)), GpuTier::Mid),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
        ];
        let recs = RecommendationEngine::analyze(&gpus, 8.0, "llama-3-8b");
        assert!(recs.iter().any(|r| r.title.contains("Multi-Instance")));
        assert!(recs.iter().any(|r| r.title.contains("SGLang Router")));
    }

    #[test]
    fn test_is_heterogeneous() {
        let gpus_same = vec![
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
        ];
        assert!(!RecommendationEngine::is_heterogeneous(&gpus_same));

        // Verify order-independent detection: [8.9, 8.9, 8.6] should still be heterogeneous
        let gpus_unsorted = vec![
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
            make_gpu("RTX 3090", 24576, Some((8, 6)), GpuTier::Mid),
        ];
        assert!(RecommendationEngine::is_heterogeneous(&gpus_unsorted));

        let gpus_diff = vec![
            make_gpu("RTX 3090", 24576, Some((8, 6)), GpuTier::Mid),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
        ];
        assert!(RecommendationEngine::is_heterogeneous(&gpus_diff));
    }

    #[test]
    fn test_recommended_deployment_mode() {
        let single = vec![make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High)];
        assert_eq!(RecommendationEngine::recommended_deployment_mode(&single, 8.0), DeploymentMode::SingleGpu);

        let hetero = vec![
            make_gpu("RTX 3090", 24576, Some((8, 6)), GpuTier::Mid),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
        ];
        assert_eq!(RecommendationEngine::recommended_deployment_mode(&hetero, 8.0), DeploymentMode::MultiInstance);
    }

    #[test]
    fn test_vram_warning() {
        let gpus = vec![make_gpu("RTX 3060", 12288, Some((8, 6)), GpuTier::Mid)];
        let recs = RecommendationEngine::analyze(&gpus, 11.0, "llama-3-70b");
        assert!(recs.iter().any(|r| r.priority == RecommendationPriority::Critical && r.title.contains("VRAM")));
    }

    #[test]
    fn test_speculative_decoding_recommendation() {
        let gpus = vec![
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
            make_gpu("RTX 4090", 24576, Some((8, 9)), GpuTier::High),
        ];
        let recs = RecommendationEngine::analyze(&gpus, 8.0, "llama-3-8b");
        assert!(recs.iter().any(|r| r.title.contains("speculative")));
    }
}
