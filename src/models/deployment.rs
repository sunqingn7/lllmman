use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Deployment mode for multi-GPU setups.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum DeploymentMode {
    #[default]
    SingleGpu,
    DataParallel,
    MultiInstance,
    PipelineParallel,
}

impl DeploymentMode {
    pub fn label(&self) -> &'static str {
        match self {
            DeploymentMode::SingleGpu => "Single GPU",
            DeploymentMode::DataParallel => "Data Parallel",
            DeploymentMode::MultiInstance => "Multi-Instance",
            DeploymentMode::PipelineParallel => "Pipeline Parallel",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            DeploymentMode::SingleGpu => "One model, one GPU (current behavior)",
            DeploymentMode::DataParallel => "Same model on identical GPUs for throughput",
            DeploymentMode::MultiInstance => "Different models or heterogeneous GPUs with router",
            DeploymentMode::PipelineParallel => "Model split across GPUs for large models",
        }
    }
}

/// Router provider type.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum RouterProvider {
    #[default]
    SglangRouter,
    Nginx,
    Custom,
}

impl RouterProvider {
    pub fn label(&self) -> &'static str {
        match self {
            RouterProvider::SglangRouter => "SGLang Router",
            RouterProvider::Nginx => "Nginx",
            RouterProvider::Custom => "Custom",
        }
    }
}

/// Router load balancing policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum RouterPolicy {
    #[default]
    CacheAware,
    RoundRobin,
    PowerOfTwo,
    Random,
}

/// Configuration for a single server instance.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InstanceConfig {
    pub gpu_indices: Vec<u32>,
    pub port: u16,
    pub memory_utilization: f32,
    pub max_num_seqs: Option<u32>,
    pub model_path: String,
    pub context_size: u32,
    pub provider: String,
}

/// Configuration for the request router.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RouterConfig {
    pub provider: RouterProvider,
    pub policy: RouterPolicy,
    pub worker_urls: Vec<String>,
    pub router_port: u16,
    #[serde(default)]
    pub pd_disaggregation: bool,
}

/// A complete deployment profile for multi-instance setups.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeploymentProfile {
    pub mode: DeploymentMode,
    pub instances: Vec<InstanceConfig>,
    pub router_config: Option<RouterConfig>,
    pub env_overrides: HashMap<String, String>,
    pub generated_script: String,
}

/// Auto-select the optimal deployment mode based on detected GPUs and model size.
pub fn auto_select_deployment_mode(
    gpu_count: u32,
    unique_archs: u32,
    model_size_gb: u64,
    largest_gpu_vram_gb: u64,
) -> DeploymentMode {
    if gpu_count <= 1 {
        return DeploymentMode::SingleGpu;
    }

    if model_size_gb > largest_gpu_vram_gb {
        return DeploymentMode::PipelineParallel;
    }

    if unique_archs > 1 {
        return DeploymentMode::MultiInstance;
    }

    if gpu_count > 1 {
        return DeploymentMode::DataParallel;
    }

    DeploymentMode::SingleGpu
}

/// Generate a default deployment profile for the given mode.
pub fn generate_default_profile(
    mode: DeploymentMode,
    provider: &str,
    model_path: &str,
    gpu_count: u32,
) -> DeploymentProfile {
    let mut profile = DeploymentProfile {
        mode,
        ..Default::default()
    };

    match mode {
        DeploymentMode::SingleGpu => {
            profile.instances = vec![InstanceConfig {
                gpu_indices: vec![0],
                port: 8080,
                memory_utilization: 0.85,
                max_num_seqs: Some(128),
                model_path: model_path.to_string(),
                context_size: 4096,
                provider: provider.to_string(),
            }];
        }
        DeploymentMode::DataParallel => {
            for i in 0..gpu_count {
                profile.instances.push(InstanceConfig {
                    gpu_indices: vec![i],
                    port: 8080 + i as u16,
                    memory_utilization: 0.85,
                    max_num_seqs: Some(128),
                    model_path: model_path.to_string(),
                    context_size: 4096,
                    provider: provider.to_string(),
                });
            }
            profile.router_config = Some(RouterConfig {
                provider: RouterProvider::SglangRouter,
                policy: RouterPolicy::CacheAware,
                worker_urls: (0..gpu_count)
                    .map(|i| format!("http://localhost:{}", 8080 + i))
                    .collect(),
                router_port: 30000,
                ..Default::default()
            });
        }
        DeploymentMode::MultiInstance => {
            for i in 0..gpu_count {
                let mem_util = if i == 0 {
                    0.90
                } else {
                    0.85 - (i as f32 * 0.05)
                };
                profile.instances.push(InstanceConfig {
                    gpu_indices: vec![i],
                    port: 8080 + i as u16,
                    memory_utilization: mem_util.max(0.70),
                    max_num_seqs: Some(if i == 0 { 256 } else { 128 - (i as u32 * 32) }),
                    model_path: model_path.to_string(),
                    context_size: 4096,
                    provider: provider.to_string(),
                });
            }
            profile.router_config = Some(RouterConfig {
                provider: RouterProvider::SglangRouter,
                policy: RouterPolicy::CacheAware,
                worker_urls: (0..gpu_count)
                    .map(|i| format!("http://localhost:{}", 8080 + i))
                    .collect(),
                router_port: 30000,
                ..Default::default()
            });
        }
        DeploymentMode::PipelineParallel => {
            profile.instances = vec![InstanceConfig {
                gpu_indices: (0..gpu_count).collect(),
                port: 8080,
                memory_utilization: 0.85,
                max_num_seqs: Some(64),
                model_path: model_path.to_string(),
                context_size: 4096,
                provider: provider.to_string(),
            }];
        }
    }

    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_select_single_gpu() {
        assert_eq!(auto_select_deployment_mode(1, 1, 8, 24), DeploymentMode::SingleGpu);
    }

    #[test]
    fn test_auto_select_data_parallel() {
        assert_eq!(
            auto_select_deployment_mode(4, 1, 8, 24),
            DeploymentMode::DataParallel
        );
    }

    #[test]
    fn test_auto_select_multi_instance() {
        assert_eq!(
            auto_select_deployment_mode(3, 2, 8, 24),
            DeploymentMode::MultiInstance
        );
    }

    #[test]
    fn test_auto_select_pipeline_parallel() {
        assert_eq!(
            auto_select_deployment_mode(4, 1, 140, 24),
            DeploymentMode::PipelineParallel
        );
    }

    #[test]
    fn test_generate_single_gpu_profile() {
        let profile = generate_default_profile(DeploymentMode::SingleGpu, "vllm", "model", 1);
        assert_eq!(profile.mode, DeploymentMode::SingleGpu);
        assert_eq!(profile.instances.len(), 1);
        assert_eq!(profile.instances[0].port, 8080);
        assert!(profile.router_config.is_none());
    }

    #[test]
    fn test_generate_multi_instance_profile() {
        let profile = generate_default_profile(DeploymentMode::MultiInstance, "vllm", "model", 3);
        assert_eq!(profile.mode, DeploymentMode::MultiInstance);
        assert_eq!(profile.instances.len(), 3);
        assert!(profile.router_config.is_some());
        let router = profile.router_config.as_ref().unwrap();
        assert_eq!(router.worker_urls.len(), 3);
        assert_eq!(router.router_port, 30000);
    }

    #[test]
    fn test_generate_data_parallel_profile() {
        let profile = generate_default_profile(DeploymentMode::DataParallel, "sglang", "model", 2);
        assert_eq!(profile.mode, DeploymentMode::DataParallel);
        assert_eq!(profile.instances.len(), 2);
        assert_eq!(profile.instances[0].port, 8080);
        assert_eq!(profile.instances[1].port, 8081);
    }

    #[test]
    fn test_deployment_mode_labels() {
        assert_eq!(DeploymentMode::SingleGpu.label(), "Single GPU");
        assert_eq!(DeploymentMode::MultiInstance.label(), "Multi-Instance");
        assert_eq!(DeploymentMode::PipelineParallel.label(), "Pipeline Parallel");
    }

    #[test]
    fn test_deployment_profile_serialization() {
        let profile = generate_default_profile(DeploymentMode::SingleGpu, "vllm", "model", 1);
        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: DeploymentProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.mode, DeploymentMode::SingleGpu);
        assert_eq!(deserialized.instances.len(), 1);
    }
}
