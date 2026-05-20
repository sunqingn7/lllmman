use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u32,
    pub index: u32,
    pub provider: GpuProvider,
    pub temperature_c: Option<f32>,
    #[serde(default)]
    pub compute_capability: Option<(u32, u32)>,
    #[serde(default)]
    pub performance_tier: GpuTier,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuTier {
    #[default]
    Low,
    Mid,
    High,
    Ultra,
}

impl GpuTier {
    pub fn label(&self) -> &'static str {
        match self {
            GpuTier::Low => "Low",
            GpuTier::Mid => "Mid",
            GpuTier::High => "High",
            GpuTier::Ultra => "Ultra",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GpuProvider {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuUsage {
    pub index: u32,
    pub used_vram_mb: u32,
    pub temperature_c: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuTemperature {
    pub index: u32,
    pub name: String,
    pub temperature_c: Option<f32>,
}
