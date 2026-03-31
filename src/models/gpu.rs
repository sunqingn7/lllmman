use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u32,
    pub index: u32,
    pub provider: GpuProvider,
    pub temperature_c: Option<f32>,
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
