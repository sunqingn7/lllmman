use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModelSource {
    Scanned,
    Manual,
    Downloaded,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    TextOnly,
    Tooling,
    Vision,
    Multimodal,
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::TextOnly
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub path: String,
    pub name: String,
    pub size_gb: f32,
    pub quantization: String,
    pub source: ModelSource,
    pub model_type: ModelType,
}
