use crate::core::{DownloadableModel, ModelDownloader, ModelInfo, ProviderError, Result};

pub struct HuggingFaceDownloader;

impl HuggingFaceDownloader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HuggingFaceDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDownloader for HuggingFaceDownloader {
    async fn search(&self, query: &str) -> Result<Vec<DownloadableModel>> {
        // HuggingFace API search endpoint
        let search_url = format!(
            "https://huggingface.co/api/models?search={}&sort=trending&limit=20", 
            urlencoding::encode(query)
        );
        
        let response = reqwest::get(&search_url)
            .await
            .map_err(|e| ProviderError::ServerError(e.to_string()))?;
        
        let models: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| ProviderError::ServerError(e.to_string()))?;
        
        let mut result = Vec::new();
        for model in models {
            // Only include GGUF models
            if model["pipeline_tag"] == "text-generation" || model["library_name"] == "llama-cpp-python" {
                let id = model["id"].as_str().unwrap_or("").to_string();
                let name = model["id"].as_str().unwrap_or("").to_string();
                let downloads = model["downloads"].as_u64().unwrap_or(0) as u32;
                
                // Estimate size (we'd need to fetch file info for accurate size)
                let size_gb = 0.0; 
                
                result.push(DownloadableModel {
                    id,
                    name,
                    size_gb,
                    downloads,
                });
            }
        }
        
        Ok(result)
    }
    
    fn download(&self, _model_id: &str, _dest_dir: &str) -> Result<ModelInfo> {
        // For now, this is a placeholder - actual download would need:
        // 1. Fetch model info from HuggingFace
        // 2. Find GGUF files in the model
        // 3. Download the file with progress
        // 4. Return ModelInfo
        
        return Err(ProviderError::ServerError(
            "Model download not fully implemented yet".to_string()
        ));
    }
}
