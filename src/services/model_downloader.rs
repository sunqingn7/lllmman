use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::core::{
    DownloadStatus, DownloadTask, ModelDownloader, ModelSource, ProviderError, Result as ProviderResult,
};

pub struct DownloadManager {
    tasks: Arc<RwLock<Vec<DownloadTask>>>,
    download_dir: String,
}

impl DownloadManager {
    pub fn new(download_dir: String) -> Self {
        Self {
            tasks: Arc::new(RwLock::new(Vec::new())),
            download_dir,
        }
    }

    pub async fn add_task(&self, source: ModelSource, file_name: String) -> String {
        let id = Uuid::new_v4().to_string();
        let file_name_clean = file_name.split('/').last().unwrap_or(&file_name).to_string();
        let dest_path = std::path::PathBuf::from(&self.download_dir).join(&file_name_clean);

        if let Some(parent) = dest_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("Failed to create download directory {}: {}", parent.display(), e);
            }
        }

        let task = DownloadTask {
            id: id.clone(),
            source,
            file_name: file_name_clean,
            dest_path: dest_path.to_string_lossy().to_string(),
            status: DownloadStatus::Pending,
            downloaded_bytes: 0,
            total_bytes: 0,
        };

        let mut tasks = self.tasks.write().await;
        tasks.push(task);
        id
    }

    pub fn add_task_sync(&self, source: ModelSource, file_name: String) -> String {
        let id = Uuid::new_v4().to_string();
        let file_name_clean = file_name.split('/').last().unwrap_or(&file_name).to_string();
        let dest_path = std::path::PathBuf::from(&self.download_dir).join(&file_name_clean);

        if let Some(parent) = dest_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                log::warn!("Failed to create download directory {}: {}", parent.display(), e);
            }
        }

        let task = DownloadTask {
            id: id.clone(),
            source,
            file_name: file_name_clean,
            dest_path: dest_path.to_string_lossy().to_string(),
            status: DownloadStatus::Pending,
            downloaded_bytes: 0,
            total_bytes: 0,
        };

        let mut tasks = self.tasks.blocking_write();
        tasks.push(task);
        id
    }

    pub fn get_tasks_sync(&self) -> Vec<DownloadTask> {
        let tasks = self.tasks.blocking_read();
        tasks.clone()
    }

    pub async fn get_tasks(&self) -> Vec<DownloadTask> {
        let tasks = self.tasks.read().await;
        tasks.clone()
    }

    pub async fn get_task(&self, id: &str) -> Option<DownloadTask> {
        let tasks = self.tasks.read().await;
        tasks.iter().find(|t| t.id == id).cloned()
    }

    pub async fn update_task(&self, id: &str, update: DownloadTaskUpdate) {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.iter_mut().find(|t| t.id == id) {
            match update {
                DownloadTaskUpdate::Status(status) => task.status = status,
                DownloadTaskUpdate::Progress { downloaded, total } => {
                    task.downloaded_bytes = downloaded;
                    task.total_bytes = total;
                }
                DownloadTaskUpdate::Complete => {
                    task.status = DownloadStatus::Completed;
                }
                DownloadTaskUpdate::Failed(err) => {
                    task.status = DownloadStatus::Failed(err);
                }
            }
        }
    }

    pub async fn remove_task(&self, id: &str) {
        let mut tasks = self.tasks.write().await;
        tasks.retain(|t| t.id != id);
    }
}

pub enum DownloadTaskUpdate {
    Status(DownloadStatus),
    Progress { downloaded: u64, total: u64 },
    Complete,
    Failed(String),
}

pub struct HuggingFaceDownloader;

impl HuggingFaceDownloader {
    pub fn new() -> Self {
        Self
    }

    pub async fn search(&self, query: &str) -> ProviderResult<Vec<crate::core::DownloadableModel>> {
        let search_url = format!(
            "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit=30",
            urlencoding::encode(query)
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&search_url)
            .send()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let models: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let mut result = Vec::new();
        for model in models {
            let id = model["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }

            let downloads = model["downloads"].as_u64().unwrap_or(0) as u32;

            result.push(crate::core::DownloadableModel {
                id: id.clone(),
                name: id.clone(),
                size_gb: 0.0,
                downloads,
                source: ModelSource::HuggingFace { repo_id: id },
            });
        }

        Ok(result)
    }

    pub fn search_sync(&self, query: &str) -> ProviderResult<Vec<crate::core::DownloadableModel>> {
        let search_url = format!(
            "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit=30",
            urlencoding::encode(query)
        );

        let response = reqwest::blocking::get(&search_url)
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let models: Vec<serde_json::Value> = response
            .json()
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let mut result = Vec::new();
        for model in models {
            let id = model["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }

            let downloads = model["downloads"].as_u64().unwrap_or(0) as u32;

            result.push(crate::core::DownloadableModel {
                id: id.clone(),
                name: id.clone(),
                size_gb: 0.0,
                downloads,
                source: ModelSource::HuggingFace { repo_id: id },
            });
        }

        Ok(result)
    }

    pub async fn list_files(&self, model_id: &str) -> ProviderResult<Vec<ModelFile>> {
        let api_url = format!(
            "https://huggingface.co/api/models/{}/tree/main?recursive=true",
            model_id
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&api_url)
            .send()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let files: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let mut result = Vec::new();
        for item in files {
            if item["type"].as_str() != Some("file") {
                continue;
            }

            let path = item["path"].as_str().unwrap_or("").to_string();
            let size = item["size"].as_u64().unwrap_or(0);

            result.push(ModelFile {
                path,
                size_bytes: size,
            });
        }

        Ok(result)
    }

    pub fn list_files_sync(&self, model_id: &str) -> ProviderResult<Vec<ModelFile>> {
        let api_url = format!("https://huggingface.co/api/models/{}", model_id);

        let response = reqwest::blocking::get(&api_url)
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let json: serde_json::Value = response
            .json()
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let siblings = json["siblings"]
            .as_array()
            .ok_or_else(|| ProviderError::NetworkError("No siblings found".to_string()))?;

        let mut result = Vec::new();
        for item in siblings {
            let path = item["rfilename"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let size = item["size"].as_u64().unwrap_or(0);

            if path.is_empty() {
                continue;
            }

            result.push(ModelFile {
                path,
                size_bytes: size,
            });
        }

        Ok(result)
    }

    pub fn get_download_url(&self, model_id: &str, file_path: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id,
            urlencoding::encode(file_path)
        )
    }
}

impl Default for HuggingFaceDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DirectUrlDownloader {
    fn default() -> Self {
        Self
    }
}

impl Default for GitHubReleaseDownloader {
    fn default() -> Self {
        Self
    }
}

pub struct ModelFile {
    pub path: String,
    pub size_bytes: u64,
}

pub struct DirectUrlDownloader;

impl DirectUrlDownloader {
    pub fn new() -> Self {
        Self
    }

    pub fn fetch_headers_sync(&self, url: &str) -> ProviderResult<DownloadHeaders> {
        let response = reqwest::blocking::Client::new()
            .head(url)
            .send()
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let content_length = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let content_disposition = response
            .headers()
            .get("content-disposition")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Ok(DownloadHeaders {
            content_length,
            file_name: content_disposition
                .and_then(|cd| Self::extract_filename(&cd))
                .unwrap_or_else(|| "download".to_string()),
        })
    }

    fn extract_filename(content_disposition: &str) -> Option<String> {
        for part in content_disposition.split(';') {
            let part = part.trim();
            if part.starts_with("filename=") {
                let filename = part.trim_start_matches("filename=");
                let filename = filename.trim_matches('"');
                return Some(filename.to_string());
            }
        }
        None
    }
}

pub struct DownloadHeaders {
    pub content_length: u64,
    pub file_name: String,
}

pub struct GitHubReleaseDownloader;

impl GitHubReleaseDownloader {
    pub fn new() -> Self {
        Self
    }

    pub fn get_latest_release_sync(&self, owner: &str, repo: &str) -> ProviderResult<GitHubRelease> {
        let url = format!("https://api.github.com/repos/{}/{}/releases/latest", owner, repo);

        let response = reqwest::blocking::Client::new()
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let release: serde_json::Value = response
            .json()
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;

        let tag = release["tag_name"]
            .as_str()
            .unwrap_or("latest")
            .to_string();

        let mut assets = Vec::new();
        if let Some(assets_arr) = release["assets"].as_array() {
            for asset in assets_arr {
                assets.push(GitHubAsset {
                    name: asset["name"].as_str().unwrap_or("").to_string(),
                    size: asset["size"].as_u64().unwrap_or(0),
                    download_url: asset["browser_download_url"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                });
            }
        }

        Ok(GitHubRelease { tag, assets })
    }
}

pub struct GitHubRelease {
    pub tag: String,
    pub assets: Vec<GitHubAsset>,
}

pub struct GitHubAsset {
    pub name: String,
    pub size: u64,
    pub download_url: String,
}

impl ModelDownloader for HuggingFaceDownloader {
    async fn search(&self, query: &str) -> ProviderResult<Vec<crate::core::DownloadableModel>> {
        HuggingFaceDownloader::search(self, query).await
    }

    fn download(&self, _model_id: &str, _dest_dir: &str) -> ProviderResult<crate::core::ModelInfo> {
        Err(ProviderError::DownloadError(
            "Use DownloadManager for async downloads with progress".to_string(),
        ))
    }
}
