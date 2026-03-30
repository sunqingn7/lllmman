use eframe::egui;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::{
    DownloadStatus, DownloadTask, LlmProvider, ModelInfo, ModelSource, ProviderConfig, ProviderRegistry,
    ServerController,
};
use crate::models::{AppSettings, GpuInfo};
use crate::services::{
    config_persistence, get_system_stats, gpu_detector,
    DownloadManager, DirectUrlDownloader, GitHubReleaseDownloader, HuggingFaceDownloader,
    detect_running_servers, parse_server_args, get_actual_gpu_layers,
    save_model_config, load_model_config, get_fallback_config,
};

pub fn run() {
    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native("LLLMMan", options, Box::new(|_cc| Box::new(App::new())));
}

pub struct App {
    models: Vec<ModelInfo>,
    gpus: Vec<GpuInfo>,
    server_config: ProviderConfig,
    server_controller: ServerController,
    settings: AppSettings,
    selected_model: Option<usize>,
    selected_provider: String,
    show_download: bool,
    show_gpu_settings: bool,
    show_plugin_config: bool,
    search_query: String,
    available_providers: Vec<(&'static str, &'static str)>,
    download_manager: Arc<RwLock<DownloadManager>>,
    search_results: Arc<RwLock<Vec<crate::core::DownloadableModel>>>,
    download_source_type: String,
    download_url: String,
    download_github_owner: String,
    download_github_repo: String,
    is_searching: bool,
    is_downloading: bool,
    frame_counter: u32,
}

impl App {
    pub fn new() -> Self {
        let settings = config_persistence::load_settings();
        let gpus = gpu_detector::detect_gpus();
        let available_providers = ProviderRegistry::list();

        let selected_provider = if available_providers.is_empty() {
            "llama.cpp".to_string()
        } else {
            available_providers[0].0.to_string()
        };

        let provider = ProviderRegistry::get(&selected_provider).unwrap_or_else(|| {
            let p = crate::providers::LlamaCppProvider::new();
            std::sync::Arc::new(p) as std::sync::Arc<dyn LlmProvider>
        });

        let mut server_config = provider.get_config_template();
        server_config.parse_additional_args();

        // Detect running servers and populate config
        let running_servers = detect_running_servers();
        for server in &running_servers {
            if server.provider_id == selected_provider {
                log::info!("Detected running {} server (PID {}): {}", server.binary, server.pid, server.command_line);
                let detected_config = parse_server_args(&server.provider_id, &server.command_line);
                
                // Merge detected config (always use detected values when external server is found)
                if !detected_config.model_path.is_empty() {
                    server_config.model_path = detected_config.model_path;
                }
                if detected_config.context_size != 4096 {
                    server_config.context_size = detected_config.context_size;
                }
                if detected_config.batch_size != 512 {
                    server_config.batch_size = detected_config.batch_size;
                }
                if detected_config.port != 8080 {
                    server_config.port = detected_config.port;
                }
                if !detected_config.host.is_empty() {
                    server_config.host = detected_config.host;
                }
                if detected_config.gpu_layers != -1 {
                    server_config.gpu_layers = detected_config.gpu_layers;
                }
                if detected_config.threads != 8 {
                    server_config.threads = detected_config.threads;
                }
                
                // If gpu_layers is -1, try to get actual layer count from server
                if server_config.gpu_layers == -1 {
                    let actual_layers = get_actual_gpu_layers(
                        &server_config.host,
                        server_config.port,
                        server_config.gpu_layers,
                    );
                    if actual_layers != -1 {
                        server_config.gpu_layers = actual_layers;
                    }
                }
                
                break;
            }
        }

        let mut models = Vec::new();

        // Scan user-configured directories first
        for dir in &settings.scan_directories {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        // Scan provider-specific default directories
        let provider_dirs = provider.default_model_directories();
        for dir in &provider_dirs {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        let download_manager = Arc::new(RwLock::new(DownloadManager::new(settings.download_directory.clone())));

        Self {
            models,
            gpus,
            server_config,
            server_controller: ServerController::new(Box::leak(
                selected_provider.clone().into_boxed_str(),
            )),
            settings,
            selected_model: None,
            selected_provider,
            show_download: false,
            show_gpu_settings: false,
            show_plugin_config: false,
            search_query: String::new(),
            available_providers,
            download_manager,
            search_results: Arc::new(RwLock::new(Vec::new())),
            download_source_type: "HuggingFace".to_string(),
            download_url: String::new(),
            download_github_owner: String::new(),
            download_github_repo: String::new(),
            is_searching: false,
            is_downloading: false,
            frame_counter: 0,
        }
    }

    fn get_current_provider(&self) -> std::sync::Arc<dyn LlmProvider> {
        ProviderRegistry::get(&self.selected_provider).unwrap_or_else(|| {
            let p = crate::providers::LlamaCppProvider::new();
            std::sync::Arc::new(p) as std::sync::Arc<dyn LlmProvider>
        })
    }

    fn save_settings(&self) {
        if let Err(e) = crate::services::save_settings(&self.settings) {
            eprintln!("Failed to save settings: {}", e);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Periodically refresh external server detection
        self.server_controller.refresh_external_detection();

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("LLLMMan");
                ui.separator();
                ui.label(format!("{} GPUs detected", self.gpus.len()));
                ui.separator();
                if ui.button("GPU Settings").clicked() {
                    self.show_gpu_settings = true;
                }
            });
        });

        egui::TopBottomPanel::bottom("monitor").show(ctx, |ui| {
            let stats = get_system_stats();
            ui.horizontal(|ui| {
                ui.label(format!(
                    "VRAM: {}/{} MB",
                    stats.vram_used_mb, stats.vram_total_mb
                ));
                ui.separator();
                ui.label(format!(
                    "RAM: {}/{} MB",
                    stats.ram_used_mb, stats.ram_total_mb
                ));
                ui.separator();
                ui.label(format!("CPU: {:.1}%", stats.cpu_percent));
                ui.separator();

                // Try to fetch server stats if server is running (throttled to every ~5 seconds: 60fps * 5 = 300 frames)
                if self.server_controller.get_status() == crate::models::ServerStatus::Running && self.frame_counter % 300 == 0 {
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
                        ui.label(format!("Queue: {}", server_stats.queue_size.unwrap_or(0)));
                    }
                }
                
                self.frame_counter = self.frame_counter.wrapping_add(1);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::SidePanel::left("models")
                .default_width(250.0)
                .show_inside(ui, |ui| {
                    ui.heading("Models");
                    ui.text_edit_singleline(&mut self.search_query);

                    ui.separator();

                    if ui.button("Download").clicked() {
                        self.show_download = true;
                    }

                    ui.separator();

                    let filtered: Vec<_> = self
                        .models
                        .iter()
                        .filter(|m| {
                            self.search_query.is_empty()
                                || m.name
                                    .to_lowercase()
                                    .contains(&self.search_query.to_lowercase())
                        })
                        .collect();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, model) in filtered.iter().enumerate() {
                            let is_selected = self.selected_model == Some(i);
                            if ui.selectable_label(is_selected, &model.name).clicked() {
                                // Save current model's config before switching
                                if let Some(old_i) = self.selected_model {
                                    if let Some(old_model) = filtered.get(old_i) {
                                        save_model_config(&old_model.path, &self.server_config).ok();
                                    }
                                }
                                
                                self.selected_model = Some(i);
                                self.server_config.model_path = model.path.clone();
                                
                                // Load config for new model or fallback to last config
                                if let Some(config) = load_model_config(&model.path) {
                                    self.server_config = config;
                                    self.server_config.model_path = model.path.clone();
                                } else if let Some(fallback) = get_fallback_config() {
                                    self.server_config.context_size = fallback.context_size;
                                    self.server_config.batch_size = fallback.batch_size;
                                    self.server_config.gpu_layers = fallback.gpu_layers;
                                    self.server_config.threads = fallback.threads;
                                    self.server_config.port = fallback.port;
                                    self.server_config.host = fallback.host;
                                    self.server_config.cache_type_k = fallback.cache_type_k;
                                    self.server_config.cache_type_v = fallback.cache_type_v;
                                    self.server_config.num_prompt_tracking = fallback.num_prompt_tracking;
                                }
                            }
                        }
                    });
                });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.heading("Server Config");
                ui.separator();

                // Provider selection with config button
                ui.horizontal(|ui| {
                    ui.label("Provider:");
                    let old_provider = self.selected_provider.clone();
                    egui::ComboBox::from_id_source("provider")
                        .selected_text(&self.selected_provider)
                        .show_ui(ui, |ui| {
                            for (id, name) in &self.available_providers {
                                ui.selectable_value(
                                    &mut self.selected_provider,
                                    (*id).to_string(),
                                    *name,
                                );
                            }
                        });

                    // Save settings when provider changes
                    if old_provider != self.selected_provider {
                        self.settings.selected_provider = self.selected_provider.clone();
                        self.save_settings();
                    }

                    if ui.button("⚙️ Config").clicked() {
                        self.show_plugin_config = true;
                    }
                });

                ui.separator();

                egui::Grid::new("config").num_columns(2).show(ui, |ui| {
                    ui.label("Model:");
                    ui.text_edit_singleline(&mut self.server_config.model_path);
                    ui.end_row();

                    ui.label("Context Size:");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.context_size)
                            .clamp_range(256..=2097152),
                    );
                    ui.end_row();

                    ui.label("Batch Size:");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.batch_size)
                            .clamp_range(1..=8192),
                    );
                    ui.end_row();

                    ui.label("GPU Layers:");
                    let mut gpu_layers_value = self.server_config.gpu_layers;
                    ui.add(egui::DragValue::new(&mut gpu_layers_value));
                    self.server_config.gpu_layers = gpu_layers_value;
                    ui.end_row();

                    ui.label("Threads:");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.threads).clamp_range(1..=64),
                    );
                    ui.end_row();

                    ui.label("K Cache Type:");
                    egui::ComboBox::from_id_source("cache_k")
                        .selected_text(&self.server_config.cache_type_k)
                        .show_ui(ui, |ui| {
                            let provider = self.get_current_provider();
                            for q in provider.supported_quantizations() {
                                ui.selectable_value(
                                    &mut self.server_config.cache_type_k,
                                    q.to_string(),
                                    q,
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("V Cache Type:");
                    egui::ComboBox::from_id_source("cache_v")
                        .selected_text(&self.server_config.cache_type_v)
                        .show_ui(ui, |ui| {
                            let provider = self.get_current_provider();
                            for q in provider.supported_quantizations() {
                                ui.selectable_value(
                                    &mut self.server_config.cache_type_v,
                                    q.to_string(),
                                    q,
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("Parallel (-np):");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.num_prompt_tracking)
                            .clamp_range(1..=64),
                    );
                    ui.end_row();

                    ui.label("Host:");
                    ui.text_edit_singleline(&mut self.server_config.host);
                    ui.end_row();

                    ui.label("Port:");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.port)
                            .clamp_range(1024..=65535),
                    );
                    ui.end_row();
                });

                ui.separator();

                ui.horizontal(|ui| {
                    let status = self.server_controller.get_status();
                    match status {
                        crate::models::ServerStatus::Running => {
                            if ui.button("Stop Server").clicked() {
                                // Save config before stopping
                                if !self.server_config.model_path.is_empty() {
                                    save_model_config(&self.server_config.model_path, &self.server_config).ok();
                                }
                                self.server_controller.stop().ok();
                            }
                        }
                        _ => {
                            if ui.button("Start Server").clicked() {
                                // Save config before starting
                                if !self.server_config.model_path.is_empty() {
                                    save_model_config(&self.server_config.model_path, &self.server_config).ok();
                                }
                                let provider = self.get_current_provider();
                                self.server_controller
                                    .start(&provider, &self.server_config)
                                    .ok();
                            }
                        }
                    }
                });
            });
        });

        // Plugin Config Popup
        if self.show_plugin_config {
            let provider = self.get_current_provider();
            let options = provider.get_options();
            let provider_name = self.selected_provider.clone();

            egui::Window::new("Provider Config")
                .open(&mut self.show_plugin_config)
                .show(ctx, |ui| {
                    ui.heading(&provider_name);
                    ui.separator();

                    for opt in &options {
                        let current_value = self.server_config.get_option(&opt.id)
                            .unwrap_or_else(|| opt.default_value.clone());
                        
                        ui.label(&opt.description);
                        match &opt.value_type {
                            crate::core::OptionValueType::String => {
                                let mut value = current_value;
                                if ui.text_edit_singleline(&mut value).changed() {
                                    self.server_config.set_option(&opt.id, value);
                                }
                            }
                            crate::core::OptionValueType::Number => {
                                let mut value: i64 = current_value.parse().unwrap_or(0);
                                if ui.add(egui::DragValue::new(&mut value)).changed() {
                                    self.server_config.set_option(&opt.id, value.to_string());
                                }
                            }
                            crate::core::OptionValueType::Bool => {
                                let mut value = current_value == "true";
                                if ui.checkbox(&mut value, &opt.name).changed() {
                                    self.server_config.set_option(&opt.id, value.to_string());
                                }
                            }
                            crate::core::OptionValueType::Select(values) => {
                                let mut value = current_value.clone();
                                egui::ComboBox::from_id_source(&opt.id)
                                    .selected_text(&value)
                                    .show_ui(ui, |ui| {
                                        for v in values {
                                            ui.selectable_value(
                                                &mut value,
                                                v.clone(),
                                                v,
                                            );
                                        }
                                    });
                                if value != current_value {
                                    self.server_config.set_option(&opt.id, value);
                                }
                            }
                        }
                        ui.end_row();
                    }

                    ui.separator();
                    ui.label("Additional CLI Args (space-separated):");
                    let mut raw_args = self.server_config.additional_args.clone();
                    if ui.text_edit_singleline(&mut raw_args).changed() {
                        self.server_config.additional_args = raw_args;
                    }
                    ui.small("Examples: --flash-attention --no-mmap");
                });
        }

        // GPU Settings Window
        if self.show_gpu_settings {
            egui::Window::new("GPU Settings")
                .open(&mut self.show_gpu_settings)
                .default_width(400.0)
                .default_height(300.0)
                .show(ctx, |ui| {
                    ui.heading("GPU Allocation Settings");
                    ui.separator();

                    // GPU allocation type selection
                    ui.label("Allocation Mode:");

                    // All GPUs
                    if ui
                        .selectable_label(
                            matches!(
                                self.server_config.gpu_allocation,
                                crate::models::GpuAllocation::All
                            ),
                            "Use All GPUs",
                        )
                        .clicked()
                    {
                        self.server_config.gpu_allocation = crate::models::GpuAllocation::All;
                    }

                    // Single GPU
                    if ui
                        .selectable_label(
                            matches!(
                                self.server_config.gpu_allocation,
                                crate::models::GpuAllocation::Single(_)
                            ),
                            "Use Single GPU",
                        )
                        .clicked()
                    {
                        self.server_config.gpu_allocation = crate::models::GpuAllocation::Single(0);
                    }

                    // VRAM Limit
                    if ui
                        .selectable_label(
                            matches!(
                                self.server_config.gpu_allocation,
                                crate::models::GpuAllocation::VramLimit { .. }
                            ),
                            "VRAM Limit Mode",
                        )
                        .clicked()
                    {
                        self.server_config.gpu_allocation =
                            crate::models::GpuAllocation::VramLimit {
                                gpu: 0,
                                max_vram_mb: 4096,
                                layers: 35,
                            };
                    }

                    ui.separator();

                    // Show GPU list
                    ui.label("Available GPUs:");
                    for gpu in &self.gpus {
                        ui.label(format!(
                            "GPU {}: {} ({:.1} GB VRAM)",
                            gpu.index,
                            gpu.name,
                            gpu.total_vram_mb as f32 / 1024.0
                        ));
                    }

                    if self.gpus.is_empty() {
                        ui.label("No GPUs detected");
                    }
                });
        }

        // Download Window
        if self.show_download {
            egui::Window::new("Download Model")
                .open(&mut self.show_download)
                .default_width(600.0)
                .default_height(500.0)
                .show(ctx, |ui| {
                    ui.heading("Download Model");
                    ui.separator();

                    ui.label("Source:");
                    egui::ComboBox::from_id_source("download_source")
                        .selected_text(&self.download_source_type)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.download_source_type, "HuggingFace".to_string(), "HuggingFace");
                            ui.selectable_value(&mut self.download_source_type, "Direct URL".to_string(), "Direct URL");
                            ui.selectable_value(&mut self.download_source_type, "GitHub Release".to_string(), "GitHub Release");
                        });

                    ui.separator();

                    match self.download_source_type.as_str() {
                        "HuggingFace" => {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.search_query)
                                        .hint_text("Search models..."),
                                );
                                if ui.button("Search").clicked() && !self.search_query.is_empty() {
                                    self.is_searching = true;
                                    let query = self.search_query.clone();
                                    let search_results = self.search_results.clone();
                                    let app_handle = ctx.clone();
                                    tokio::spawn(async move {
                                        let downloader = HuggingFaceDownloader::new();
                                        match downloader.search(&query).await {
                                            Ok(results) => {
                                                let mut results_lock = search_results.write().await;
                                                *results_lock = results;
                                            }
                                            Err(e) => {
                                                log::error!("Search failed: {}", e);
                                            }
                                        }
                                        app_handle.request_repaint();
                                    });
                                }
                            });

                            ui.separator();

                            // Display search results
                            let search_results_ref = self.search_results.clone();
                            let results_guard = search_results_ref.blocking_read();
                            if !results_guard.is_empty() {
                                ui.label("Search Results:");
                                egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                                    for result in results_guard.iter() {
                                        if ui.button(&result.name).clicked() {
                                            self.download_url = result.id.clone();
                                        }
                                    }
                                });
                            }

                            ui.separator();
                            ui.label("Or enter model ID directly:");
                            ui.text_edit_singleline(&mut self.download_url);
                            ui.label("Example: TheBloke/Mistral-7B-Instruct-v0.1-GGUF");

                            if ui.button("Start Download").clicked() && !self.download_url.is_empty() {
                                let model_id = self.download_url.clone();
                                let manager = self.download_manager.clone();
                                
                                tokio::spawn(async move {
                                    let downloader = HuggingFaceDownloader::new();
                                    match downloader.list_files(&model_id).await {
                                        Ok(files) => {
                                            for file in files {
                                                let source = ModelSource::HuggingFace { repo_id: model_id.clone() };
                                                let id = manager.write().await.add_task(source, file.path.clone()).await;
                                                // TODO: Start actual download with progress
                                                log::info!("Added download task: {}", id);
                                            }
                                        }
                                        Err(e) => {
                                            log::error!("Failed to list files: {}", e);
                                        }
                                    }
                                });
                            }
                        }
                        "Direct URL" => {
                            ui.label("Enter direct download URL:");
                            ui.text_edit_singleline(&mut self.download_url);
                            
                            if ui.button("Start Download").clicked() && !self.download_url.is_empty() {
                                let url = self.download_url.clone();
                                let manager = self.download_manager.clone();
                                
                                tokio::spawn(async move {
                                    let downloader = DirectUrlDownloader::new();
                                    match downloader.fetch_headers(&url).await {
                                        Ok(headers) => {
                                            let file_name = headers.file_name.clone();
                                            let content_length = headers.content_length;
                                            let source = ModelSource::DirectUrl { url: url.clone() };
                                            let _id = manager.write().await.add_task(source, file_name.clone()).await;
                                            log::info!("Started download: {} ({} bytes)", file_name, content_length);
                                        }
                                        Err(e) => {
                                            log::error!("Failed to fetch headers: {}", e);
                                        }
                                    }
                                });
                            }
                        }
                        "GitHub Release" => {
                            ui.horizontal(|ui| {
                                ui.label("Owner:");
                                ui.text_edit_singleline(&mut self.download_github_owner);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Repository:");
                                ui.text_edit_singleline(&mut self.download_github_repo);
                            });
                            
                            if ui.button("Fetch Release").clicked() && !self.download_github_owner.is_empty() && !self.download_github_repo.is_empty() {
                                let owner = self.download_github_owner.clone();
                                let repo = self.download_github_repo.clone();
                                let manager = self.download_manager.clone();
                                
                                tokio::spawn(async move {
                                    let downloader = GitHubReleaseDownloader::new();
                                    match downloader.get_latest_release(&owner, &repo).await {
                                        Ok(release) => {
                                            log::info!("Found release: {} with {} assets", release.tag, release.assets.len());
                                            for asset in release.assets {
                                                let source = ModelSource::GitHubRelease {
                                                    owner: owner.clone(),
                                                    repo: repo.clone(),
                                                    tag: release.tag.clone(),
                                                    asset_name: asset.name.clone(),
                                                };
                                                let _id = manager.write().await.add_task(source, asset.name).await;
                                            }
                                        }
                                        Err(e) => {
                                            log::error!("Failed to fetch release: {}", e);
                                        }
                                    }
                                });
                            }
                        }
                        _ => {}
                    }

                    ui.separator();
                    ui.heading("Download Queue");
                    
                    let manager = self.download_manager.clone();
                    let tasks = manager.blocking_read().get_tasks_sync();
                    
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for task in &tasks {
                            ui.horizontal(|ui| {
                                ui.label(&task.file_name);
                                ui.separator();
                                match &task.status {
                                    crate::core::DownloadStatus::Pending => ui.label("Pending"),
                                    crate::core::DownloadStatus::Downloading => {
                                        let progress = if task.total_bytes > 0 {
                                            task.downloaded_bytes as f32 / task.total_bytes as f32
                                        } else {
                                            0.0
                                        };
                                        ui.add(egui::ProgressBar::new(progress).text(&format!("{:.1}%", progress * 100.0)))
                                    }
                                    crate::core::DownloadStatus::Completed => ui.label("Completed"),
                                    crate::core::DownloadStatus::Failed(e) => ui.label(format!("Failed: {}", e)),
                                    crate::core::DownloadStatus::Cancelled => ui.label("Cancelled"),
                                }
                            });
                        }
                    });
                });
        }

        ctx.request_repaint();
    }
}
