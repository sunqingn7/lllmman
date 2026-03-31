use eframe::egui;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::{
    LogBuffer, LogLevel, LlmProvider, ModelInfo, ModelSource, ProviderConfig, ProviderRegistry,
    ProviderSettings, ServerController,
};
use crate::models::{AppSettings, GpuInfo};
use crate::services::{
    config_persistence, get_system_stats, gpu_detector,
    DownloadManager, DirectUrlDownloader, GitHubReleaseDownloader, HuggingFaceDownloader,
    detect_running_servers, parse_server_args,
    save_model_config, load_model_config, get_fallback_config,
    load_provider_settings_for, save_provider_settings_for,
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
    provider_settings: ProviderSettings,
    settings: AppSettings,
    selected_model: Option<usize>,
    selected_provider: String,
    show_download: bool,
    show_gpu_settings: bool,
    show_plugin_config: bool,
    show_provider_settings: bool,
    show_log_view: bool,
    log_buffer: LogBuffer,
    search_query: String,
    available_providers: Vec<(&'static str, &'static str)>,
    download_manager: Arc<RwLock<DownloadManager>>,
    search_results: Arc<RwLock<Vec<crate::core::DownloadableModel>>>,
    download_source_type: String,
    download_url: String,
    download_github_owner: String,
    download_github_repo: String,
    is_searching: bool,
    frame_counter: u32,
    bottom_view: BottomView,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomView {
    Monitor,
    Log,
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

        let (models, server_config) = Self::build_for_provider(&selected_provider, &settings);
        let provider_settings = load_provider_settings_for(&selected_provider);

        let provider = Self::get_provider_static(&selected_provider);
        let mut server_controller = ServerController::new();
        server_controller.set_provider(provider.clone());
        server_controller.set_provider_settings(provider_settings.clone());
        let log_buffer = server_controller.get_log_buffer();

        Self {
            models,
            gpus,
            server_config,
            server_controller,
            provider_settings,
            settings: settings.clone(),
            selected_model: None,
            selected_provider,
            show_download: false,
            show_gpu_settings: false,
            show_plugin_config: false,
            show_provider_settings: false,
            show_log_view: false,
            log_buffer,
            search_query: String::new(),
            available_providers,
            download_manager: Arc::new(RwLock::new(DownloadManager::new(settings.download_directory.clone()))),
            search_results: Arc::new(RwLock::new(Vec::new())),
            download_source_type: "HuggingFace".to_string(),
            download_url: String::new(),
            download_github_owner: String::new(),
            download_github_repo: String::new(),
            is_searching: false,
            frame_counter: 0,
            bottom_view: BottomView::Monitor,
        }
    }

    fn build_for_provider(provider_id: &str, settings: &AppSettings) -> (Vec<ModelInfo>, ProviderConfig) {
        let provider = Self::get_provider_static(provider_id);

        let mut server_config = provider.get_config_template();
        server_config.parse_additional_args();

        let running_servers = detect_running_servers();
        for server in &running_servers {
            if server.provider_id == provider_id {
                log::info!("Detected running {} server (PID {}): {}", server.binary, server.pid, server.command_line);
                let detected_config = parse_server_args(&server.provider_id, &server.command_line);

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

                break;
            }
        }

        let mut models = Vec::new();

        for dir in &settings.scan_directories {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        let provider_dirs = provider.default_model_directories();
        for dir in &provider_dirs {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        (models, server_config)
    }

    fn get_provider_static(provider_id: &str) -> Arc<dyn LlmProvider> {
        ProviderRegistry::get(provider_id).unwrap_or_else(|| {
            let p = crate::providers::LlamaCppProvider::new();
            Arc::new(p) as Arc<dyn LlmProvider>
        })
    }

    fn get_current_provider(&self) -> Arc<dyn LlmProvider> {
        Self::get_provider_static(&self.selected_provider)
    }

    fn switch_provider(&mut self, new_provider_id: &str) {
        if let Some(old_i) = self.selected_model {
            if let Some(model) = self.models.get(old_i) {
                save_model_config(&model.path, &self.server_config).ok();
            }
        }

        self.selected_provider = new_provider_id.to_string();
        self.settings.selected_provider = new_provider_id.to_string();
        self.save_settings();

        let (models, server_config) = Self::build_for_provider(new_provider_id, &self.settings);
        self.models = models;
        self.server_config = server_config;
        self.selected_model = None;

        self.provider_settings = load_provider_settings_for(new_provider_id);

        let provider = self.get_current_provider();
        self.server_controller = ServerController::new();
        self.log_buffer = self.server_controller.get_log_buffer();
        self.server_controller.set_provider(provider);
        self.server_controller.set_provider_settings(self.provider_settings.clone());
    }

    fn save_settings(&self) {
        if let Err(e) = crate::services::save_settings(&self.settings) {
            eprintln!("Failed to save settings: {}", e);
        }
    }

    fn render_bottom_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(
                    self.bottom_view == BottomView::Monitor,
                    "Monitor",
                )
                .clicked()
            {
                self.bottom_view = BottomView::Monitor;
            }
            if ui
                .selectable_label(
                    self.bottom_view == BottomView::Log,
                    "Log",
                )
                .clicked()
            {
                self.bottom_view = BottomView::Log;
            }
            ui.separator();

            if self.bottom_view == BottomView::Monitor {
                let stats = get_system_stats();
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
                let cpu_temp_str = stats.cpu_temperature
                    .map(|t| format!(" ({:.0}°C)", t))
                    .unwrap_or_default();
                ui.label(format!("CPU: {:.1}%{}", stats.cpu_percent, cpu_temp_str));
                ui.separator();

                for gpu_temp in &stats.gpu_temperatures {
                    let temp_str = gpu_temp
                        .temperature_c
                        .map(|t| format!("{:.0}°C", t))
                        .unwrap_or_else(|| "N/A".to_string());
                    ui.label(format!("GPU{}: {}", gpu_temp.index, temp_str));
                }
                if !stats.gpu_temperatures.is_empty() {
                    ui.separator();
                }

                if self.server_controller.get_status()
                    == crate::models::ServerStatus::Running
                    && self.frame_counter % 300 == 0
                {
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
                        ui.label(format!(
                            "Queue: {}",
                            server_stats.queue_size.unwrap_or(0)
                        ));
                    }
                }
            } else {
                let entries = self.log_buffer.get_entries();
                ui.label(format!("{} entries", entries.len()));
                if ui.button("Clear").clicked() {
                    self.log_buffer.clear();
                }
            }

            self.frame_counter = self.frame_counter.wrapping_add(1);
        });

        ui.separator();

        if self.bottom_view == BottomView::Log {
            let entries = self.log_buffer.get_entries();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if entries.is_empty() {
                        ui.label(
                            egui::RichText::new("No logs yet. Start a server to see output here.")
                                .color(egui::Color32::GRAY)
                                .italics(),
                        );
                    } else {
                        for entry in &entries {
                            let time_str = entry
                                .timestamp
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| {
                                    let secs = d.as_secs();
                                    let h = secs / 3600;
                                    let m = (secs % 3600) / 60;
                                    let s = secs % 60;
                                    format!("{:02}:{:02}:{:02}", h, m, s)
                                })
                                .unwrap_or_default();

                            let color = match entry.level {
                                LogLevel::Error => egui::Color32::RED,
                                LogLevel::Warn => egui::Color32::YELLOW,
                                LogLevel::Info => egui::Color32::LIGHT_GRAY,
                            };

                            ui.label(
                                egui::RichText::new(format!(
                                    "{} [{}] {}",
                                    time_str,
                                    entry.level.as_str(),
                                    entry.message
                                ))
                                .color(color)
                                .monospace(),
                            );
                        }
                    }
                });
        }
    }

    fn render_main_content(&mut self, ui: &mut egui::Ui) {
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
                            if let Some(old_i) = self.selected_model {
                                if let Some(old_model) = filtered.get(old_i) {
                                    save_model_config(&old_model.path, &self.server_config).ok();
                                }
                            }

                            self.selected_model = Some(i);
                            self.server_config.model_path = model.path.clone();

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

                if old_provider != self.selected_provider {
                    let new_id = self.selected_provider.clone();
                    self.switch_provider(&new_id);
                }

                if ui.button("⚙️ Provider Settings").clicked() {
                    self.show_provider_settings = true;
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

                ui.label("GPU Memory Utilization:");
                let mut gpu_mem_value = if self.server_config.gpu_layers <= 0 {
                    90
                } else {
                    self.server_config.gpu_layers
                };
                ui.add(
                    egui::DragValue::new(&mut gpu_mem_value)
                        .clamp_range(1..=100)
                        .suffix("%"),
                );
                self.server_config.gpu_layers = gpu_mem_value;
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
                            if !self.server_config.model_path.is_empty() {
                                save_model_config(&self.server_config.model_path, &self.server_config).ok();
                            }
                            self.server_controller.stop().ok();
                        }
                    }
                    _ => {
                        if ui.button("Start Server").clicked() {
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
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::TopBottomPanel::bottom("bottom_panel")
                .resizable(true)
                .default_height(200.0)
                .show_inside(ui, |ui| {
                    self.render_bottom_panel(ui);
                });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.render_main_content(ui);
            });
        });

        if self.show_provider_settings {
            egui::Window::new("Provider Settings")
                .open(&mut self.show_provider_settings)
                .default_width(500.0)
                .show(ctx, |ui| {
                    ui.heading(format!("{} Settings", self.selected_provider));
                    ui.separator();

                    ui.label("Binary Path:");
                    ui.text_edit_singleline(&mut self.provider_settings.binary_path);
                    ui.small("Full path to the server binary (e.g., /home/qing/Projects/llama.cpp/build/bin/llama-server)");

                    ui.add_space(8.0);

                    ui.label("Environment Script:");
                    ui.text_edit_singleline(&mut self.provider_settings.env_script);
                    ui.small("Path to env script to source before starting (e.g., /home/qing/Projects/.venv/bin/activate)");

                    ui.add_space(8.0);

                    ui.label("Additional Arguments:");
                    ui.text_edit_singleline(&mut self.provider_settings.additional_args);
                    ui.small("Extra CLI args passed to the server (space-separated, e.g., --flash-attention --no-mmap)");

                    ui.add_space(12.0);

                    if ui.button("Save").clicked() {
                        self.server_controller.set_provider_settings(self.provider_settings.clone());
                        save_provider_settings_for(&self.selected_provider, &self.provider_settings).ok();
                    }
                });
        }

        if self.show_log_view {
            let entries = self.log_buffer.get_entries();
            egui::Window::new("Log View")
                .open(&mut self.show_log_view)
                .default_width(900.0)
                .default_height(500.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Server Logs");
                        ui.separator();
                        ui.label(format!("{} entries", entries.len()));
                        if ui.button("Clear").clicked() {
                            self.log_buffer.clear();
                        }
                    });
                    ui.separator();

                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for entry in &entries {
                                let time_str = entry
                                    .timestamp
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| {
                                        let secs = d.as_secs();
                                        let h = secs / 3600;
                                        let m = (secs % 3600) / 60;
                                        let s = secs % 60;
                                        format!("{:02}:{:02}:{:02}", h, m, s)
                                    })
                                    .unwrap_or_default();

                                let color = match entry.level {
                                    LogLevel::Error => egui::Color32::RED,
                                    LogLevel::Warn => egui::Color32::YELLOW,
                                    LogLevel::Info => egui::Color32::LIGHT_GRAY,
                                };

                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} [{}] {}",
                                        time_str,
                                        entry.level.as_str(),
                                        entry.message
                                    ))
                                    .color(color)
                                    .monospace(),
                                );
                            }
                        });
                });
        }

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

        if self.show_gpu_settings {
            egui::Window::new("GPU Settings")
                .open(&mut self.show_gpu_settings)
                .default_width(400.0)
                .default_height(300.0)
                .show(ctx, |ui| {
                    ui.heading("GPU Allocation Settings");
                    ui.separator();

                    ui.label("Allocation Mode:");

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
