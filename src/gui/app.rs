#![allow(dead_code)]
use eframe::egui;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::{
    LlmProvider, LogBuffer, LogLevel, ModelInfo, ModelSource, ProviderConfig, ProviderRegistry,
    ProviderSettings, ServerController,
};
use crate::models::{AppSettings, GpuInfo, MonitorStats};
use crate::services::{
    check_provider_installed, config_persistence, detect_running_servers, get_fallback_config,
    get_provider_install_info, get_system_info_summary, get_system_stats, gpu_detector,
    load_model_config, load_provider_settings_for, parse_server_args, recommend_gpu_layers,
    save_model_config, save_provider_settings_for, DirectUrlDownloader, DownloadManager,
    GitHubReleaseDownloader, HuggingFaceDownloader,
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
    show_provider_setup: bool,
    provider_setup_provider: String,
    provider_installed_cache: Option<(String, bool)>,
    show_cmdline_dialog: bool,
    cmdline_input: String,
    started_hf_model: Option<String>,
    cached_stats: Option<(u32, MonitorStats)>,
    previous_server_status: crate::models::ServerStatus,
    needs_repaint: Arc<AtomicBool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomView {
    Log,
}

impl App {
    pub fn new() -> Self {
        let settings = config_persistence::load_settings();
        let gpus = gpu_detector::detect_gpus();
        let available_providers = ProviderRegistry::list();

        // First, detect running servers across ALL providers
        let running_servers = detect_running_servers();
        let detected_provider = running_servers.first().map(|s| s.provider_id.clone());

        let selected_provider = detected_provider.unwrap_or_else(|| {
            if available_providers.is_empty() {
                "llama.cpp".to_string()
            } else {
                // First check if saved provider is still available
                let saved_provider = &settings.selected_provider;
                if !saved_provider.is_empty()
                    && available_providers
                        .iter()
                        .any(|(id, _)| *id == saved_provider)
                {
                    saved_provider.clone()
                } else {
                    available_providers[0].0.to_string()
                }
            }
        });

        log::info!("Selected provider for startup: {}", selected_provider);

        let (models, server_config) = Self::build_for_provider(&selected_provider, &settings);

        let selected_model = models.iter().position(|m| {
            m.path == server_config.model_path || m.path.contains(&server_config.model_path)
        });

        let provider = Self::get_provider_static(&selected_provider);
        let provider_defaults = provider.default_settings();
        let loaded_settings = load_provider_settings_for(&selected_provider);
        let provider_settings = ProviderSettings {
            binary_path: if !loaded_settings.binary_path.is_empty() {
                loaded_settings.binary_path.clone()
            } else {
                provider_defaults.binary_path.clone()
            },
            env_script: if !loaded_settings.env_script.is_empty() {
                loaded_settings.env_script.clone()
            } else {
                provider_defaults.env_script.clone()
            },
            additional_args: if !loaded_settings.additional_args.is_empty() {
                loaded_settings.additional_args.clone()
            } else {
                provider_defaults.additional_args.clone()
            },
            health_endpoint: if !loaded_settings.health_endpoint.is_empty() {
                loaded_settings.health_endpoint.clone()
            } else {
                provider_defaults.health_endpoint.clone()
            },
            heartbeat_interval_secs: if loaded_settings.heartbeat_interval_secs > 0 {
                loaded_settings.heartbeat_interval_secs
            } else {
                provider_defaults.heartbeat_interval_secs
            },
            venv_path: if !loaded_settings.venv_path.is_empty() {
                loaded_settings.venv_path.clone()
            } else {
                provider_defaults.venv_path.clone()
            },
        };

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
            selected_model,
            selected_provider,
            show_download: false,
            show_gpu_settings: false,
            show_plugin_config: false,
            show_provider_settings: false,
            show_log_view: false,
            log_buffer,
            search_query: String::new(),
            available_providers,
            download_manager: Arc::new(RwLock::new(DownloadManager::new(
                settings.download_directory.clone(),
            ))),
            search_results: Arc::new(RwLock::new(Vec::new())),
            download_source_type: "HuggingFace".to_string(),
            download_url: String::new(),
            download_github_owner: String::new(),
            download_github_repo: String::new(),
            is_searching: false,
            frame_counter: 0,
            bottom_view: BottomView::Log,
            show_provider_setup: false,
            provider_setup_provider: String::new(),
            provider_installed_cache: None,
            show_cmdline_dialog: false,
            cmdline_input: String::new(),
            started_hf_model: None,
            cached_stats: None,
            previous_server_status: crate::models::ServerStatus::Stopped,
            needs_repaint: Arc::new(AtomicBool::new(false)),
        }
    }

    fn build_for_provider(
        provider_id: &str,
        settings: &AppSettings,
    ) -> (Vec<ModelInfo>, ProviderConfig) {
        let provider = Self::get_provider_static(provider_id);

        let mut server_config = provider.get_config_template();
        server_config.parse_additional_args();

        let running_servers = detect_running_servers();
        for server in &running_servers {
            if server.provider_id == provider_id {
                log::info!(
                    "Detected running {} server (PID {}): {}",
                    server.binary,
                    server.pid,
                    server.command_line
                );
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
                if detected_config.temperature.is_some() {
                    server_config.temperature = detected_config.temperature;
                }
                if detected_config.top_k.is_some() {
                    server_config.top_k = detected_config.top_k;
                }
                if detected_config.top_p.is_some() {
                    server_config.top_p = detected_config.top_p;
                }
                if detected_config.min_p.is_some() {
                    server_config.min_p = detected_config.min_p;
                }
                if detected_config.presence_penalty.is_some() {
                    server_config.presence_penalty = detected_config.presence_penalty;
                }
                if detected_config.repetition_penalty.is_some() {
                    server_config.repetition_penalty = detected_config.repetition_penalty;
                }
                if detected_config.enable_thinking.is_some() {
                    server_config.enable_thinking = detected_config.enable_thinking;
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

    fn build_gpu_selection_options(&self) -> Vec<(String, Option<u32>)> {
        let mut options = vec![("Auto".to_string(), None)];
        for gpu in &self.gpus {
            options.push((format!("GPU {}", gpu.index), Some(gpu.index)));
        }
        options
    }

    fn switch_provider(&mut self, new_provider_id: &str) {
        if let Some(old_i) = self.selected_model {
            if let Some(model) = self.models.get(old_i) {
                save_model_config(&model.path, &self.server_config, new_provider_id).ok();
            }
        }

        self.selected_provider = new_provider_id.to_string();
        self.settings.selected_provider = new_provider_id.to_string();
        self.save_settings();

        // Invalidate provider install cache when switching providers
        self.provider_installed_cache = None;

        let (models, server_config) = Self::build_for_provider(new_provider_id, &self.settings);
        self.models = models;
        self.server_config = server_config;
        self.selected_model = None;

        self.provider_settings = load_provider_settings_for(new_provider_id);

        let provider = self.get_current_provider();
        let provider_defaults = provider.default_settings();
        let loaded = self.provider_settings.clone();
        self.provider_settings = ProviderSettings {
            binary_path: if !loaded.binary_path.is_empty() {
                loaded.binary_path.clone()
            } else {
                provider_defaults.binary_path.clone()
            },
            env_script: if !loaded.env_script.is_empty() {
                loaded.env_script.clone()
            } else {
                provider_defaults.env_script.clone()
            },
            additional_args: if !loaded.additional_args.is_empty() {
                loaded.additional_args.clone()
            } else {
                provider_defaults.additional_args.clone()
            },
            health_endpoint: if !loaded.health_endpoint.is_empty() {
                loaded.health_endpoint.clone()
            } else {
                provider_defaults.health_endpoint.clone()
            },
            heartbeat_interval_secs: if loaded.heartbeat_interval_secs > 0 {
                loaded.heartbeat_interval_secs
            } else {
                provider_defaults.heartbeat_interval_secs
            },
            venv_path: if !loaded.venv_path.is_empty() {
                loaded.venv_path.clone()
            } else {
                provider_defaults.venv_path.clone()
            },
        };

        self.server_controller = ServerController::new();
        self.log_buffer = self.server_controller.get_log_buffer();
        self.server_controller.set_provider(provider);
        self.server_controller
            .set_provider_settings(self.provider_settings.clone());
    }

    fn save_settings(&self) {
        if let Err(e) = crate::services::save_settings(&self.settings) {
            eprintln!("Failed to save settings: {}", e);
        }
    }

    fn render_bottom_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.bottom_view == BottomView::Log, "Log")
                .clicked()
            {
                self.bottom_view = BottomView::Log;
            }
            ui.separator();

            let entries = self.log_buffer.get_entries();
            ui.label(format!("{} entries", entries.len()));
            if ui.button("Clear").clicked() {
                self.log_buffer.clear();
            }

            
        });

        ui.separator();

        if self.bottom_view == BottomView::Log {
            let entries = self.log_buffer.get_entries();
            ui.push_id("log_panel", |ui| {
                let mut log_text = String::new();
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

                    log_text.push_str(&format!(
                        "{} [{}] {}\n",
                        time_str,
                        entry.level.as_str(),
                        entry.message
                    ));
                }

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::TextEdit::multiline(&mut log_text)
                            .desired_width(f32::INFINITY)
                            .desired_rows(entries.len().max(10))
                            .frame(false)
                            .show(ui);
                    });
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

                ui.horizontal(|ui| {
                    if ui.button("Download").clicked() {
                        self.show_download = true;
                    }

                    if ui.button("Refresh").clicked() {
                        let provider = self.get_current_provider();
                        let mut new_models = Vec::new();
                        for dir in &self.settings.scan_directories {
                            let found = provider.scan_models(dir);
                            new_models.extend(found);
                        }
                        let provider_dirs = provider.default_model_directories();
                        for dir in &provider_dirs {
                            let found = provider.scan_models(dir);
                            new_models.extend(found);
                        }
                        self.models = new_models;
                    }
                });

                ui.separator();

                let provider_supports_gguf = self.get_current_provider().supports_gguf();
                let filtered: Vec<_> = self
                    .models
                    .iter()
                    .filter(|m| {
                        let is_gguf = m.path.to_lowercase().ends_with(".gguf");
                        if is_gguf && !provider_supports_gguf {
                            return false;
                        }
                        self.search_query.is_empty()
                            || m.name
                                .to_lowercase()
                                .contains(&self.search_query.to_lowercase())
                    })
                    .collect();

                ui.push_id("model_list", |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, model) in filtered.iter().enumerate() {
                            let is_selected = self.selected_model == Some(i);

                            let bg_color = if is_selected {
                                egui::Color32::from_rgb(40, 44, 52)
                            } else {
                                egui::Color32::from_rgb(30, 32, 38)
                            };

                            let text_color = if is_selected {
                                egui::Color32::from_rgb(130, 170, 255)
                            } else {
                                egui::Color32::from_rgb(230, 230, 230)
                            };

                            let subtext_color = if is_selected {
                                egui::Color32::from_rgb(140, 150, 170)
                            } else {
                                egui::Color32::from_rgb(140, 140, 150)
                            };

                            ui.add_space(4.0);

                            let card_frame = egui::Frame::none()
                                .fill(bg_color)
                                .stroke(egui::Stroke::new(
                                    1.0,
                                    if is_selected {
                                        egui::Color32::from_rgb(48, 63, 159)
                                    } else {
                                        egui::Color32::from_rgb(224, 224, 224)
                                    },
                                ))
                                .inner_margin(egui::Margin::same(10.0))
                                .rounding(8.0);

                            let response = ui.allocate_response(
                                egui::vec2(ui.available_width(), 52.0),
                                egui::Sense::click(),
                            );

                            let frame_shape = card_frame.paint(response.rect);
                            ui.painter().add(frame_shape);

                            let (creator, model_name) = if let Some(pos) = model.name.rfind('/') {
                    let (c, n) = model.name.split_at(pos);
                    (c, &n[1..]) // Skip the '/' character
                } else {
                    ("", model.name.as_str())
                };

                // Creator line (small font)
                if !creator.is_empty() {
                    let creator_rect = egui::Rect::from_min_size(
                        response.rect.min + egui::vec2(8.0, 2.0),
                        egui::vec2(response.rect.width() - 16.0, 12.0),
                    );
                    ui.painter().text(
                        creator_rect.min,
                        egui::Align2::LEFT_TOP,
                        creator,
                        egui::FontId::new(10.0, egui::FontFamily::Proportional),
                        subtext_color,
                    );
                }

                // Model name line (big bold font)
                let model_name_rect = egui::Rect::from_min_size(
                    response.rect.min + egui::vec2(8.0, if creator.is_empty() { 2.0 } else { 14.0 }),
                    egui::vec2(response.rect.width() - 16.0, 18.0),
                );
                ui.painter().text(
                    model_name_rect.min,
                    egui::Align2::LEFT_TOP,
                    model_name,
                    egui::FontId::new(13.0, egui::FontFamily::Monospace),
                    text_color,
                );

                // Details line
                let detail_text = format!(
                    "{:.1} GB • {} • {}",
                    model.size_gb,
                    model.quantization,
                    match model.model_type {
                        crate::models::ModelType::TextOnly => "Text",
                        crate::models::ModelType::Tooling => "Tool",
                        crate::models::ModelType::Vision => "Vision",
                        crate::models::ModelType::Multimodal => "Multi",
                    }
                );
                let detail_y = if creator.is_empty() { 20.0 } else { 32.0 };
                let detail_rect = egui::Rect::from_min_size(
                    response.rect.min + egui::vec2(8.0, detail_y),
                    egui::vec2(response.rect.width() - 16.0, 14.0),
                );
                ui.painter().text(
                    detail_rect.min,
                    egui::Align2::LEFT_TOP,
                    &detail_text,
                    egui::FontId::new(10.0, egui::FontFamily::Proportional),
                    subtext_color,
                );

                if response.clicked() {
                                if let Some(old_i) = self.selected_model {
                                    if let Some(old_model) = filtered.get(old_i) {
                                        save_model_config(
                                            &old_model.path,
                                            &self.server_config,
                                            &self.selected_provider,
                                        )
                                        .ok();
                                    }
                                }

                self.selected_model = Some(i);
                self.server_config.model_path = model.path.clone();

                // Auto-detect mmproj for llama.cpp models
                if self.selected_provider == "llama.cpp" {
                    if let Some(mmproj_path) =
                        crate::providers::llama_cpp::find_mmproj_for_model(&model.path)
                    {
                        self.server_config.mmproj_path = mmproj_path;
                        self.log_buffer.push_info(format!(
                            "[Auto-detect] Found mmproj: {}",
                            self.server_config.mmproj_path
                        ));
                    } else {
                        self.server_config.mmproj_path.clear();
                    }
                }

                if let Some(config) =
                    load_model_config(&model.path, &self.selected_provider)
                {
                                    self.server_config = config;
                                    self.server_config.model_path = model.path.clone();
                                } else if let Some(fallback) =
                                    get_fallback_config(&self.selected_provider)
                                {
                                    self.server_config.context_size = fallback.context_size;
                                    self.server_config.batch_size = fallback.batch_size;
                                    self.server_config.gpu_layers = fallback.gpu_layers;
                                    self.server_config.threads = fallback.threads;
                                    self.server_config.port = fallback.port;
                                    self.server_config.host = fallback.host;
                                    self.server_config.cache_type_k = fallback.cache_type_k;
                                    self.server_config.cache_type_v = fallback.cache_type_v;
                                    self.server_config.num_prompt_tracking =
                                        fallback.num_prompt_tracking;
                                    self.server_config.additional_args = fallback.additional_args;
                                    self.server_config.tokenizer = fallback.tokenizer;
                                }

                                if self.selected_provider == "llama.cpp" {
                                    let model_size_gb = model.size_gb;
                                    let available_vram: u32 =
                                        self.gpus.iter().map(|g| g.total_vram_mb).sum();
                                    let total_layers =
                                        crate::providers::llama_cpp::read_gguf_n_layer(&model.path)
                                            .map(|l| l as i32)
                                            .unwrap_or(-1);

                                    if available_vram > 0 && model_size_gb > 0.0 {
                                        let (suggested_mode, suggested_layers) =
                                            crate::services::calculate_cpu_offload_mode(
                                                model_size_gb,
                                                available_vram,
                                                total_layers,
                                                self.server_config.context_size,
                                            );

                                        if !matches!(
                                            suggested_mode,
                                            crate::core::CpuOffloadMode::Auto
                                        ) {
                                            self.server_config.cpu_offload = suggested_mode;
                                            if suggested_layers >= 0 {
                                                self.server_config.gpu_layers = suggested_layers;
                                            }
                                        }
                                    }
                                }

                                if self.selected_provider == "vllm" {
                                    crate::services::apply_vllm_smart_config(
                                        model.size_gb,
                                        self.server_config.context_size,
                                        model.is_moe,
                                        Some(&self.gpus),
                                        &mut self.server_config,
                                    );
                                    let is_gguf = model.path.to_lowercase().contains("gguf");
                                    self.log_buffer.push_info(format!(
                                        "[vLLM GGUF] Checking model path: {} (looks like GGUF: {})",
                                        model.path, is_gguf
                                    ));
                                    if is_gguf {
                                        if let Some((hf_id, tokenizer, logs)) =
                                            crate::providers::vllm::get_gguf_tokenizer_info(
                                                &model.path,
                                            )
                                        {
                                            for log_msg in &logs {
                                                self.log_buffer.push_info(log_msg.clone());
                                            }
                                            self.server_config.huggingface_id = hf_id;
                                            self.server_config.tokenizer = tokenizer;
                                        } else {
                                            self.log_buffer
                                                .push_warn("Could not auto-detect HuggingFace ID or tokenizer for GGUF model".to_string());
                                        }
                                    }
                                }
                            }

                            ui.add_space(4.0);
                        }
                    });
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
                    let provider = self.get_current_provider();
                    let defaults = provider.default_settings();
                    let current = self.server_controller.get_provider_settings();
                    self.provider_settings = ProviderSettings {
                        binary_path: if !current.binary_path.is_empty() {
                            current.binary_path.clone()
                        } else {
                            defaults.binary_path
                        },
                        env_script: if !current.env_script.is_empty() {
                            current.env_script.clone()
                        } else {
                            defaults.env_script
                        },
                        additional_args: if !current.additional_args.is_empty() {
                            current.additional_args.clone()
                        } else {
                            defaults.additional_args
                        },
                        health_endpoint: if !current.health_endpoint.is_empty() {
                            current.health_endpoint.clone()
                        } else {
                            defaults.health_endpoint
                        },
                        heartbeat_interval_secs: if current.heartbeat_interval_secs > 0 {
                            current.heartbeat_interval_secs
                        } else {
                            defaults.heartbeat_interval_secs
                        },
                        venv_path: if !current.venv_path.is_empty() {
                            current.venv_path.clone()
                        } else {
                            defaults.venv_path
                        },
                    };
                    self.show_provider_settings = true;
                }

                if ui.button("Setup").clicked() {
                    self.provider_installed_cache = None;
                    self.show_provider_setup = true;
                    self.provider_setup_provider = self.selected_provider.clone();
                }
            });

            ui.separator();

            egui::Grid::new("config").num_columns(2).show(ui, |ui| {
                ui.label("Model:");
                ui.text_edit_singleline(&mut self.server_config.model_path);
                ui.end_row();

                ui.label("HuggingFace ID:");
                ui.text_edit_singleline(&mut self.server_config.huggingface_id);
                ui.end_row();

                ui.label("Tokenizer:");
                ui.text_edit_singleline(&mut self.server_config.tokenizer);
                ui.end_row();

                ui.label("Context Size:");
                ui.add(
                    egui::DragValue::new(&mut self.server_config.context_size)
                        .clamp_range(256..=2097152),
                );
                ui.end_row();

                ui.label("Batch Size:");
                ui.add(
                    egui::DragValue::new(&mut self.server_config.batch_size).clamp_range(1..=8192),
                );
                ui.end_row();

                ui.label("GPU Layers:");
                ui.horizontal(|ui| {
                    let mut gpu_layers_val = if self.server_config.gpu_layers < 0 {
                        -1
                    } else {
                        self.server_config.gpu_layers
                    };
                    let display_text = if gpu_layers_val < 0 {
                        "All".to_string()
                    } else {
                        gpu_layers_val.to_string()
                    };

                    if ui.button("All").clicked() {
                        gpu_layers_val = -1;
                    }

                    ui.add(
                        egui::DragValue::new(&mut gpu_layers_val)
                            .custom_formatter(|n, _| {
                                let n = n as i32;
                                if n < 0 {
                                    "All".to_string()
                                } else {
                                    n.to_string()
                                }
                            })
                            .custom_parser(|s: &str| {
                                if s.eq_ignore_ascii_case("all") {
                                    Some(-1.0)
                                } else {
                                    s.parse::<i32>().ok().map(|v| v as f64)
                                }
                            }),
                    );
                    self.server_config.gpu_layers = gpu_layers_val;

                    if ui.button("Auto").clicked() {
                        let model_size_gb =
                            if let Ok(meta) = std::fs::metadata(&self.server_config.model_path) {
                                meta.len() as f32 / (1024.0 * 1024.0 * 1024.0)
                            } else {
                                7.0
                            };

                        let total_layers = if self.selected_provider == "llama.cpp" {
                            crate::providers::llama_cpp::read_gguf_n_layer(
                                &self.server_config.model_path,
                            )
                            .map(|l| l as i32)
                            .unwrap_or(-1)
                        } else {
                            -1
                        };

                        let recommended = recommend_gpu_layers(model_size_gb, total_layers);
                        self.server_config.gpu_layers = recommended;
                    }

                    ui.label(format!("({})", display_text));
                });
                ui.end_row();

                ui.label("GPU:");
                ui.horizontal(|ui| {
                    let gpu_options = self.build_gpu_selection_options();
                    let current_label = match self.server_config.selected_gpu {
                        None => "Auto".to_string(),
                        Some(idx) => format!("GPU {}", idx),
                    };
                    egui::ComboBox::from_id_source("gpu_selection")
                        .selected_text(&current_label)
                        .show_ui(ui, |ui| {
                            for (label, value) in &gpu_options {
                                ui.selectable_value(
                                    &mut self.server_config.selected_gpu,
                                    *value,
                                    label,
                                );
                            }
                        });
                });
                ui.end_row();

                ui.label("Threads:");
                ui.add(egui::DragValue::new(&mut self.server_config.threads).clamp_range(1..=64));
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

                ui.label("Parallel:");
                ui.add(
                    egui::DragValue::new(&mut self.server_config.num_prompt_tracking)
                        .clamp_range(1..=64),
                );
                ui.end_row();

                ui.label("CPU Offload:");
                ui.horizontal(|ui| {
                    let offload_options = vec![
                        ("Auto", crate::core::CpuOffloadMode::Auto),
                        ("Offload", crate::core::CpuOffloadMode::Offload),
                        ("Full GPU", crate::core::CpuOffloadMode::Disabled),
                        ("CPU Only", crate::core::CpuOffloadMode::FullOffload),
                    ];
                    let current_label = match self.server_config.cpu_offload {
                        crate::core::CpuOffloadMode::Auto => "Auto",
                        crate::core::CpuOffloadMode::Offload => "Offload",
                        crate::core::CpuOffloadMode::Disabled => "Full GPU",
                        crate::core::CpuOffloadMode::FullOffload => "CPU Only",
                    };
                    egui::ComboBox::from_id_source("cpu_offload")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for (label, value) in &offload_options {
                                ui.selectable_value(
                                    &mut self.server_config.cpu_offload,
                                    value.clone(),
                                    *label,
                                );
                            }
                        });
                    if self.selected_provider == "llama.cpp" {
                        let model_size_gb = if !self.server_config.model_path.is_empty() {
                            std::fs::metadata(&self.server_config.model_path)
                                .map(|m| m.len() as f32 / (1024.0 * 1024.0 * 1024.0))
                                .unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        let available_vram: u32 = self.gpus.iter().map(|g| g.total_vram_mb).sum();
                        if model_size_gb > 0.0 && available_vram > 0 {
                            let vram_needed = (model_size_gb * 1024.0 * 1.2) as u32;
                            if available_vram < vram_needed {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "(VRAM needed: {}MB, available: {}MB)",
                                        vram_needed, available_vram
                                    ))
                                    .color(egui::Color32::YELLOW),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("(VRAM sufficient)")
                                        .color(egui::Color32::GREEN),
                                );
                            }
                        }
                    } else {
                        ui.label(
                            egui::RichText::new("(Not supported by this provider)")
                                .color(egui::Color32::GRAY)
                                .italics(),
                        );
                    }
                });
                ui.end_row();

                ui.label("Host:");
                ui.text_edit_singleline(&mut self.server_config.host);
                ui.end_row();

                ui.label("Port:");
                ui.add(egui::DragValue::new(&mut self.server_config.port).clamp_range(1..=65535));
                ui.end_row();

                ui.label("Temperature:");
                let mut temp_val = self.server_config.temperature.unwrap_or(0.7);
                ui.add(
                    egui::DragValue::new(&mut temp_val)
                        .speed(0.1)
                        .clamp_range(0.0..=2.0),
                )
                .on_hover_text("Sampling temperature (0.0-2.0)");
                if temp_val != self.server_config.temperature.unwrap_or(0.7) {
                    self.server_config.temperature = Some(temp_val);
                } else if self.server_config.temperature.is_none() {
                    self.server_config.temperature = Some(0.7);
                }
                ui.end_row();

                ui.label("Top-K:");
                let mut top_k_val = self.server_config.top_k.unwrap_or(40);
                ui.add(
                    egui::DragValue::new(&mut top_k_val)
                        .speed(1.0)
                        .clamp_range(0..=100),
                )
                .on_hover_text("Top-K sampling (0-100)");
                if top_k_val != self.server_config.top_k.unwrap_or(40) {
                    self.server_config.top_k = Some(top_k_val);
                } else if self.server_config.top_k.is_none() {
                    self.server_config.top_k = Some(40);
                }
                ui.end_row();

                ui.label("Top-P:");
                let mut top_p_val = self.server_config.top_p.unwrap_or(0.95);
                ui.add(
                    egui::DragValue::new(&mut top_p_val)
                        .speed(0.01)
                        .clamp_range(0.0..=1.0),
                )
                .on_hover_text("Top-P (nucleus) sampling (0.0-1.0)");
                if top_p_val != self.server_config.top_p.unwrap_or(0.95) {
                    self.server_config.top_p = Some(top_p_val);
                } else if self.server_config.top_p.is_none() {
                    self.server_config.top_p = Some(0.95);
                }
                ui.end_row();

                ui.label("Min-P:");
                let mut min_p_val = self.server_config.min_p.unwrap_or(0.05);
                ui.add(
                    egui::DragValue::new(&mut min_p_val)
                        .speed(0.01)
                        .clamp_range(0.0..=1.0),
                )
                .on_hover_text("Min-P sampling (0.0-1.0)");
                if min_p_val != self.server_config.min_p.unwrap_or(0.05) {
                    self.server_config.min_p = Some(min_p_val);
                } else if self.server_config.min_p.is_none() {
                    self.server_config.min_p = Some(0.05);
                }
                ui.end_row();

                ui.label("Presence Penalty:");
                let mut presence_penalty_val = self.server_config.presence_penalty.unwrap_or(0.0);
                ui.add(
                    egui::DragValue::new(&mut presence_penalty_val)
                        .speed(0.1)
                        .clamp_range(-2.0..=2.0),
                )
                .on_hover_text("Presence penalty (-2.0-2.0)");
                if presence_penalty_val != self.server_config.presence_penalty.unwrap_or(0.0) {
                    self.server_config.presence_penalty = Some(presence_penalty_val);
                } else if self.server_config.presence_penalty.is_none() {
                    self.server_config.presence_penalty = Some(0.0);
                }
                ui.end_row();

                ui.label("Repetition Penalty:");
                let mut repetition_penalty_val =
                    self.server_config.repetition_penalty.unwrap_or(1.1);
                ui.add(
                    egui::DragValue::new(&mut repetition_penalty_val)
                        .speed(0.1)
                        .clamp_range(0.0..=5.0),
                )
                .on_hover_text("Repetition penalty (0.0-5.0)");
                if repetition_penalty_val != self.server_config.repetition_penalty.unwrap_or(1.1) {
                    self.server_config.repetition_penalty = Some(repetition_penalty_val);
                } else if self.server_config.repetition_penalty.is_none() {
                    self.server_config.repetition_penalty = Some(1.1);
                }
                ui.end_row();

                ui.label("Enable Thinking:");
                ui.horizontal(|ui| {
                    let mut enabled = self.server_config.enable_thinking.unwrap_or(false);
                    if ui.checkbox(&mut enabled, "").changed() {
                        self.server_config.enable_thinking = Some(enabled);
                    }
                    ui.label("Enable thinking/deepseek reasoning");
                });
                ui.end_row();

            ui.label("Additional Args:");
            ui.text_edit_singleline(&mut self.server_config.additional_args);
            ui.end_row();

            // Show mmproj path for multimodal models (only for llama.cpp)
            if self.selected_provider == "llama.cpp" {
                ui.label("Vision Model:");
                ui.horizontal(|ui| {
                    let mut mmproj_path = self.server_config.mmproj_path.clone();
                    let text_edit = ui.text_edit_singleline(&mut mmproj_path);
                    if text_edit.changed() {
                        self.server_config.mmproj_path = mmproj_path;
                    }
                    if !self.server_config.mmproj_path.is_empty() {
                        ui.label("✓").on_hover_text("Vision support enabled");
                    }
                });
                ui.end_row();
            }
        });

        ui.separator();

        ui.horizontal(|ui| {
            let status = self.server_controller.get_status();
                match status {
                    crate::models::ServerStatus::Running => {
                        if ui.button("Stop Server").clicked() {
                            if !self.server_config.model_path.is_empty()
                                || !self.server_config.huggingface_id.is_empty()
                            {
                                let key = if !self.server_config.huggingface_id.is_empty() {
                                    self.server_config.huggingface_id.clone()
                                } else {
                                    self.server_config.model_path.clone()
                                };
                                save_model_config(
                                    &key,
                                    &self.server_config,
                                    &self.selected_provider,
                                )
                                .ok();
                            }
                            self.server_controller.stop().ok();
                            self.started_hf_model = None;
                        }
                    }
                    _ => {
                        if ui.button("Start Server").clicked() {
                            self.log_buffer.clear();
                            if !self.server_config.model_path.is_empty()
                                || !self.server_config.huggingface_id.is_empty()
                            {
                                let key = if !self.server_config.huggingface_id.is_empty() {
                                    self.server_config.huggingface_id.clone()
                                } else {
                                    self.server_config.model_path.clone()
                                };
                                save_model_config(
                                    &key,
                                    &self.server_config,
                                    &self.selected_provider,
                                )
                                .ok();
                            }
                            let provider = self.get_current_provider();
                            self.server_controller
                                .start(&provider, &self.server_config)
                                .ok();
                            if !self.server_config.huggingface_id.is_empty() {
                                self.started_hf_model =
                                    Some(self.server_config.huggingface_id.clone());
                            }
                        }
                    }
                }

                if ui.button("Recommended").clicked() {
                    let model_path = &self.server_config.model_path;
                    let quantization = if let Some(idx) = self.selected_model {
                        if let Some(model) = self.models.get(idx) {
                            &model.quantization
                        } else {
                            "q4_0"
                        }
                    } else {
                        "q4_0"
                    };
                    let params = crate::services::get_recommended_params(model_path, quantization);
                    crate::services::apply_recommended_params(&mut self.server_config, &params);
                }

                if ui.button("Example").clicked() {
                    self.show_cmdline_dialog = true;
                    self.cmdline_input = String::new();
                }
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Always request repaint to keep UI responsive for background updates (logs, stats, etc.)
        // This is lightweight - actual expensive work is throttled below
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        // Increment frame counter at start of each frame
        self.frame_counter = self.frame_counter.wrapping_add(1);

        // Throttle expensive operations to every ~1 second (at 60fps)
        if self.frame_counter % 60 == 0 {
            self.server_controller.refresh_external_detection();
        }

        // Update cached stats every ~0.5 seconds for smooth display
        if self.frame_counter % 30 == 0 {
            let new_stats = get_system_stats();
            self.cached_stats = Some((self.frame_counter, new_stats));
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("LLLMMan");
                ui.separator();
                ui.label(format!("{} GPUs detected", self.gpus.len()));
                ui.separator();
                if ui.button("GPU Settings").clicked() {
                    self.show_gpu_settings = true;
                }
                ui.separator();
                // Use cached stats (updated every ~0.5s in update())
                let stats = self.cached_stats.as_ref().map(|(_, s)| s.clone()).unwrap_or_else(|| {
                    // Fallback if cache is empty
                    MonitorStats::default()
                });

                // System Resources - Single line layout with inline bars
                // CPU
                let cpu_color = usage_color(stats.cpu_percent);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("CPU").strong().size(9.0));
                    let bar_width = 40.0;
                    let bar_height = 8.0;
                    let (response, painter) = ui.allocate_painter(
                        egui::vec2(bar_width, bar_height),
                        egui::Sense::hover(),
                    );
                    let bar_rect = response.rect;
                    painter.rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
                    let fill_width = bar_width * (stats.cpu_percent / 100.0).min(1.0);
                    if fill_width > 0.0 {
                        painter.rect_filled(
                            egui::Rect::from_min_size(bar_rect.min, egui::vec2(fill_width, bar_height)),
                            2.0,
                            cpu_color,
                        );
                    }
                    ui.label(
                        egui::RichText::new(format!("{:.0}%", stats.cpu_percent))
                            .color(cpu_color)
                            .strong()
                            .size(9.0),
                    );
                    if let Some(cpu_temp) = stats.cpu_temperature {
                        ui.label(
                            egui::RichText::new(format!("{:.0}°C", cpu_temp))
                                .color(temp_color(cpu_temp))
                                .size(9.0),
                        );
                    }
                });

                ui.separator();

                // RAM
                let ram_percent = if stats.ram_total_mb > 0 {
                    (stats.ram_used_mb as f32 / stats.ram_total_mb as f32) * 100.0
                } else {
                    0.0
                };
                let ram_color = usage_color(ram_percent);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("RAM").strong().size(9.0));
                    let bar_width = 50.0;
                    let bar_height = 8.0;
                    let (response, painter) = ui.allocate_painter(
                        egui::vec2(bar_width, bar_height),
                        egui::Sense::hover(),
                    );
                    let bar_rect = response.rect;
                    painter.rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
                    let fill_width = bar_width * (ram_percent / 100.0).min(1.0);
                    if fill_width > 0.0 {
                        painter.rect_filled(
                            egui::Rect::from_min_size(bar_rect.min, egui::vec2(fill_width, bar_height)),
                            2.0,
                            ram_color,
                        );
                    }
                    ui.label(
                        egui::RichText::new(format!("{:.0}%", ram_percent))
                            .color(ram_color)
                            .strong()
                            .size(9.0),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "{:.1}/{:.1}G",
                            stats.ram_used_mb as f32 / 1024.0,
                            stats.ram_total_mb as f32 / 1024.0
                        ))
                        .color(egui::Color32::GRAY)
                        .size(8.0),
                    );
                });

                // GPUs
                for gpu_info in &self.gpus {
                    ui.separator();
                    let index = gpu_info.index as usize;
                    let (used_vram, total_vram) = if index < stats.gpu_vram_usage.len() {
                        (stats.gpu_vram_usage[index].1, stats.gpu_vram_usage[index].2)
                    } else {
                        (0, 0)
                    };
                    let vram_percent = if total_vram > 0 {
                        (used_vram as f32 / total_vram as f32) * 100.0
                    } else {
                        0.0
                    };
                    let temp = if index < stats.gpu_temperatures.len() {
                        stats.gpu_temperatures[index].temperature_c.unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let vram_color = usage_color(vram_percent);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("GPU{}", gpu_info.index))
                                .strong()
                                .size(9.0),
                        );
                        let bar_width = 60.0;
                        let bar_height = 8.0;
                        let (response, painter) = ui.allocate_painter(
                            egui::vec2(bar_width, bar_height),
                            egui::Sense::hover(),
                        );
                        let bar_rect = response.rect;
                        painter.rect_filled(bar_rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
                        let fill_width = bar_width * (vram_percent / 100.0).min(1.0);
                        if fill_width > 0.0 {
                            painter.rect_filled(
                                egui::Rect::from_min_size(bar_rect.min, egui::vec2(fill_width, bar_height)),
                                2.0,
                                vram_color,
                            );
                        }
                        ui.label(
                            egui::RichText::new(format!("{:.0}%", vram_percent))
                                .color(vram_color)
                                .strong()
                                .size(9.0),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.1}/{:.1}G",
                                used_vram as f32 / 1024.0,
                                total_vram as f32 / 1024.0
                            ))
                            .color(egui::Color32::GRAY)
                            .size(8.0),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.0}°C", temp))
                                .color(temp_color(temp))
                                .size(9.0),
                        );
                    });
                }
                let current_status = self.server_controller.get_status();
                let status_just_became_running =
                    matches!(current_status, crate::models::ServerStatus::Running)
                        && !matches!(
                            self.previous_server_status,
                            crate::models::ServerStatus::Running
                        );

                if status_just_became_running {
                    if let Some(hf_id) = self.started_hf_model.clone() {
                        let provider = self.get_current_provider();

                        // Scan directories for downloaded model
                        let all_dirs: Vec<String> = self
                            .settings
                            .scan_directories
                            .clone()
                            .into_iter()
                            .chain(provider.default_model_directories())
                            .collect();

                        for dir in &all_dirs {
                            let found = provider.scan_models(dir);
                            for model in found {
                                if model.path.contains(&hf_id)
                                    || model.name.contains(hf_id.split('/').last().unwrap_or(""))
                                {
                                    if !self.models.iter().any(|m| m.path == model.path) {
                                        self.models.push(model.clone());
                                    }
                                    if self.server_config.huggingface_id == hf_id {
                                        self.server_config.model_path = model.path.clone();
                                        self.server_config.huggingface_id = String::new();
                                        self.started_hf_model = None;
                                        self.selected_model =
                                            self.models.iter().position(|m| m.path == model.path);
                                    }
                                }
                            }
                        }
                    }
                }

                if matches!(current_status, crate::models::ServerStatus::Running)
                    && self.frame_counter
                        % (self.provider_settings.heartbeat_interval_secs as u32 * 50)
                        == 0
                {
                    if let Some(server_stats) = crate::services::fetch_server_stats(
                        &self.server_config.host,
                        self.server_config.port,
                        &self.provider_settings.health_endpoint,
                    ) {
                        let tps = server_stats
                            .time_per_token
                            .map(|t| if t > 0.0 { 1000.0 / t } else { 0.0 })
                            .unwrap_or(0.0);
                        ui.separator();
                        ui.label(format!(
                            "Tokens/s: {:.1} | Queue: {}",
                            tps,
                            server_stats.queue_size.unwrap_or(0)
                        ));
                    }
                }

                self.previous_server_status = current_status.clone();
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

                    ui.label("Venv Path:");
                    ui.text_edit_singleline(&mut self.provider_settings.venv_path);
                    ui.small("Path to Python virtualenv directory (e.g., /home/qing/Project/vllm/.venv). Activated automatically before starting the server.");

                    ui.add_space(8.0);

                    ui.label("Environment Script:");
                    ui.text_edit_singleline(&mut self.provider_settings.env_script);
                    ui.small("Path to env script to source before starting (e.g., /home/qing/Projects/.venv/bin/activate)");

                    ui.add_space(8.0);

                    ui.label("Additional Arguments:");
                    ui.text_edit_singleline(&mut self.provider_settings.additional_args);
                    ui.small("Extra CLI args passed to the server (space-separated, e.g., --flash-attention --no-mmap)");

                    ui.add_space(8.0);

                    ui.label("Health Endpoint:");
                    ui.text_edit_singleline(&mut self.provider_settings.health_endpoint);
                    ui.small("HTTP endpoint used to check if server is alive (e.g., /health, /v1/models)");

                    ui.add_space(8.0);

                    ui.label("Heartbeat Interval (seconds):");
                    ui.add(
                        egui::DragValue::new(&mut self.provider_settings.heartbeat_interval_secs)
                            .clamp_range(1..=60),
                    );
                    ui.small("How often to ping the server for status (1-60 seconds)");

                    ui.add_space(12.0);

                    if ui.button("Save").clicked() {
                        self.server_controller.set_provider_settings(self.provider_settings.clone());
                        save_provider_settings_for(&self.selected_provider, &self.provider_settings).ok();
                        // Invalidate cache so setup dialog checks with new settings
                        self.provider_installed_cache = None;
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

                    ui.push_id("log_window", |ui| {
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
                        let current_value = self
                            .server_config
                            .get_option(&opt.id)
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
                                            ui.selectable_value(&mut value, v.clone(), v);
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

                    ui.label("System Info:");
                    ui.label(get_system_info_summary());

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
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Download Model");
                        ui.separator();

                        ui.label("Source:");
                        egui::ComboBox::from_id_source("download_source")
                            .selected_text(&self.download_source_type)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.download_source_type,
                                    "HuggingFace".to_string(),
                                    "HuggingFace",
                                );
                                ui.selectable_value(
                                    &mut self.download_source_type,
                                    "Direct URL".to_string(),
                                    "Direct URL",
                                );
                                ui.selectable_value(
                                    &mut self.download_source_type,
                                    "GitHub Release".to_string(),
                                    "GitHub Release",
                                );
                            });

                        ui.separator();

                        match self.download_source_type.as_str() {
                            "HuggingFace" => {
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.search_query)
                                            .hint_text("Search models..."),
                                    );
                                    if ui.button("Search").clicked()
                                        && !self.search_query.is_empty()
                                    {
                                        self.is_searching = true;
                                        let query = self.search_query.clone();
                                        let search_results = self.search_results.clone();
                                        let app_handle = ctx.clone();
                                        std::thread::spawn(move || {
                                            let downloader = HuggingFaceDownloader::new();
                                            match downloader.search_sync(&query) {
                                                Ok(results) => {
                                                    let mut results_lock =
                                                        search_results.blocking_write();
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
                                    ui.label(format!("Search Results ({}):", results_guard.len()));
                                    ui.push_id("search_results", |ui| {
                                        egui::ScrollArea::vertical().show(ui, |ui| {
                                            for result in results_guard.iter() {
                                                ui.horizontal(|ui| {
                                                    if ui.button(&result.name).clicked() {
                                                        self.download_url = result.id.clone();
                                                    }
                                                    ui.label(
                                                        egui::RichText::new(format!(
                                                            "{} downloads",
                                                            result.downloads
                                                        ))
                                                        .size(10.0)
                                                        .color(egui::Color32::from_rgb(
                                                            140, 140, 150,
                                                        )),
                                                    );
                                                });
                                            }
                                        });
                                    });
                                }

                                ui.separator();
                                ui.label("Or enter model ID directly:");
                                ui.text_edit_singleline(&mut self.download_url);
                                ui.label("Example: TheBloke/Mistral-7B-Instruct-v0.1-GGUF");

                                if ui.button("Set as HuggingFace Model").clicked()
                                    && !self.download_url.is_empty()
                                {
                                    let hf_id = self.download_url.clone();
                                    self.server_config.huggingface_id = hf_id;
                                    self.server_config.model_path = String::new();
                                    self.server_config.gpu_layers = -1;
                                }
                            }
                            "Direct URL" => {
                                ui.label("Enter direct download URL:");
                                ui.text_edit_singleline(&mut self.download_url);

                                if ui.button("Start Download").clicked()
                                    && !self.download_url.is_empty()
                                {
                                    let url = self.download_url.clone();
                                    let manager = self.download_manager.clone();

                                    std::thread::spawn(move || {
                                        let downloader = DirectUrlDownloader::new();
                                        match downloader.fetch_headers_sync(&url) {
                                            Ok(headers) => {
                                                let file_name = headers.file_name.clone();
                                                let content_length = headers.content_length;
                                                let source =
                                                    ModelSource::DirectUrl { url: url.clone() };
                                                let manager = manager.blocking_read();
                                                let _id = manager
                                                    .add_task_sync(source, file_name.clone());
                                                log::info!(
                                                    "Started download: {} ({} bytes)",
                                                    file_name,
                                                    content_length
                                                );
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
                                    ui.label("Repository:");
                                    ui.text_edit_singleline(&mut self.download_github_repo);
                                });

                                if ui.button("Fetch Release").clicked()
                                    && !self.download_github_owner.is_empty()
                                    && !self.download_github_repo.is_empty()
                                {
                                    let owner = self.download_github_owner.clone();
                                    let repo = self.download_github_repo.clone();
                                    let manager = self.download_manager.clone();

                                    std::thread::spawn(move || {
                                        let downloader = GitHubReleaseDownloader::new();
                                        match downloader.get_latest_release_sync(&owner, &repo) {
                                            Ok(release) => {
                                                log::info!(
                                                    "Found release: {} with {} assets",
                                                    release.tag,
                                                    release.assets.len()
                                                );
                                                let manager = manager.blocking_read();
                                                for asset in release.assets {
                                                    let source = ModelSource::GitHubRelease {
                                                        owner: owner.clone(),
                                                        repo: repo.clone(),
                                                        tag: release.tag.clone(),
                                                        asset_name: asset.name.clone(),
                                                    };
                                                    let _id =
                                                        manager.add_task_sync(source, asset.name);
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

                        ui.push_id("download_queue", |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for task in &tasks {
                                    ui.horizontal(|ui| {
                                        ui.label(&task.file_name);
                                        ui.separator();
                                        match &task.status {
                                            crate::core::DownloadStatus::Pending => {
                                                ui.label("Pending")
                                            }
                                            crate::core::DownloadStatus::Downloading => {
                                                let progress = if task.total_bytes > 0 {
                                                    task.downloaded_bytes as f32
                                                        / task.total_bytes as f32
                                                } else {
                                                    0.0
                                                };
                                                ui.add(
                                                    egui::ProgressBar::new(progress)
                                                        .text(&format!("{:.1}%", progress * 100.0)),
                                                )
                                            }
                                            crate::core::DownloadStatus::Completed => {
                                                ui.label("Completed")
                                            }
                                            crate::core::DownloadStatus::Failed(e) => {
                                                ui.label(format!("Failed: {}", e))
                                            }
                                            crate::core::DownloadStatus::Cancelled => {
                                                ui.label("Cancelled")
                                            }
                                        }
                                    });
                                }
                            });
                        });
                    });
                });
        }

        if self.show_provider_setup {
            let provider_id = &self.provider_setup_provider;

            let installed = if let Some((cached_id, cached_result)) = &self.provider_installed_cache
            {
                if cached_id == provider_id {
                    Some(*cached_result)
                } else {
                    None
                }
            } else {
                None
            };

            let installed = match installed {
                Some(result) => result,
                None => {
                    let result = check_provider_installed(provider_id);
                    self.provider_installed_cache = Some((provider_id.clone(), result));
                    result
                }
            };

            if let Some(info) = get_provider_install_info(provider_id) {
                egui::Window::new(format!("Setup: {}", info.provider_name))
                    .open(&mut self.show_provider_setup)
                    .default_width(600.0)
                    .default_height(400.0)
                    .show(ctx, |ui| {
                        ui.heading(format!("{} Setup", info.provider_name));
                        ui.separator();

                        if installed {
                            ui.colored_label(
                                egui::Color32::GREEN,
                                format!("✓ {} is installed", info.provider_name),
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::RED,
                                format!("✗ {} is NOT installed", info.provider_name),
                            );
                        }

                        ui.separator();
                        ui.heading("Quick Install");
                        ui.label(info.simple_description);
                        ui.horizontal(|ui| {
                            ui.code(info.simple_command);
                            if ui.button("Copy").clicked() {
                                ui.output_mut(|o| o.copied_text = info.simple_command.to_string());
                            }
                        });

                        ui.separator();
                        ui.heading("Advanced Install");
                        ui.label(info.advanced_description);
                        ui.horizontal(|ui| {
                            ui.code(info.advanced_command);
                            if ui.button("Copy").clicked() {
                                ui.output_mut(|o| {
                                    o.copied_text = info.advanced_command.to_string()
                                });
                            }
                        });
                    });
            }
        }

        if self.show_cmdline_dialog {
            let mut close_dialog = false;
            let mut apply_cmdline = false;

            egui::Window::new("Import from Command Line")
                .open(&mut self.show_cmdline_dialog)
                .default_width(600.0)
                .default_height(300.0)
                .show(ctx, |ui| {
                    ui.heading("Import from Command Line");
                    ui.separator();
                    ui.label(
                        "Paste a complete command line. Known parameters will be extracted and filled in.",
                    );
                    ui.label(
                        "Example: llama-server -m /path/to/model.gguf -c 8192 -ngl 35 -t 4 --port 8080",
                    );
                    ui.add_space(8.0);

                    ui.text_edit_multiline(&mut self.cmdline_input);

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() && !self.cmdline_input.is_empty() {
                            apply_cmdline = true;
                        }
                        if ui.button("Cancel").clicked() {
                            close_dialog = true;
                        }
                    });
                });

            if close_dialog {
                self.show_cmdline_dialog = false;
            }

            if apply_cmdline {
                let parsed = crate::services::parse_command_line(&self.cmdline_input);
                self.server_config.model_path = parsed.model_path;
                if parsed.context_size != 4096 {
                    self.server_config.context_size = parsed.context_size;
                }
                if parsed.batch_size != 512 {
                    self.server_config.batch_size = parsed.batch_size;
                }
                if parsed.gpu_layers != -1 {
                    self.server_config.gpu_layers = parsed.gpu_layers;
                }
                if parsed.threads != 8 {
                    self.server_config.threads = parsed.threads;
                }
                if parsed.port != 8080 {
                    self.server_config.port = parsed.port;
                }
                if parsed.host != "0.0.0.0" {
                    self.server_config.host = parsed.host;
                }
                if !parsed.cache_type_k.is_empty() {
                    self.server_config.cache_type_k = parsed.cache_type_k;
                }
                if !parsed.cache_type_v.is_empty() {
                    self.server_config.cache_type_v = parsed.cache_type_v;
                }
                if parsed.num_prompt_tracking != 1 {
                    self.server_config.num_prompt_tracking = parsed.num_prompt_tracking;
                }
                if parsed.temperature.is_some() {
                    self.server_config.temperature = parsed.temperature;
                }
                if parsed.top_k.is_some() {
                    self.server_config.top_k = parsed.top_k;
                }
                if parsed.top_p.is_some() {
                    self.server_config.top_p = parsed.top_p;
                }
                if parsed.min_p.is_some() {
                    self.server_config.min_p = parsed.min_p;
                }
                if parsed.presence_penalty.is_some() {
                    self.server_config.presence_penalty = parsed.presence_penalty;
                }
                if parsed.repetition_penalty.is_some() {
                    self.server_config.repetition_penalty = parsed.repetition_penalty;
                }
                if parsed.enable_thinking.is_some() {
                    self.server_config.enable_thinking = parsed.enable_thinking;
                }
                if !parsed.additional_args.is_empty() {
                    self.server_config.additional_args = parsed.additional_args;
                }
                if !parsed.tokenizer.is_empty() {
                    self.server_config.tokenizer = parsed.tokenizer;
                }
                self.show_cmdline_dialog = false;
            }
        }
    }
}

fn temp_color(temp: f32) -> egui::Color32 {
    if temp < 50.0 {
        egui::Color32::from_rgb(0, 200, 0)
    } else if temp < 80.0 {
        egui::Color32::from_rgb(255, 200, 0)
    } else {
        egui::Color32::from_rgb(255, 50, 50)
    }
}

fn usage_color(percent: f32) -> egui::Color32 {
    if percent < 50.0 {
        egui::Color32::from_rgb(0, 200, 0)
    } else if percent < 80.0 {
        egui::Color32::from_rgb(255, 200, 0)
    } else {
        egui::Color32::from_rgb(255, 50, 50)
    }
}
