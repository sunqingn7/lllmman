#![allow(dead_code)]
use eframe::egui;
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
                available_providers[0].0.to_string()
            }
        });

        let (models, server_config) = Self::build_for_provider(&selected_provider, &settings);

        let selected_model = models.iter().position(|m| {
            m.path == server_config.model_path || m.path.contains(&server_config.model_path)
        });

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

            self.frame_counter = self.frame_counter.wrapping_add(1);
        });

        ui.separator();

        if self.bottom_view == BottomView::Log {
            let entries = self.log_buffer.get_entries();
            ui.push_id("log_panel", |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if entries.is_empty() {
                            ui.label(
                                egui::RichText::new(
                                    "No logs yet. Start a server to see output here.",
                                )
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

                            let name_rect = egui::Rect::from_min_size(
                                response.rect.min + egui::vec2(8.0, 4.0),
                                egui::vec2(response.rect.width() - 16.0, 20.0),
                            );
                            ui.painter().text(
                                name_rect.min,
                                egui::Align2::LEFT_TOP,
                                &model.name,
                                egui::FontId::new(14.0, egui::FontFamily::Monospace),
                                text_color,
                            );

                            let detail_text = format!(
                                "{:.1} GB  •  {}  •  {}",
                                model.size_gb,
                                model.quantization,
                                match model.model_type {
                                    crate::models::ModelType::TextOnly => "Text",
                                    crate::models::ModelType::Tooling => "Tool",
                                    crate::models::ModelType::Vision => "Vision",
                                    crate::models::ModelType::Multimodal => "Multi",
                                }
                            );
                            let detail_rect = egui::Rect::from_min_size(
                                response.rect.min + egui::vec2(8.0, 26.0),
                                egui::vec2(response.rect.width() - 16.0, 16.0),
                            );
                            ui.painter().text(
                                detail_rect.min,
                                egui::Align2::LEFT_TOP,
                                &detail_text,
                                egui::FontId::new(11.0, egui::FontFamily::Proportional),
                                subtext_color,
                            );

                            if response.clicked() {
                                if let Some(old_i) = self.selected_model {
                                    if let Some(old_model) = filtered.get(old_i) {
                                        save_model_config(&old_model.path, &self.server_config)
                                            .ok();
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
                                    self.server_config.num_prompt_tracking =
                                        fallback.num_prompt_tracking;
                                    self.server_config.additional_args = fallback.additional_args;
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
                    self.show_provider_settings = true;
                }

                if ui.button("Setup").clicked() {
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

                ui.label("Host:");
                ui.text_edit_singleline(&mut self.server_config.host);
                ui.end_row();

                ui.label("Port:");
                ui.add(egui::DragValue::new(&mut self.server_config.port).clamp_range(1..=65535));
                ui.end_row();

                ui.label("Additional Args:");
                ui.text_edit_singleline(&mut self.server_config.additional_args);
                ui.end_row();
            });

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Example").clicked() {
                    self.show_cmdline_dialog = true;
                    self.cmdline_input = String::new();
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
                                save_model_config(&key, &self.server_config).ok();
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
                                save_model_config(&key, &self.server_config).ok();
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
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Throttle expensive operations
        if self.frame_counter % 60 == 0 {
            self.server_controller.refresh_external_detection();
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
                // Cache system stats for 30 frames (~0.5s at 60fps)
                let stats = match &self.cached_stats {
                    Some((last_frame, cached)) if self.frame_counter - last_frame < 30 => {
                        cached.clone()
                    }
                    _ => {
                        let new_stats = get_system_stats();
                        self.cached_stats = Some((self.frame_counter, new_stats.clone()));
                        new_stats
                    }
                };

                for gpu_info in &self.gpus {
                    let index = gpu_info.index as usize;
                    let (used, total) = if index < stats.gpu_vram_usage.len() {
                        (stats.gpu_vram_usage[index].1, stats.gpu_vram_usage[index].2)
                    } else {
                        (0, 0)
                    };
                    let vram_percent = if total > 0 {
                        (used as f32 / total as f32) * 100.0
                    } else {
                        0.0
                    };

                    let temp = if index < stats.gpu_temperatures.len() {
                        stats.gpu_temperatures[index].temperature_c.unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    ui.label(
                        egui::RichText::new(format!(" GPU{}", gpu_info.index))
                            .color(egui::Color32::WHITE),
                    );
                    ui.label(
                        egui::RichText::new(format!(" ({})", gpu_info.name))
                            .color(egui::Color32::GRAY),
                    );
                    // VRAM Progress bar
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.0}%( {}/{} MB)",
                                vram_percent, used, total
                            ))
                            .size(9.0)
                            .color(usage_color(vram_percent)),
                        );
                        let rect = ui.available_rect_before_wrap();
                        let bar_width = 100.0;
                        let bar_height = 8.0;
                        let bar_rect = egui::Rect::from_min_size(
                            rect.min + egui::vec2(0.0, (rect.height() - bar_height) / 2.0),
                            egui::vec2(bar_width, bar_height),
                        );
                        ui.painter().rect_filled(
                            bar_rect.shrink(1.0),
                            2.0,
                            usage_color(vram_percent).linear_multiply(0.3),
                        );
                        let fill_width = (bar_width * (vram_percent / 100.0)).min(bar_width);
                        ui.painter().rect_filled(
                            egui::Rect::from_min_size(
                                bar_rect.min,
                                egui::vec2(fill_width, bar_height),
                            )
                            .shrink(1.0),
                            2.0,
                            usage_color(vram_percent),
                        );
                        ui.advance_cursor_after_rect(bar_rect);
                    });
                    ui.label(
                        egui::RichText::new(format!(" {:.0}°C", temp)).color(temp_color(temp)),
                    );
                    ui.separator();
                }

                let ram_percent = if stats.ram_total_mb > 0 {
                    (stats.ram_used_mb as f32 / stats.ram_total_mb as f32) * 100.0
                } else {
                    0.0
                };
                ui.label(egui::RichText::new("RAM: ").color(egui::Color32::WHITE));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{:.0}%( {}/{} MB)",
                            ram_percent, stats.ram_used_mb, stats.ram_total_mb
                        ))
                        .size(9.0)
                        .color(usage_color(ram_percent)),
                    );
                    ui.add(
                        egui::ProgressBar::new(ram_percent / 100.0)
                            .animate(true)
                            .show_percentage()
                            .desired_width(100.0),
                    );
                });
                ui.separator();

                ui.label(egui::RichText::new("CPU: ").color(egui::Color32::WHITE));
                ui.label(
                    egui::RichText::new(format!("{:.1}%", stats.cpu_percent))
                        .color(usage_color(stats.cpu_percent)),
                );
                if let Some(cpu_temp) = stats.cpu_temperature {
                    ui.label(
                        egui::RichText::new(format!(" ({:.0}°C)", cpu_temp))
                            .color(temp_color(cpu_temp)),
                    );
                }
                if self.server_controller.get_status() == crate::models::ServerStatus::Running
                    && self.frame_counter % 300 == 0
                {
                    if let Some(hf_id) = self.started_hf_model.clone() {
                        let provider = self.get_current_provider();
                        let mut found_model = false;

                        // Scan directories for downloaded model
                        let all_dirs: Vec<String> = self
                            .settings
                            .scan_directories
                            .clone()
                            .into_iter()
                            .chain(provider.default_model_directories())
                            .collect();

                        for dir in &all_dirs {
                            if found_model {
                                break;
                            }
                            let found = provider.scan_models(dir);
                            for model in found {
                                if model.path.contains(&hf_id)
                                    || model.name.contains(hf_id.split('/').last().unwrap_or(""))
                                {
                                    if !self.models.iter().any(|m| m.path == model.path) {
                                        self.models.push(model.clone());
                                        found_model = true;
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

                    if let Some(server_stats) = crate::services::fetch_server_stats(
                        &self.server_config.host,
                        self.server_config.port,
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
                                    ui.label("Owner:");
                                    ui.text_edit_singleline(&mut self.download_github_owner);
                                });
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
                if !parsed.additional_args.is_empty() {
                    self.server_config.additional_args = parsed.additional_args;
                }
                self.show_cmdline_dialog = false;
            }
        }

        ctx.request_repaint();
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
