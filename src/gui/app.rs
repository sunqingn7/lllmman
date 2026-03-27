use eframe::egui;

use crate::core::{LlmProvider, ModelInfo, ProviderConfig, ProviderRegistry, ServerController};
use crate::models::{AppSettings, GpuInfo};
use crate::services::{config_persistence, get_system_stats, gpu_detector};

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

        let server_config = provider.get_config_template();

        let mut models = Vec::new();
        for dir in &settings.scan_directories {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

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

                // Try to fetch server stats if server is running
                if self.server_controller.get_status() == crate::models::ServerStatus::Running {
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
                                self.selected_model = Some(i);
                                self.server_config.model_path = model.path.clone();
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
                            .clamp_range(256..=128000),
                    );
                    ui.end_row();

                    ui.label("Batch Size:");
                    ui.add(
                        egui::DragValue::new(&mut self.server_config.batch_size)
                            .clamp_range(1..=8192),
                    );
                    ui.end_row();

                    ui.label("GPU Layers:");
                    ui.add(egui::DragValue::new(&mut self.server_config.gpu_layers));
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
                                self.server_controller.stop().ok();
                            }
                        }
                        _ => {
                            if ui.button("Start Server").clicked() {
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
                        ui.label(&opt.description);
                        match &opt.value_type {
                            crate::core::OptionValueType::String => {
                                ui.text_edit_singleline(&mut self.server_config.additional_args);
                            }
                            crate::core::OptionValueType::Number => {
                                // handled elsewhere
                            }
                            crate::core::OptionValueType::Bool => {
                                // handled elsewhere
                            }
                            crate::core::OptionValueType::Select(values) => {
                                egui::ComboBox::from_id_source(&opt.id)
                                    .selected_text(&self.server_config.additional_args)
                                    .show_ui(ui, |ui| {
                                        for v in values {
                                            ui.selectable_value(
                                                &mut self.server_config.additional_args,
                                                v.clone(),
                                                v,
                                            );
                                        }
                                    });
                            }
                        }
                        ui.end_row();
                    }

                    ui.separator();
                    ui.label("Additional CLI Args (space-separated):");
                    ui.text_edit_singleline(&mut self.server_config.additional_args);
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
                .default_width(500.0)
                .default_height(400.0)
                .show(ctx, |ui| {
                    ui.heading("Download Model from HuggingFace");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search_query)
                                .hint_text("Search for models..."),
                        );
                        if ui.button("Search").clicked() {
                            // Search functionality would go here
                            ui.label("Search functionality coming soon...");
                        }
                    });

                    ui.separator();

                    ui.label("Enter a model URL or ID to download:");
                    ui.text_edit_multiline(&mut String::new());

                    ui.separator();
                    ui.small("Note: Full download functionality coming soon.");
                    ui.small("For now, please download GGUF models manually from HuggingFace.");
                });
        }

        ctx.request_repaint();
    }
}
