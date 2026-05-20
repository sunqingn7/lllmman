use eframe::egui::{self, Color32, FontId, Frame, Margin, Sense, Stroke, Ui, Vec2, Align2, RichText, ScrollArea};

use crate::models::deployment::{DeploymentMode, DeploymentProfile, InstanceConfig, RouterConfig, RouterPolicy, RouterProvider};

#[derive(Clone, Copy, PartialEq, Eq)]
enum WizardStep {
    SelectMode,
    ConfigureInstances,
    ConfigureRouter,
    PreviewAndLaunch,
}

pub struct DeploymentWizard {
    pub open: bool,
    step: WizardStep,
    selected_mode: DeploymentMode,
    instance_model_path: String,
    instance_provider: String,
    instance_context_size: u32,
    instance_memory_util: f32,
    instances_to_add: Vec<InstanceConfig>,
    next_port: u16,
    enable_router: bool,
    router_provider: RouterProvider,
    router_policy: RouterPolicy,
    router_port: u16,
    launch_result: Option<String>,
    gpu_count: u32,
    heterogeneous: bool,
}

impl DeploymentWizard {
    pub fn new(gpu_count: u32, heterogeneous: bool) -> Self {
        Self {
            open: false,
            step: WizardStep::SelectMode,
            selected_mode: if gpu_count > 1 {
                if heterogeneous {
                    DeploymentMode::MultiInstance
                } else {
                    DeploymentMode::DataParallel
                }
            } else {
                DeploymentMode::SingleGpu
            },
            instance_model_path: String::new(),
            instance_provider: "vLLM".to_string(),
            instance_context_size: 4096,
            instance_memory_util: 0.9,
            instances_to_add: Vec::new(),
            next_port: 8000,
            enable_router: gpu_count > 1,
            router_provider: RouterProvider::SglangRouter,
            router_policy: RouterPolicy::CacheAware,
            router_port: 9000,
            launch_result: None,
            gpu_count,
            heterogeneous,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, instance_manager: &mut crate::core::InstanceManager) {
        if !self.open {
            return;
        }

        let mut open = true;
        egui::Window::new("🚀 Deployment Wizard")
            .open(&mut open)
            .default_width(600.0)
            .default_height(500.0)
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                self.render_content(ui, instance_manager);
            });

        if !open {
            self.open = false;
        }
    }

    fn render_content(&mut self, ui: &mut Ui, instance_manager: &mut crate::core::InstanceManager) {
        // Step indicator
        self.render_step_indicator(ui);
        ui.separator();

        match self.step {
            WizardStep::SelectMode => self.render_select_mode(ui),
            WizardStep::ConfigureInstances => self.render_configure_instances(ui),
            WizardStep::ConfigureRouter => self.render_configure_router(ui),
            WizardStep::PreviewAndLaunch => self.render_preview_and_launch(ui, instance_manager),
        }

        ui.separator();
        self.render_navigation(ui);
    }

    fn render_step_indicator(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let steps = [
                (WizardStep::SelectMode, "1. Mode"),
                (WizardStep::ConfigureInstances, "2. Instances"),
                (WizardStep::ConfigureRouter, "3. Router"),
                (WizardStep::PreviewAndLaunch, "4. Launch"),
            ];

            for (i, (step, label)) in steps.iter().enumerate() {
                let is_current = *step == self.step;
                let is_past = matches!(
                    (self.step, step),
                    (WizardStep::ConfigureInstances, WizardStep::SelectMode)
                        | (WizardStep::ConfigureRouter, WizardStep::SelectMode)
                        | (WizardStep::ConfigureRouter, WizardStep::ConfigureInstances)
                        | (WizardStep::PreviewAndLaunch, _)
                );

                let color = if is_current {
                    Color32::from_rgb(80, 140, 255)
                } else if is_past {
                    Color32::from_rgb(80, 200, 120)
                } else {
                    Color32::from_rgb(100, 100, 110)
                };

                ui.label(RichText::new(*label).color(color).strong().size(10.0));

                if i < steps.len() - 1 {
                    ui.label(RichText::new("→").color(Color32::from_rgb(80, 80, 90)));
                }
            }
        });
    }

    fn render_select_mode(&mut self, ui: &mut Ui) {
        ui.heading("Select Deployment Mode");
        ui.add_space(8.0);

        if self.heterogeneous {
            let warn_frame = Frame::none()
                .fill(Color32::from_rgb(50, 40, 20))
                .inner_margin(Margin::same(8.0))
                .rounding(4.0);
            warn_frame.show(ui, |ui: &mut Ui| {
                ui.label(RichText::new("⚠ Heterogeneous GPUs detected").color(Color32::YELLOW).size(11.0));
                ui.label(RichText::new("Multi-Instance mode is recommended for different GPU architectures").color(Color32::from_rgb(200, 200, 150)).size(10.0));
            });
            ui.add_space(8.0);
        }

        let modes = [
            DeploymentMode::SingleGpu,
            DeploymentMode::DataParallel,
            DeploymentMode::MultiInstance,
            DeploymentMode::PipelineParallel,
        ];

        for mode in modes {
            let is_selected = self.selected_mode == mode;
            let bg_color = if is_selected {
                Color32::from_rgb(30, 40, 80)
            } else {
                Color32::from_rgb(28, 28, 32)
            };

            let card_frame = Frame::none()
                .fill(bg_color)
                .stroke(Stroke::new(1.0, if is_selected { Color32::from_rgb(80, 120, 220) } else { Color32::from_rgb(50, 50, 55) }))
                .inner_margin(Margin::same(10.0))
                .rounding(6.0);

            let response = ui.allocate_response(
                Vec2::new(ui.available_width(), 60.0),
                Sense::click(),
            );

            let painter = ui.painter_at(response.rect);
            let _ = card_frame.paint(response.rect);

            let min = response.rect.min;
            painter.text(
                min + Vec2::new(8.0, 4.0),
                Align2::LEFT_TOP,
                mode.label(),
                FontId::new(12.0, FontId::monospace(12.0).family),
                if is_selected { Color32::WHITE } else { Color32::from_rgb(200, 200, 200) },
            );
            painter.text(
                min + Vec2::new(8.0, 24.0),
                Align2::LEFT_TOP,
                mode.description(),
                FontId::new(9.0, FontId::monospace(9.0).family),
                Color32::from_rgb(160, 160, 170),
            );

            if is_selected {
                painter.text(
                    response.rect.min + Vec2::new(response.rect.width() - 30.0, 4.0),
                    Align2::RIGHT_TOP,
                    "✓",
                    FontId::new(14.0, FontId::monospace(14.0).family),
                    Color32::from_rgb(80, 200, 120),
                );
            }

            if response.clicked() {
                self.selected_mode = mode;
            }

            ui.add_space(4.0);
        }
    }

    fn render_configure_instances(&mut self, ui: &mut Ui) {
        ui.heading("Configure Instances");
        ui.add_space(8.0);

        ui.label("Model path:");
        ui.text_edit_singleline(&mut self.instance_model_path);
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Provider:");
            egui::ComboBox::from_id_source("wizard_provider")
                .selected_text(&self.instance_provider)
                .show_ui(ui, |ui| {
                    for provider in &["vLLM", "SGLang", "llama.cpp"] {
                        ui.selectable_value(&mut self.instance_provider, provider.to_string(), *provider);
                    }
                });
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Context size:");
            ui.add(egui::DragValue::new(&mut self.instance_context_size).clamp_range(256..=131072));
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Memory utilization:");
            ui.add(egui::DragValue::new(&mut self.instance_memory_util).clamp_range(0.1..=1.0).speed(0.01));
            ui.label(format!("{:.0}%", self.instance_memory_util * 100.0));
        });
        ui.add_space(8.0);

        if ui.button("➕ Add Instance").clicked() && !self.instance_model_path.is_empty() {
            self.instances_to_add.push(InstanceConfig {
                gpu_indices: vec![self.instances_to_add.len() as u32 % self.gpu_count.max(1)],
                port: self.next_port,
                memory_utilization: self.instance_memory_util,
                max_num_seqs: None,
                model_path: self.instance_model_path.clone(),
                context_size: self.instance_context_size,
                provider: self.instance_provider.clone(),
            });
            self.next_port += 1;
        }

        ui.add_space(8.0);

        if !self.instances_to_add.is_empty() {
            ui.label(RichText::new("Instances to deploy:").strong().size(11.0));
            ui.add_space(4.0);

            let mut to_remove = None;
            ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                for (i, inst) in self.instances_to_add.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Instance #{}", i + 1)).monospace().size(10.0));
                        ui.label(RichText::new(format!("GPU:{:?} Port:{}", inst.gpu_indices, inst.port)).size(9.0).color(Color32::from_rgb(160, 160, 170)));
                        ui.label(RichText::new(&inst.model_path).size(9.0).color(Color32::from_rgb(140, 140, 150)));

                        if ui.small_button("✕").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }
            });
            if let Some(idx) = to_remove {
                self.instances_to_add.remove(idx);
            }
        }
    }

    fn render_configure_router(&mut self, ui: &mut Ui) {
        ui.heading("Configure Router (Optional)");
        ui.add_space(8.0);

        ui.checkbox(&mut self.enable_router, "Enable load balancing router");

        if self.enable_router {
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label("Router provider:");
                egui::ComboBox::from_id_source("wizard_router_provider")
                    .selected_text(self.router_provider.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.router_provider, RouterProvider::SglangRouter, "SGLang Router");
                        ui.selectable_value(&mut self.router_provider, RouterProvider::Nginx, "Nginx");
                    });
            });
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Load balancing policy:");
                egui::ComboBox::from_id_source("wizard_router_policy")
                    .selected_text(match self.router_policy {
                        RouterPolicy::CacheAware => "Cache-Aware",
                        RouterPolicy::RoundRobin => "Round Robin",
                        RouterPolicy::PowerOfTwo => "Power of Two",
                        RouterPolicy::Random => "Random",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.router_policy, RouterPolicy::CacheAware, "Cache-Aware");
                        ui.selectable_value(&mut self.router_policy, RouterPolicy::RoundRobin, "Round Robin");
                        ui.selectable_value(&mut self.router_policy, RouterPolicy::PowerOfTwo, "Power of Two");
                        ui.selectable_value(&mut self.router_policy, RouterPolicy::Random, "Random");
                    });
            });
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Router port:");
                ui.add(egui::DragValue::new(&mut self.router_port).clamp_range(1024..=65535));
            });
        }
    }

    fn render_preview_and_launch(&mut self, ui: &mut Ui, instance_manager: &mut crate::core::InstanceManager) {
        ui.heading("Preview & Launch");
        ui.add_space(8.0);

        ui.label(RichText::new("Deployment Summary").strong().size(11.0));
        ui.add_space(4.0);

        ui.label(format!("Mode: {}", self.selected_mode.label()));
        ui.label(format!("Instances: {}", self.instances_to_add.len()));
        if self.enable_router {
            ui.label(format!("Router: {} (Port {})", self.router_provider.label(), self.router_port));
        }
        ui.add_space(8.0);

        ui.label(RichText::new("Instance Details:").strong().size(10.0));
        for (i, inst) in self.instances_to_add.iter().enumerate() {
            ui.label(format!("  #{}: {} on GPU:{:?} port:{} mem:{:.0}%",
                i + 1, inst.model_path, inst.gpu_indices, inst.port, inst.memory_utilization * 100.0));
        }

        ui.add_space(12.0);

        if let Some(ref result) = self.launch_result {
            let frame = Frame::none()
                .fill(if result.starts_with("✓") { Color32::from_rgb(20, 40, 20) } else { Color32::from_rgb(50, 20, 20) })
                .inner_margin(Margin::same(8.0))
                .rounding(4.0);
            frame.show(ui, |ui: &mut Ui| {
                ui.label(RichText::new(result).size(10.0).monospace().color(if result.starts_with("✓") { Color32::from_rgb(150, 220, 150) } else { Color32::RED }));
            });
            ui.add_space(8.0);
        }

        ui.horizontal(|ui| {
            if ui.button("🚀 Launch Deployment").clicked() {
                self.launch_result = Some(self.launch_deployment(instance_manager));
            }
            if ui.button("Generate Script Only").clicked() {
                self.launch_result = Some("Script generated (check logs)".to_string());
            }
        });
    }

    fn launch_deployment(&mut self, instance_manager: &mut crate::core::InstanceManager) -> String {
        let mut launched = 0;
        let mut errors = Vec::new();

        for config in &self.instances_to_add {
            let provider = crate::core::ProviderRegistry::get(&config.provider);
            let provider_ref = match provider {
                Some(p) => p,
                None => {
                    errors.push(format!("Unknown provider: {}", config.provider));
                    continue;
                }
            };

            match instance_manager.launch_instance(config, provider_ref.as_ref(), &crate::core::ProviderSettings::default()) {
                Ok(id) => {
                    launched += 1;
                    log::info!("Launched instance #{} with ID {}", config.model_path, id);
                }
                Err(e) => errors.push(format!("Instance {}: {}", config.model_path, e)),
            }
        }

        if self.enable_router && !self.instances_to_add.is_empty() {
            let worker_urls: Vec<String> = self.instances_to_add.iter()
                .map(|i| format!("http://127.0.0.1:{}", i.port))
                .collect();

            let router_config = RouterConfig {
                provider: self.router_provider.clone(),
                policy: self.router_policy.clone(),
                worker_urls,
                router_port: self.router_port,
                pd_disaggregation: false,
            };

            if let Err(e) = instance_manager.launch_router(&router_config) {
                errors.push(format!("Router: {}", e));
            } else {
                launched += 1;
            }
        }

        if errors.is_empty() {
            format!("✓ Successfully launched {} component(s)", launched)
        } else {
            format!("⚠ Launched {}, errors: {}", launched, errors.join("; "))
        }
    }

    fn render_navigation(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if self.step != WizardStep::SelectMode {
                if ui.button("← Back").clicked() {
                    self.step = match self.step {
                        WizardStep::ConfigureInstances => WizardStep::SelectMode,
                        WizardStep::ConfigureRouter => WizardStep::ConfigureInstances,
                        WizardStep::PreviewAndLaunch => WizardStep::ConfigureRouter,
                        _ => self.step,
                    };
                    self.launch_result = None;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can_advance = match self.step {
                    WizardStep::SelectMode => true,
                    WizardStep::ConfigureInstances => !self.instances_to_add.is_empty(),
                    WizardStep::ConfigureRouter => true,
                    WizardStep::PreviewAndLaunch => false,
                };

                if can_advance {
                    if ui.button("Next →").clicked() {
                        self.step = match self.step {
                            WizardStep::SelectMode => WizardStep::ConfigureInstances,
                            WizardStep::ConfigureInstances => {
                                if self.instances_to_add.len() > 1 {
                                    WizardStep::ConfigureRouter
                                } else {
                                    WizardStep::PreviewAndLaunch
                                }
                            }
                            WizardStep::ConfigureRouter => WizardStep::PreviewAndLaunch,
                            _ => self.step,
                        };
                    }
                }
            });
        });
    }

    pub fn reset(&mut self) {
        self.step = WizardStep::SelectMode;
        self.instances_to_add.clear();
        self.next_port = 8000;
        self.launch_result = None;
        self.instance_model_path.clear();
    }
}
