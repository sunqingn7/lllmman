use eframe::egui::{Color32, FontId, Frame, Margin, Sense, Stroke, Ui, Vec2, Align2, RichText, ScrollArea};

use crate::core::instance_manager::{InstanceMetrics, InstanceStatus};

#[derive(Clone)]
pub struct InstanceUiState {
    pub id: u32,
    pub model_name: String,
    pub port: u16,
    pub gpu_id: Option<u32>,
    pub status: InstanceStatus,
    pub metrics: InstanceMetrics,
    pub provider: String,
}

#[derive(Clone)]
pub struct RouterUiState {
    pub status: InstanceStatus,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub connected_workers: u32,
    pub total_qps: f32,
}

pub struct MultiInstanceManagerPanel {
    pub instances: Vec<InstanceUiState>,
    pub router: Option<RouterUiState>,
    pub selected_instance: Option<u32>,
    pub show_logs: bool,
    pub log_filter: String,
    pub show_wizard: bool,
}

impl MultiInstanceManagerPanel {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            router: None,
            selected_instance: None,
            show_logs: false,
            log_filter: String::new(),
            show_wizard: false,
        }
    }

    pub fn sync_from_manager(&mut self, manager: &crate::core::InstanceManager) {
        self.instances.clear();

        for handle in manager.instances() {
            let status = handle.status.lock().unwrap().clone();
            let metrics = handle.metrics.lock().unwrap().clone();

            self.instances.push(InstanceUiState {
                id: handle.id,
                model_name: handle.config.model_path.clone(),
                port: handle.port,
                gpu_id: handle.config.gpu_indices.first().copied(),
                status,
                metrics,
                provider: handle.config.provider.clone(),
            });
        }

        if let Some(router) = manager.router() {
            let status = router.status.lock().unwrap().clone();
            self.router = Some(RouterUiState {
                status,
                port: router.config.router_port,
                worker_urls: router.config.worker_urls.clone(),
                connected_workers: router.connected_workers,
                total_qps: router.total_qps,
            });
        } else {
            self.router = None;
        }
    }

    pub fn show(&mut self, ui: &mut Ui, manager: &mut crate::core::InstanceManager) {
        ui.heading("Multi-Instance Manager");
        ui.separator();

        // Toolbar
        ui.horizontal(|ui| {
            if ui.button("➕ Add Instance").clicked() {
                self.show_wizard = true;
            }
            if ui.button("🔄 Refresh").clicked() {
                self.sync_from_manager(manager);
            }
            if ui.button("⏹ Stop All").clicked() {
                manager.stop_all();
                self.sync_from_manager(manager);
            }
        });

        ui.add_space(8.0);

        // Router status
        if let Some(router) = &self.router {
            self.render_router_card(ui, router);
            ui.add_space(8.0);
        }

        // Instance list
        ui.label(RichText::new("Instances").strong().size(11.0));
        ui.add_space(4.0);

        if self.instances.is_empty() {
            ui.label("No instances running. Use the Deployment Wizard to start instances.");
            return;
        }

        for instance in &self.instances {
            let is_selected = self.selected_instance == Some(instance.id);
            let clicked = self.render_instance_card(ui, instance, is_selected);
            if clicked {
                if self.selected_instance == Some(instance.id) {
                    self.selected_instance = None;
                    self.show_logs = false;
                } else {
                    self.selected_instance = Some(instance.id);
                    self.show_logs = true;
                }
            }
            ui.add_space(4.0);
        }

        // Selected instance details/logs
        if self.show_logs {
            ui.separator();
            self.render_instance_logs(ui, manager);
        }
    }

    fn render_router_card(&self, ui: &mut Ui, router: &RouterUiState) {
        let status_color = match router.status {
            InstanceStatus::Running => Color32::from_rgb(80, 200, 120),
            InstanceStatus::Error(_) => Color32::RED,
            _ => Color32::from_rgb(160, 160, 170),
        };

        let card_frame = Frame::none()
            .fill(Color32::from_rgb(25, 30, 50))
            .stroke(Stroke::new(1.0, Color32::from_rgb(60, 80, 140)))
            .inner_margin(Margin::same(8.0))
            .rounding(6.0);

        let response = ui.allocate_response(
            Vec2::new(ui.available_width(), 50.0),
            Sense::click(),
        );

        let painter = ui.painter_at(response.rect);
        let _ = card_frame.paint(response.rect);

        let min = response.rect.min;

        painter.text(
            min + Vec2::new(8.0, 4.0),
            Align2::LEFT_TOP,
            "🔀 SGLang Router",
            FontId::new(11.0, FontId::monospace(11.0).family),
            Color32::WHITE,
        );

        painter.text(
            min + Vec2::new(8.0, 22.0),
            Align2::LEFT_TOP,
            format!("Port: {} | Workers: {}/{} | QPS: {:.1}",
                router.port, router.connected_workers, router.worker_urls.len(), router.total_qps),
            FontId::new(9.0, FontId::monospace(9.0).family),
            Color32::from_rgb(160, 160, 170),
        );

        let status_text = match &router.status {
            InstanceStatus::Running => "Running",
            InstanceStatus::Stopped => "Stopped",
            InstanceStatus::Starting => "Starting",
            InstanceStatus::Error(e) => e.as_str(),
            InstanceStatus::Restarting => "Restarting",
        };
        painter.text(
            response.rect.min + Vec2::new(response.rect.width() - 80.0, 4.0),
            Align2::RIGHT_TOP,
            status_text,
            FontId::new(10.0, FontId::monospace(10.0).family),
            status_color,
        );
    }

    fn render_instance_card(&self, ui: &mut Ui, instance: &InstanceUiState, is_selected: bool) -> bool {
        let bg_color = if is_selected {
            Color32::from_rgb(30, 40, 80)
        } else {
            Color32::from_rgb(28, 28, 32)
        };
        let border_color = if is_selected {
            Color32::from_rgb(80, 120, 220)
        } else {
            Color32::from_rgb(50, 50, 55)
        };

        let status_color = match instance.status {
            InstanceStatus::Running => Color32::from_rgb(80, 200, 120),
            InstanceStatus::Error(_) => Color32::RED,
            InstanceStatus::Starting => Color32::YELLOW,
            InstanceStatus::Restarting => Color32::from_rgb(220, 180, 60),
            InstanceStatus::Stopped => Color32::from_rgb(160, 160, 170),
        };

        let card_frame = Frame::none()
            .fill(bg_color)
            .stroke(Stroke::new(1.0, border_color))
            .inner_margin(Margin::same(8.0))
            .rounding(6.0);

        let response = ui.allocate_response(
            Vec2::new(ui.available_width(), 70.0),
            Sense::click(),
        );

        let painter = ui.painter_at(response.rect);
        let _ = card_frame.paint(response.rect);

        let min = response.rect.min;

        // Model name (truncated)
        let model_display = if instance.model_name.len() > 40 {
            format!("...{}", &instance.model_name[instance.model_name.len()-37..])
        } else {
            instance.model_name.clone()
        };

        painter.text(
            min + Vec2::new(8.0, 4.0),
            Align2::LEFT_TOP,
            &model_display,
            FontId::new(10.0, FontId::monospace(10.0).family),
            Color32::WHITE,
        );

        // GPU and port info
        let gpu_text = match instance.gpu_id {
            Some(gpu) => format!("GPU:{} | Port:{}", gpu, instance.port),
            None => format!("Port:{}", instance.port),
        };
        painter.text(
            min + Vec2::new(8.0, 20.0),
            Align2::LEFT_TOP,
            &gpu_text,
            FontId::new(9.0, FontId::monospace(9.0).family),
            Color32::from_rgb(160, 160, 170),
        );

        // Provider badge
        painter.text(
            min + Vec2::new(8.0, 34.0),
            Align2::LEFT_TOP,
            &instance.provider,
            FontId::new(8.0, FontId::monospace(8.0).family),
            Color32::from_rgb(100, 140, 200),
        );

        // Status indicator
        let status_text = match &instance.status {
            InstanceStatus::Running => "Running",
            InstanceStatus::Stopped => "Stopped",
            InstanceStatus::Starting => "Starting",
            InstanceStatus::Error(e) => e.as_str(),
            InstanceStatus::Restarting => "Restarting",
        };
        painter.text(
            response.rect.min + Vec2::new(response.rect.width() - 80.0, 4.0),
            Align2::RIGHT_TOP,
            status_text,
            FontId::new(10.0, FontId::monospace(10.0).family),
            status_color,
        );

        // Metrics line
        if matches!(instance.status, InstanceStatus::Running) {
            let metrics_text = format!(
                "TPS: {:.1} | Queue: {} | VRAM: {:.0}%",
                instance.metrics.tokens_per_second,
                instance.metrics.queue_size,
                if instance.metrics.vram_total_mb > 0 {
                    (instance.metrics.vram_used_mb as f32 / instance.metrics.vram_total_mb as f32) * 100.0
                } else {
                    0.0
                }
            );
            painter.text(
                min + Vec2::new(8.0, 50.0),
                Align2::LEFT_TOP,
                &metrics_text,
                FontId::new(8.0, FontId::monospace(8.0).family),
                Color32::from_rgb(140, 140, 150),
            );
        }

        response.clicked()
    }

    fn render_instance_logs(&mut self, ui: &mut Ui, manager: &crate::core::InstanceManager) {
        if let Some(instance_id) = self.selected_instance {
            if let Some(handle) = manager.get_instance(instance_id) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Instance Logs").strong().size(11.0));
                    ui.text_edit_singleline(&mut self.log_filter);
                });
                ui.add_space(4.0);

                ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        let logs = handle.log_buffer.get_entries();
                        for entry in logs {
                            let color = match entry.level {
                                crate::core::LogLevel::Error => Color32::RED,
                                crate::core::LogLevel::Warn => Color32::YELLOW,
                                crate::core::LogLevel::Info => Color32::from_rgb(180, 180, 180),
                            };
                            ui.label(RichText::new(&entry.message).color(color).size(9.0).monospace());
                        }
                    });
            }
        }
    }
}
