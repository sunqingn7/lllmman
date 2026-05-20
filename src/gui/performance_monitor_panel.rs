use eframe::egui::{self, Color32, FontId, Frame, Margin, Sense, Stroke, Ui, Vec2, Align2, RichText};

use crate::core::instance_manager::InstanceStatus;

#[derive(Clone, Debug)]
pub struct PerfMetricPoint {
    pub timestamp: f64,
    pub value: f32,
}

#[derive(Clone, Debug)]
pub struct PerfMetricHistory {
    pub points: Vec<PerfMetricPoint>,
    pub max_capacity: usize,
}

impl PerfMetricHistory {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(max_capacity),
            max_capacity,
        }
    }

    pub fn push(&mut self, timestamp: f64, value: f32) {
        if self.points.len() >= self.max_capacity {
            self.points.remove(0);
        }
        self.points.push(PerfMetricPoint { timestamp, value });
    }

    pub fn max_value(&self) -> f32 {
        self.points.iter().map(|p| p.value).fold(0.0f32, f32::max)
    }

    pub fn avg_value(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points.iter().map(|p| p.value).sum::<f32>() / self.points.len() as f32
    }

    pub fn latest(&self) -> f32 {
        self.points.last().map(|p| p.value).unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct InstancePerfData {
    pub id: u32,
    pub model_name: String,
    pub tps_history: PerfMetricHistory,
    pub latency_history: PerfMetricHistory,
    pub queue_history: PerfMetricHistory,
    pub vram_history: PerfMetricHistory,
    pub gpu_util_history: PerfMetricHistory,
}

impl InstancePerfData {
    pub fn new(id: u32, model_name: String) -> Self {
        Self {
            id,
            model_name,
            tps_history: PerfMetricHistory::new(120),
            latency_history: PerfMetricHistory::new(120),
            queue_history: PerfMetricHistory::new(120),
            vram_history: PerfMetricHistory::new(120),
            gpu_util_history: PerfMetricHistory::new(120),
        }
    }
}

pub struct PerformanceMonitorPanel {
    pub instances: Vec<InstancePerfData>,
    pub selected_instance: Option<u32>,
    pub auto_refresh: bool,
    pub refresh_interval_secs: u64,
    last_refresh: f64,
}

impl PerformanceMonitorPanel {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            selected_instance: None,
            auto_refresh: true,
            refresh_interval_secs: 2,
            last_refresh: 0.0,
        }
    }

    pub fn sync_from_manager(&mut self, manager: &crate::core::InstanceManager, now: f64) {
        if !self.auto_refresh {
            return;
        }

        if now - self.last_refresh < self.refresh_interval_secs as f64 {
            return;
        }
        self.last_refresh = now;

        for handle in manager.instances() {
            let status = handle.status.lock().unwrap().clone();
            let metrics = handle.metrics.lock().unwrap().clone();

            if !matches!(status, InstanceStatus::Running) {
                continue;
            }

            let perf = self.instances.iter_mut().find(|p| p.id == handle.id);
            if let Some(perf) = perf {
                perf.tps_history.push(now, metrics.tokens_per_second);
                perf.latency_history.push(now, metrics.avg_latency_ms);
                perf.queue_history.push(now, metrics.queue_size as f32);
                let vram_pct = if metrics.vram_total_mb > 0 {
                    (metrics.vram_used_mb as f32 / metrics.vram_total_mb as f32) * 100.0
                } else {
                    0.0
                };
                perf.vram_history.push(now, vram_pct);
                perf.gpu_util_history.push(now, metrics.gpu_utilization);
            } else {
                let mut new_perf = InstancePerfData::new(handle.id, handle.config.model_path.clone());
                new_perf.tps_history.push(now, metrics.tokens_per_second);
                new_perf.latency_history.push(now, metrics.avg_latency_ms);
                new_perf.queue_history.push(now, metrics.queue_size as f32);
                let vram_pct = if metrics.vram_total_mb > 0 {
                    (metrics.vram_used_mb as f32 / metrics.vram_total_mb as f32) * 100.0
                } else {
                    0.0
                };
                new_perf.vram_history.push(now, vram_pct);
                new_perf.gpu_util_history.push(now, metrics.gpu_utilization);
                self.instances.push(new_perf);
            }
        }

        self.instances.retain(|p| {
            manager.instances().iter().any(|h| h.id == p.id)
        });
    }

    pub fn show(&mut self, ui: &mut Ui, manager: &crate::core::InstanceManager, now: f64) {
        self.sync_from_manager(manager, now);

        ui.heading("Performance Monitor");
        ui.separator();

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_refresh, "Auto-refresh");
            ui.label("Every");
            ui.add(egui::DragValue::new(&mut self.refresh_interval_secs).clamp_range(1..=30).speed(1));
            ui.label("seconds");

            if ui.button("Refresh Now").clicked() {
                self.sync_from_manager(manager, now);
            }
        });

        ui.add_space(8.0);

        if self.instances.is_empty() {
            ui.label("No running instances to monitor. Launch instances first.");
            return;
        }

        // Instance selector
        ui.horizontal(|ui| {
            ui.label("Instance:");
            ui.selectable_value(&mut self.selected_instance, None, "All");
            for perf in &self.instances {
                let label = format!("Instance #{} ({})", perf.id, perf.model_name.chars().take(20).collect::<String>());
                ui.selectable_value(&mut self.selected_instance, Some(perf.id), &label);
            }
        });

        ui.add_space(8.0);

        if let Some(id) = self.selected_instance {
            if let Some(perf) = self.instances.iter().find(|p| p.id == id) {
                self.render_instance_charts(ui, perf);
            }
        } else {
            self.render_overview(ui);
        }
    }

    fn render_overview(&self, ui: &mut Ui) {
        ui.label(RichText::new("Overview (All Instances)").strong().size(11.0));
        ui.add_space(4.0);

        for perf in &self.instances {
            self.render_mini_dashboard(ui, perf);
            ui.add_space(8.0);
        }
    }

    fn render_mini_dashboard(&self, ui: &mut Ui, perf: &InstancePerfData) {
        let frame = Frame::none()
            .fill(Color32::from_rgb(28, 28, 32))
            .stroke(Stroke::new(1.0, Color32::from_rgb(50, 50, 55)))
            .inner_margin(Margin::same(8.0))
            .rounding(6.0);

        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Instance #{}", perf.id)).monospace().size(10.0).strong());
                ui.label(RichText::new(&perf.model_name).size(9.0).color(Color32::from_rgb(160, 160, 170)));
            });
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                self.render_metric_box(ui, "TPS", perf.tps_history.latest(), "tok/s", Color32::from_rgb(80, 200, 120));
                self.render_metric_box(ui, "Latency", perf.latency_history.latest(), "ms", Color32::from_rgb(80, 160, 255));
                self.render_metric_box(ui, "Queue", perf.queue_history.latest(), "", Color32::from_rgb(220, 180, 60));
                self.render_metric_box(ui, "VRAM", perf.vram_history.latest(), "%", Color32::from_rgb(200, 120, 220));
                self.render_metric_box(ui, "GPU", perf.gpu_util_history.latest(), "%", Color32::from_rgb(120, 200, 200));
            });
        });
    }

    fn render_metric_box(&self, ui: &mut Ui, label: &str, value: f32, unit: &str, color: Color32) {
        let frame = Frame::none()
            .fill(Color32::from_rgb(35, 35, 40))
            .inner_margin(Margin::symmetric(8.0, 4.0))
            .rounding(4.0);

        frame.show(ui, |ui| {
            ui.label(RichText::new(label).size(8.0).color(Color32::from_rgb(140, 140, 150)));
            let display = if unit.is_empty() {
                format!("{:.0}", value)
            } else {
                format!("{:.1}{}", value, unit)
            };
            ui.label(RichText::new(display).size(12.0).strong().color(color).monospace());
        });
    }

    fn render_instance_charts(&self, ui: &mut Ui, perf: &InstancePerfData) {
        ui.label(RichText::new(format!("Instance #{} - {}", perf.id, perf.model_name)).strong().size(11.0));
        ui.add_space(4.0);

        self.render_chart(ui, "Tokens per Second", &perf.tps_history, "tok/s", Color32::from_rgb(80, 200, 120));
        ui.add_space(8.0);

        self.render_chart(ui, "Avg Latency", &perf.latency_history, "ms", Color32::from_rgb(80, 160, 255));
        ui.add_space(8.0);

        self.render_chart(ui, "Queue Size", &perf.queue_history, "", Color32::from_rgb(220, 180, 60));
        ui.add_space(8.0);

        self.render_chart(ui, "VRAM Usage", &perf.vram_history, "%", Color32::from_rgb(200, 120, 220));
        ui.add_space(8.0);

        self.render_chart(ui, "GPU Utilization", &perf.gpu_util_history, "%", Color32::from_rgb(120, 200, 200));

        ui.add_space(8.0);
        ui.label(RichText::new("Statistics").strong().size(10.0));
        ui.horizontal(|ui| {
            ui.label(format!("TPS Avg: {:.1} | Max: {:.1}", perf.tps_history.avg_value(), perf.tps_history.max_value()));
            ui.label(format!("Latency Avg: {:.1}ms", perf.latency_history.avg_value()));
        });
    }

    fn render_chart(&self, ui: &mut Ui, title: &str, history: &PerfMetricHistory, unit: &str, color: Color32) {
        ui.label(RichText::new(title).size(10.0).strong());

        let chart_height = 60.0;
        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), chart_height),
            Sense::hover(),
        );

        let rect = response.rect;
        let padding = 4.0;
        let chart_rect = rect.shrink(padding);

        painter.rect_filled(rect, 2.0, Color32::from_rgb(22, 22, 26));

        if history.points.is_empty() {
            painter.text(
                chart_rect.center(),
                Align2::CENTER_CENTER,
                "No data yet",
                FontId::new(10.0, FontId::monospace(10.0).family),
                Color32::from_rgb(100, 100, 110),
            );
            return;
        }

        let max_val = history.max_value().max(0.001);

        for i in 0..history.points.len().saturating_sub(1) {
            let p1 = &history.points[i];
            let p2 = &history.points[i + 1];

            let x1 = chart_rect.min.x + (i as f32 / (history.max_capacity - 1) as f32) * chart_rect.width();
            let x2 = chart_rect.min.x + ((i + 1) as f32 / (history.max_capacity - 1) as f32) * chart_rect.width();
            let y1 = chart_rect.max.y - (p1.value / max_val) * chart_rect.height();
            let y2 = chart_rect.max.y - (p2.value / max_val) * chart_rect.height();

            painter.line_segment(
                [egui::Pos2::new(x1, y1), egui::Pos2::new(x2, y2)],
                Stroke::new(1.5, color),
            );
        }

        let latest = history.latest();
        let display = if unit.is_empty() {
            format!("{:.0}", latest)
        } else {
            format!("{:.1} {}", latest, unit)
        };
        painter.text(
            chart_rect.min + Vec2::new(4.0, 2.0),
            Align2::LEFT_TOP,
            display,
            FontId::new(9.0, FontId::monospace(9.0).family),
            color,
        );
    }
}
