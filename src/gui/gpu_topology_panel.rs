use eframe::egui::{Color32, FontId, Frame, Margin, Rect, Sense, Stroke, Ui, Vec2, Align2, RichText};
use crate::models::gpu::GpuTier;

fn tier_color(tier: &GpuTier) -> Color32 {
    match tier {
        GpuTier::Low => Color32::from_rgb(180, 180, 180),
        GpuTier::Mid => Color32::from_rgb(80, 160, 220),
        GpuTier::High => Color32::from_rgb(80, 200, 120),
        GpuTier::Ultra => Color32::from_rgb(220, 180, 60),
    }
}

#[derive(Clone, Debug)]
pub struct GpuTopologyEntry {
    pub gpu_id: u32,
    pub name: String,
    pub arch: String,
    pub sm_version: String,
    pub vram_total_gb: f32,
    pub vram_used_gb: f32,
    pub vram_percent: f32,
    pub gpu_util_percent: f32,
    pub temp_celsius: u32,
    pub power_watts: f32,
    pub power_limit_watts: f32,
    pub tier: GpuTier,
}

pub struct GpuTopologyPanel {
    pub gpus: Vec<GpuTopologyEntry>,
    pub heterogeneous: bool,
    pub selected_gpu: Option<u32>,
    pub show_detail: bool,
}

impl GpuTopologyPanel {
    pub fn new() -> Self {
        Self {
            gpus: Vec::new(),
            heterogeneous: false,
            selected_gpu: None,
            show_detail: false,
        }
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("GPU Topology");
        ui.separator();

        if self.gpus.is_empty() {
            ui.label("No GPUs detected");
            return;
        }

        if self.heterogeneous {
            let warn_frame = Frame::none()
                .fill(Color32::from_rgb(50, 40, 20))
                .inner_margin(Margin::same(6.0))
                .rounding(4.0);
            warn_frame.show(ui, |ui: &mut Ui| {
                ui.label(RichText::new("⚠ Heterogeneous GPUs").color(Color32::YELLOW));
                ui.label(RichText::new("Multi-instance + router recommended").color(Color32::from_rgb(200, 200, 150)).size(10.0));
            });
            ui.add_space(4.0);
        }

        let mut clicked_gpu_id = None;
        for gpu in &self.gpus {
            let is_selected = self.selected_gpu == Some(gpu.gpu_id);
            let clicked = Self::render_gpu_card(ui, gpu, is_selected);
            if clicked {
                clicked_gpu_id = Some(gpu.gpu_id);
            }
            ui.add_space(6.0);
        }

        if let Some(gpu_id) = clicked_gpu_id {
            if self.selected_gpu == Some(gpu_id) {
                self.selected_gpu = None;
                self.show_detail = false;
            } else {
                self.selected_gpu = Some(gpu_id);
                self.show_detail = true;
            }
        }

        if self.show_detail {
            ui.separator();
            self.render_detail_panel(ui);
        }
    }

    fn render_gpu_card(ui: &mut Ui, gpu: &GpuTopologyEntry, is_selected: bool) -> bool {
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

        let card_frame = Frame::none()
            .fill(bg_color)
            .stroke(Stroke::new(1.0, border_color))
            .inner_margin(Margin::same(8.0))
            .rounding(6.0);

        let response = ui.allocate_response(
            Vec2::new(ui.available_width(), 90.0),
            Sense::click(),
        );

        let painter = ui.painter_at(response.rect);
        let _ = card_frame.paint(response.rect);

        let min = response.rect.min;

        // GPU name and tier badge
        let name_rect = Rect::from_min_size(min + Vec2::new(8.0, 4.0), Vec2::new(response.rect.width() - 80.0, 16.0));
        painter.text(
            name_rect.min,
            Align2::LEFT_TOP,
            &gpu.name,
            FontId::new(11.0, FontId::monospace(11.0).family),
            Color32::WHITE,
        );

        // Tier badge
        let tc = tier_color(&gpu.tier);
        let badge_rect = Rect::from_min_size(
            min + Vec2::new(response.rect.width() - 60.0, 4.0),
            Vec2::new(50.0, 16.0),
        );
        painter.rect_filled(badge_rect, 4.0, tc.linear_multiply(0.2));
        painter.rect_stroke(badge_rect, 4.0, Stroke::new(1.0, tc));
        painter.text(
            badge_rect.center(),
            Align2::CENTER_CENTER,
            gpu.tier.label(),
            FontId::new(9.0, FontId::monospace(9.0).family),
            tc,
        );

        // Arch info line
        let arch_rect = Rect::from_min_size(min + Vec2::new(8.0, 22.0), Vec2::new(response.rect.width() - 16.0, 12.0));
        painter.text(
            arch_rect.min,
            Align2::LEFT_TOP,
            format!("{} • SM {} • Temp: {}°C", gpu.arch, gpu.sm_version, gpu.temp_celsius),
            FontId::new(9.0, FontId::monospace(9.0).family),
            Color32::from_rgb(160, 160, 170),
        );

        // VRAM bar
        let vram_bar_y = 38.0;
        let vram_bar_height = 8.0;
        let vram_bar_rect = Rect::from_min_size(
            min + Vec2::new(8.0, vram_bar_y),
            Vec2::new(response.rect.width() - 16.0, vram_bar_height),
        );
        painter.rect_filled(vram_bar_rect, 3.0, Color32::from_rgb(40, 40, 45));

        let vram_fill_width = (vram_bar_rect.width() * gpu.vram_percent / 100.0).min(vram_bar_rect.width());
        let vram_fill_rect = Rect::from_min_size(vram_bar_rect.min, Vec2::new(vram_fill_width, vram_bar_height));
        let vram_color = if gpu.vram_percent > 90.0 {
            Color32::RED
        } else if gpu.vram_percent > 70.0 {
            Color32::YELLOW
        } else {
            Color32::from_rgb(80, 180, 120)
        };
        painter.rect_filled(vram_fill_rect, 3.0, vram_color);

        // VRAM text
        let vram_text_rect = Rect::from_min_size(
            min + Vec2::new(8.0, vram_bar_y + vram_bar_height + 2.0),
            Vec2::new(response.rect.width() - 16.0, 12.0),
        );
        painter.text(
            vram_text_rect.min,
            Align2::LEFT_TOP,
            format!("VRAM: {:.1}/{:.1} GB ({:.0}%)", gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent),
            FontId::new(9.0, FontId::monospace(9.0).family),
            Color32::from_rgb(160, 160, 170),
        );

        // GPU utilization bar
        let util_bar_y = 62.0;
        let util_bar_height = 6.0;
        let util_bar_rect = Rect::from_min_size(
            min + Vec2::new(8.0, util_bar_y),
            Vec2::new(response.rect.width() - 16.0, util_bar_height),
        );
        painter.rect_filled(util_bar_rect, 2.0, Color32::from_rgb(40, 40, 45));

        let util_fill_width = (util_bar_rect.width() * gpu.gpu_util_percent / 100.0).min(util_bar_rect.width());
        let util_fill_rect = Rect::from_min_size(util_bar_rect.min, Vec2::new(util_fill_width, util_bar_height));
        let util_color = if gpu.gpu_util_percent > 80.0 {
            Color32::from_rgb(80, 200, 120)
        } else if gpu.gpu_util_percent > 40.0 {
            Color32::from_rgb(220, 180, 60)
        } else {
            Color32::from_rgb(160, 160, 170)
        };
        painter.rect_filled(util_fill_rect, 2.0, util_color);

        // Util text
        let util_text_rect = Rect::from_min_size(
            min + Vec2::new(8.0, util_bar_y + util_bar_height + 2.0),
            Vec2::new(response.rect.width() - 16.0, 12.0),
        );
        painter.text(
            util_text_rect.min,
            Align2::LEFT_TOP,
            format!("GPU: {:.0}% • Power: {:.0}/{:.0}W", gpu.gpu_util_percent, gpu.power_watts, gpu.power_limit_watts),
            FontId::new(9.0, FontId::monospace(9.0).family),
            Color32::from_rgb(160, 160, 170),
        );

        response.clicked()
    }

    fn render_detail_panel(&self, ui: &mut Ui) {
        if let Some(gpu_id) = self.selected_gpu {
            if let Some(gpu) = self.gpus.iter().find(|g| g.gpu_id == gpu_id) {
                ui.label(RichText::new("GPU Details").size(12.0).strong());
                ui.add_space(4.0);

                let details = vec![
                    ("Name", gpu.name.clone()),
                    ("Architecture", gpu.arch.clone()),
                    ("SM Version", gpu.sm_version.clone()),
                    ("Tier", gpu.tier.label().to_string()),
                    ("VRAM Used", format!("{:.1} / {:.1} GB", gpu.vram_used_gb, gpu.vram_total_gb)),
                    ("VRAM %", format!("{:.0}%", gpu.vram_percent)),
                    ("GPU Utilization", format!("{:.0}%", gpu.gpu_util_percent)),
                    ("Temperature", format!("{}°C", gpu.temp_celsius)),
                    ("Power", format!("{:.0} / {:.0} W", gpu.power_watts, gpu.power_limit_watts)),
                ];

                for (label, value) in details {
                    ui.horizontal(|ui: &mut Ui| {
                        ui.label(RichText::new(label).size(10.0).color(Color32::from_rgb(140, 140, 150)));
                        ui.label(RichText::new(value).size(10.0).monospace());
                    });
                }
            }
        }
    }
}
