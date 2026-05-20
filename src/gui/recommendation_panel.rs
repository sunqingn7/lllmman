use eframe::egui::{self, Color32, FontId, Frame, Margin, Sense, Stroke, Ui, Vec2, Align2, RichText, ScrollArea};

use crate::services::{RecommendationEngine, Recommendation, RecommendationCategory, RecommendationPriority};

pub struct RecommendationPanel {
    pub recommendations: Vec<Recommendation>,
    pub show: bool,
    pub filter_category: Option<RecommendationCategory>,
    pub filter_priority: Option<RecommendationPriority>,
}

impl RecommendationPanel {
    pub fn new() -> Self {
        Self {
            recommendations: Vec::new(),
            show: false,
            filter_category: None,
            filter_priority: None,
        }
    }

    pub fn analyze(&mut self, gpus: &[crate::models::gpu::GpuInfo], model_size_gb: f32, model_name: &str) {
        self.recommendations = RecommendationEngine::analyze(gpus, model_size_gb, model_name);
    }

    pub fn show(&mut self, ui: &mut Ui) {
        if !self.show {
            return;
        }

        ui.heading("💡 Smart Recommendations");
        ui.separator();

        // Filters
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.selectable_value(&mut self.filter_category, None, "All");
            ui.selectable_value(&mut self.filter_category, Some(RecommendationCategory::Deployment), "🚀 Deploy");
            ui.selectable_value(&mut self.filter_category, Some(RecommendationCategory::Provider), "⚙️ Provider");
            ui.selectable_value(&mut self.filter_category, Some(RecommendationCategory::Router), "🔀 Router");
            ui.selectable_value(&mut self.filter_category, Some(RecommendationCategory::Performance), "⚡ Perf");
            ui.selectable_value(&mut self.filter_category, Some(RecommendationCategory::Safety), "🛡️ Safety");
        });

        ui.add_space(4.0);

        let filtered: Vec<_> = self.recommendations.iter()
            .filter(|r| {
                if let Some(cat) = self.filter_category {
                    if r.category != cat { return false; }
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            ui.label("No recommendations match the current filter.");
            return;
        }

        ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
            for rec in &filtered {
                self.render_recommendation_card(ui, rec);
                ui.add_space(6.0);
            }
        });
    }

    fn render_recommendation_card(&self, ui: &mut Ui, rec: &Recommendation) {
        let (border_color, bg_color) = match rec.priority {
            RecommendationPriority::Critical => (Color32::RED, Color32::from_rgb(50, 20, 20)),
            RecommendationPriority::High => (Color32::from_rgb(255, 140, 50), Color32::from_rgb(50, 35, 20)),
            RecommendationPriority::Medium => (Color32::YELLOW, Color32::from_rgb(50, 45, 20)),
            RecommendationPriority::Low => (Color32::from_rgb(100, 160, 220), Color32::from_rgb(25, 30, 45)),
            RecommendationPriority::Info => (Color32::from_rgb(100, 100, 120), Color32::from_rgb(28, 28, 32)),
        };

        let card_frame = Frame::none()
            .fill(bg_color)
            .stroke(Stroke::new(1.0, border_color))
            .inner_margin(Margin::same(10.0))
            .rounding(6.0);

        card_frame.show(ui, |ui| {
            // Header line: icon + title + priority badge
            ui.horizontal(|ui| {
                ui.label(RichText::new(rec.category.icon()).size(14.0));
                ui.label(RichText::new(&rec.title).strong().size(11.0));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let badge_color = match rec.priority {
                        RecommendationPriority::Critical => Color32::RED,
                        RecommendationPriority::High => Color32::from_rgb(255, 140, 50),
                        RecommendationPriority::Medium => Color32::YELLOW,
                        RecommendationPriority::Low => Color32::from_rgb(100, 160, 220),
                        RecommendationPriority::Info => Color32::from_rgb(100, 100, 120),
                    };
                    let badge_frame = Frame::none()
                        .fill(badge_color.linear_multiply(0.2))
                        .inner_margin(Margin::symmetric(6.0, 2.0))
                        .rounding(3.0);
                    badge_frame.show(ui, |ui| {
                        ui.label(RichText::new(rec.priority.color_label()).size(8.0).strong().color(badge_color));
                    });
                });
            });

            ui.add_space(4.0);

            // Description
            ui.label(RichText::new(&rec.description).size(10.0).color(Color32::from_rgb(200, 200, 200)));

            // Action button if available
            if let Some(ref action) = rec.action {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Action:").size(9.0).strong().color(Color32::from_rgb(160, 160, 170)));
                    ui.label(RichText::new(action).size(9.0).monospace().color(Color32::from_rgb(120, 180, 255)));
                    if ui.small_button("📋 Copy").clicked() {
                        // Could copy to clipboard if clipboard feature is available
                    }
                });
            }
        });
    }
}
