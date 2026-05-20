use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs},
    Frame, Terminal,
};

use crate::core::{
    LlmProvider, LogBuffer, LogLevel, ModelInfo, ProviderConfig, ProviderRegistry, ProviderSettings, ServerController,
};
use crate::models::{AppSettings, GpuInfo, ServerStatus};
use crate::services::{
    config_persistence, detect_running_servers, get_fallback_config, get_system_stats,
    gpu_detector, load_model_config, load_provider_settings_for, parse_server_args,
    save_model_config, save_provider_settings_for,
};

#[derive(PartialEq, Eq, Clone, Copy)]
enum TuiView {
    Models,
    Instances,
    Logs,
    Metrics,
}

#[derive(PartialEq, Eq)]
enum InputMode {
    Normal,
    ModelSearch,
}

pub struct TuiApp {
    models: Vec<ModelInfo>,
    filtered_models: Vec<ModelInfo>,
    gpus: Vec<GpuInfo>,
    server_config: ProviderConfig,
    server_controller: ServerController,
    provider_settings: ProviderSettings,
    settings: AppSettings,
    selected_model_index: Option<usize>,
    selected_provider: String,
    available_providers: Vec<(String, String)>,
    input_mode: InputMode,
    search_query: String,
    scroll_offset: usize,
    status_message: String,
    // Multi-panel state
    active_view: TuiView,
    instance_scroll: usize,
    log_scroll: usize,
    log_buffer: LogBuffer,
    // Instance manager
    instance_manager: crate::core::InstanceManager,
}

impl TuiApp {
    pub fn new() -> Self {
        let settings = config_persistence::load_settings();
        let gpus = gpu_detector::detect_gpus();

        let available_providers: Vec<(String, String)> = ProviderRegistry::list()
            .into_iter()
            .map(|(id, name)| (id.to_string(), name.to_string()))
            .collect();
        let selected_provider = if available_providers.is_empty() {
            "llama.cpp".to_string()
        } else {
            available_providers[0].0.clone()
        };

        let provider = ProviderRegistry::get(&selected_provider).unwrap_or_else(|| {
            let p = crate::providers::LlamaCppProvider::new();
            std::sync::Arc::new(p) as std::sync::Arc<dyn LlmProvider>
        });

        let mut server_config = provider.get_config_template();

        if let Some(saved_config) = get_fallback_config(&selected_provider) {
            server_config.context_size = saved_config.context_size;
            server_config.batch_size = saved_config.batch_size;
            server_config.gpu_layers = saved_config.gpu_layers;
            server_config.threads = saved_config.threads;
            server_config.port = saved_config.port;
            server_config.host = saved_config.host;
            server_config.cache_type_k = saved_config.cache_type_k;
            server_config.cache_type_v = saved_config.cache_type_v;
            server_config.num_prompt_tracking = saved_config.num_prompt_tracking;
        }

        let running_servers = detect_running_servers();
        for server in &running_servers {
            if server.provider_id == selected_provider {
                let detected_config = parse_server_args(&server.provider_id, &server.command_line);
                if server_config.model_path.is_empty() {
                    server_config.model_path = detected_config.model_path;
                }
                if server_config.context_size == 4096 && detected_config.context_size != 4096 {
                    server_config.context_size = detected_config.context_size;
                }
                if server_config.port == 8080 && detected_config.port != 8080 {
                    server_config.port = detected_config.port;
                }
                if !detected_config.host.is_empty() && detected_config.host != "0.0.0.0" {
                    server_config.host = detected_config.host;
                }
                if detected_config.gpu_layers != 35 {
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
        for dir in &provider.default_model_directories() {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        let filtered_models = models.clone();
        let provider_settings = load_provider_settings_for(&selected_provider);

        let mut server_controller = ServerController::new();
        server_controller.set_provider(provider.clone());
        server_controller.set_provider_settings(provider_settings.clone());
        let log_buffer = server_controller.get_log_buffer();

        Self {
            models,
            filtered_models,
            gpus,
            server_config,
            server_controller,
            provider_settings,
            settings,
            selected_model_index: None,
            selected_provider,
            available_providers,
            input_mode: InputMode::Normal,
            search_query: String::new(),
            scroll_offset: 0,
            status_message: String::from("Ready"),
            active_view: TuiView::Models,
            instance_scroll: 0,
            log_scroll: 0,
            log_buffer,
            instance_manager: crate::core::InstanceManager::new(),
        }
    }

    fn filter_models(&mut self) {
        let query = self.search_query.to_lowercase();
        let provider_supports_gguf = ProviderRegistry::get(&self.selected_provider)
            .map(|p| p.supports_gguf())
            .unwrap_or(true);
        self.filtered_models = self
            .models
            .iter()
            .filter(|m| {
                let is_gguf = m.path.to_lowercase().ends_with(".gguf");
                if is_gguf && !provider_supports_gguf {
                    return false;
                }
                query.is_empty() || m.name.to_lowercase().contains(&query)
            })
            .cloned()
            .collect();
    }

    fn switch_provider(&mut self, provider_id: &str) {
        if let Some(provider) = ProviderRegistry::get(provider_id) {
            self.selected_provider = provider_id.to_string();
            self.server_config = provider.get_config_template();
            self.provider_settings = load_provider_settings_for(provider_id);
            self.server_controller = ServerController::new();
            self.server_controller.set_provider(provider.clone());
            self.server_controller.set_provider_settings(self.provider_settings.clone());
            self.log_buffer = self.server_controller.get_log_buffer();

            let mut models = Vec::new();
            for dir in &self.settings.scan_directories {
                let found = provider.scan_models(dir);
                models.extend(found);
            }
            for dir in &provider.default_model_directories() {
                let found = provider.scan_models(dir);
                models.extend(found);
            }
            self.models = models;
            self.filtered_models = self.models.clone();
            self.selected_model_index = None;
            self.scroll_offset = 0;
            self.status_message = format!("Switched to {}", provider_id);
        }
    }

    fn start_server(&mut self) {
        let provider = ProviderRegistry::get(&self.selected_provider).unwrap_or_else(|| {
            let p = crate::providers::LlamaCppProvider::new();
            std::sync::Arc::new(p) as std::sync::Arc<dyn LlmProvider>
        });

        match self.server_controller.start(&provider, &self.server_config) {
            Ok(_) => self.status_message = "Server started".to_string(),
            Err(e) => self.status_message = format!("Failed to start: {}", e),
        }
    }

    fn stop_server(&mut self) {
        match self.server_controller.stop() {
            Ok(_) => self.status_message = "Server stopped".to_string(),
            Err(e) => self.status_message = format!("Failed to stop: {}", e),
        }
    }

    fn handle_input(&mut self, key: crossterm::event::KeyEvent) -> bool {
        match &self.input_mode {
            InputMode::Normal => {
                match key.code {
                    crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => {
                        return true;
                    }

                    // View switching: 1=Models, 2=Instances, 3=Logs, 4=Metrics
                    crossterm::event::KeyCode::Char('1') => self.active_view = TuiView::Models,
                    crossterm::event::KeyCode::Char('2') => self.active_view = TuiView::Instances,
                    crossterm::event::KeyCode::Char('3') => self.active_view = TuiView::Logs,
                    crossterm::event::KeyCode::Char('4') => self.active_view = TuiView::Metrics,

                    // Tab navigation
                    crossterm::event::KeyCode::Tab => {
                        self.active_view = match self.active_view {
                            TuiView::Models => TuiView::Instances,
                            TuiView::Instances => TuiView::Logs,
                            TuiView::Logs => TuiView::Metrics,
                            TuiView::Metrics => TuiView::Models,
                        };
                    }
                    crossterm::event::KeyCode::BackTab => {
                        self.active_view = match self.active_view {
                            TuiView::Models => TuiView::Metrics,
                            TuiView::Instances => TuiView::Models,
                            TuiView::Logs => TuiView::Instances,
                            TuiView::Metrics => TuiView::Logs,
                        };
                    }

                    // Model list navigation
                    crossterm::event::KeyCode::Char('j') | crossterm::event::KeyCode::Down => {
                        if self.active_view == TuiView::Models {
                            if let Some(idx) = self.selected_model_index {
                                if idx < self.filtered_models.len() - 1 {
                                    self.selected_model_index = Some(idx + 1);
                                    if idx + 1 >= self.scroll_offset + 10 {
                                        self.scroll_offset += 1;
                                    }
                                }
                            } else if !self.filtered_models.is_empty() {
                                self.selected_model_index = Some(0);
                            }
                        } else if self.active_view == TuiView::Logs {
                            self.log_scroll += 1;
                        } else if self.active_view == TuiView::Instances {
                            self.instance_scroll += 1;
                        }
                    }

                    crossterm::event::KeyCode::Char('k') | crossterm::event::KeyCode::Up => {
                        if self.active_view == TuiView::Models {
                            if let Some(idx) = self.selected_model_index {
                                if idx > 0 {
                                    self.selected_model_index = Some(idx - 1);
                                    if idx - 1 < self.scroll_offset {
                                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
                                    }
                                }
                            }
                        } else if self.active_view == TuiView::Logs {
                            self.log_scroll = self.log_scroll.saturating_sub(1);
                        } else if self.active_view == TuiView::Instances {
                            self.instance_scroll = self.instance_scroll.saturating_sub(1);
                        }
                    }

                    crossterm::event::KeyCode::Char('g') => {
                        if self.active_view == TuiView::Models && !self.filtered_models.is_empty() {
                            self.selected_model_index = Some(0);
                            self.scroll_offset = 0;
                        } else if self.active_view == TuiView::Logs {
                            self.log_scroll = 0;
                        } else if self.active_view == TuiView::Instances {
                            self.instance_scroll = 0;
                        }
                    }
                    crossterm::event::KeyCode::Char('G') => {
                        if self.active_view == TuiView::Models && !self.filtered_models.is_empty() {
                            let last = self.filtered_models.len() - 1;
                            self.selected_model_index = Some(last);
                            self.scroll_offset = last.saturating_sub(10);
                        }
                    }

                    crossterm::event::KeyCode::Char('/') => {
                        if self.active_view == TuiView::Models {
                            self.input_mode = InputMode::ModelSearch;
                            self.search_query.clear();
                        }
                    }

                    crossterm::event::KeyCode::Enter => {
                        if self.active_view == TuiView::Models {
                            if let Some(idx) = self.selected_model_index {
                                if let Some(model) = self.filtered_models.get(idx) {
                                    self.server_config.model_path = model.path.clone();
                                    if let Some(saved) = load_model_config(&model.path, &self.selected_provider) {
                                        self.server_config.context_size = saved.context_size;
                                        self.server_config.batch_size = saved.batch_size;
                                        self.server_config.gpu_layers = saved.gpu_layers;
                                        self.server_config.threads = saved.threads;
                                        self.server_config.port = saved.port;
                                        self.server_config.host = saved.host.clone();
                                        self.server_config.cache_type_k = saved.cache_type_k.clone();
                                        self.server_config.cache_type_v = saved.cache_type_v.clone();
                                        self.server_config.num_prompt_tracking = saved.num_prompt_tracking;
                                        self.server_config.additional_args = saved.additional_args.clone();
                                    }
                                    self.status_message = format!("Selected: {}", model.name);
                                }
                            }
                        } else if self.active_view == TuiView::Instances {
                            let status = self.server_controller.get_status();
                            if matches!(status, ServerStatus::Running) {
                                self.stop_server();
                            } else {
                                if !self.server_config.model_path.is_empty() {
                                    save_model_config(&self.server_config.model_path, &self.server_config, &self.selected_provider).ok();
                                }
                                self.start_server();
                            }
                        }
                    }

                    crossterm::event::KeyCode::Char('s') => {
                        if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                            let status = self.server_controller.get_status();
                            if matches!(status, ServerStatus::Running) {
                                self.stop_server();
                            } else {
                                self.start_server();
                            }
                        }
                    }
                    crossterm::event::KeyCode::Char('x') => {
                        if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                            self.stop_server();
                        }
                    }

                    crossterm::event::KeyCode::Char('c') => {
                        if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                            let current_idx = self.available_providers.iter().position(|(id, _)| id == &self.selected_provider).unwrap_or(0);
                            let next_idx = (current_idx + 1) % self.available_providers.len();
                            let next_provider = self.available_providers[next_idx].0.clone();
                            self.switch_provider(&next_provider);
                        }
                    }

                    _ => {}
                }
            }
            InputMode::ModelSearch => match key.code {
                crossterm::event::KeyCode::Esc => {
                    self.input_mode = InputMode::Normal;
                    self.search_query.clear();
                    self.filter_models();
                }
                crossterm::event::KeyCode::Backspace => {
                    self.search_query.pop();
                    self.filter_models();
                }
                crossterm::event::KeyCode::Enter => {
                    self.input_mode = InputMode::Normal;
                }
                crossterm::event::KeyCode::Char(c) => {
                    self.search_query.push(c);
                    self.filter_models();
                }
                _ => {}
            },
        }
        false
    }
}

pub fn run() -> std::io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::style::ResetColor
    )?;

    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = run_inner(&mut terminal);

    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;

    result
}

fn run_inner(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> std::io::Result<()> {
    let mut app = TuiApp::new();

    loop {
        terminal.draw(|f| {
            let size = f.size();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(1),
                    Constraint::Min(0),
                    Constraint::Length(3),
                ])
                .split(size);

            render_header(f, chunks[0], &app);
            render_tabs(f, chunks[1], &app);
            render_main_content(f, chunks[2], &mut app);
            render_footer(f, chunks[3], &app);

            if let InputMode::ModelSearch = app.input_mode {
                render_search_overlay(f, size, &app.search_query);
            }
        })?;

        if crossterm::event::poll(std::time::Duration::from_millis(100))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if app.handle_input(key) {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn render_header(f: &mut Frame, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20),
            Constraint::Min(0),
            Constraint::Length(40),
        ])
        .split(area);

    let provider_text = format!("Provider: {}", app.selected_provider);
    let gpu_text = format!("GPUs: {} | {}", app.gpus.len(), if app.gpus.len() > 1 { "Multi-GPU" } else { "Single" });

    f.render_widget(
        Paragraph::new("LLLMMan").style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        chunks[0],
    );
    f.render_widget(Paragraph::new(provider_text), chunks[1]);
    f.render_widget(
        Paragraph::new(gpu_text).alignment(ratatui::layout::Alignment::Right),
        chunks[2],
    );
}

fn render_tabs(f: &mut Frame, area: Rect, app: &TuiApp) {
    let titles = vec!["[1] Models", "[2] Instances", "[3] Logs", "[4] Metrics"];
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL))
        .select(match app.active_view {
            TuiView::Models => 0,
            TuiView::Instances => 1,
            TuiView::Logs => 2,
            TuiView::Metrics => 3,
        })
        .style(Style::default())
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, area);
}

fn render_main_content(f: &mut Frame, area: Rect, app: &mut TuiApp) {
    match app.active_view {
        TuiView::Models => render_models_view(f, area, app),
        TuiView::Instances => render_instances_view(f, area, app),
        TuiView::Logs => render_logs_view(f, area, app),
        TuiView::Metrics => render_metrics_view(f, area, app),
    }
}

fn render_models_view(f: &mut Frame, area: Rect, app: &mut TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    let model_items: Vec<ListItem> = app
        .filtered_models
        .iter()
        .skip(app.scroll_offset)
        .take(area.height.saturating_sub(2) as usize)
        .enumerate()
        .map(|(i, model)| {
            let idx = app.scroll_offset + i;
            let prefix = if app.selected_model_index == Some(idx) { "► " } else { "  " };
            let size_text = if model.size_gb > 0.0 {
                format!("{:.1}GB", model.size_gb)
            } else {
                "?GB".to_string()
            };
            ListItem::new(format!("{}{} [{}]", prefix, model.name, size_text))
        })
        .collect();

    let models_block = Block::default()
        .borders(Borders::ALL)
        .title(format!("Models ({})", app.filtered_models.len()));

    let model_list = List::new(model_items).block(models_block);
    f.render_widget(model_list, chunks[0]);

    let server_status = app.server_controller.get_status();
    let is_running = matches!(server_status, ServerStatus::Running);

    let config_content = vec![
        Line::from(vec![Span::raw("Model: "), Span::raw(truncate_path(&app.server_config.model_path, chunks[1].width as usize))]),
        Line::from(vec![Span::raw("Context: "), Span::raw(app.server_config.context_size.to_string())]),
        Line::from(vec![Span::raw("Batch: "), Span::raw(app.server_config.batch_size.to_string())]),
        Line::from(vec![Span::raw("GPU Layers: "), Span::raw(app.server_config.gpu_layers.to_string())]),
        Line::from(vec![Span::raw("Threads: "), Span::raw(app.server_config.threads.to_string())]),
        Line::from(vec![Span::raw("Port: "), Span::raw(app.server_config.port.to_string())]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            if is_running { Span::raw("Running").green() } else { Span::raw("Stopped").yellow() },
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("[").fg(if is_running { Color::Red } else { Color::Green }),
            Span::raw(if is_running { "STOP" } else { "START" }).fg(if is_running { Color::Red } else { Color::Green }).add_modifier(Modifier::BOLD),
            Span::raw("]").fg(if is_running { Color::Red } else { Color::Green }),
            Span::raw(" [Enter] or [Ctrl+S]"),
        ]),
    ];

    let config_block = Block::default().borders(Borders::ALL).title("Server Config");
    let config_para = Paragraph::new(config_content).block(config_block);
    f.render_widget(config_para, chunks[1]);
}

fn render_instances_view(f: &mut Frame, area: Rect, _app: &mut TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let instance_lines = vec![
        Line::from(vec![Span::raw("Instance Manager").fg(Color::Cyan).add_modifier(Modifier::BOLD)]),
        Line::from(""),
        Line::from("No instances running."),
        Line::from(""),
        Line::from("Use the GUI Deployment Wizard to launch"),
        Line::from("multi-instance deployments."),
        Line::from(""),
        Line::from("Shortcuts:"),
        Line::from("  [Enter] - Start/Stop server"),
        Line::from("  [Ctrl+S] - Toggle server"),
        Line::from("  [Ctrl+C] - Cycle provider"),
    ];

    let block = Block::default().borders(Borders::ALL).title("Instances");
    let para = Paragraph::new(instance_lines).block(block);
    f.render_widget(para, chunks[0]);

    let gpu_lines: Vec<Line> = std::iter::once(Line::from(vec![Span::raw("GPU Topology").fg(Color::Cyan).add_modifier(Modifier::BOLD)]))
        .chain(std::iter::once(Line::from("")))
        .chain(_app.gpus.iter().map(|gpu| {
            let vram_gb = gpu.total_vram_mb as f32 / 1024.0;
            let cap = gpu.compute_capability.map(|(m, n)| format!("SM {}.{}", m, n)).unwrap_or("N/A".to_string());
            let temp = gpu.temperature_c.map(|t| format!("{:.0}°C", t)).unwrap_or("N/A".to_string());
            Line::from(format!("  GPU{}: {} | {:.0}GB | {} | {}", gpu.index, gpu.name, vram_gb, cap, temp))
        }))
        .collect();

    let gpu_block = Block::default().borders(Borders::ALL).title("GPU Topology");
    let gpu_para = Paragraph::new(gpu_lines).block(gpu_block);
    f.render_widget(gpu_para, chunks[1]);
}

fn render_logs_view(f: &mut Frame, area: Rect, app: &mut TuiApp) {
    let entries = app.log_buffer.get_entries();
    let visible_height = area.height.saturating_sub(2) as usize;

    let log_lines: Vec<Line> = entries
        .iter()
        .skip(app.log_scroll)
        .take(visible_height)
        .map(|entry| {
            let level_color = match entry.level {
                LogLevel::Error => Color::Red,
                LogLevel::Warn => Color::Yellow,
                LogLevel::Info => Color::White,
            };
            let level_text = match entry.level {
                LogLevel::Error => "ERR",
                LogLevel::Warn => "WRN",
                LogLevel::Info => "INF",
            };
            Line::from(vec![
                Span::raw(format!("[{}] ", level_text)).fg(level_color),
                Span::raw(&entry.message),
            ])
        })
        .collect();

    let block = Block::default().borders(Borders::ALL).title(format!("Logs ({})", entries.len()));
    let para = Paragraph::new(log_lines).block(block);
    f.render_widget(para, area);
}

fn render_metrics_view(f: &mut Frame, area: Rect, _app: &mut TuiApp) {
    let stats = get_system_stats();

    let vram_pct = if stats.vram_total_mb > 0 {
        (stats.vram_used_mb as f32 / stats.vram_total_mb as f32) * 100.0
    } else {
        0.0
    };
    let ram_pct = if stats.ram_total_mb > 0 {
        (stats.ram_used_mb as f32 / stats.ram_total_mb as f32) * 100.0
    } else {
        0.0
    };

    let bar = |pct: f32, width: u16| -> String {
        let filled = (pct / 100.0 * width as f32) as usize;
        let empty = width as usize - filled;
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    };

    let lines = vec![
        Line::from(vec![Span::raw("System Metrics").fg(Color::Cyan).add_modifier(Modifier::BOLD)]),
        Line::from(""),
        Line::from(format!("CPU:    {:.1}% {}", stats.cpu_percent, bar(stats.cpu_percent, 30))),
        Line::from(format!("RAM:    {:.1}% {} ({}/{} MB)", ram_pct, bar(ram_pct, 30), stats.ram_used_mb, stats.ram_total_mb)),
        Line::from(format!("VRAM:   {:.1}% {} ({}/{} MB)", vram_pct, bar(vram_pct, 30), stats.vram_used_mb, stats.vram_total_mb)),
        Line::from(""),
    ];

    let gpu_lines: Vec<Line> = if stats.gpu_temperatures.is_empty() {
        vec![Line::from("No GPU data available")]
    } else {
        stats.gpu_temperatures.iter().map(|gpu| {
            let temp = gpu.temperature_c.map(|t| format!("{:.0}°C", t)).unwrap_or("N/A".to_string());
            Line::from(format!("  GPU{}: {}", gpu.index, temp))
        }).collect()
    };

    let all_lines: Vec<Line> = lines.into_iter().chain(gpu_lines).collect();

    let block = Block::default().borders(Borders::ALL).title("Performance Metrics");
    let para = Paragraph::new(all_lines).block(block);
    f.render_widget(para, area);
}

fn render_footer(f: &mut Frame, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Min(0),
        ])
        .split(area);

    let stats = get_system_stats();
    let vram_text = format!("VRAM: {}/{} MB", stats.vram_used_mb, stats.vram_total_mb);
    let ram_cpu_text = format!("RAM: {}/{} MB | CPU: {:.1}%", stats.ram_used_mb, stats.ram_total_mb, stats.cpu_percent);

    let gpu_text = if stats.gpu_temperatures.is_empty() {
        "No GPU".to_string()
    } else {
        stats.gpu_temperatures.iter().map(|gpu| {
            let temp = gpu.temperature_c.map(|t| format!("{:.0}°C", t)).unwrap_or_else(|| "N/A".to_string());
            format!("GPU{}: {}", gpu.index, temp)
        }).collect::<Vec<_>>().join(" | ")
    };

    f.render_widget(Paragraph::new(vram_text), chunks[0]);
    f.render_widget(Paragraph::new(ram_cpu_text), chunks[1]);
    f.render_widget(Paragraph::new(gpu_text), chunks[2]);
    f.render_widget(
        Paragraph::new(app.status_message.as_str())
            .alignment(ratatui::layout::Alignment::Right)
            .style(Style::default().fg(Color::Yellow)),
        chunks[3],
    );
}

fn render_search_overlay(f: &mut Frame, size: Rect, query: &str) {
    let area = Rect::new(size.x + 5, size.y + 3, size.width - 10, 3);
    f.render_widget(Clear, area);
    f.render_widget(
        Block::default().borders(Borders::ALL).title("Search Models"),
        area,
    );
    let search_text = if query.is_empty() { "/".to_string() } else { format!("/{}", query) };
    f.render_widget(
        Paragraph::new(search_text.as_str()).fg(Color::Yellow),
        Rect::new(area.x + 1, area.y + 1, area.width - 2, 1),
    );
}

fn truncate_path(path: &str, max_width: usize) -> String {
    if path.len() <= max_width {
        return path.to_string();
    }
    let file_name = path.split('/').last().unwrap_or(path);
    if file_name.len() + 4 <= max_width {
        format!(".../{}", file_name)
    } else {
        let start = file_name.len() + 4 - max_width;
        format!("...{}", &file_name[start..])
    }
}
