use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
    Frame, Terminal,
};
use std::sync::Arc;

use crate::core::{
    LlmProvider, ModelInfo, ProviderConfig, ProviderRegistry, ProviderSettings, ServerController,
};
use crate::models::{AppSettings, GpuInfo, ServerStatus};
use crate::services::{
    config_persistence, detect_running_servers, get_system_stats, gpu_detector,
    load_provider_settings_for, parse_server_args, save_provider_settings_for,
};

enum InputMode {
    Normal,
    ModelSearch,
    ProviderSettingsEdit { field: ProviderSettingsField },
}

#[derive(Clone, Copy)]
enum ProviderSettingsField {
    BinaryPath,
    EnvScript,
}

#[derive(PartialEq, Eq)]
enum FocusPanel {
    Models,
    Config,
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
    input_mode: InputMode,
    focus_panel: FocusPanel,
    search_query: String,
    scroll_offset: usize,
    status_message: String,
}

impl TuiApp {
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

        let mut server_config = provider.get_config_template();

        // Detect running servers and populate config
        let running_servers = detect_running_servers();
        for server in &running_servers {
            if server.provider_id == selected_provider {
                log::info!(
                    "Detected running {} server (PID {}): {}",
                    server.binary,
                    server.pid,
                    server.command_line
                );
                let detected_config = parse_server_args(&server.provider_id, &server.command_line);

                // Merge detected config (only use detected values if current is default/empty)
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

        // Scan user-configured directories first
        for dir in &settings.scan_directories {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        // Scan provider-specific default directories
        let provider_dirs = provider.default_model_directories();
        for dir in &provider_dirs {
            let found = provider.scan_models(dir);
            models.extend(found);
        }

        let filtered_models = models.clone();

        let provider_settings = load_provider_settings_for(&selected_provider);

        let mut server_controller = ServerController::new();
        server_controller.set_provider(provider.clone());
        server_controller.set_provider_settings(provider_settings.clone());

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
            input_mode: InputMode::Normal,
            focus_panel: FocusPanel::Models,
            search_query: String::new(),
            scroll_offset: 0,
            status_message: String::from("Ready"),
        }
    }

    fn filter_models(&mut self) {
        let query = self.search_query.to_lowercase();
        self.filtered_models = if query.is_empty() {
            self.models.clone()
        } else {
            self.models
                .iter()
                .filter(|m| m.name.to_lowercase().contains(&query))
                .cloned()
                .collect()
        };
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

    fn handle_input(&mut self, key: crossterm::event::KeyEvent) {
        match &self.input_mode {
            InputMode::Normal => {
                match key.code {
                    // Quit
                    crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => {
                        return;
                    }

                    // Panel navigation
                    crossterm::event::KeyCode::Tab => {
                        self.focus_panel = match self.focus_panel {
                            FocusPanel::Models => FocusPanel::Config,
                            FocusPanel::Config => FocusPanel::Models,
                        };
                    }
                    crossterm::event::KeyCode::BackTab => {
                        self.focus_panel = match self.focus_panel {
                            FocusPanel::Config => FocusPanel::Models,
                            FocusPanel::Models => FocusPanel::Config,
                        };
                    }

                    // Model list navigation (when focused on models panel)
                    crossterm::event::KeyCode::Char('j')
                    | crossterm::event::KeyCode::Down
                    | crossterm::event::KeyCode::Char('\n') => {
                        if self.focus_panel == FocusPanel::Models {
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
                        }
                    }

                    crossterm::event::KeyCode::Char('k') | crossterm::event::KeyCode::Up => {
                        if self.focus_panel == FocusPanel::Models {
                            if let Some(idx) = self.selected_model_index {
                                if idx > 0 {
                                    self.selected_model_index = Some(idx - 1);
                                    if idx - 1 < self.scroll_offset {
                                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
                                    }
                                }
                            }
                        }
                    }

                    // Jump to top/bottom
                    crossterm::event::KeyCode::Char('g') => {
                        if self.focus_panel == FocusPanel::Models {
                            if !self.filtered_models.is_empty() {
                                self.selected_model_index = Some(0);
                                self.scroll_offset = 0;
                            }
                        }
                    }
                    crossterm::event::KeyCode::Char('G') => {
                        if self.focus_panel == FocusPanel::Models {
                            if !self.filtered_models.is_empty() {
                                let last = self.filtered_models.len() - 1;
                                self.selected_model_index = Some(last);
                                self.scroll_offset = last.saturating_sub(10);
                            }
                        }
                    }

                    // Search mode
                    crossterm::event::KeyCode::Char('/') => {
                        if self.focus_panel == FocusPanel::Models {
                            self.input_mode = InputMode::ModelSearch;
                            self.search_query.clear();
                        }
                    }

                    // Provider settings mode
                    crossterm::event::KeyCode::Char('p') => {
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                        {
                            self.input_mode = InputMode::ProviderSettingsEdit {
                                field: ProviderSettingsField::BinaryPath,
                            };
                        }
                    }

                    // Select model / Start server
                    crossterm::event::KeyCode::Enter => {
                        if self.focus_panel == FocusPanel::Models {
                            if let Some(idx) = self.selected_model_index {
                                if let Some(model) = self.filtered_models.get(idx) {
                                    self.server_config.model_path = model.path.clone();
                                    self.status_message = format!("Selected: {}", model.name);
                                }
                            }
                        } else if self.focus_panel == FocusPanel::Config {
                            // Start/Stop server
                            let status = self.server_controller.get_status();
                            if matches!(status, ServerStatus::Running) {
                                self.stop_server();
                            } else {
                                self.start_server();
                            }
                        }
                    }

                    // Start/Stop server shortcuts
                    crossterm::event::KeyCode::Char('s') => {
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                        {
                            let status = self.server_controller.get_status();
                            if matches!(status, ServerStatus::Running) {
                                self.stop_server();
                            } else {
                                self.start_server();
                            }
                        }
                    }
                    crossterm::event::KeyCode::Char('x') => {
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                        {
                            self.stop_server();
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
            InputMode::ProviderSettingsEdit { field } => match key.code {
                crossterm::event::KeyCode::Esc => {
                    self.input_mode = InputMode::Normal;
                }
                crossterm::event::KeyCode::Enter => {
                    self.input_mode = InputMode::Normal;
                    save_provider_settings_for(&self.selected_provider, &self.provider_settings)
                        .ok();
                    self.server_controller
                        .set_provider_settings(self.provider_settings.clone());
                    self.status_message = "Provider settings saved".to_string();
                }
                crossterm::event::KeyCode::Backspace => match field {
                    ProviderSettingsField::BinaryPath => {
                        self.provider_settings.binary_path.pop();
                    }
                    ProviderSettingsField::EnvScript => {
                        self.provider_settings.env_script.pop();
                    }
                },
                crossterm::event::KeyCode::Tab => match field {
                    ProviderSettingsField::BinaryPath => {
                        self.input_mode = InputMode::ProviderSettingsEdit {
                            field: ProviderSettingsField::EnvScript,
                        };
                    }
                    ProviderSettingsField::EnvScript => {
                        self.input_mode = InputMode::ProviderSettingsEdit {
                            field: ProviderSettingsField::BinaryPath,
                        };
                    }
                },
                crossterm::event::KeyCode::Char(c) => match field {
                    ProviderSettingsField::BinaryPath => {
                        self.provider_settings.binary_path.push(c);
                    }
                    ProviderSettingsField::EnvScript => {
                        self.provider_settings.env_script.push(c);
                    }
                },
                _ => {}
            },
        }
    }
}

pub fn run() -> std::io::Result<()> {
    // Setup terminal
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

    // Restore terminal
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;

    result
}

fn run_inner(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> std::io::Result<()> {
    let mut app = TuiApp::new();

    // Main loop
    loop {
        // Draw frame
        terminal.draw(|f| {
            let size = f.size();

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(3),
                ])
                .split(size);

            render_header(f, chunks[0], &app);
            render_main_content(f, chunks[1], &mut app);
            render_footer(f, chunks[2], &app);

            // Render search overlay
            if let InputMode::ModelSearch = app.input_mode {
                render_search_overlay(f, size, &app.search_query);
            }

            // Render provider settings overlay
            if let InputMode::ProviderSettingsEdit { .. } = app.input_mode {
                render_provider_settings_overlay(f, size, &app);
            }
        })?;

        // Non-blocking event poll with 100ms timeout
        if crossterm::event::poll(std::time::Duration::from_millis(100))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                app.handle_input(key);
            }
        }
    }
}

fn render_header(f: &mut Frame, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20),
            Constraint::Min(0),
            Constraint::Length(30),
        ])
        .split(area);

    let provider_text = format!("Provider: {}", app.selected_provider);
    let help_text = "[Tab]switch [/]search [Ctrl+P]settings [Enter]start [q]quit";

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
        Paragraph::new(help_text).alignment(ratatui::layout::Alignment::Right),
        chunks[2],
    );
}

fn render_main_content(f: &mut Frame, area: Rect, app: &mut TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Min(0)])
        .split(area);

    // Models panel
    let model_items: Vec<ListItem> = app
        .filtered_models
        .iter()
        .skip(app.scroll_offset)
        .take(area.height.saturating_sub(2) as usize)
        .enumerate()
        .map(|(i, model)| {
            let idx = app.scroll_offset + i;
            let prefix = if app.selected_model_index == Some(idx) {
                "► "
            } else {
                "  "
            };
            ListItem::new(format!("{}{}", prefix, model.name))
        })
        .collect();

    let models_block = Block::default()
        .borders(Borders::ALL)
        .title("Models")
        .title_style(if app.focus_panel == FocusPanel::Models {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        });

    let model_list = List::new(model_items)
        .block(models_block)
        .style(Style::default().fg(Color::White));

    f.render_widget(model_list, chunks[0]);

    // Config panel
    let server_status = app.server_controller.get_status();
    let is_running = matches!(server_status, ServerStatus::Running);

    let config_block = Block::default()
        .borders(Borders::ALL)
        .title("Server Config")
        .title_style(if app.focus_panel == FocusPanel::Config {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        });

    let config_content = vec![
        Line::from(vec![
            Span::raw("Model: "),
            Span::raw(truncate_path(
                &app.server_config.model_path,
                chunks[1].width as usize,
            )),
        ]),
        Line::from(vec![
            Span::raw("Context: "),
            Span::raw(app.server_config.context_size.to_string()),
        ]),
        Line::from(vec![
            Span::raw("Batch: "),
            Span::raw(app.server_config.batch_size.to_string()),
        ]),
        Line::from(vec![
            Span::raw("GPU Layers: "),
            Span::raw(app.server_config.gpu_layers.to_string()),
        ]),
        Line::from(vec![
            Span::raw("Threads: "),
            Span::raw(app.server_config.threads.to_string()),
        ]),
        Line::from(vec![
            Span::raw("Port: "),
            Span::raw(app.server_config.port.to_string()),
        ]),
        Line::from(vec![
            Span::raw("Host: "),
            Span::raw(&app.server_config.host),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Status: "),
            if is_running {
                Span::raw("Running").green()
            } else {
                Span::raw("Stopped").yellow()
            },
        ]),
        Line::from(""),
        // Start/Stop button
        Line::from(vec![
            Span::raw("[").fg(if is_running { Color::Red } else { Color::Green }),
            Span::raw(if is_running { "STOP" } else { "START " })
                .fg(if is_running { Color::Red } else { Color::Green })
                .add_modifier(Modifier::BOLD),
            Span::raw("]").fg(if is_running { Color::Red } else { Color::Green }),
            Span::raw(" [Enter]"),
        ]),
    ];

    let config_para = Paragraph::new(config_content).block(config_block);
    f.render_widget(config_para, chunks[1]);
}

fn render_footer(f: &mut Frame, area: Rect, app: &TuiApp) {
    let stats = get_system_stats();

    let vram_text = format!("VRAM: {}/{} MB", stats.vram_used_mb, stats.vram_total_mb);

    let ram_cpu_text = if let Some(cpu_temp) = stats.cpu_temperature {
        format!(
            "RAM: {}/{} MB | CPU: {:.1}% ({:.0}°C)",
            stats.ram_used_mb, stats.ram_total_mb, stats.cpu_percent, cpu_temp
        )
    } else {
        format!(
            "RAM: {}/{} MB | CPU: {:.1}%",
            stats.ram_used_mb, stats.ram_total_mb, stats.cpu_percent
        )
    };

    let gpu_text = if stats.gpu_temperatures.is_empty() {
        "No GPU".to_string()
    } else {
        stats
            .gpu_temperatures
            .iter()
            .map(|gpu| {
                let temp = gpu
                    .temperature_c
                    .map(|t| format!("{:.0}°C", t))
                    .unwrap_or_else(|| "N/A".to_string());
                format!("GPU{}: {}", gpu.index, temp)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
            Constraint::Min(0),
        ])
        .split(area);

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
        Block::default()
            .borders(Borders::ALL)
            .title("Search Models"),
        area,
    );
    let search_text = if query.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", query)
    };
    f.render_widget(
        Paragraph::new(search_text.as_str()).fg(Color::Yellow),
        Rect::new(area.x + 1, area.y + 1, area.width - 2, 1),
    );
}

fn render_provider_settings_overlay(f: &mut Frame, size: Rect, app: &TuiApp) {
    let area = Rect::new(size.x + 5, size.y + 5, size.width - 10, 7);
    f.render_widget(Clear, area);

    let binary_label = if let InputMode::ProviderSettingsEdit {
        field: ProviderSettingsField::BinaryPath,
    } = app.input_mode
    {
        format!("> Binary Path: {}", app.provider_settings.binary_path)
    } else {
        format!("  Binary Path: {}", app.provider_settings.binary_path)
    };
    let env_label = if let InputMode::ProviderSettingsEdit {
        field: ProviderSettingsField::EnvScript,
    } = app.input_mode
    {
        format!("> Env Script: {}", app.provider_settings.env_script)
    } else {
        format!("  Env Script: {}", app.provider_settings.env_script)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title("Provider Settings (Ctrl+P)");

    let content = vec![
        Line::from(binary_label.as_str()),
        Line::from(env_label.as_str()),
        Line::from(""),
        Line::from("[Tab] switch | [Enter] save | [Esc] cancel"),
    ];

    let para = Paragraph::new(content).block(block);
    f.render_widget(para, area);
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
