use ratatui::{*, widgets::*};
use ratatui::layout::{Constraint, Direction, Layout};
use std::sync::Arc;

pub fn run() -> std::io::Result<()> {
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    
    let mut selected_provider = "llama.cpp".to_string();
    let mut running = false;
    
    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(3),
                ])
                .split(f.area());
            
            // Header
            f.render_widget(Paragraph::new("LLLMMan - Local LLM Manager"), chunks[0]);
            
            // Main content
            let content = if running {
                "Server Running - Use --features gui for full interface"
            } else {
                "Use --features gui for full GUI interface"
            };
            f.render_widget(Paragraph::new(content), chunks[1]);
            
            // Footer
            f.render_widget(Paragraph::new("Press q to quit | Space to toggle server"), chunks[2]);
        })?;
        
        // Handle input
        if let Ok(event) = crossterm::event::read() {
            if let crossterm::event::KeyEvent { code, .. } = event {
                match code {
                    crossterm::event::KeyCode::Char('q') => break,
                    _ => {}
                }
            }
        }
    }
    
    Ok(())
}
