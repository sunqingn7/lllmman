#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "tui")]
mod tui;
mod core;
mod models;
mod services;
mod providers;

fn main() {
    env_logger::init();
    providers::register_all_providers();
    
    // GUI takes priority when both features are enabled
    #[cfg(all(feature = "gui", not(feature = "tui")))]
    {
        gui::run();
    }
    
    #[cfg(all(feature = "tui", not(feature = "gui")))]
    {
        tui::run().expect("TUI error");
    }
    
    // When both gui and tui are enabled, prefer GUI
    #[cfg(all(feature = "gui", feature = "tui"))]
    {
        gui::run();
    }
    
    #[cfg(not(any(feature = "gui", feature = "tui")))]
    {
        eprintln!("Please compile with --features gui or --features tui");
    }
}