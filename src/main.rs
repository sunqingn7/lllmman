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
    
    // TUI takes priority when explicitly requested with --features tui
    #[cfg(all(feature = "gui", feature = "tui"))]
    {
        tui::run().expect("TUI error");
    }
    
    #[cfg(all(feature = "gui", not(feature = "tui")))]
    {
        gui::run();
    }
    
    #[cfg(all(feature = "tui", not(feature = "gui")))]
    {
        tui::run().expect("TUI error");
    }
    
    #[cfg(not(any(feature = "gui", feature = "tui")))]
    {
        eprintln!("Please compile with --features gui or --features tui");
        eprintln!("  cargo run --features gui   # for GUI");
        eprintln!("  cargo run --features tui   # for TUI");
    }
}
