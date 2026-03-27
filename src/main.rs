#[cfg(feature = "gui")]
mod gui;
#[cfg(feature = "tui")]
mod tui;
mod core;
mod models;
mod services;
mod providers;

#[cfg(feature = "gui")]
fn main() {
    env_logger::init();
    providers::register_all_providers();
    gui::run();
}

#[cfg(feature = "tui")]
fn main() {
    env_logger::init();
    providers::register_all_providers();
    tui::run().expect("TUI error");
}

#[cfg(not(any(feature = "gui", feature = "tui")))]
fn main() {
    eprintln!("Please compile with --features gui or --features tui");
    eprintln!("  cargo run --features gui   # for GUI");
    eprintln!("  cargo run --features tui   # for TUI");
}
