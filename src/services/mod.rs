#![allow(dead_code)]

pub mod auto_params;
pub mod cmdline_parser;
pub mod config_persistence;
pub mod gpu_detector;
pub mod monitor;
pub mod model_downloader;
pub mod process_detector;
pub mod provider_installer;

pub use auto_params::{get_system_info_summary, recommend_gpu_layers};
pub use cmdline_parser::parse_command_line;
pub use config_persistence::{save_settings, save_model_config, load_model_config, get_fallback_config, load_provider_settings_for, save_provider_settings_for};
pub use monitor::{fetch_server_stats, get_system_stats};
pub use model_downloader::{DownloadManager, DirectUrlDownloader, GitHubReleaseDownloader, HuggingFaceDownloader};
pub use process_detector::{detect_running_servers, parse_server_args};
pub use provider_installer::{get_provider_install_info, check_provider_installed};
