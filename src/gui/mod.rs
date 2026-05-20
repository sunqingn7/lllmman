#![allow(dead_code)]
#![cfg_attr(all(feature = "gui", feature = "tui"), allow(unused_imports))]

pub mod app;
pub mod deployment_wizard;
pub mod gpu_topology_panel;
pub mod instance_manager_panel;
pub mod performance_monitor_panel;
pub mod recommendation_panel;

pub use app::run;
