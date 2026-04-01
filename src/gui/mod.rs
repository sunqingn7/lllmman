#![allow(dead_code)]
#![cfg_attr(all(feature = "gui", feature = "tui"), allow(unused_imports))]

pub mod app;

pub use app::run;
