pub mod provider;
pub use provider::{
    find_huggingface_model_path,
    SglangProvider
};

use crate::core::LlmProvider;
use crate::register_provider;
use once_cell::sync::Lazy;
use std::sync::Arc;

static SGLANG: Lazy<Arc<SglangProvider>> = Lazy::new(|| Arc::new(SglangProvider::new()));

pub fn register() {
    register_provider!(SGLANG.clone() as Arc<dyn LlmProvider>);
}
