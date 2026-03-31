pub mod provider;

pub use provider::VllmProvider;

use crate::core::LlmProvider;
use crate::register_provider;
use once_cell::sync::Lazy;
use std::sync::Arc;

static VLLM: Lazy<Arc<VllmProvider>> = Lazy::new(|| Arc::new(VllmProvider::new()));

pub fn register() {
    register_provider!(VLLM.clone() as Arc<dyn LlmProvider>);
}
