pub mod provider;

pub use provider::LlamaCppProvider;

use std::sync::Arc;
use once_cell::sync::Lazy;
use crate::core::LlmProvider;
use crate::register_provider;

static LLAMA_CPP: Lazy<Arc<LlamaCppProvider>> = Lazy::new(|| Arc::new(LlamaCppProvider::new()));

pub fn register() {
    register_provider!(LLAMA_CPP.clone() as Arc<dyn LlmProvider>);
}
