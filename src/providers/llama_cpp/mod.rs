pub mod provider;

pub use provider::{read_gguf_n_layer, LlamaCppProvider};

use crate::core::LlmProvider;
use crate::register_provider;
use once_cell::sync::Lazy;
use std::sync::Arc;

static LLAMA_CPP: Lazy<Arc<LlamaCppProvider>> = Lazy::new(|| Arc::new(LlamaCppProvider::new()));

pub fn register() {
    register_provider!(LLAMA_CPP.clone() as Arc<dyn LlmProvider>);
}
