pub mod llama_cpp;

pub use llama_cpp::LlamaCppProvider;
pub use llama_cpp::register as register_llama_cpp;

pub fn register_all_providers() {
    register_llama_cpp();
}
