pub mod llama_cpp;
pub mod vllm;

pub use llama_cpp::register as register_llama_cpp;
pub use llama_cpp::LlamaCppProvider;
pub use vllm::register as register_vllm;

pub fn register_all_providers() {
    register_llama_cpp();
    register_vllm();
}
