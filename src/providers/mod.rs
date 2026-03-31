pub mod llama_cpp;
pub mod sglang;
pub mod vllm;

pub use llama_cpp::register as register_llama_cpp;
pub use llama_cpp::LlamaCppProvider;
pub use sglang::register as register_sglang;
pub use vllm::register as register_vllm;

pub fn register_all_providers() {
    register_llama_cpp();
    register_vllm();
    register_sglang();
}
