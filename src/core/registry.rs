use crate::core::provider::LlmProvider;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

static PROVIDER_REGISTRY: Lazy<RwLock<HashMap<&'static str, Arc<dyn LlmProvider>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub struct ProviderRegistry;

impl ProviderRegistry {
    pub fn register(provider: Arc<dyn LlmProvider>) {
        let id = provider.id();
        PROVIDER_REGISTRY.write().unwrap().insert(id, provider);
    }

    pub fn get(id: &str) -> Option<Arc<dyn LlmProvider>> {
        PROVIDER_REGISTRY.read().unwrap().get(id).cloned()
    }

    pub fn list() -> Vec<(&'static str, &'static str)> {
        PROVIDER_REGISTRY
            .read()
            .unwrap()
            .iter()
            .map(|(id, p)| (*id, p.name()))
            .collect()
    }

    pub fn available_ids() -> Vec<&'static str> {
        PROVIDER_REGISTRY.read().unwrap().keys().copied().collect()
    }
}

#[macro_export]
macro_rules! register_provider {
    ($provider:expr) => {
        $crate::core::ProviderRegistry::register($provider);
    };
}
