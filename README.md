# LLMMan - Local LLM Manager

A desktop application for managing local Large Language Models, featuring both GUI and TUI interfaces.

## Features

- **Multi-Provider Support**: Extensible architecture for different LLM backends (llama.cpp/GGUF currently supported)
- **Model Discovery**: Automatic scanning of directories for GGUF model files
- **Server Management**: Start/stop LLM servers with configurable parameters
- **Real-time Monitoring**: System resource usage (CPU, RAM, VRAM) and server metrics
- **GPU Detection**: Automatic detection of available GPUs with VRAM information
- **Settings Persistence**: Configuration saved to `~/.config/lllmman/config.json`

## Installation

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs))
- **llama-server** binary in PATH (from [llama.cpp](https://github.com/ggerganov/llama.cpp))

### Build and Run

```bash
# Clone the repository
git clone <repository-url>
cd lllmman

# Run with GUI (default)
cargo run --features gui

# Run with TUI
cargo run --features tui
```

## Usage

### GUI Interface

1. **Model Selection**: Browse and select GGUF models from the left panel
2. **Server Configuration**: Configure server parameters in the center panel:
   - Context size (256-128000)
   - Batch size (1-8192)
   - GPU layers (number of layers offloaded to GPU)
   - Threads (CPU thread count)
   - Cache types for K and V caches
   - Host and port
3. **Start/Stop Server**: Use the buttons at the bottom to control the server
4. **GPU Settings**: Click "GPU Settings" in the top panel for advanced GPU configuration
5. **Download Models**: Click "Download" in the left panel (feature in progress)

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Context Size | Maximum context window | 4096 | 256-128000 |
| Batch Size | Token batch size | 512 | 1-8192 |
| GPU Layers | Layers to offload to GPU | 35 | 0+ |
| Threads | CPU threads | 8 | 1-64 |
| Port | Server port | 8080 | 1024-65535 |
| Host | Bind address | 0.0.0.0 | - |

## Architecture

```
lllmman/
├── src/
│   ├── main.rs              # Entry point, feature-based routing
│   ├── lib.rs               # Library exports
│   ├── core/                # Core functionality
│   │   ├── provider.rs      # Provider traits and types
│   │   ├── registry.rs      # Provider registry
│   │   ├── server.rs        # Server controller
│   │   └── mod.rs           # Module exports
│   ├── models/              # Data models
│   │   ├── config.rs        # Configuration structs
│   │   ├── gpu.rs           # GPU-related types
│   │   ├── model.rs         # Model types
│   │   └── mod.rs           # Module exports
│   ├── providers/           # LLM provider implementations
│   │   └── llama_cpp/       # llama.cpp provider
│   ├── services/            # Utility services
│   │   ├── gpu_detector.rs  # GPU detection
│   │   ├── monitor.rs       # System monitoring
│   │   ├── config_persistence.rs
│   │   └── model_downloader.rs
│   ├── gui/                 # GUI implementation (eframe/egui)
│   │   └── app.rs           # Main GUI application
│   └── tui/                 # TUI implementation (ratatui)
│       └── app.rs           # Main TUI application
└── Cargo.toml               # Dependencies and features
```

## Adding New Providers

To add support for a new LLM provider:

1. Create a new module under `src/providers/`
2. Implement the `LlmProvider` trait:

```rust
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn id(&self) -> &'static str;
    fn get_config_template(&self) -> ProviderConfig;
    fn validate_config(&self, config: &ProviderConfig) -> Result<()>;
    fn build_start_command(&self, config: &ProviderConfig) -> Command;
    fn supported_quantizations(&self) -> Vec<&'static str>;
    fn scan_models(&self, path: &str) -> Vec<ModelInfo>;
    fn add_model(&self, path: &str) -> Result<ModelInfo>;
}
```

3. Register the provider using the `register_provider!` macro

## API Endpoints

When the llama.cpp server is running, the following endpoints are available:

- `GET /stats` - Server statistics including:
  - Queue size
  - Tokens generated
  - Time per token
  - Cache hits/misses

## Configuration File

Settings are persisted to `~/.config/lllmman/config.json`:

```json
{
  "scan_directories": [],
  "download_directory": "/home/user/.cache/lllmman/models",
  "default_port": 8080,
  "default_context_size": 4096,
  "default_batch_size": 512,
  "default_gpu_layers": 35,
  "default_threads": 8,
  "default_cache_type_k": "q4_0",
  "default_cache_type_v": "q4_0",
  "gpu_strategy": "All",
  "selected_provider": "llama.cpp"
}
```

## Roadmap

- [ ] Complete model download functionality
- [ ] Add model management (add/remove models manually)
- [ ] Implement provider-specific options UI
- [ ] Add model performance metrics display
- [ ] Support for additional providers (Ollama, LM Studio)

## License

MIT License
