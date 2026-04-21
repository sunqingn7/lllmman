# LLMMan - Local LLM Manager

A desktop application for managing local Large Language Models, featuring both GUI and TUI interfaces.

## Features

- **Multi-Provider Support**: Extensible architecture supporting multiple LLM backends:
  - **llama.cpp/GGUF**: Fast, efficient local inference
  - **vLLM**: High-throughput serving for transformer models
  - **SGLang**: Optimized for structured generation
- **Model Discovery**: Automatic scanning of directories for model files
- **Server Management**: Start/stop LLM servers with configurable parameters
- **Real-time Monitoring**: System resource usage (CPU, RAM, VRAM) with temperature monitoring
- **GPU Detection**: Automatic detection of available GPUs with VRAM and temperature information
- **Smart Auto-Parameters**: Automatic calculation of optimal GPU layers based on available VRAM
- **Provider Setup Wizard**: Guided installation for providers
- **Settings Persistence**: Configuration saved to `~/.config/lllmman/config.json`

## Installation

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs))
- At least one supported LLM backend:
  - **llama.cpp**: `llama-server` binary in PATH
  - **vLLM**: `vllm` Python package installed
  - **SGLang**: `sglang` Python package installed

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

1. **Model Selection**: Browse and select models from the left panel
2. **Server Configuration**: Configure server parameters in the center panel:
   - Context size (256-128000)
   - Batch size (1-8192)
   - GPU layers (number of layers offloaded to GPU) with **Auto** calculation
   - Threads (CPU thread count)
   - Cache types for K and V caches
   - Host and port
   - Sampling parameters (temperature, top_k, top_p, etc.)
3. **Start/Stop Server**: Use the buttons at the bottom to control the server
4. **GPU Settings**: Click "GPU Settings" in the top panel for advanced GPU configuration
5. **Provider Setup**: Click "Setup" next to the provider dropdown for installation guidance
6. **Download Models**: Click "Download" in the left panel to download models

### TUI Interface

Navigate through panels using keyboard shortcuts:
- Arrow keys or vim bindings for navigation
- Enter to select actions
- Real-time resource monitoring in the footer with temperature display

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| Context Size | Maximum context window | 4096 | 256-128000 |
| Batch Size | Token batch size | 512 | 1-8192 |
| GPU Layers | Layers to offload to GPU | 35 | 0-100+ (or -1 for all) |
| Threads | CPU threads | 8 | 1-64 |
| Port | Server port | 8080 | 1024-65535 |
| Host | Bind address | 0.0.0.0 | - |
| Temperature | Sampling temperature | - | 0.0-2.0 |
| Top K | Top-k sampling | - | 0-100 |
| Top P | Nucleus sampling | - | 0.0-1.0 |

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
│   │   ├── log_buffer.rs    # Log management
│   │   └── mod.rs
│   ├── models/              # Data models
│   │   ├── config.rs        # Configuration structs
│   │   ├── gpu.rs           # GPU-related types with temperature
│   │   ├── model.rs         # Model types
│   │   └── mod.rs
│   ├── providers/           # LLM provider implementations
│   │   ├── llama_cpp/       # llama.cpp provider
│   │   ├── vllm/            # vLLM provider
│   │   └── sglang/          # SGLang provider
│   ├── services/            # Utility services
│   │   ├── gpu_detector.rs  # GPU detection with temperature
│   │   ├── monitor.rs       # System monitoring
│   │   ├── config_persistence.rs
│   │   ├── model_downloader.rs
│   │   ├── model_metadata.rs    # GGUF metadata extraction
│   │   ├── process_detector.rs  # Running server detection
│   │   ├── provider_installer.rs # Setup wizard
│   │   ├── auto_params.rs       # Smart parameter calculation
│   │   └── recommended_params.rs
│   ├── gui/                 # GUI implementation (eframe/egui)
│   │   └── app.rs
│   └── tui/                 # TUI implementation (ratatui)
│       └── app.rs
└── Cargo.toml
```

## Providers

### llama.cpp
- Best for: GGUF quantized models
- Requirements: `llama-server` binary
- Supports: CPU and GPU inference, various quantizations (Q4_0, Q5_K, etc.)

### vLLM
- Best for: High-throughput serving, continuous batching
- Requirements: `vllm` Python package
- Supports: HuggingFace models, PagedAttention optimization

### SGLang
- Best for: Structured generation, JSON mode
- Requirements: `sglang` Python package
- Supports: HuggingFace models, advanced sampling

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

3. Register the provider using the `register_provider!` macro in `src/providers/mod.rs`

## API Endpoints

When a server is running, the following endpoints are available:

- `GET /stats` - Server statistics including:
  - Queue size
  - Tokens generated
  - Time per token
  - Cache hits/misses
  - GPU utilization

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

Model-specific configurations are stored separately and include:
- Sampling parameters (temperature, top_k, top_p, etc.)
- GPU allocation settings
- Provider-specific options

## Roadmap

- [x] GPU/CPU temperature monitoring
- [x] SGLang provider support
- [x] Provider setup wizard
- [x] Smart auto-parameter calculation
- [x] Model metadata extraction
- [x] Process detection for running servers
- [x] Recommended parameter suggestions
- [ ] Additional providers (Ollama, LM Studio)
- [ ] Model performance metrics display
- [ ] Web interface option

## License

MIT License
