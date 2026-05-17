# Heterogeneous GPU & Enhanced Provider Setup Plan

> **Date:** 2026-05-17
> **Project:** lllmman
> **Status:** Draft — awaiting review

---

## 1. Overview

This plan transforms lllmman from a single-instance, single-GPU LLM manager into a **heterogeneous GPU-aware orchestration tool** that can:

1. **Detect and classify** all GPUs in the system by architecture, compute capability, and performance tier
2. **Generate optimal deployment strategies** for mixed-GPU setups (3090 + 4090 + 5090 + RTX Pro 6000, etc.)
3. **Automate vLLM/SGLang installation** with GPU-aware build configuration (CUDA arch lists, precompiled wheels, source builds)
4. **Produce ready-to-run multi-instance launch scripts** with safety flags, per-GPU KV cache tuning, and router configuration
5. **Warn users** about Tensor Parallelism limitations on heterogeneous hardware and suggest correct alternatives

---

## 2. Current State Analysis

### What lllmman Does Today
- Detects GPUs via `nvidia-smi` / `rocm-smi` / sysfs (name, VRAM, index, temperature)
- Starts **one** server instance at a time
- Supports llama.cpp, vLLM, SGLang providers
- Basic provider setup wizard with static install commands
- Auto GPU layer calculation for llama.cpp
- Smart vLLM config based on model size

### What It Cannot Do
- Detect GPU compute capability (SM version)
- Recognize heterogeneous GPU clusters
- Start multiple server instances simultaneously
- Generate SGLang router / vLLM multi-instance configurations
- Set `TORCH_CUDA_ARCH_LIST` or other build-time env vars during installation
- Auto-inject safety flags (`VLLM_SKIP_P2P_CHECK=1`, `--enforce-eager`, `--disable-cuda-graph`)
- Configure per-GPU memory utilization in a multi-GPU scenario
- Warn against Tensor Parallelism on mixed architectures

---

## 3. Research Findings (2025–2026 Ecosystem)

### 3.1 vLLM Installation Landscape

| Aspect | Current State |
|--------|---------------|
| **Default CUDA** | 13.0 (was 12.9; cu129 now a variant) |
| **Precompiled wheels** | `manylinux_2_24` (fixed glibc regression) |
| **Recommended install** | `uv pip install vllm --torch-backend=auto` |
| **Dev install** | `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` |
| **Auto-detection** | `VLLM_PRECOMPILED_WHEEL_VARIANT` auto-detected from torch/nvidia-smi (PR #32948 merged) |
| **CUDA arch list (x86)** | `7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX` |
| **Nightly wheels** | Available at `https://wheels.vllm.ai/nightly` for every commit since v0.5.3 |

**Key limitation:** vLLM does **NOT** support heterogeneous GPUs in a single inference instance. Issues #13760 and #27239 (Heterogeneous TP per Pipeline Stage) were closed as `not_planned`.

### 3.2 SGLang Installation Landscape

| Aspect | Current State |
|--------|---------------|
| **Kernel build system** | CMake with separate SM90 and SM100+ builds |
| **Default arch** | SM90; SM80/89 via `ENABLE_BELOW_SM90` flag |
| **SM100a/SM120a** | Require CUDA ≥ 12.8 |
| **CUDA 13.0 support** | Adds SM103a, SM110a, SM121a (PR #9721) |
| **Multi-GPU approach** | `sglang_router.launch_server --dp-size N` (recommended) |
| **Heterogeneous strategy** | Separate instances + `sglang_router.launch_router` |

**Key limitation:** Same as vLLM — no native heterogeneous GPU support within a single instance.

### 3.3 Heterogeneous GPU Reality

| Strategy | Viability | Notes |
|----------|-----------|-------|
| **Tensor Parallelism (TP)** | ❌ Blocked | All-reduce sync creates wood-bucket effect; fastest GPU waits for slowest |
| **Pipeline Parallelism (PP)** | ⚠️ Limited | Works if layers are balanced by compute; vLLM has partial support |
| **Multi-Instance + Router** | ✅ Recommended | Each GPU runs independent instance; router distributes by speed |
| **MIG (Datacenter GPUs)** | ✅ Workaround | Slice A100/H100/RTX Pro 6000 into uniform instances to match smaller GPUs |
| **Prefill-Decode Disaggregation** | ✅ Advanced | Separate prefill GPUs from decode GPUs; SGLang supports this |

### 3.4 Compute Capability Reference

| GPU | Architecture | SM Version | VRAM | Tier |
|-----|-------------|------------|------|------|
| RTX 3090 | Ampere | 8.6 | 24 GB | Mid |
| RTX 3080 | Ampere | 8.6 | 10/12 GB | Mid |
| RTX 4090 | Ada Lovelace | 8.9 | 24 GB | High |
| RTX 4080 | Ada Lovelace | 8.9 | 16 GB | High |
| RTX 5090 | Blackwell | 12.0 | 32 GB | Ultra |
| RTX Pro 6000 | Blackwell | 12.0 | 96 GB | Ultra |
| A100 | Ampere | 8.0 | 40/80 GB | High |
| H100 | Hopper | 9.0 | 80 GB | Ultra |
| L40S | Ada Lovelace | 8.9 | 48 GB | High |

---

## 4. Implementation Phases

### Phase 1: GPU Architecture Detection & Classification

**New file:** `src/services/gpu_arch.rs`

```rust
pub struct GpuArchInfo {
    pub index: u32,
    pub name: String,
    pub compute_capability: (u32, u32),  // e.g. (8, 6)
    pub sm_string: String,               // e.g. "sm_86"
    pub vram_mb: u32,
    pub performance_tier: GpuTier,
}

pub enum GpuTier {
    Low,     // SM < 8.0, < 12GB VRAM
    Mid,     // SM 8.x, 12-24GB VRAM
    High,    // SM 8.9/9.0, 24-48GB VRAM
    Ultra,   // SM 9.0/10.0/12.0, 48GB+ VRAM
}
```

**Functions:**
- `detect_gpu_architectures() -> Vec<GpuArchInfo>` — queries `nvidia-smi --query-gpu=compute_cap` or parses from GPU name
- `is_heterogeneous_cluster(gpus: &[GpuArchInfo]) -> bool` — true if different SM versions or VRAM ratios > 2x
- `get_combined_arch_list(gpus: &[GpuArchInfo]) -> String` — returns `"8.6 8.9 12.0"` for build env
- `get_cuda_driver_version() -> Option<String>` — parses `nvidia-smi` header for "CUDA Version: 12.8"
- `classify_performance_tier(sm: (u32,u32), vram: u32) -> GpuTier`

**Changes to `src/models/gpu.rs`:**
```rust
pub struct GpuInfo {
    pub name: String,
    pub total_vram_mb: u32,
    pub index: u32,
    pub provider: GpuProvider,
    pub temperature_c: Option<f32>,
    pub compute_capability: Option<(u32, u32)>,  // NEW
    pub performance_tier: GpuTier,                // NEW
}
```

**Priority:** High — foundational for all other phases

---

### Phase 2: CUDA Compatibility Checker

**New file:** `src/services/cuda_compat.rs`

```rust
pub struct CudaCompatibilityReport {
    pub driver_version: String,              // e.g. "12.8"
    pub runtime_version: Option<String>,     // e.g. "12.4" (from torch)
    pub torch_installed: bool,
    pub torch_cuda_version: Option<String>,  // e.g. "cu124"
    pub uv_available: bool,
    pub max_supported_sm: String,            // e.g. "sm_120"
    pub gpu_archs: Vec<String>,              // e.g. ["sm_86", "sm_89", "sm_120"]
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}
```

**Checks performed:**
1. Is CUDA driver new enough for all detected GPUs? (e.g., SM120 requires driver ≥ 570)
2. Is PyTorch installed? What CUDA was it built with?
3. Can precompiled wheels be used, or is source build required?
4. Will Blackwell (SM120) work with current CUDA/toolchain?
5. Is `uv` available for faster installs?

**Output example:**
```
⚠️ Issues:
  - CUDA driver 12.4 is too old for RTX 5090 (SM120). Upgrade to driver 570+.
  - PyTorch built with cu121; SM120 requires cu128+.

✓ Recommendations:
  - Upgrade NVIDIA driver to 570+ for Blackwell support
  - Install torch with cu128: pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Priority:** High — prevents user frustration from install failures

---

### Phase 3: Enhanced Provider Installer (Dynamic Setup Plans)

**Rewrite:** `src/services/provider_installer.rs`

Replace static `ProviderInstallInfo` with dynamic `SetupPlan`:

```rust
pub struct SetupPlan {
    pub provider: String,
    pub install_mode: InstallMode,
    pub commands: Vec<SetupCommand>,
    pub env_vars: Vec<(String, String)>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
    pub estimated_time: &'static str,
}

pub enum InstallMode {
    PrecompiledWheel,           // pip install vllm (fastest, ~30s)
    SourceWithPrecompiledKernels, // VLLM_USE_PRECOMPILED=1 pip install -e . (~2min)
    SourceFullBuild,            // Full compile (~10-30min)
    HeterogeneousBuild,         // Multi-arch compile + safety flags (~15-45min)
}

pub struct SetupCommand {
    pub description: String,
    pub command: String,
    pub is_critical: bool,      // if false, can be skipped
    pub estimated_seconds: u32,
}
```

**Dynamic command generation matrix:**

| Scenario | vLLM Command | SGLang Command |
|----------|-------------|----------------|
| Single GPU, precompiled OK | `uv pip install vllm --torch-backend=auto` | `uv pip install sglang --torch-backend=auto` |
| Single GPU, dev mode | `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` | `cd sglang && pip install -e "python[all]"` |
| Heterogeneous (3090+4090+5090) | `export TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0" && pip install --no-build-isolation -e .` | `export TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0" && cd sgl-kernel && make build && cd ../python && pip install -e ".[all]"` |
| CUDA 13.0 system | `uv pip install vllm --torch-backend=cu130` | `uv pip install sglang --torch-backend=cu130` |
| ROCm system (AMD) | `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/` | `pip install sglang[rocm]` |

**Safety flags auto-injection for heterogeneous setups:**

```
vLLM:
  VLLM_SKIP_P2P_CHECK=1
  VLLM_USE_V1=0 (if needed for compatibility)
  --enforce-eager (disable CUDA graph on heterogeneous)

SGLang:
  --disable-cuda-graph
  --enable-p2p-check (force P2P verification)
```

**Priority:** High — core user-facing feature

---

### Phase 4: Deployment Profile System

**New file:** `src/models/deployment.rs`

```rust
pub struct DeploymentProfile {
    pub mode: DeploymentMode,
    pub instances: Vec<InstanceConfig>,
    pub router_config: Option<RouterConfig>,
    pub env_overrides: HashMap<String, String>,
    pub generated_script: String,
}

pub enum DeploymentMode {
    SingleGpu,           // One model, one GPU (current behavior)
    DataParallel,        // Same model on identical GPUs (TP=1, DP=N)
    PipelineParallel,    // Model split across GPUs (PP=N, TP=1)
    MultiInstance,       // Different models or heterogeneous GPUs
}

pub struct InstanceConfig {
    pub gpu_indices: Vec<u32>,
    pub port: u16,
    pub memory_utilization: f32,      // 0.70 - 0.95
    pub max_num_seqs: Option<u32>,    // concurrency limit
    pub model_path: String,
    pub context_size: u32,
    pub provider: String,
}

pub struct RouterConfig {
    pub provider: RouterProvider,     // SglangRouter, Nginx, Custom
    pub policy: RouterPolicy,         // cache_aware, round_robin, power_of_two
    pub worker_urls: Vec<String>,
    pub router_port: u16,
    pub pd_disaggregation: bool,      // prefill-decode split
}

pub enum RouterPolicy {
    CacheAware,
    RoundRobin,
    PowerOfTwo,
    Random,
}

pub enum RouterProvider {
    SglangRouter,
    Nginx,
    Custom,
}

pub enum RouterStatus {
    Stopped,
    Starting,
    Running,
    Error(String),
}
```

**Auto-profile selection logic:**

```
IF single GPU → SingleGpu
IF multiple GPUs, all same arch + same VRAM → DataParallel
IF multiple GPUs, different arch OR VRAM ratio > 2x → MultiInstance
IF model size > largest GPU VRAM × 0.85 → PipelineParallel (warn about heterogeneity)
IF user explicitly requests TP on heterogeneous → BLOCK with warning
```

**Priority:** Medium — enables multi-instance orchestration

---

### Phase 5: Build Script Generator

**New file:** `src/services/build_script_gen.rs`

Generates complete, ready-to-run shell scripts:

```bash
#!/bin/bash
# Generated by LLLMMan on 2026-05-17
# Heterogeneous GPU Setup: RTX 3090 (sm_86), RTX 4090 (sm_89), RTX 5090 (sm_120)
# Strategy: Multi-Instance with SGLang Router

set -e

# ============================================================
# Step 1: Environment Setup
# ============================================================
export TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0"
export FORCE_CUDA=1
export VLLM_SKIP_P2P_CHECK=1

# ============================================================
# Step 2: Install vLLM (multi-arch from source)
# ============================================================
echo "Installing vLLM with multi-arch support..."
cd /path/to/vllm
pip install --no-build-isolation -e .
echo "vLLM installed successfully."

# ============================================================
# Step 3: Install SGLang Router
# ============================================================
echo "Installing SGLang Router..."
pip install sglang-router
echo "SGLang Router installed successfully."

# ============================================================
# Step 4: Launch Instances
# ============================================================

# Instance 1: RTX 5090 (fastest) — high concurrency
echo "Starting instance on GPU 2 (RTX 5090)..."
CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3-8B-Instruct \
  --port 8080 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enforce-eager \
  --host 0.0.0.0 &

# Instance 2: RTX 4090 — medium concurrency
echo "Starting instance on GPU 1 (RTX 4090)..."
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3-8B-Instruct \
  --port 8081 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 128 \
  --enforce-eager \
  --host 0.0.0.0 &

# Instance 3: RTX 3090 (slowest) — low concurrency
echo "Starting instance on GPU 0 (RTX 3090)..."
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3-8B-Instruct \
  --port 8082 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 64 \
  --enforce-eager \
  --host 0.0.0.0 &

# Wait for all instances to be healthy
echo "Waiting for instances to start..."
sleep 10

# ============================================================
# Step 5: Launch Router
# ============================================================
echo "Starting SGLang Router on port 30000..."
python -m sglang_router.launch_router \
  --worker-urls http://localhost:8080 http://localhost:8081 http://localhost:8082 \
  --port 30000 \
  --host 0.0.0.0 \
  --policy cache_aware &

echo "Setup complete!"
echo "  - Router: http://localhost:30000/v1/chat/completions"
echo "  - Instance 1 (5090): http://localhost:8080"
echo "  - Instance 2 (4090): http://localhost:8081"
echo "  - Instance 3 (3090): http://localhost:8082"

# Wait for all background processes
wait
```

**Also generates SGLang-native variant:**
```bash
# SGLang workers + router (alternative to vLLM)
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --port 8080 --mem-fraction-static 0.90 --disable-cuda-graph &

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --port 8081 --mem-fraction-static 0.85 --disable-cuda-graph &

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --port 8082 --mem-fraction-static 0.75 --disable-cuda-graph &

python -m sglang_router.launch_router \
  --worker-urls http://localhost:8080 http://localhost:8081 http://localhost:8082 \
  --port 30000 --policy cache_aware
```

**Priority:** Medium — high-value output for users

---

### Phase 6: Setup Wizard UI Overhaul

**Modify:** `src/gui/app.rs` — provider setup dialog (lines ~2324-2391)

#### 6.1 System Detection Panel (auto-run on dialog open)

```
┌─────────────────────────────────────────────────────────┐
│  System Detection                                       │
├─────────────────────────────────────────────────────────┤
│  GPUs Detected: 3                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ GPU 0: NVIDIA GeForce RTX 3090    SM 8.6  24GB   │   │
│  │ GPU 1: NVIDIA GeForce RTX 4090    SM 8.9  24GB   │   │
│  │ GPU 2: NVIDIA GeForce RTX 5090    SM 12.0 32GB   │   │
│  └──────────────────────────────────────────────────┘   │
│  CUDA Driver: 12.8                                      │
│  Python: 3.12.3  |  uv: available                      │
│  PyTorch: 2.6.0+cu124                                   │
├─────────────────────────────────────────────────────────┤
│  ⚠️ HETEROGENEOUS CLUSTER DETECTED                      │
│  Mixed architectures: SM 8.6, 8.9, 12.0                │
│  Tensor Parallelism will NOT work efficiently.          │
│  Recommended: Multi-Instance + Router                   │
└─────────────────────────────────────────────────────────┘
```

#### 6.2 Install Mode Selector

```
┌─────────────────────────────────────────────────────────┐
│  Installation Mode                                      │
├─────────────────────────────────────────────────────────┤
│  ○ Quick Install (Precompiled Wheel)                    │
│    Fastest (~30s). Uses pre-built binaries.             │
│    May not include latest features.                     │
│                                                         │
│  ○ Development Install (Precompiled Kernels)            │
│    Editable source with pre-built CUDA kernels (~2min). │
│    Good for Python-level development.                   │
│                                                         │
│  ○ Full Source Build                                    │
│    Compile everything from source (~10-30min).          │
│    Required for custom kernel modifications.            │
│                                                         │
│  ● Heterogeneous Build (Multi-Arch)                     │
│    Compiles for ALL detected GPU architectures.         │
│    Sets TORCH_CUDA_ARCH_LIST automatically.             │
│    Includes safety flags for mixed GPUs. (~15-45min)    │
└─────────────────────────────────────────────────────────┘
```

#### 6.3 Multi-Instance Setup Generator (new section)

```
┌─────────────────────────────────────────────────────────┐
│  Multi-Instance Configuration                           │
├─────────────────────────────────────────────────────────┤
│  Model: meta-llama/Llama-3-8B-Instruct                  │
│  Provider: vLLM                                         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Instance │ GPU     │ Port │ VRAM % │ Max Seqs   │   │
│  ├──────────────────────────────────────────────────┤   │
│  │ #1       │ 5090:2  │ 8080 │  90%   │ 256        │   │
│  │ #2       │ 4090:1  │ 8081 │  85%   │ 128        │   │
│  │ #3       │ 3090:0  │ 8082 │  75%   │ 64         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Router: SGLang Router (port 30000, cache_aware)       │
│                                                         │
│  [ Generate Script ]  [ Copy to Clipboard ]             │
│  [ Save as .sh file ]   [ Launch All Instances ]        │
└─────────────────────────────────────────────────────────┘
```

#### 6.4 Safety Flags Display

```
┌─────────────────────────────────────────────────────────┐
│  Safety Flags (auto-applied for heterogeneous setup)    │
├─────────────────────────────────────────────────────────┤
│  ✓ VLLM_SKIP_P2P_CHECK=1                                │
│    Disables untested direct GPU-to-GPU communication    │
│    between different architectures.                     │
│                                                         │
│  ✓ --enforce-eager                                      │
│    Disables CUDA Graph capture which may fail on        │
│    mixed-architecture systems.                          │
│                                                         │
│  ✓ --disable-cuda-graph (SGLang)                        │
│    Prevents CUDA Graph crashes on heterogeneous GPUs.   │
└─────────────────────────────────────────────────────────┘
```

**Priority:** High — primary user interface for all new features

---

## 5. Data Model Changes Summary

### `src/models/gpu.rs`
```diff
 pub struct GpuInfo {
     pub name: String,
     pub total_vram_mb: u32,
     pub index: u32,
     pub provider: GpuProvider,
     pub temperature_c: Option<f32>,
+    pub compute_capability: Option<(u32, u32)>,
+    pub performance_tier: GpuTier,
 }

+pub enum GpuTier { Low, Mid, High, Ultra }
```

### `src/models/deployment.rs` (NEW)
```rust
pub struct DeploymentProfile { ... }
pub enum DeploymentMode { SingleGpu, DataParallel, PipelineParallel, MultiInstance }
pub struct InstanceConfig { ... }
pub struct RouterConfig { ... }
pub enum RouterPolicy { CacheAware, RoundRobin, PowerOfTwo, Random }
```

### `src/models/config.rs`
```diff
+ // Add to AppSettings:
+ pub deployment_mode: DeploymentMode,
+ pub auto_detect_heterogeneous: bool,

+ // Add to ModelConfigEntry:
+ pub deployment_profile: Option<DeploymentProfile>,
+ pub router_port: Option<u16>,
```

---

## 6. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/models/gpu.rs` | Modify | Add `compute_capability`, `GpuTier` enum |
| `src/models/deployment.rs` | **NEW** | Deployment profiles, instance configs, router config |
| `src/models/config.rs` | Modify | Add deployment-related fields to settings |
| `src/services/gpu_arch.rs` | **NEW** | GPU architecture detection, SM version parsing |
| `src/services/cuda_compat.rs` | **NEW** | CUDA compatibility checker, torch/driver validation |
| `src/services/build_script_gen.rs` | **NEW** | Shell script generator for multi-instance setups |
| `src/services/provider_installer.rs` | **Rewrite** | Dynamic `SetupPlan` generation based on detected GPUs |
| `src/services/metrics_collector.rs` | **NEW** | Poll /stats endpoints for all instances + GPU metrics |
| `src/services/mod.rs` | Modify | Export new modules |
| `src/gui/app.rs` | **Major Rewrite** | Multi-tab layout, instance manager integration |
| `src/gui/instance_manager.rs` | **NEW** | Instance lifecycle, metrics collection, shared state |
| `src/gui/gpu_topology_panel.rs` | **NEW** | Right sidebar GPU visualization with VRAM bars |
| `src/gui/deployment_wizard.rs` | **NEW** | Step-by-step guided setup modal (4 steps) |
| `src/gui/performance_monitor.rs` | **NEW** | Real-time metrics dashboard with charts |
| `src/gui/recommendation_engine.rs` | **NEW** | Smart suggestion system based on live metrics |
| `src/tui/app.rs` | **Major Rewrite** | Multi-panel layout, instance management |
| `src/tui/instance_panel.rs` | **NEW** | TUI instance list with per-instance controls |
| `src/tui/gpu_status_bar.rs` | **NEW** | TUI footer GPU status with VRAM/temp |
| `src/tui/performance_panel.rs` | **NEW** | TUI performance metrics overlay |
| `src/providers/vllm/provider.rs` | Modify | Add heterogeneous safety flags, PP support options |
| `src/providers/sglang/provider.rs` | Modify | Add router launch options, `--dp-size` support |
| `Cargo.toml` | Modify | Upgrade `reqwest` 0.11→0.12; add `rmcp` 1.7, `async-stream`, `schemars`; add `ai` feature flag |

---

## 7. User Flow After Implementation

### 7.1 First-Time User (Heterogeneous Setup)

```
1. User opens lllmman for the first time

2. System auto-detects hardware (takes ~1 second):
   "3 GPUs found: RTX 3090 (SM86, 24GB), RTX 4090 (SM89, 24GB), RTX 5090 (SM120, 32GB)"

3. Right sidebar shows GPU Topology Panel with all 3 GPUs

4. User clicks [⚡ Auto-Configure]

5. Deployment Wizard opens (Step 1/4):
   User selects "Small Model (≤ 32B) — fits on a single GPU"

6. Step 2/4:
   Warning: "⚠️ Heterogeneous GPUs detected! Tensor Parallelism will be severely bottlenecked."
   Recommendation: "Multi-Instance mode with SGLang Router"
   User clicks [Use Recommendation]

7. Step 3/4:
   "vLLM is NOT installed. Install mode: Heterogeneous Build (Multi-Arch)"
   "TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0" will be set automatically"
   User clicks [Install Now]
   → Progress bar shows installation status

8. Step 4/4:
   Summary: "Mode: Multi-Instance + SGLang Router, 3 instances, Router port 30000"
   User clicks [Launch Now]

9. Main interface shows:
   - GPU Topology Panel: all 3 GPUs with live VRAM/temp
   - Multi-Instance Manager tab: 3 instances + router, all running
   - Performance Monitor: real-time tok/s, queue, cache hit rate
   - Status bar: "3 instances running | Router: active | QPS: 308.2"

10. Smart Recommendation Banner appears:
    "RTX 5090 VRAM at 90%. Increase --max-num-seqs to 320 for higher throughput. [Apply]"
```

### 7.2 Experienced User (Quick Setup)

```
1. User selects model from left panel
2. Clicks [⚡ Auto-Configure] → system auto-detects optimal config
3. Clicks [Launch Now] in wizard summary
4. Everything starts automatically
5. User monitors performance in the dashboard
```

### 7.3 TUI User Workflow

```
1. User runs: cargo run --features tui
2. Sees GPU status bar at bottom with all GPUs
3. Presses 'w' to launch deployment wizard
4. Follows wizard steps (same as GUI, terminal-adapted)
5. After launch, sees instance manager panel with all instances
6. Presses 'm' to toggle performance monitor overlay
7. Uses j/k to navigate, s/S to start/stop instances
8. Presses 'g' to generate and save launch script
```

---

## 8. GUI/TUI Redesign — Multi-Instance Orchestration Dashboard

The current GUI is designed for **single-instance, single-GPU** usage. To support heterogeneous GPU orchestration, we need a fundamental redesign that makes multi-instance management intuitive while keeping simple setups effortless.

### 8.1 Core Design Principles

1. **Progressive Disclosure** — Simple users see simple controls; advanced users can drill down
2. **Visual Hierarchy** — GPU health and instance status are always visible at a glance
3. **Smart Defaults** — Auto-configure optimal settings based on detected hardware + model
4. **One-Click Optimization** — "Best Configuration" button that does everything automatically
5. **No Silent Failures** — Clear warnings when user attempts suboptimal configurations

### 8.2 GUI Architecture Redesign

**Current layout:**
```
┌────────────────────────────────────────────────────┐
│ LLLMMan  [Provider ▼] [GPU Settings] [Setup]       │
├──────────┬─────────────────────────────────────────┤
│ Models   │  Server Config                          │
│ (left)   │  (center - single instance only)        │
│          │                                         │
│          │  [Start Server] [Stop Server]           │
├──────────┴─────────────────────────────────────────┤
│ Log entries                        [Clear]         │
└────────────────────────────────────────────────────┘
```

**New layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│ LLLMMan  [Provider ▼] [⚡ Auto-Configure] [Setup] [📊 Dashboard] │
├──────────┬────────────────────────────┬─────────────────────────┤
│ Models   │  Active View (tabbed):     │  GPU Topology Panel     │
│          │  • Instance Config         │  (always visible)       │
│          │  • Multi-Instance Manager  │                         │
│          │  • Router Control          │  [GPU0] ████░ 18/24GB   │
│          │  • Performance Monitor     │  [GPU1] ██░░░ 10/24GB   │
│          │                            │  [GPU2] █████ 28/32GB   │
│          │                            │                         │
│          │                            │  Temp: 62°C 58°C 71°C   │
├──────────┴────────────────────────────┴─────────────────────────┤
│ Status Bar: 3 instances running | Router: active | QPS: 47.2   │
└──────────────────────────────────────────────────────────────────┘
```

### 8.3 New GUI Components

#### 8.3.1 GPU Topology Panel (Right Sidebar — Always Visible)

Shows real-time status of all GPUs at a glance:

```
┌─────────────────────────────────────┐
│  GPU TOPOLOGY                       │
├─────────────────────────────────────┤
│  GPU 0  RTX 3090    SM 8.6  24GB   │
│  ┌───────────────────────────────┐  │
│  │ VRAM  ████████░░░░  18.2/24   │  │
│  │ Temp  62°C  │  Util  94%      │  │
│  │ Instance: #3 (port 8082)      │  │
│  └───────────────────────────────┘  │
│                                     │
│  GPU 1  RTX 4090    SM 8.9  24GB   │
│  ┌───────────────────────────────┐  │
│  │ VRAM  ██████░░░░░░  12.1/24   │  │
│  │ Temp  58°C  │  Util  87%      │  │
│  │ Instance: #2 (port 8081)      │  │
│  └───────────────────────────────┘  │
│                                     │
│  GPU 2  RTX 5090    SM 12.0 32GB   │
│  ┌───────────────────────────────┐  │
│  │ VRAM  ████████████░ 28.4/32   │  │
│  │ Temp  71°C  │  Util  98%      │  │
│  │ Instance: #1 (port 8080)      │  │
│  └───────────────────────────────┘  │
│                                     │
│  ⚠️ Heterogeneous cluster           │
│     Multi-Instance mode active      │
└─────────────────────────────────────┘
```

**Color coding:**
- Green: VRAM < 80%, Temp < 75°C
- Yellow: VRAM 80-90%, Temp 75-85°C
- Red: VRAM > 90%, Temp > 85°C

#### 8.3.2 Multi-Instance Manager (Tab View)

Replaces the single-instance config when multiple instances are configured:

```
┌─────────────────────────────────────────────────────────────────┐
│  MULTI-INSTANCE MANAGER                    [+ Add Instance]     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Instance #1 ──────────────────────────────────────────┐ ●   │
│  │  GPU: RTX 5090 (GPU 2)    Port: 8080    Status: Running │    │
│  │  Model: Llama-3-8B-Instruct                            │    │
│  │  VRAM: 90% (28.8/32GB)    Max Seqs: 256                │    │
│  │  Tokens/s: 142.3    Queue: 3    Cache Hit: 67%         │    │
│  │  [Stop] [Restart] [Logs] [Open Browser]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ Instance #2 ──────────────────────────────────────────┐ ●   │
│  │  GPU: RTX 4090 (GPU 1)    Port: 8081    Status: Running │    │
│  │  Model: Llama-3-8B-Instruct                            │    │
│  │  VRAM: 85% (20.4/24GB)    Max Seqs: 128                │    │
│  │  Tokens/s: 98.7     Queue: 1    Cache Hit: 54%         │    │
│  │  [Stop] [Restart] [Logs] [Open Browser]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ Instance #3 ──────────────────────────────────────────┐ ●   │
│  │  GPU: RTX 3090 (GPU 0)    Port: 8082    Status: Running │    │
│  │  Model: Llama-3-8B-Instruct                            │    │
│  │  VRAM: 75% (18.0/24GB)    Max Seqs: 64                 │    │
│  │  Tokens/s: 67.2     Queue: 0    Cache Hit: 41%         │    │
│  │  [Stop] [Restart] [Logs] [Open Browser]                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ Router ───────────────────────────────────────────────┐ ●   │
│  │  SGLang Router    Port: 30000    Policy: cache_aware   │    │
│  │  Workers: 3/3 active    Total QPS: 308.2               │    │
│  │  Endpoint: http://localhost:30000/v1/chat/completions  │    │
│  │  [Stop] [Restart] [Logs] [Copy URL]                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  [Start All]  [Stop All]  [Regenerate Script]  [Export Config]  │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.3.3 Deployment Wizard (Modal — First-Run or "Auto-Configure")

Step-by-step guided flow for optimal setup:

```
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT WIZARD                                    Step 1/4  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  What do you want to run?                                       │
│                                                                 │
│  ○ Small Model (≤ 32B) — fits on a single GPU                   │
│    → Recommended: Multi-Instance for maximum throughput         │
│                                                                 │
│  ○ Large Model (70B+) — needs multiple GPUs                     │
│    → Recommended: Pipeline Parallelism                          │
│                                                                 │
│  ○ Multiple Different Models                                    │
│    → Recommended: Multi-Instance, one per GPU                   │
│                                                                 │
│  [Back]                    [Next →]                             │
└─────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT WIZARD                                    Step 2/4  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Detected Hardware:                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ GPU 0: RTX 3090  SM 8.6  24GB  → Tier: Mid               │  │
│  │ GPU 1: RTX 4090  SM 8.9  24GB  → Tier: High              │  │
│  │ GPU 2: RTX 5090  SM 12.0 32GB  → Tier: Ultra             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ⚠️ Heterogeneous GPUs detected!                                │
│     Tensor Parallelism will be severely bottlenecked.           │
│     We recommend Multi-Instance mode.                           │
│                                                                 │
│  Recommended Configuration:                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Instance per GPU with SGLang Router                       │  │
│  │ • Each GPU runs independent server                        │  │
│  │ • Router distributes requests by speed                    │  │
│  │ • Fastest GPU gets more traffic                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [← Back]        [Use Recommendation]    [Custom Setup]        │
└─────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT WIZARD                                    Step 3/4  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Provider Installation                                          │
│                                                                 │
│  vLLM is NOT installed on this system.                          │
│                                                                 │
│  Install mode (auto-selected for your hardware):                │
│  ● Heterogeneous Build (Multi-Arch)                             │
│    Compiles for SM 8.6, 8.9, 12.0                               │
│    Estimated time: 15-45 minutes                                │
│                                                                 │
│  Environment variables to be set:                               │
│    TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0"                          │
│    VLLM_SKIP_P2P_CHECK=1                                        │
│                                                                 │
│  [← Back]              [Install Now →]                          │
│                                                                 │
│  Progress: ████████████░░░░░░░░  45%  (est. 12 min remaining)  │
└─────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT WIZARD                                    Step 4/4  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ready to Launch!                                               │
│                                                                 │
│  Configuration Summary:                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Mode:       Multi-Instance + SGLang Router                │  │
│  │ Model:      Llama-3-8B-Instruct                           │  │
│  │ Provider:   vLLM                                          │  │
│  │ Instances:  3 (one per GPU)                               │  │
│  │ Router:     port 30000, cache_aware policy                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [← Back]        [Save Script]    [Launch Now]                  │
│                                                                 │
│  Your setup script has been saved to:                           │
│  ~/.config/lllmman/launch_heterogeneous.sh                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.3.4 Performance Monitor Dashboard (Tab View)

Real-time metrics for tuning and debugging:

```
┌─────────────────────────────────────────────────────────────────┐
│  PERFORMANCE MONITOR                         [Live] [Paused]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Throughput (tokens/s)                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  350 ┤                                                  │    │
│  │  300 ┤        ╭──╮                                      │    │
│  │  250 ┤       ╭╯   ╰╮                                    │    │
│  │  200 ┤  ╭────╯      ╰────╮                              │    │
│  │  150 ┤ ╭╯                ╰──╮                            │    │
│  │  100 ┤╭╯                    ╰──────                      │    │
│  │   50 ┤                                                │    │
│  │    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┤    │    │
│  │       :00  :05  :10  :15  :20  :25  :30  :35  :40     │    │
│  │                                                  │    │
│  │  ● Total  ● Instance #1  ● Instance #2  ● #3   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Per-Instance Metrics                                           │
│  ┌──────────┬─────────┬────────┬────────┬────────┬─────────┐   │
│  │ Instance │ tok/s   │ Queue  │ Cache% │ VRAM%  │ Latency │   │
│  ├──────────┼─────────┼────────┼────────┼────────┼─────────┤   │
│  │ #1 (5090)│ 142.3   │   3    │  67%   │  90%   │  23ms   │   │
│  │ #2 (4090)│  98.7   │   1    │  54%   │  85%   │  31ms   │   │
│  │ #3 (3090)│  67.2   │   0    │  41%   │  75%   │  45ms   │   │
│  ├──────────┼─────────┼────────┼────────┼────────┼─────────┤   │
│  │ TOTAL    │ 308.2   │   4    │  54%   │  83%   │  33ms   │   │
│  └──────────┴─────────┴────────┴────────┴────────┴─────────┘   │
│                                                                 │
│  Router Distribution                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Instance #1 (5090): ████████████████████  46% of traffic│    │
│  │ Instance #2 (4090): ██████████████░░░░░░  32% of traffic│    │
│  │ Instance #3 (3090): ██████████░░░░░░░░░░  22% of traffic│    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  [Export CSV]  [Reset Stats]  [Benchmark]                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.3.5 Smart Recommendation Banner (Context-Aware)

Appears at the top of the config panel when suboptimal settings are detected:

```
┌─────────────────────────────────────────────────────────────────┐
│  💡 Optimization Suggestions                                    │
├─────────────────────────────────────────────────────────────────┤
│  • RTX 5090 VRAM is at 90%. Increase --max-num-seqs to 320     │
│    for higher throughput. [Apply]                               │
│  • Router is sending equal traffic to all GPUs. Switch to       │
│    cache_aware policy for 2x better cache hit rate. [Apply]     │
│  • RTX 3090 temperature is 87°C. Consider reducing max-num-seqs │
│    to 48 to lower thermal throttling. [Apply]                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.3.6 Instance Quick-Edit Dialog

Click any instance to adjust on the fly:

```
┌─────────────────────────────────────────────────────┐
│  Edit Instance #2                                   │
├─────────────────────────────────────────────────────┤
│  GPU: RTX 4090 (GPU 1)                              │
│  Port: [8081]                                       │
│  Memory Utilization: [████████░░] 85%               │
│  Max Concurrent Seqs: [128]                         │
│  Context Length: [4096]                             │
│                                                     │
│  Safety Flags:                                      │
│  ☑ --enforce-eager                                  │
│  ☐ --disable-log-requests                           │
│                                                     │
│  [Cancel]  [Apply (Restart Required)]               │
└─────────────────────────────────────────────────────┘
```

### 8.4 TUI Redesign

The TUI must provide equivalent functionality for terminal users.

**New TUI layout:**

```
┌─ LLLMMan v0.2.0 ─────────────────────────────────────────────────┐
│ Provider: [vLLM]  Mode: [Multi-Instance]  GPUs: 3  [F1 Help]    │
├──────────────────────┬───────────────────────────────────────────┤
│ MODELS               │ INSTANCE MANAGER                          │
│                      │                                           │
│ > Llama-3-8B         │ ┌─ #1 GPU2:5090 ──── ● Running ─────────┐│
│   Llama-3-70B        │ │ Port:8080  VRAM:90%  tok/s:142.3      ││
│   Qwen2.5-32B        │ │ [Stop] [Restart] [Logs] [Edit]        ││
│   Mistral-7B         │ └───────────────────────────────────────┘│
│                      │                                           │
│ [a] Add scan dir     │ ┌─ #2 GPU1:4090 ──── ● Running ─────────┐│
│ [r] Refresh          │ │ Port:8081  VRAM:85%  tok/s:98.7       ││
│ [d] Download         │ │ [Stop] [Restart] [Logs] [Edit]        ││
│                      │ └───────────────────────────────────────┘│
│                      │                                           │
│                      │ ┌─ #3 GPU0:3090 ──── ● Running ─────────┐│
│                      │ │ Port:8082  VRAM:75%  tok/s:67.2       ││
│                      │ │ [Stop] [Restart] [Logs] [Edit]        ││
│                      │ └───────────────────────────────────────┘│
│                      │                                           │
│                      │ ┌─ Router ──────────── ● Active ─────────┐│
│                      │ │ Port:30000  Workers:3/3  QPS:308.2    ││
│                      │ │ [Stop] [Logs]                          ││
│                      │ └───────────────────────────────────────┘│
│                      │                                           │
│                      │ [Start All] [Stop All] [Wizard] [Script]  │
├──────────────────────┴───────────────────────────────────────────┤
│ GPU0:3090 ████░ 18/24GB 62°C │ GPU1:4090 ███░ 10/24GB 58°C     │
│ GPU2:5090 █████ 28/32GB 71°C │ Total QPS: 308.2  Avg Lat: 33ms │
└──────────────────────────────────────────────────────────────────┘
```

**TUI Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| `Tab` | Switch between panels |
| `↑/↓` or `j/k` | Navigate within panel |
| `Enter` | Select/activate |
| `s` | Start selected instance |
| `S` | Stop selected instance |
| `r` | Restart selected instance |
| `l` | View logs for selected instance |
| `e` | Edit selected instance config |
| `w` | Launch deployment wizard |
| `g` | Generate launch script |
| `A` | Start all instances |
| `X` | Stop all instances |
| `1/2/3` | Switch to GPU 1/2/3 view |
| `m` | Toggle performance monitor |
| `q` | Quit |

**TUI Performance Monitor (toggle with `m`):**

```
┌─ Performance Monitor ────────────────────────────────────────────┐
│ Throughput (tok/s): 308.2  │  Queue: 4  │  Cache Hit: 54%       │
│                                                                    │
│ Instance   tok/s   Queue   Cache%   VRAM%   Latency   GPU        │
│ ─────────────────────────────────────────────────────────────── │
│ #1 (5090)  142.3     3      67%      90%     23ms     ████░░░░  │
│ #2 (4090)   98.7     1      54%      85%     31ms     ███░░░░░  │
│ #3 (3090)   67.2     0      41%      75%     45ms     ██░░░░░░  │
│ ─────────────────────────────────────────────────────────────── │
│ TOTAL      308.2     4      54%      83%     33ms               │
│                                                                    │
│ Router Distribution:                                               │
│   #1 (5090): ████████████████████ 46%                             │
│   #2 (4090): ██████████████░░░░░░ 32%                             │
│   #3 (3090): ██████████░░░░░░░░░░ 22%                             │
└───────────────────────────────────────────────────────────────────┘
```

### 8.5 New GUI/TUI Data Structures

**New file:** `src/gui/instance_manager.rs`

```rust
pub struct InstanceManager {
    pub instances: Vec<InstanceHandle>,
    pub router: Option<RouterHandle>,
    pub deployment_profile: DeploymentProfile,
}

pub struct InstanceHandle {
    pub id: u32,
    pub config: InstanceConfig,
    pub status: InstanceStatus,
    pub metrics: InstanceMetrics,
    pub process: Option<Child>,
    pub log_buffer: LogBuffer,
}

pub struct InstanceMetrics {
    pub tokens_per_second: f32,
    pub queue_size: u32,
    pub cache_hit_rate: f32,
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub avg_latency_ms: f32,
    pub gpu_utilization: f32,
    pub temperature_c: f32,
    pub requests_total: u64,
    pub requests_failed: u64,
}

pub enum InstanceStatus {
    Stopped,
    Starting,
    Running,
    Error(String),
    Restarting,
}

pub struct RouterHandle {
    pub config: RouterConfig,
    pub status: RouterStatus,
    pub process: Option<Child>,
    pub log_buffer: LogBuffer,
    pub connected_workers: u32,
    pub total_workers: u32,
    pub total_qps: f32,
}
```

### 8.6 GUI/TUI File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/gui/app.rs` | **Major Rewrite** | Multi-tab layout, instance manager integration |
| `src/gui/instance_manager.rs` | **NEW** | Instance lifecycle, metrics collection |
| `src/gui/gpu_topology_panel.rs` | **NEW** | Right sidebar GPU visualization |
| `src/gui/deployment_wizard.rs` | **NEW** | Step-by-step guided setup modal |
| `src/gui/performance_monitor.rs` | **NEW** | Real-time metrics dashboard |
| `src/gui/recommendation_engine.rs` | **NEW** | Smart suggestion system |
| `src/tui/app.rs` | **Major Rewrite** | Multi-panel layout, instance management |
| `src/tui/instance_panel.rs` | **NEW** | TUI instance list and controls |
| `src/tui/gpu_status_bar.rs` | **NEW** | TUI footer GPU status |
| `src/tui/performance_panel.rs` | **NEW** | TUI performance metrics overlay |
| `src/services/metrics_collector.rs` | **NEW** | Poll /stats endpoints for all instances |

### 8.7 Metrics Collection Architecture

```
┌─────────────────────────────────────────────────┐
│  MetricsCollector (background tokio task)       │
├─────────────────────────────────────────────────┤
│                                                 │
│  Every 2 seconds:                               │
│  1. Poll http://localhost:8080/stats            │
│  2. Poll http://localhost:8081/stats            │
│  3. Poll http://localhost:8082/stats            │
│  4. Poll http://localhost:30000/metrics         │
│  5. Query nvidia-smi for GPU temps/utilization  │
│  6. Update shared state (Arc<RwLock<...>>)      │
│                                                 │
│  GUI/TUI reads shared state for display         │
└─────────────────────────────────────────────────┘
```

### 8.8 "One-Click Best Configuration" Flow

The most important UX improvement — a single button that does everything:

```
User clicks: [⚡ Auto-Configure]

System does:
1. Detects all GPUs + architectures
2. Checks if vLLM/SGLang is installed → offers install if needed
3. Analyzes selected model size vs available VRAM
4. Determines optimal deployment mode:
   - Model fits on single GPU → SingleGpu mode
   - Multiple identical GPUs → DataParallel mode
   - Heterogeneous GPUs → MultiInstance mode
   - Model too large → PipelineParallel mode (with warning)
5. Auto-generates instance configs:
   - Assigns GPUs by performance tier
   - Sets memory utilization based on VRAM
   - Sets max-num-seqs based on compute capability
   - Assigns sequential ports
6. Configures router with cache_aware policy
7. Applies safety flags for heterogeneous setups
8. Shows summary → user confirms → launches everything

Result: User goes from zero to running heterogeneous cluster in 3 clicks.
```

---

## 9. Alternative: Pipeline Parallelism Support

For scenarios where the model is too large for any single GPU (e.g., 70B+ model), lllmman should also offer Pipeline Parallelism as an option:

```
┌─────────────────────────────────────────────────────────┐
│  Pipeline Parallelism Configuration                     │
│  (For models too large for a single GPU)                │
├─────────────────────────────────────────────────────────┤
│  Model: meta-llama/Llama-3-70B-Instruct (~140GB)        │
│                                                         │
│  GPU Assignment (ordered by performance):               │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Stage │ GPU     │ Layers  │ Role                 │   │
│  ├──────────────────────────────────────────────────┤   │
│  │ 0     │ 3090:0  │ 0-15    │ Input/Prefill        │   │
│  │ 1     │ 4090:1  │ 16-40   │ Middle layers        │   │
│  │ 2     │ 5090:2  │ 41-60   │ Output/Generation    │   │
│  │ 3     │ 6000:3  │ 61-80   │ Final layers         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Command: vllm serve <model> --pipeline-parallel-size 4 │
│           --enforce-eager                                │
│                                                         │
│  ⚠️ Note: Slower GPUs in the pipeline will bottleneck   │
│     overall throughput. Multi-Instance is preferred     │
│     if model fits on individual GPUs.                   │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| `nvidia-smi` doesn't report compute cap on older drivers | Medium | Fallback: parse from GPU name using known mapping table |
| Source build OOM on systems with < 16GB RAM | High | Auto-set `MAX_JOBS=2` for low-memory systems; warn user |
| PTX forward compatibility breaks on future GPUs | Low | Always include `+PTX` suffix for newest arch in list |
| SGLang router version mismatch with workers | Medium | Pin router version to match worker version in generated script |
| User has mixed NVIDIA + AMD GPUs | High | Detect and block; show "mixed vendor not supported" error |
| CUDA driver too old for Blackwell (SM120) | High | Detect early, show clear upgrade instructions before install |
| GUI performance degradation with many instances | Medium | Throttle metrics polling to 2s interval; use Arc<RwLock> for shared state |
| TUI rendering lag on slow terminals | Low | Use buffered rendering; skip frames if rendering takes > 50ms |

---

## 11. Suggested Implementation Order

| Phase | Priority | Estimated Effort | Dependencies |
|-------|----------|------------------|--------------|
| 1. GPU Arch Detection | High | 1-2 days | None |
| 2. CUDA Compatibility Checker | High | 1 day | Phase 1 |
| 3. Enhanced Provider Installer | High | 2-3 days | Phase 1, 2 |
| 4. Deployment Profile System | Medium | 2 days | Phase 1 |
| 5. Build Script Generator | Medium | 1-2 days | Phase 3, 4 |
| 6. Setup Wizard UI (dialog) | High | 3-4 days | Phase 1-5 |
| **7. GUI/TUI Redesign** | **High** | **5-7 days** | **Phase 1-6** |
| **8. AI Copilot** | **High** | **7-10 days** | **Phase 1-7** |
| 9. PP Support (optional) | Low | 1-2 days | Phase 4, 7 |

**Total estimated effort:** 23-32 days

### Phase 7 Breakdown (GUI/TUI Redesign):

| Sub-phase | Effort | Notes |
|-----------|--------|-------|
| 7a. InstanceManager + metrics collector | 1-2 days | Backend plumbing |
| 7b. GPU Topology Panel (GUI) | 1 day | Right sidebar |
| 7c. Multi-Instance Manager tab (GUI) | 1-2 days | Core UI |
| 7d. Deployment Wizard modal (GUI) | 1 day | Guided flow |
| 7e. Performance Monitor tab (GUI) | 1 day | Charts + tables |
| 7f. Smart recommendation engine | 0.5 day | Context-aware tips |
| 7g. TUI multi-panel layout | 1-2 days | Terminal equivalent |
| 7h. Integration testing | 1 day | End-to-end flows |

### Phase 8 Breakdown (AI Copilot):

| Sub-phase | Effort | Notes |
|-----------|--------|-------|
| 8a. LLM config + search engine config | 0.5 day | Settings panel, API key management |
| 8b. MCP server + tool implementations | 2-3 days | `rmcp` integration, 10+ tools |
| 8c. AI agent planner + executor | 2 days | Task routing, conversation management |
| 8d. Debug workflow (error → diagnose → fix) | 1-2 days | Error parsing, solution proposal, auto-fix |
| 8e. Optimization workflow (research → benchmark → apply) | 1-2 days | Iterative tuning, rollback |
| 8f. Chat interface (GUI + TUI) | 1 day | Streaming responses, message history |
| 8g. Action log + audit trail + rollback | 0.5 day | Snapshot system, undo |
| 8h. Safety layer + command classification | 0.5 day | Whitelist, rate limiting, confirmation flows |
| 8i. System prompts + integration testing | 1 day | End-to-end AI workflows |

---

## 12. AI Copilot — LLM-Powered Setup, Debugging & Optimization

The AI Copilot transforms lllmman from a configuration tool into an **intelligent LLM infrastructure engineer** that can diagnose errors, auto-tune parameters, and research best practices — all within the application.

### 12.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  LLLMMan AI Copilot                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  LLM Config  │───▶│  AI Agent    │───▶│  MCP Server       │  │
│  │  (User Set)  │    │  (Planner)   │    │  (Local Tools)    │  │
│  └──────────────┘    └──────┬───────┘    └───────────────────┘  │
│                             │                      │            │
│                      ┌──────▼───────┐              │            │
│                      │  Web Search  │              │            │
│                      │  Engine      │              │            │
│                      └──────────────┘              │            │
│                                                    │            │
│                              ┌─────────────────────┘            │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Available MCP Tools                                      │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐  │  │
│  │  │ run_command │ │ read_logs    │ │ check_gpu_status   │  │  │
│  │  └─────────────┘ └──────────────┘ └────────────────────┘  │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐  │  │
│  │  │ apply_config│ │ restart_svc  │ │ benchmark_config   │  │  │
│  │  └─────────────┘ └──────────────┘ └────────────────────┘  │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐  │  │
│  │  │ search_web  │ │ get_metrics  │ │ generate_script    │  │  │
│  │  └─────────────┘ └──────────────┘ └────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Safety Layer                                             │  │
│  │  • All destructive actions require user confirmation      │  │
│  │  • Command whitelist for auto-execution                   │  │
│  │  • Rollback capability for failed changes                 │  │
│  │  • Audit log of all AI actions                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 LLM Configuration Panel

**New settings page** where users configure the AI assistant:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Copilot Configuration                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Backend                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Provider: [OpenAI Compatible ▼]                          │    │
│  │                                                         │    │
│  │ API Endpoint: [http://localhost:11434/v1_____________]   │    │
│  │ API Key:    [sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx] 🔑   │    │
│  │ Model:      [qwen2.5-coder-32b-instruct______________]   │    │
│  │                                                         │    │
│  │ [Test Connection] ✓ Connected (latency: 45ms)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Supported Providers:                                           │
│  ○ OpenAI Compatible (any OpenAI-compatible API)               │    │
│  ○ OpenAI (api.openai.com)                                     │    │
│  ○ Ollama (localhost:11434)                                    │    │
│  ○ LM Studio (localhost:1234)                                  │    │
│  ○ vLLM (local inference endpoint)                             │    │
│  ○ Custom URL                                                  │    │
│                                                                 │
│  Search Engine                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Provider: [Tavily ▼]                                     │    │
│  │ API Key:    [tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx] 🔑   │    │
│  │ Max Results: [5]                                         │    │
│  │                                                         │    │
│  │ Alternative: Google SerpAPI, DuckDuckGo (free), None    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  AI Behavior                                                    │
│  ☑ Auto-debug installation errors                              │    │
│  ☑ Suggest performance optimizations                           │    │
│  ☑ Research latest best practices on web                       │    │
│  ☐ Auto-apply safe configurations (requires confirmation)      │    │
│  ☑ Log all AI actions                                          │    │
│                                                                 │
│  Safety                                                         │
│  Auto-execute commands (no confirmation needed):                │    │
│  ☑ Read logs, check status, get metrics                        │    │
│  ☑ Search web, read documentation                              │    │
│  ☐ Modify configuration files                                  │    │
│  ☐ Restart services                                            │    │
│  ☐ Install packages                                            │    │
│                                                                 │
│  [Save Configuration]  [Reset to Defaults]                      │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 Feature Input Panel

Users specify what features they want to optimize for. The AI uses this to research and apply configurations:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Optimization Goals                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  What features do you want to enable?                           │
│  (Check all that apply — AI will research and configure)        │
│                                                                 │
│  Speculative Decoding                                           │
│  ☐ MTP (Multi-Token Prediction) — 1.5-2x speedup on supported  │    │
│    models (DeepSeek, GLM, MiMo)                                 │    │
│  ☐ EAGLE-2/EAGLE-3 — Draft model speculative decoding          │    │
│  ☐ AutoSpec — Automatic runtime parameter tuning               │    │
│  ☐ N-gram Speculation — Lightweight, no draft model needed     │    │
│  ☐ Suffix Decoding — Dynamic speculation depth                 │    │
│                                                                 │
│  Performance Optimization                                       │
│  ☐ KV Cache Optimization — Maximize concurrent requests        │    │
│  ☐ CUDA Graph — Graph capture for small batches                │    │
│  ☐ PagedAttention — Efficient memory management                │    │
│  ☐ Chunked Prefill — Handle long prompts without OOM           │    │
│  ☐ Prefix Caching — Cache shared prompt prefixes               │    │
│                                                                 │
│  Advanced Features                                              │
│  ☐ LoRA Adapters — Multiple fine-tuned models on one base      │    │
│  ☐ Structured Output — JSON/schema-constrained generation      │    │
│  ☐ Tool Calling — Function calling support                     │    │
│  ☐ Reasoning/Thinking — DeepSeek-style reasoning mode          │    │
│  ☐ Multimodal — Vision/language models                         │    │
│                                                                 │
│  Quantization                                                   │
│  ☐ FP8 (E4M3/E5M2) — 2x throughput, minimal quality loss       │    │
│  ☐ AWQ — Activation-aware quantization                         │    │
│  ☐ GPTQ — Post-training quantization                           │    │
│  ☐ Marlin — Optimized INT4 kernel                              │    │
│                                                                 │
│  Parallelism                                                    │
│  ☐ Data Parallelism — Multiple replicas for throughput         │    │
│  ☐ Pipeline Parallelism — Split large models across GPUs       │    │
│  ☐ Expert Parallelism — MoE model optimization                 │    │
│  ☐ Prefill-Decode Disaggregation — Separate P/D workloads      │    │
│                                                                 │
│  [Research & Apply Selected Features]                           │
│                                                                 │
│  AI will:                                                       │
│  1. Search web for latest configuration guides                  │    │
│  2. Check model compatibility with selected features            │    │
│  3. Test configurations on your hardware                        │    │
│  4. Apply optimal settings with rollback capability             │    │
└─────────────────────────────────────────────────────────────────┘
```

### 12.4 MCP Server — Local Tool Interface

**New file:** `src/ai/mcp_server.rs`

Built using the official `rmcp` Rust SDK (v1.7). Exposes lllmman's capabilities as MCP tools:

```rust
// Available MCP Tools:

#[tool(description = "Execute a shell command and return stdout/stderr")]
async fn run_command(command: String, timeout_secs: Option<u32>) -> ToolOutput {
    // Executes command, captures output
    // Safety: checks against whitelist
}

#[tool(description = "Read the latest logs from a running instance")]
async fn read_logs(instance_id: u32, lines: Option<u32>) -> ToolOutput {
    // Reads from instance log buffer
}

#[tool(description = "Get current GPU status (VRAM, temp, utilization)")]
async fn check_gpu_status(gpu_index: Option<u32>) -> ToolOutput {
    // Returns GPU metrics as structured JSON
}

#[tool(description = "Apply a configuration change to an instance")]
async fn apply_config(instance_id: u32, config_changes: serde_json::Value) -> ToolOutput {
    // Modifies instance config, requires restart
}

#[tool(description = "Restart a specific instance")]
async fn restart_instance(instance_id: u32) -> ToolOutput {
    // Graceful restart
}

#[tool(description = "Run a benchmark test on the current configuration")]
async fn benchmark_config(instance_id: u32, test_type: String) -> ToolOutput {
    // Runs tok/s benchmark, returns metrics
}

#[tool(description = "Search the web for information about LLM inference")]
async fn search_web(query: String, max_results: Option<u32>) -> ToolOutput {
    // Uses configured search engine (Tavily, SerpAPI, etc.)
}

#[tool(description = "Get real-time performance metrics for all instances")]
async fn get_metrics() -> ToolOutput {
    // Returns current metrics from MetricsCollector
}

#[tool(description = "Generate a launch script for the current configuration")]
async fn generate_script(format: String) -> ToolOutput {
    // Uses BuildScriptGenerator
}

#[tool(description = "Detect GPU architectures and recommend optimal settings")]
async fn detect_hardware() -> ToolOutput {
    // Returns GPU arch info + recommended config
}

#[tool(description = "Check if a specific feature is supported by the current model and hardware")]
async fn check_feature_support(feature: String, model: String) -> ToolOutput {
    // Checks compatibility matrix
}

#[tool(description = "Get the current deployment configuration")]
async fn get_current_config() -> ToolOutput {
    // Returns full deployment profile as JSON
}
```

### 12.5 AI Agent — Planning & Execution Engine

**New file:** `src/ai/agent.rs`

The AI agent orchestrates MCP tools to accomplish tasks:

```rust
pub struct AiAgent {
    pub llm_config: LlmConfig,
    pub search_engine: Option<SearchEngineConfig>,
    pub mcp_tools: Vec<McpTool>,
    pub conversation_history: Vec<ChatMessage>,
    pub safety_policy: SafetyPolicy,
}

pub enum AiTask {
    DebugInstallError {
        provider: String,
        error_log: String,
    },
    OptimizePerformance {
        instance_id: u32,
        goals: Vec<OptimizationGoal>,
    },
    ResearchAndApply {
        features: Vec<String>,
        model: String,
        hardware: Vec<GpuArchInfo>,
    },
    DiagnoseIssue {
        symptoms: Vec<String>,
        instance_id: Option<u32>,
    },
    GenerateConfiguration {
        requirements: UserRequirements,
    },
}
```

### 12.6 AI-Powered Debug Flow

When an installation or runtime error occurs, the AI follows a structured debugging workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Debug Workflow                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Error Detected: "CUDA error: no kernel image is available"     │
│                                                                 │
│  Step 1: Analyze Error                                          │
│  ├─ Read full error log (MCP: read_logs)                        │
│  ├─ Check GPU architectures (MCP: detect_hardware)              │
│  └─ Check current config (MCP: get_current_config)              │
│                                                                 │
│  Step 2: Diagnose Root Cause                                    │
│  ├─ AI identifies: "Missing SM120 in TORCH_CUDA_ARCH_LIST"      │
│  ├─ Searches web for solution (MCP: search_web)                 │
│  └─ Finds: vLLM docs on multi-arch builds                       │
│                                                                 │
│  Step 3: Propose Fix                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Diagnosis: The vLLM binary was compiled without Blackwell │  │
│  │ (SM120) support. Your RTX 5090 requires SM120 kernels.   │  │
│  │                                                          │  │
│  │ Proposed Fix:                                             │  │
│  │ 1. Set TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0"               │  │
│  │ 2. Rebuild vLLM: pip install --no-build-isolation -e .   │  │
│  │ 3. Estimated time: 15-30 minutes                         │  │
│  │                                                          │  │
│  │ [Apply Fix]  [Show Details]  [Cancel]                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Step 4: Execute (after user confirmation)                      │
│  ├─ Sets environment variable (MCP: run_command)                │
│  ├─ Rebuilds vLLM (MCP: run_command, streams progress)         │
│  └─ Monitors build output for errors                            │
│                                                                 │
│  Step 5: Verify                                                 │
│  ├─ Checks if build succeeded                                   │
│  ├─ Restarts instance (MCP: restart_instance)                   │
│  └─ Confirms GPU is recognized (MCP: check_gpu_status)          │
│                                                                 │
│  Result: ✓ Fixed! RTX 5090 now detected and usable.             │
└─────────────────────────────────────────────────────────────────┘
```

### 12.7 AI-Powered Performance Optimization Flow

The AI acts as an **auto-tuning engineer**:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Performance Optimization                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Request: "Optimize for maximum throughput with MTP"       │
│                                                                 │
│  Phase 1: Research                                              │
│  ├─ Searches web for latest MTP guides (MCP: search_web)        │
│  ├─ Reads vLLM/SGLang MTP documentation                         │
│  ├─ Checks model compatibility (is model MTP-capable?)          │
│  └─ Finds optimal draft models for this architecture            │
│                                                                 │
│  Phase 2: Baseline Measurement                                  │
│  ├─ Records current throughput (MCP: get_metrics)               │
│  ├─ Notes current config (MCP: get_current_config)              │
│  └─ Baseline: 142 tok/s on RTX 5090                             │
│                                                                 │
│  Phase 3: Apply & Test (iterative)                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Iteration 1: Enable MTP with num_speculative_tokens=2     │  │
│  │ → Apply config → Benchmark → 198 tok/s (+39%) ✓          │  │
│  │                                                          │  │
│  │ Iteration 2: Increase to num_speculative_tokens=3         │  │
│  │ → Apply config → Benchmark → 221 tok/s (+56%) ✓          │  │
│  │                                                          │  │
│  │ Iteration 3: Increase to num_speculative_tokens=4         │  │
│  │ → Apply config → Benchmark → 215 tok/s (regression ✗)    │  │
│  │ → Rollback to iteration 2                                  │  │
│  │                                                          │  │
│  │ Iteration 4: Try mem_fraction_static=0.92                 │  │
│  │ → Apply config → Benchmark → 234 tok/s (+65%) ✓          │  │
│  │                                                          │  │
│  │ Iteration 5: Enable CUDA graph max_bs=256                 │  │
│  │ → Apply config → Benchmark → OOM ✗                        │  │
│  │ → Reduce to max_bs=128 → 238 tok/s (+68%) ✓              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Phase 4: Final Apply                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Optimization Complete!                                    │  │
│  │                                                          │  │
│  │ Before: 142 tok/s                                        │  │
│  │ After:  238 tok/s (+68% improvement)                     │  │
│  │                                                          │  │
│  │ Applied Changes:                                          │  │
│  │ • MTP enabled (num_speculative_tokens=3)                  │  │
│  │ • mem_fraction_static=0.92                                │  │
│  │ • cuda_graph_max_bs=128                                   │  │
│  │                                                          │  │
│  │ [Apply Permanently]  [Revert to Baseline]  [View Details] │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.8 AI Chat Interface (Embedded in GUI/TUI)

A persistent chat panel where users can ask questions and request actions:

**GUI Chat Panel (bottom panel, expandable):**

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Copilot                                    [⚙️] [📋] [×]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AI: I've detected your system has 3 heterogeneous GPUs.        │
│      Would you like me to configure the optimal multi-instance  │
│      setup?                                                     │
│                                                                 │
│  You: Yes, set it up for Llama-3-8B with maximum throughput    │
│                                                                 │
│  AI: I'll configure a multi-instance setup with SGLang Router.  │
│      Here's the plan:                                           │
│      • GPU 2 (5090): Instance #1, port 8080, 90% VRAM          │
│      • GPU 1 (4090): Instance #2, port 8081, 85% VRAM          │
│      • GPU 0 (3090): Instance #3, port 8082, 75% VRAM          │
│      • Router: port 30000, cache_aware policy                   │
│      I'll also enable MTP if your model supports it.            │
│      Shall I proceed?                                           │
│                                                                 │
│  You: Go ahead                                                  │
│                                                                 │
│  AI: ✓ Instance #1 started on GPU 2 (5090)                      │
│      ✓ Instance #2 started on GPU 1 (4090)                      │
│      ✓ Instance #3 started on GPU 0 (3090)                      │
│      ✓ Router started on port 30000                             │
│      All instances healthy. Total throughput: 308 tok/s         │
│                                                                 │
│  You: Can you optimize it further with speculative decoding?    │
│                                                                 │
│  AI: Checking model compatibility...                            │
│      Llama-3-8B supports EAGLE-3 speculative decoding.          │
│      I'll search for the best draft model and auto-tune...      │
│      [Searching web...] [Testing config...] [Benchmarking...]   │
│      ✓ Applied EAGLE-3 with Qwen2-1.5B-EAGLE-3 draft model      │
│      New throughput: 412 tok/s (+34% improvement)               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  [Type a message...]                              [Send] 🎤     │
└─────────────────────────────────────────────────────────────────┘
```

**TUI Chat (toggle with `c` key):**

```
┌─ AI Copilot ─────────────────────────────────────────────────────┐
│ AI: I've detected your system has 3 heterogeneous GPUs.         │
│     Would you like me to configure the optimal multi-instance   │
│     setup? (y/n)                                                │
│                                                                 │
│ > y                                                             │
│                                                                 │
│ AI: Configuring multi-instance setup...                         │
│     ✓ Instance #1 started on GPU 2 (5090)                       │
│     ✓ Instance #2 started on GPU 1 (4090)                       │
│     ✓ Instance #3 started on GPU 0 (3090)                       │
│     ✓ Router started on port 30000                              │
│     All instances healthy. Total throughput: 308 tok/s          │
│                                                                 │
│ > Can you optimize with speculative decoding?                   │
│                                                                 │
│ AI: Checking compatibility... Llama-3-8B supports EAGLE-3.      │
│     Searching for best draft model...                           │
│     Testing configuration...                                    │
│     ✓ Applied EAGLE-3. Throughput: 412 tok/s (+34%)             │
│                                                                 │
│ >                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### 12.9 AI Action Log & Audit Trail

Every AI action is logged for transparency and rollback:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Action Log                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2026-05-17 14:32:01  [DIAGNOSE]  Error: CUDA no kernel image  │
│                       Root cause: Missing SM120 in arch list    │
│                       Confidence: 95%                           │
│                                                                 │
│  2026-05-17 14:32:15  [PROPOSE]   Set TORCH_CUDA_ARCH_LIST     │
│                       = "8.6 8.9 12.0" + rebuild                │
│                       User approved: ✓                          │
│                                                                 │
│  2026-05-17 14:32:16  [EXECUTE]   export TORCH_CUDA_ARCH_LIST  │
│                       = "8.6 8.9 12.0"                          │
│                       Status: ✓ Success                         │
│                                                                 │
│  2026-05-17 14:32:17  [EXECUTE]   pip install --no-build-      │
│                       isolation -e .                            │
│                       Status: ✓ Success (18 min 32 sec)         │
│                                                                 │
│  2026-05-17 14:50:49  [VERIFY]    Restarted instance, GPU 2    │
│                       now detected                              │
│                       Status: ✓ Fixed                           │
│                                                                 │
│  2026-05-17 14:51:02  [OPTIMIZE]  Enabled MTP with             │
│                       num_speculative_tokens=3                  │
│                       Throughput: 142 → 221 tok/s (+56%)        │
│                       User approved: ✓                          │
│                                                                 │
│  [Export Log]  [Rollback Last Action]  [Clear Log]              │
└─────────────────────────────────────────────────────────────────┘
```

### 12.10 System Prompts

The AI uses specialized system prompts for different tasks:

**Debug System Prompt:**
```
You are an expert LLM inference engineer debugging a vLLM/SGLang installation issue.

System Context:
- Detected GPUs: {gpu_list}
- CUDA Driver: {cuda_version}
- Provider: {provider}
- Python: {python_version}

Error Log:
{error_log}

Your task:
1. Analyze the error and identify the root cause
2. Search for the latest solution if needed
3. Propose a specific fix with exact commands
4. After applying, verify the fix worked

Rules:
- Always explain what went wrong in simple terms
- Provide exact commands, not vague instructions
- Check for safety before proposing destructive actions
- If unsure, search the web for current best practices
```

**Optimization System Prompt:**
```
You are an expert LLM performance engineer optimizing inference throughput.

System Context:
- GPUs: {gpu_list}
- Model: {model}
- Current throughput: {current_tok_s}
- Current config: {current_config}

User Goals: {optimization_goals}

Your task:
1. Research the latest optimization techniques for this model/hardware
2. Test configurations iteratively, measuring throughput each time
3. Apply the best configuration
4. Report before/after metrics

Available techniques to consider:
- Speculative decoding (MTP, EAGLE, draft models, N-gram)
- KV cache tuning (mem_fraction_static, chunked_prefill)
- CUDA graph optimization
- Quantization (FP8, AWQ, GPTQ)
- Parallelism strategies

Rules:
- Always benchmark before and after each change
- Rollback if a change degrades performance
- Keep a log of all tested configurations
- Prioritize stability over marginal gains
```

### 12.11 AI Copilot File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/ai/mod.rs` | **NEW** | AI module exports |
| `src/ai/config.rs` | **NEW** | LLM config, search engine config, safety policy |
| `src/ai/agent.rs` | **NEW** | AI agent with task planning and execution |
| `src/ai/mcp_server.rs` | **NEW** | MCP server exposing lllmman tools via `rmcp` |
| `src/ai/mcp_tools.rs` | **NEW** | MCP tool implementations (run_command, read_logs, etc.) |
| `src/ai/debugger.rs` | **NEW** | AI-powered error diagnosis and auto-fix |
| `src/ai/optimizer.rs` | **NEW** | AI-powered performance tuning with iterative benchmarking |
| `src/ai/search.rs` | **NEW** | Web search integration (Tavily, SerpAPI, DuckDuckGo) |
| `src/ai/chat.rs` | **NEW** | Chat interface state management |
| `src/ai/action_log.rs` | **NEW** | Audit trail for all AI actions with rollback |
| `src/ai/system_prompts.rs` | **NEW** | System prompts for different AI tasks |
| `src/gui/ai_config_panel.rs` | **NEW** | GUI panel for LLM/search configuration |
| `src/gui/feature_input_panel.rs` | **NEW** | GUI panel for selecting optimization goals |
| `src/gui/ai_chat_panel.rs` | **NEW** | Embedded AI chat interface |
| `src/gui/ai_action_log.rs` | **NEW** | AI action log viewer |
| `src/tui/ai_chat.rs` | **NEW** | TUI AI chat overlay |
| `src/services/ai_benchmark.rs` | **NEW** | Benchmark runner for AI optimization loop |
| `src/models/ai.rs` | **NEW** | AI-related data models |
| `Cargo.toml` | Modify | Upgrade `reqwest` 0.11→0.12; add `rmcp` 1.7, `async-stream`, `schemars`; add `ai` feature flag |

### 12.12 Dependencies

> **Note:** See Section 14.3 for dependency corrections vs. actual codebase state.

```toml
[dependencies]
# MCP (Model Context Protocol) — Rust SDK
# Upgrade note: verify API at v1.7 vs plan's original v0.16
rmcp = { version = "1.7", features = ["server", "client", "macros"] }

# HTTP client — UPGRADE existing reqwest 0.11 → 0.12
# Must keep "blocking" feature for existing sync code; add "stream" for AI chat
reqwest = { version = "0.12", features = ["json", "stream", "blocking"] }

# Streaming for LLM responses
async-stream = "0.3"

# For structured LLM output
schemars = "0.8"

# serde_json already present in Cargo.toml — no action needed
```

### 12.13 Safety & Security Design

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Safety Architecture                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Command Classification:                                        │
│                                                                 │
│  SAFE (auto-execute):                                           │
│  • Read logs, check status, get metrics                        │
│  • Search web, read documentation                              │
│  • Analyze configurations                                      │
│                                                                 │
│  REVIEW (show to user, one-click approve):                      │
│  • Modify configuration parameters                             │
│  • Restart services                                            │
│  • Apply benchmark results                                     │
│                                                                 │
│  CONFIRM (explicit approval with details):                      │
│  • Install/upgrade packages                                    │
│  • Delete files or directories                                 │
│  • Modify system environment variables                         │
│                                                                 │
│  BLOCKED (never auto-execute):                                  │
│  • rm -rf, format, dd                                          │
│  • sudo commands                                               │
│  • Network access outside localhost                            │
│  • API key modifications                                       │
│                                                                 │
│  Rollback System:                                               │
│  • Every config change saves a snapshot                        │
│  • "Undo last AI action" always available                      │
│  • Full config history with timestamps                         │
│  • One-click revert to any previous state                      │
│                                                                 │
│  Rate Limiting:                                                 │
│  • Max 5 AI actions per minute                                 │
│  • Max 3 consecutive failed actions before pause               │
│  • Cooldown period after destructive actions                   │
└─────────────────────────────────────────────────────────────────┘
```

### 12.14 AI Copilot User Flows

**Flow 1: Auto-Debug Installation Error**
```
1. User clicks "Install vLLM" in setup wizard
2. Build fails with "CUDA error: no kernel image is available"
3. AI automatically detects the error
4. AI reads logs, checks GPU arch, searches web
5. AI proposes: "Set TORCH_CUDA_ARCH_LIST and rebuild"
6. User clicks "Apply Fix"
7. AI executes fix, monitors progress
8. AI verifies: "Build succeeded, all GPUs detected"
```

**Flow 2: Research & Apply MTP**
```
1. User checks "MTP" in Feature Input Panel
2. User clicks "Research & Apply Selected Features"
3. AI searches web for latest MTP guides
4. AI checks if model supports MTP natively
5. AI finds optimal num_speculative_tokens via benchmarking
6. AI proposes: "Enable MTP with 3 speculative tokens (+56% throughput)"
7. User approves
8. AI applies config, restarts instance, verifies improvement
```

**Flow 3: Chat-Based Optimization**
```
1. User types in AI chat: "My 3090 instance is slow, help optimize"
2. AI reads metrics for that instance
3. AI identifies bottleneck (e.g., low KV cache utilization)
4. AI proposes specific changes
5. User approves
6. AI applies and verifies
```

---

## 13. Future Enhancements (Out of Scope)

- **MIG configuration helper** — auto-generate `nvidia-smi mig` commands to slice datacenter GPUs into uniform instances
- **Dynamic worker scaling** — add/remove instances at runtime via SGLang router API without restarting
- **Prefill-Decode disaggregation UI** — configure separate prefill/decode GPU pools with dedicated router
- **Automated GPU benchmarking** — run tok/s benchmarks on each GPU at startup to auto-tune `max-num-seqs`
- **Docker compose generator** — output `docker-compose.yml` for containerized multi-instance deployment
- **NGINX config generator** — alternative to SGLang router for users who prefer reverse proxy load balancing
- **GPU topology optimization** — parse `nvidia-smi topo -m` to place instances on GPUs with best PCIe/NVLink proximity
- **Model download integration** — download models directly from wizard with auto-placement to fastest GPU
- **Export/Import deployment profiles** — share configurations between machines
- **Remote GPU management** — manage instances on remote machines via SSH
- **Multi-agent AI debugging** — Instrumentation → Analysis → Repair agent chain (TraceCoder-style) for complex failures
- **AutoSpec integration** — leverage SGLang's built-in AutoSpec for real-time speculative decoding parameter tuning
- **AI cost estimator** — predict token/s and VRAM usage before launching, suggest optimal model/quantization
- **Continuous learning** — AI learns from past optimizations to improve future recommendations
- **Federated benchmark sharing** — anonymous community benchmark data to improve AI recommendations

---

## 14. Plan Review — Gaps & Enrichments

> **Added:** 2026-05-17 — Detailed review against actual codebase state. Identifies critical discrepancies, missing sections, and inline fixes needed before implementation begins.

### 14.1 Critical Issues (Block Implementation)

#### 14.1.1 Async Runtime Integration — Plan is Silent

**Problem:** The current codebase is **synchronous**. `main.rs` has no `#[tokio::main]`. The GUI and TUI event loops are fully synchronous. `tokio` is only used for `tokio::sync::RwLock` in the `DownloadManager`. The `rmcp` MCP server (Phase 8) and `MetricsCollector` (background polling task) **require a proper tokio runtime**, but the plan never addresses how to integrate async code into the existing synchronous architecture.

**Impact:** Without resolving this, Phase 7 (MetricsCollector) and Phase 8 (MCP server) cannot work.

**Recommended approach:**
```
Option A: Dual-runtime (RECOMMENDED)
  - Keep the main GUI/TUI event loop synchronous (egui requires it)
  - Spawn a tokio Runtime in a background thread at app startup
  - Use tokio::sync::mpsc channels to communicate between sync UI and async services
  - MetricsCollector and MCP server run on the tokio runtime
  - UI polls channels every frame for updates

  main.rs:
    let rt = tokio::runtime::Runtime::new().unwrap();
    let metrics_tx = /* channel sender */;
    rt.spawn(metrics_collector_loop(metrics_tx));
    rt.spawn(mcp_server_loop());
    // GUI/TUI runs synchronously, reads from channels

Option B: Full async migration
  - Convert main.rs to #[tokio::main]
  - Make all provider operations async
  - Requires rewriting ServerController, monitor, and all providers
  - Much larger scope, not recommended for this plan
```

**Action:** Add new Phase 0.5 or Phase 7 prerequisite: "Async Runtime Integration" — spawn tokio runtime in background thread, create channel-based bridge between sync UI and async services.

#### 14.1.2 `reqwest` Version Mismatch

**Problem:** Plan originally specified `reqwest = { version = "0.12", features = ["json", "stream"] }` in Section 12.12. But the codebase already has `reqwest = { version = "0.11", features = ["json", "blocking"] }`. Current crates.io shows `reqwest = "0.13.3"`.

**Impact:** 
- 0.11 → 0.12 is a **breaking change** (API changes, feature flag changes)
- The codebase uses `reqwest::blocking::get` in: `monitor.rs` (fetch_server_stats), `vllm/provider.rs` (get_gguf_tokenizer_info), and potentially `config_persistence.rs`
- The `stream` feature in 0.12+ is needed for AI chat streaming
- The `blocking` feature in 0.11 is used by synchronous code throughout

**Recommended approach:**
```toml
# Upgrade to reqwest 0.12 with BOTH features
reqwest = { version = "0.12", features = ["json", "stream", "blocking"] }

# Or upgrade to 0.13 (latest) with both features
reqwest = { version = "0.13", features = ["json", "stream", "blocking"] }
```

**Migration checklist:**
- `reqwest::blocking::get(url)` → `reqwest::blocking::Client::new().get(url).send()` (API may have changed)
- Test all HTTP call sites after upgrade
- The `stream` feature enables `reqwest` async streaming for AI chat responses
- Keep `blocking` feature for existing synchronous call sites until they're migrated to async

**Action:** Update Section 12.12 dependencies. Add migration checklist to Phase 8 prerequisites.

#### 14.1.3 `LlmProvider` Trait Only Supports Single Instance

**Problem:** The current `LlmProvider` trait (`src/core/provider.rs`) has:
```rust
fn start_server(&self, config: &ProviderConfig, settings: &AppSettings) -> Result<std::process::Child>;
```
This returns a single `Child`, which fundamentally only supports one instance. The `ServerController` holds one `Option<Child>`. The plan's multi-instance architecture requires managing `Vec<InstanceHandle>` with per-instance processes.

**Impact:** Phase 4 (Deployment Profiles), Phase 5 (Build Script Generator), Phase 7 (GUI/TUI Redesign) all assume multi-instance capability that doesn't exist at the trait level.

**Recommended approach:**
```rust
// Option A: Add multi-instance methods to LlmProvider trait
trait LlmProvider {
    // Existing single-instance method (backward compatible)
    fn start_server(&self, config: &ProviderConfig, settings: &AppSettings) -> Result<Child>;

    // NEW: Multi-instance support
    fn build_instance_command(&self, instance: &InstanceConfig, settings: &AppSettings) -> Result<String>;
    fn start_instance(&self, instance: &InstanceConfig, settings: &AppSettings) -> Result<Child>;
    fn build_router_command(&self, router: &RouterConfig, workers: &[String]) -> Result<String>;
    fn start_router(&self, router: &RouterConfig, workers: &[String]) -> Result<Child>;
}

// Option B: Separate InstanceManager that uses LlmProvider internally
struct InstanceManager {
    instances: Vec<InstanceHandle>,
    router: Option<RouterHandle>,
    provider: Box<dyn LlmProvider>,
}
impl InstanceManager {
    fn launch_instance(&mut self, config: &InstanceConfig, settings: &AppSettings) -> Result<u32>;
    fn launch_router(&mut self, config: &RouterConfig) -> Result<()>;
    fn stop_instance(&mut self, id: u32) -> Result<()>;
    fn stop_all(&mut self) -> Result<()>;
}
```

**Recommendation:** Option B is better — keep `LlmProvider` trait focused on building commands, and let `InstanceManager` handle process lifecycle. This preserves backward compatibility and avoids breaking the existing single-instance flow.

**Action:** Add Phase 4.5: "InstanceManager & Process Orchestration" — design `InstanceManager` that wraps `LlmProvider` and manages multiple processes.

#### 14.1.4 GUI is a 2510-line Monolith — Modularization Strategy Missing

**Problem:** All GUI code lives in `src/gui/app.rs` (2510 lines). The plan says "Major Rewrite" and proposes 6 new GUI files, but doesn't explain how to break up the monolith. The `App` struct has 30+ fields, all accessed from methods throughout the file. Breaking into modules requires careful state management.

**Impact:** Phase 7 (GUI/TUI Redesign) is estimated at 5-7 days but the modularization alone could take 2-3 days of that.

**Recommended modularization strategy:**
```
Phase 7a: Extract state into sub-structs (no behavior change)
  App {
      model_panel: ModelPanelState,     // models, search, download
      server_panel: ServerPanelState,   // server_config, provider_settings
      instance_manager: InstanceManager, // NEW: multi-instance state
      gpu_panel: GpuPanelState,         // gpus, gpu_usage
      log_panel: LogPanelState,         // log_buffer
      ai_state: AiState,               // NEW: AI copilot state
      settings: AppSettings,
  }

Phase 7b: Move rendering into separate modules
  gui/app.rs          → App struct definition + update() + event handling
  gui/model_panel.rs  → render_model_panel()
  gui/server_panel.rs → render_server_panel()
  gui/instance_panel.rs → render_instance_manager()
  gui/gpu_topology.rs  → render_gpu_topology_panel()
  gui/perf_monitor.rs  → render_performance_monitor()
  gui/deployment_wizard.rs → render_deployment_wizard()
  gui/ai_chat.rs       → render_ai_chat_panel()

Phase 7c: Add new functionality incrementally
  - Add GPU topology panel (right sidebar)
  - Add instance manager tab
  - Add performance monitor tab
  - Add AI chat panel (bottom)
```

**Action:** Add this modularization strategy as a prerequisite sub-phase within Phase 7.

### 14.2 Type Overlaps & Reconciliation

#### 14.2.1 `MonitorStats` vs `InstanceMetrics`

**Existing (`src/models/config.rs`):**
```rust
pub struct MonitorStats {
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub tokens_per_second: f32,
    pub active_connections: u32,
    pub gpu_temperatures: Vec<f32>,
    pub gpu_vram_usage: Vec<(u32, u32)>, // (used, total)
    pub requests_total: u64,
    pub requests_failed: u64,
    pub avg_latency_ms: f32,
}
```

**Plan's proposed `InstanceMetrics`:**
```rust
pub struct InstanceMetrics {
    pub tokens_per_second: f32,
    pub queue_size: u32,
    pub cache_hit_rate: f32,
    pub vram_used_mb: u32,
    pub vram_total_mb: u32,
    pub avg_latency_ms: f32,
    pub gpu_utilization: f32,
    pub temperature_c: f32,
    pub requests_total: u64,
    pub requests_failed: u64,
}
```

**Overlap:** `vram_used_mb`, `vram_total_mb`, `tokens_per_second`, `avg_latency_ms`, `requests_total`, `requests_failed` are shared.

**Resolution:** Extend `MonitorStats` with new fields rather than creating a parallel struct. Add `queue_size`, `cache_hit_rate`, `gpu_utilization` as `Option<u32>`/`Option<f32>` fields. Rename to `InstanceMetrics` when Phase 7 lands, but keep backward compatibility via type alias:
```rust
// In Phase 4-5: extend MonitorStats
pub struct MonitorStats {
    // ... existing fields ...
    pub queue_size: Option<u32>,
    pub cache_hit_rate: Option<f32>,
    pub gpu_utilization: Option<f32>,
}

// In Phase 7: type alias for migration
pub type InstanceMetrics = MonitorStats;
```

#### 14.2.2 `ServerStatus` vs `InstanceStatus`

**Existing (`src/models/config.rs`):**
```rust
pub enum ServerStatus { Stopped, Starting, Running, Error(String) }
```

**Plan's proposed `InstanceStatus`:**
```rust
pub enum InstanceStatus { Stopped, Starting, Running, Error(String), Restarting }
```

**Resolution:** Extend `ServerStatus` with `Restarting` variant. Add type alias `InstanceStatus = ServerStatus` for Phase 7.

#### 14.2.3 `GpuAllocation` vs `DeploymentMode`

**Existing (`src/models/config.rs`):**
```rust
pub enum GpuAllocation { Single, Multi, All, VramLimit }
```

**Plan's proposed `DeploymentMode`:**
```rust
pub enum DeploymentMode { SingleGpu, DataParallel, PipelineParallel, MultiInstance }
```

**Overlap:** `GpuAllocation::Single` ≈ `DeploymentMode::SingleGpu`, `GpuAllocation::Multi` ≈ `DeploymentMode::DataParallel`.

**Resolution:** These serve different purposes — `GpuAllocation` controls which GPUs to use; `DeploymentMode` controls how they're used. Keep both, but clarify in documentation:
- `GpuAllocation` = "which GPUs?" (selection)
- `DeploymentMode` = "how to use them?" (strategy)

When `DeploymentMode::MultiInstance` is set, `GpuAllocation` should be `All` (each instance gets its own GPU via `CUDA_VISIBLE_DEVICES`).

### 14.3 Dependency Corrections

| Dependency | Plan Says | Actual Needed | Notes |
|------------|-----------|---------------|-------|
| `reqwest` | Add `0.12` with `["json", "stream"]` | **Upgrade** existing `0.11` → `0.12` with `["json", "stream", "blocking"]` | Must keep `blocking` feature for existing sync code |
| `serde_json` | Add as new | **Already present** (`1`) | Remove from "add" list; already exists |
| `rmcp` | Add `0.16` | Add `1.7` | crates.io has `rmcp = "1.7.0"` (not 0.16) |
| `async-stream` | Add `0.3` | Add `0.3` | Correct |
| `schemars` | Add `0.8` | Add `0.8` | Correct |

**Updated dependency block for Section 12.12:**
```toml
[dependencies]
# Upgrade: reqwest 0.11 → 0.12 (breaking — add "stream" + keep "blocking")
reqwest = { version = "0.12", features = ["json", "stream", "blocking"] }

# NEW: MCP (Model Context Protocol) — Rust SDK
rmcp = { version = "1.7", features = ["server", "client", "macros"] }

# NEW: Streaming for async iterators
async-stream = "0.3"

# NEW: JSON Schema generation for structured LLM output
schemars = "0.8"

# REMOVED from "add" list: serde_json (already present)
```

### 14.4 Missing Sections in Plan

#### 14.4.1 Config File Migration & Backward Compatibility

**Problem:** `AppSettings` and `ModelConfigEntry` both have `Default` impls and are serialized to disk (`~/.config/lllmman/`). Adding new fields (`deployment_mode`, `auto_detect_heterogeneous`, `deployment_profile`, `router_port`) will break deserialization of existing config files unless handled.

**Resolution — add to Phase 4:**
```rust
// Use serde defaults for new fields
#[derive(Serialize, Deserialize)]
pub struct AppSettings {
    // ... existing fields ...

    #[serde(default = "default_deployment_mode")]
    pub deployment_mode: DeploymentMode,

    #[serde(default)]
    pub auto_detect_heterogeneous: bool,
}

fn default_deployment_mode() -> DeploymentMode {
    DeploymentMode::SingleGpu // backward compatible default
}

// ModelConfigEntry new fields
#[derive(Serialize, Deserialize)]
pub struct ModelConfigEntry {
    // ... existing fields ...

    #[serde(default)]
    pub deployment_profile: Option<DeploymentProfile>,

    #[serde(default)]
    pub router_port: Option<u16>,
}
```

**Action:** Add "Config Migration" subsection to Phase 4.

#### 14.4.2 Feature Flags for Optional Components

**Problem:** The AI copilot (Phase 8) adds heavy dependencies (`rmcp`, `async-stream`, `schemars`). Not all users need AI features. The current `Cargo.toml` has `[features] default = ["gui"]` and `tui = [...]`. No AI feature flag exists.

**Resolution — add to Phase 8:**
```toml
[features]
default = ["gui"]
gui = ["eframe"]
tui = ["ratatui", "crossterm"]
ai = ["rmcp", "async-stream", "schemars"]
# ai feature is optional — AI copilot panel is hidden when disabled
```

**In code:**
```rust
// src/gui/app.rs
#[cfg(feature = "ai")]
ai_state: Option<AiState>,

#[cfg(feature = "ai")]
fn render_ai_chat_panel(&mut self, ctx: &egui::Context) { ... }

// Always show AI config button but disable it gracefully when feature is off
fn render_settings(&mut self, ctx: &egui::Context) {
    if cfg!(feature = "ai") {
        ui.button("⚙️ AI Copilot Config").clicked();
    } else {
        ui.add_enabled(false, egui::Button::new("⚙️ AI Copilot (install with --features ai)"));
    }
}
```

#### 14.4.3 Testing Strategy

**Problem:** The plan has no testing section. No unit tests, integration tests, or manual test procedures are defined. With 23-32 days of work across 9 phases, testing must be planned.

**Recommended testing approach:**

| Phase | Unit Tests | Integration Tests | Manual Tests |
|-------|-----------|-------------------|-------------|
| 1. GPU Arch Detection | `classify_performance_tier()`, `get_combined_arch_list()`, `is_heterogeneous_cluster()` | Mock `nvidia-smi` output, test parsing | Run on system with 1 GPU, 2 same GPUs, 2 different GPUs |
| 2. CUDA Compat Checker | `check_driver_compat()`, `check_torch_cuda()` | Mock driver/torch versions | Run on fresh system, system with old driver |
| 3. Provider Installer | `generate_setup_plan()` for each `InstallMode` | Mock GPU scenarios → verify generated commands | Actually install on test system |
| 4. Deployment Profiles | `auto_select_mode()`, profile serialization | Config file round-trip test | Load old config → verify defaults |
| 5. Build Script Gen | `generate_script()` output validation | Compare generated script to expected | Run generated script on test system |
| 6. Setup Wizard UI | — | — | Walk through wizard on each scenario |
| 7. GUI/TUI Redesign | — | — | Visual inspection + keyboard shortcut test |
| 8. AI Copilot | `safety_classification()`, `command_whitelist_check()` | Mock LLM responses → verify tool calls | Full debug flow + optimization flow |

**Action:** Add "Testing Strategy" section.

#### 14.4.4 Error Handling Strategy

**Problem:** The codebase uses `thiserror` + `log` + `env_logger`. The plan adds async code, streaming, and AI components that benefit from `tracing` (structured logging, span-based context, async-compatible). The plan does not discuss whether to migrate.

**Recommendation:** Do NOT migrate from `log` to `tracing` in this plan — it's a large scope-creep risk. Instead:
- Keep `log` + `env_logger` for existing code
- Use `log` macros in new AI code too (they're compatible with `tracing` subscriber if migrated later)
- Add structured error context to `ProviderError` variants for AI debugging:
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum ProviderError {
      // ... existing variants ...

      #[error("CUDA compute capability mismatch: GPU {gpu} requires SM {required_sm} but binary supports up to SM {max_sm}")]
      ComputeCapabilityMismatch {
          gpu: String,
          required_sm: String,
          max_sm: String,
          suggestion: String, // AI-readable suggestion
      },

      #[error("Heterogeneous GPU TP not supported: {gpu_list}")]
      HeterogeneousTpNotSupported {
          gpu_list: String,
          suggestion: String,
      },
  }
  ```

#### 14.4.5 AMD/Intel Compute Capability Fallback

**Problem:** Plan Phase 1 only addresses NVIDIA `nvidia-smi --query-gpu=compute_cap`. For AMD GPUs (`rocm-smi`) and Intel GPUs (sysfs), there's no compute capability equivalent. The current `gpu_detector.rs` has AMD/Intel fallbacks that return `total_vram_mb: 0` for VRAM.

**Resolution — add to Phase 1:**
```rust
// For AMD GPUs: gfx architecture version (not SM, but analogous)
// rocm-smi --showproductname returns "gfx900", "gfx90a", "gfx1030", "gfx1100", "gfx1200"
// These map to AMD compute capability equivalents

pub enum GpuArch {
    Nvidia { compute_cap: (u32, u32) }, // e.g. (8, 6)
    Amd { gfx_arch: String },           // e.g. "gfx90a"
    Intel { pci_id: String },           // e.g. "0x7d55"
    Unknown,
}

// GpuTier classification for AMD:
// gfx900/ggfx906 = Low (GCN5, similar to SM70)
// gfx90a/gfx940 = High/Ultra (CDNA2/3, similar to SM90)
// gfx1030/gfx1100 = Mid/High (RDNA2/3, similar to SM86/89)
// gfx1200 = High (RDNA4, similar to SM120)
```

**Action:** Add AMD/Intel compute capability detection to Phase 1.

### 14.5 Inline Corrections

#### 14.5.1 GpuTier Classification Ambiguity

**Problem:** Plan Section 1 defines `GpuTier` as:
```
Low:  SM < 8.0, < 12GB VRAM
Mid:  SM 8.x, 12-24GB VRAM
High: SM 8.9/9.0, 24-48GB VRAM
Ultra: SM 9.0/10.0/12.0, 48GB+ VRAM
```
But Section 3.4 lists H100 (SM 9.0, 80GB) as "Ultra" while the formula says SM 9.0 is "High" unless 48GB+. And RTX 4090 (SM 8.9, 24GB) fits both "Mid" and "High" — 24GB is at the boundary of 12-24GB and 24-48GB.

**Resolution — use primary key = SM version, secondary key = VRAM:**
```rust
fn classify_performance_tier(sm: (u32, u32), vram_mb: u32) -> GpuTier {
    match sm {
        (major, _) if major < 8 => GpuTier::Low,
        (8, minor) if minor <= 6 => GpuTier::Mid,   // 8.0, 8.6 — Ampere
        (8, 8..=9) => {
            // SM 8.9 — Ada Lovelace. Tier depends on VRAM.
            if vram_mb >= 24 * 1024 { GpuTier::High }  // 4090, L40S
            else { GpuTier::Mid }                        // 4080 (16GB)
        }
        (9, 0) => GpuTier::Ultra,   // H100 Hopper
        (10, _) => GpuTier::Ultra,  // future
        (12, 0) => {
            // Blackwell. Tier depends on VRAM.
            if vram_mb >= 48 * 1024 { GpuTier::Ultra }  // 5090 (32GB), RTX Pro 6000 (96GB)
            else { GpuTier::High }                        // hypothetical 16GB Blackwell
        }
        _ => GpuTier::Low,
    }
}
```

**Note:** RTX 5090 (32GB) — should this be High or Ultra? With 32GB it's below 48GB threshold but it's the newest architecture. Recommend: **SM 12.0 is always Ultra regardless of VRAM**, since it's the latest gen. Adjust formula above.

#### 14.5.2 vLLM V1 Flag Version Sensitivity

**Problem:** Plan Phase 3 lists `VLLM_USE_V1=0` as a safety flag for heterogeneous setups. However:
- vLLM V1 is the **default** since v0.8.0 (released late 2025)
- V1 has significant performance improvements
- Setting `VLLM_USE_V1=0` disables V1 and may cause regressions on newer vLLM versions
- The V1 architecture handles heterogeneous GPUs differently than V0

**Resolution:** Make this version-conditional:
```rust
fn vllm_safety_flags(version: &str, is_heterogeneous: bool) -> Vec<String> {
    let mut flags = vec!["VLLM_SKIP_P2P_CHECK=1".to_string()];

    if is_heterogeneous {
        flags.push("--enforce-eager".to_string());

        // VLLM_USE_V1=0 only for vLLM < 0.8.0
        // V1 (0.8.0+) handles eager mode natively
        if version < "0.8.0" {
            flags.push("VLLM_USE_V1=0".to_string());
        }
    }

    flags
}
```

#### 14.5.3 rmcp Version Correction

**Problem:** Plan originally specified `rmcp = { version = "0.16", features = [...] }` in Section 12.12. The actual crate on crates.io is at **version 1.7.0** (checked 2026-05-17). The plan referenced "v1.6.0" in the description text.

**Resolution:**
```toml
rmcp = { version = "1.7", features = ["server", "client", "macros"] }
```

**Also verify:** The `rmcp` API at v1.7 may differ from v0.16 (which may never have existed). The `#[tool]` macro syntax, server builder patterns, and transport handling should be re-verified against v1.7 docs before implementation.

### 14.6 Additional Missing Considerations

#### 14.6.1 Process Lifecycle Management

The current `ServerController` has a robust start/stop/health-check pattern but for a single process only. The plan needs to address:

- **Port conflict detection:** Before launching instance on port 8080, check if port is already in use
- **Process cleanup on crash:** If lllmman crashes, orphaned vLLM/SGLang processes keep running. Need PID file tracking and cleanup on startup
- **Graceful shutdown order:** Router must stop AFTER workers (otherwise workers have no router to deregister from). Or: stop workers first, then router.
- **Zombie process prevention:** `Child::wait()` must be called for all spawned processes

**Add to Phase 4.5 (InstanceManager):**
```rust
impl InstanceManager {
    fn check_port_available(port: u16) -> bool;
    fn write_pid_file(&self, path: &Path);
    fn cleanup_orphans(&mut self); // on startup, kill stale PIDs
    fn graceful_shutdown_order(&self) -> Vec<u32>; // worker IDs first, then router
}
```

#### 14.6.2 SGLang Binary Discovery Inconsistency

**Problem:** VllmProvider hardcodes `"vllm"` as the binary name. SglangProvider uses `which sglang` to dynamically find the binary. This inconsistency means:
- If vLLM is installed in a venv but not on PATH, it fails
- If SGLang is installed in a venv, `which sglang` fails but the user has configured `binary_path`

**Resolution:** Both providers should follow the same discovery pattern:
```
1. Use explicit binary_path if configured
2. Try venv binary_path if venv is configured
3. Fall back to `which <provider_name>`
4. Fall back to hardcoded binary name
```

#### 14.6.3 Router Health Monitoring

The plan's `MetricsCollector` polls `/stats` on each instance but doesn't mention monitoring the **router itself**. The SGLang router exposes:
- `http://router:port/health` — basic liveness
- `http://router:port/get_worker_info` — connected workers list
- Router metrics (QPS, latency distribution, routing decisions)

**Add to Section 8.7 Metrics Collection:**
```
Every 2 seconds:
  1-3. Poll instance /stats endpoints
  4. Poll router /health + /get_worker_info
  5. If worker count < expected, flag warning
  6. Query nvidia-smi for GPU temps/utilization
  7. Update shared state
```

#### 14.6.4 Model Compatibility Matrix

The AI copilot's `check_feature_support` tool (Section 12.4) needs a model compatibility matrix, but the plan doesn't define one. Critical for:
- MTP: Only works with models trained for multi-token prediction (DeepSeek-V3, GLM-4, MiMo)
- EAGLE: Requires specific EAGLE draft model for the base model
- Structured output: vLLM vs SGLang have different implementations
- FP8: Requires GPU with FP8 support (SM 8.9+ for hardware, or SM 9.0+ for native)

**Add to Phase 8:**
```rust
pub struct FeatureCompatMatrix {
    pub feature: String,
    pub min_sm_version: Option<(u32, u32)>,
    pub min_provider_version: HashMap<String, String>, // "vllm" -> "0.6.0"
    pub model_requirements: Vec<String>, // "mtp_capable", "eagle_draft_available"
    pub notes: String,
}
```

### 14.7 Section Numbering Fixes

The plan has inconsistent subsection numbering:
- Section 8 (GUI/TUI Redesign) uses subsections 7.1 through 7.8 — should be 8.1 through 8.8
- Section 12 (AI Copilot) uses subsections 11.1 through 11.14 — should be 12.1 through 12.14

These should be corrected before implementation to avoid confusion.

### 14.8 Summary of Required Plan Updates

> **Status:** All items below have been applied inline. Section references reflect the corrected numbering.

| Priority | Update | Section Affected | Status |
|----------|--------|------------------|--------|
| **CRITICAL** | Add async runtime integration strategy (Phase 0.5) | Section 14.1.1 | Documented |
| **CRITICAL** | Fix reqwest version (0.11 → 0.12 with blocking+stream) | Section 12.12, Section 6 | Applied |
| **CRITICAL** | Add InstanceManager design (wraps LlmProvider) | Section 14.1.3 | Documented |
| **CRITICAL** | Add GUI modularization strategy (3-step approach) | Section 14.1.4 | Documented |
| **HIGH** | Reconcile MonitorStats/InstanceMetrics overlap | Section 14.2.1 | Documented |
| **HIGH** | Reconcile ServerStatus/InstanceStatus overlap | Section 14.2.2 | Documented |
| **HIGH** | Reconcile GpuAllocation/DeploymentMode overlap | Section 14.2.3 | Documented |
| **HIGH** | Add config migration strategy (serde defaults) | Section 14.4.1 | Documented |
| **HIGH** | Add feature flags (`ai` feature) | Section 14.4.2 | Documented |
| **HIGH** | Add testing strategy section | Section 14.4.3 | Documented |
| **HIGH** | Add error handling strategy (keep log, extend ProviderError) | Section 14.4.4 | Documented |
| **MEDIUM** | Fix GpuTier classification logic (SM 12.0 = always Ultra) | Section 14.5.1 | Documented |
| **MEDIUM** | Add AMD/Intel compute capability detection | Section 14.4.5 | Documented |
| **MEDIUM** | Fix rmcp version (0.16 → 1.7) | Section 12.12 | Applied |
| **MEDIUM** | Fix subsection numbering (7.x → 8.x, 11.x → 12.x) | Sections 8, 12 | Applied |
| **MEDIUM** | Remove serde_json from "add" list (already present) | Section 12.12, Section 6 | Applied |
| **MEDIUM** | Add process lifecycle management details | Section 14.6.1 | Documented |
| **MEDIUM** | Add router health monitoring to MetricsCollector | Section 8.7 | Documented |
| **MEDIUM** | Add model compatibility matrix for AI copilot | Section 14.6.4 | Documented |
| **LOW** | Make VLLM_USE_V1=0 version-conditional | Section 14.5.2 | Documented |
| **LOW** | Fix SGLang binary discovery inconsistency | Section 14.6.2 | Documented |
| **LOW** | Update reqwest blocking→async migration checklist | Section 14.1.2 | Documented |
