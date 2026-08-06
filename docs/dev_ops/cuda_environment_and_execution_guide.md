# CUDA Environment & Execution Guide

This document summarizes the verified environment, CUDA extension build instructions, verified training commands, and codebase extension points for adapting model architectures in `modalities`.

---

## 1. Environment & Hardware Specifications

- **OS / Host**: Linux (`dgx2`)
- **GPUs**: 8x NVIDIA A100-SXM4-80GB (Ampere Architecture, Compute Capability `8.0` / `sm_80`)
- **CUDA Version**: 12.8 (`nvcc` release 12.8)
- **Python**: 3.11 (virtual environment at `.venv`)
- **Core Dependencies**:
  - `torch==2.10.0+cu128`
  - `torchvision==0.25.0+cu128`
  - `flash-attn==2.8.3`
  - `mamba-ssm==2.2.5`
  - `causal-conv1d==1.5.0.post8`
  - `modalities==0.5.0` (installed in editable mode from workspace root)

---

## 2. CUDA Extensions & Build Environment Instructions

### Why Custom Build Flags are Required
1. **Prebuilt Wheel 404s**: PyPI / GitHub releases do not host prebuilt binary wheels for PyTorch 2.10 + CUDA 12.8. Setup scripts fall back to building C++/CUDA extensions from source.
2. **Resource Limits & OOM**: By default, `flash-attn`, `mamba-ssm`, and `causal-conv1d` attempt to compile kernels for all CUDA architectures (`sm_80`, `sm_90`, `sm_100`, `sm_120`) using `ninja -j 128`. This spawns 128 concurrent `nvcc` instances, causing host RAM OOM and Ninja exit code 255.
3. **Build Flags Solution**:
   - `TORCH_CUDA_ARCH_LIST="8.0"` restricts `nvcc` to compile ONLY `sm_80` kernels for A100 GPUs (reducing compile workload by ~75%).
   - `MAX_JOBS=8` limits Ninja workers to 8 concurrent processes.
   - `--config-file /dev/null` prevents `uv` from triggering project build dependency overrides during single-package installs.

### Reproducible Rebuild Commands

```bash
source .venv/bin/activate

# 1. Sync primary dependencies and build flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 uv sync --extra cu128

# 2. Install setuptools and wheel in venv
uv pip install --config-file /dev/null setuptools wheel

# 3. Install Mamba-2 fused kernel extras (required for ssd_backend="fused")
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 uv pip install --config-file /dev/null --no-build-isolation --no-deps mamba-ssm==2.2.5
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8 uv pip install --config-file /dev/null --no-build-isolation --no-deps "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8"
```

---

## 3. Verified Execution Command

To launch training on a single GPU (e.g., GPU index 5) using FSDP2 / torchrun:

```bash
source .venv/bin/activate
mkdir -p /home/markus_frey/Github/modalities/results

CUDA_VISIBLE_DEVICES=5 torchrun \
  --rdzv-endpoint localhost:1235 \
  --nnodes 1 \
  --nproc_per_node 1 \
  $(which modalities) run \
  --experiments_root_path /home/markus_frey/Github/modalities/results \
  --config_file_path config_files/nemotron/config_fineweb_nemotron_nano_fsdp2_1gpu.yaml
```

---

## 4. Codebase Architecture & Extension Points

When adapting or introducing new model architectures, refer to the following codebase layout:

### Key Directories & Files
- **Main CLI & Gym Entry Points**:
  - `src/modalities/__main__.py` — Click CLI entry points (`modalities run`).
  - `src/modalities/main.py` — Component resolution and execution orchestrator.
  - `src/modalities/gym.py` & `trainer.py` — Training loop execution and loss backward pass.

- **Component & Model Instantiation Factory**:
  - `src/modalities/config/component_factory.py` — Resolves YAML config definitions into Python object instances using standard `component_key` / `variant_key` mappings.

- **Model Implementations**:
  - `src/modalities/models/` — Directory containing model implementations.
  - **Nemotron / Hybrid Architecture**:
    - `src/modalities/models/nemotron/nemotron_model_factory.py`
    - `src/modalities/models/nemotron/nemotron_model.py`
    - `src/modalities/models/nemotron/nemotron_layer_specs.py`
  - **Mamba / State Space Models**:
    - `src/modalities/models/components/mamba2/mamba2_mixer.py` — Implements `ssd_backend="fused"` vs `ssd_backend="native"`.

- **Configuration Files**:
  - `config_files/nemotron/` — YAML configs for Nemotron hybrid architectures.

---

## 5. Next Steps for Architectural Adaptation

When modifying or extending model architectures:
1. **Adding/Modifying Layer Blocks**: Add or update layer spec classes in `src/modalities/models/<model_name>/` or `components/`.
2. **Config Schema**: Expose new parameters in pydantic/yaml schema under `config_files/`.
3. **Component Registration**: Ensure any new model variants are registered in `component_factory.py` or the corresponding `model_factory`.
