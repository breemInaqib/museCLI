# Local GPT‑OSS‑20B Deployment Guide

## Overview
GPT‑OSS‑20B is an open‑weight, 21‑billion‑parameter model from OpenAI designed to run on consumer hardware. Its Mixture‑of‑Experts architecture and aggressive MXFP4 quantization mean that the weights fit in around 12–14 GB of VRAM. With the right tuning you can serve it locally for private inference, experimentation or building tools like museCLI.

## Hardware Requirements
| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | ≥ 16 GB (4‑bit weights) | 24 GB+ for comfortable context sizes and offloading |
| System RAM | ≥ 16 GB (for CPU offload) | 32‑64 GB; more RAM helps when offloading KV cache or running other workloads |
| Storage | ~20 GB free for model weights and cache | NVMe SSD for faster load times |
| OS | 64‑bit Linux (Ubuntu 22.04/24.04), macOS 13+, or Windows via WSL2 | Linux with recent CUDA drivers for vLLM |

Recommended GPUs include NVIDIA RTX 4090, RTX 3090/3090 Ti or RTX 4080, which provide 16‑24 GB of GDDR6X memory and high bandwidth. AMD RX 7900 XTX/XT with ≥ 20 GB VRAM are also viable. Lower‑end cards or integrated GPUs will force heavy CPU offload and significantly reduce throughput.

## Operating System Setup
1. **Install a 64‑bit OS:** Ubuntu 22.04/24.04 LTS is recommended. macOS users can run via LM Studio or Ollama; Windows users should enable WSL2 with a recent Ubuntu image.
2. **Install GPU Drivers:** Ensure the NVIDIA driver supports CUDA 12.1+ (or the driver recommended by vLLM). On AMD, install ROCm ≥ 5.7.
3. **Install Python:** Use Python 3.10 or newer. Create a virtual environment for isolation:
   ```bash
   python3 -m venv ~/gpt-oss-env
   source ~/gpt-oss-env/bin/activate
   pip install --upgrade pip
   ```
4. **Install CUDA toolkit (optional):** Some instructions require a specific CUDA version (e.g., 12.8 for Ada GPUs). Follow the vLLM recipe if using an Ada Lovelace card.

## Quantization Options
GPT‑OSS‑20B is trained with MXFP4 (4‑bit) weights; this is the default when you pull the model from Hugging Face or via LM Studio. If you need a compromise between quality and memory, community builds may provide Q5 (5‑bit) or Q6 quantizations via GGUF/GPTQ. Lower bit‑widths reduce VRAM usage but slightly decrease model fidelity. Use Q4 on GPUs with 16‑20 GB of VRAM; use Q5 if you have ≥ 24 GB VRAM and want improved accuracy. When using vLLM, MXFP4 is currently required on Hopper and Blackwell GPUs; alternatives like AWQ/GPTQ are experimental.

### KV Cache Tuning
During generation, the key‑value (KV) cache grows with the context length and can become a bottleneck. Strategies:
- **Quantize the KV cache:** Tools like Unsloth allow quantizing the KV cache to 4 bits to reduce VRAM usage and memory transfers, improving throughput.
- **Reduce context length:** Limit `--max-model-len` or `--max-num-seqs` to fit within GPU memory.
- **Hybrid cache manager:** vLLM implements a hybrid KV cache allocator that shares memory between full attention and sliding‑window layers, reducing fragmentation.
- **CPU offload:** On GPUs with 16 GB VRAM you may offload part of the KV cache to system RAM, at the cost of latency.

## Dependencies

| Dependency | Purpose |
|---|---|
| **vLLM ≥ 0.10.2** | High‑throughput inference engine supporting GPT‑OSS models; provides PagedAttention, hybrid KV cache, and asynchronous scheduling. |
| **flashinfer** | Optional acceleration library providing custom MXFP4 MoE and attention kernels for Hopper/Blackwell GPUs. |
| **LM Studio or Ollama** | GUI tools to download and run GPT‑OSS locally without manual setup (suitable for macOS and Windows). |
| **Harmony / transformers** | Tokenization library required for proper chat formatting; installed automatically by vLLM. |
| **Python packages:** `torch`, `uv`, `accelerate`, `huggingface_hub` (handled automatically by vLLM). |

## Installation Steps (vLLM)

1. **Create a Python environment** (see OS setup above).
2. **Install vLLM and optional flashinfer**:
   ```bash
   # Use uv for faster installations (optional)
   pip install uv

   # Install vLLM with the appropriate backend; specify flashinfer for Blackwell GPUs
   uv pip install vllm==0.10.2 --torch-backend=auto
   # For Blackwell GPUs (B200), include FlashInfer kernels
   uv pip install "vllm[flashinfer]==0.10.2" --torch-backend=auto
   # Install flashinfer separately if using Ada GPUs
   uv pip install flashinfer-python==0.2.10
   ```
3. **Download model weights**:
   ```bash
   # using huggingface CLI (requires HF token for large downloads)
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/gpt-oss-20b', local_dir='models/gpt-oss-20b', allow_patterns=['*.safetensors','*.json'])"
   ```
   Alternatively, LM Studio or Ollama can download the model with one click and handle quantization.
4. **Run the server**:
   ```bash
   # Serve GPT-OSS-20B on port 8000 with asynchronous scheduling
   python -m vllm.entrypoints.openai.api_server \
     --model openai/gpt-oss-20b \
     --port 8000 \
     --max-model-len 128000 \
     --dtype auto \
     --tensor-parallel-size 1 \
     --async-scheduling
   ```
   Adjust `--tensor-parallel-size` to the number of GPUs you have (e.g., 2 or 4). For Hopper/Blackwell GPUs, the MXFP4 kernels are enabled by default.

5. **Test the API**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-..." \
     -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hello!"}]}'
   ```

## Sample Run Script

Below is a simple Bash script that installs vLLM, downloads GPT OSS‑20B, and starts a local server. Modify the environment variables as needed.

```bash
#!/usr/bin/env bash

set -e

# Create and activate virtual environment
python3 -m venv ~/gpt-oss-env
source ~/gpt-oss-env/bin/activate
pip install --upgrade pip uv

# Install vLLM and flashinfer (optional)
uv pip install vllm==0.10.2 --torch-backend=auto
uv pip install flashinfer-python==0.2.10

# Download model weights (requires HF token)
export HF_HOME=~/hf-cache
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/gpt-oss-20b', local_dir='~/models/gpt-oss-20b', allow_patterns=['*.safetensors','*.json'])"

# Serve the model with asynchronous scheduling
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --port 8000 \
  --max-model-len 128000 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --async-scheduling
```

## Linking from the Main Documentation

Once this file is committed, link it in the main `README.md` or documentation by adding a line such as:

```
- [Local GPT OSS‑20B Deployment Guide](./LOCAL-OSS-DEPLOY.md)  
```

This ensures users can easily find these instructions.

---

This guide consolidates the key steps and best practices for running GPT‑OSS‑20B on your own machine, with attention to hardware suitability, quantization trade‑offs, KV cache tuning, and recommended dependencies. Use it as a starting point for integrating the model into your projects, automation scripts, or experiments.
