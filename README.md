# Lunference

Lunference is a local, GGUF-first inference service and benchmark lab built to own the full stack from model loading to runtime behavior, tokenizer resolution, benchmarking, and compatibility inspection.

It is designed for local model bring-up, parity validation, and benchmark comparison with a strong focus on visibility, control, and correctness.

## Features

- Native GGUF loading and dequantization paths
- Native tokenizer resolution with GGUF-first fallback logic
- Custom runtime support for:
  - Dense models
  - MoE models
  - Hybrid Qwen variants
  - Early multimodal model paths
- OpenAI-compatible chat completions API
- Browser-based benchmark dashboard with:
  - Per-GGUF status
  - Confidence scoring
  - Saved benchmark reports
  - Live debug events
  - Compatibility metadata
  - Recommendations and warnings

## Current Support

The following model families are currently wired through Lunference's internal tokenizer and runtime resolver stack:

- Qwen3.5
- Qwen3-Coder-Next
- Qwen3-Coder-30B-A3B
- Qwen3VL
- Phi-3
- Phi-4
- GLM
- MiniCPM-V

## What Works

### Chat API

- FastAPI-based local chat service
- Streaming and non-streaming chat completions
- OpenAI-style request format

### Benchmark Lab

The dashboard available at `/` can:

- Scan GGUF models from a target folder
- Display per-model status and confidence
- Run local benchmark sweeps with adjustable generation settings
- Save benchmark reports to `models\benchmarks`
- Browse previous benchmark history
- Inspect:
  - Debug warnings
  - Recommendations
  - Raw benchmark metrics
  - GGUF compatibility metadata

## Quick Start

From this folder, run:

```powershell
python -m lunference.main --host 127.0.0.1 --port 8080
