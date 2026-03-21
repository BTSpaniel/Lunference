# 🌙 Lunference

**A minimal, high-performance LLM inference engine. 404 lines. One file. Zero bloat.**

Lunference is a from-scratch Python inference engine built around a single philosophy: **own every layer**. No black boxes, no framework magic, no config hell. Just PyTorch tensors, raw CUDA, and clean code you can read in an afternoon.

Built as the inference backbone for [Particle Realms Engine](https://particlerealms.online).

---

## Why Lunference?

| | llama.cpp | vLLM | Ollama | **Lunference** |
|---|---|---|---|---|
| Single file | ❌ | ❌ | ❌ | ✅ |
| Zero config | ❌ | ❌ | ❌ | ✅ |
| Readable in a day | ❌ | ❌ | ❌ | ✅ |
| GGUF support | ✅ | ❌ | ✅ | ✅ |
| Q4 KV cache | ❌ | ❌ | ❌ | ✅ |
| Sink eviction | ❌ | ❌ | ❌ | ✅ |
| OpenAI-compatible API | ✅ | ✅ | ✅ | ✅ |
| Embeddable | ❌ | ❌ | ❌ | ✅ |

---

## Features

- **GGUF native** — reads model weights directly off disk via `mmap`, no conversion needed
- **Full dequantization** — F32, F16, Q4_0, Q8_0, Q4_K, Q4_K_M, Q6_K
- **GQA attention** — Grouped Query Attention, supports Llama, Qwen, Mistral architectures
- **Q4 KV cache** — Key/Value tensors stored in int8 with per-block scales, ~4× less VRAM than F16
- **Attention sink eviction** — keeps first 4 tokens + last 512, evicts the middle. Effectively unlimited context on any VRAM budget
- **BPE tokenizer** — loaded directly from `tokenizer.json`, no HuggingFace dependency
- **Chat templates** — Llama-3 and ChatML formats auto-detected
- **OpenAI-compatible HTTP API** — streaming SSE + non-streaming, drop-in replacement for any OpenAI client
- **VRAM budget manager** — real-time tracking via `torch.cuda.memory_allocated()`
- **Multi-model support** — load/unload models at runtime via API

---

## Quickstart

```bash
pip install torch numpy aiohttp
python main.py --gguf ./qwen3-4b.gguf --tok ./qwen3-4b --port 8080
```

That's it. No install, no config file, no daemon.

---

## Usage

```bash
python main.py \
  --gguf    ./models/qwen3-4b.gguf \   # path to GGUF model file
  --tok     ./models/qwen3-4b \        # path to tokenizer directory
  --name    qwen3-4b \                 # model name (used in API calls)
  --vram-mb 2000 \                     # VRAM budget in MB
  --kv-max  8192 \                     # max KV cache tokens before eviction
  --port    8080
```

---

## API

Lunference exposes an OpenAI-compatible REST API. Any OpenAI client works out of the box.

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Explain attention sinks"}],
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### VRAM Status

```bash
curl http://localhost:8080/v1/vram
# {"used_mb": 1842.3, "free_mb": 157.7, "models": ["qwen3-4b"]}
```

### Load Model at Runtime

```bash
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"name": "coder", "gguf_path": "./qwen-coder-3b.gguf", "tokenizer_path": "./qwen-coder-3b"}'
```

### Unload Model

```bash
curl -X DELETE http://localhost:8080/v1/models/qwen3-4b
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

stream = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Plan a game engine architecture"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Supported Models

Any model with a GGUF export and a Llama/Qwen/Mistral architecture works out of the box.

| Model | Size | VRAM (Q4_K_M) | Notes |
|---|---|---|---|
| Qwen3-4B | 4B | ~2.8GB | Recommended — thinking mode, best reasoning/GB |
| Qwen2.5-Coder-3B | 3B | ~2.0GB | Best for code tasks, 2 KV heads = huge context |
| Qwen3-8B | 8B | ~5.0GB | Full reasoning model |
| Llama 3.1 8B | 8B | ~4.8GB | Broad community support |
| Mistral 7B | 7B | ~4.3GB | Fast, reliable general use |

---

## Architecture

```
main.py (404 lines)
│
├── GGUFReader          — mmap-based binary parser, reads metadata + tensor offsets
├── dequantize()        — F32/F16/Q4_0/Q8_0/Q4_K/Q6_K dequant fns
├── Tokenizer           — BPE encode/decode, chat template rendering
│
├── rms_norm()          — x / sqrt(mean(x²) + ε) * weight
├── build_rope()        — rotary position embeddings
├── gqa()               — grouped query attention, auto-expands KV heads
├── swiglu()            — SiLU(gate) * up @ down.T
│
├── Q4KVCache           — int8 KV storage + per-block scales
│   ├── _quant()        — f16 → int8 + scale
│   ├── _dequant()      — int8 + scale → f16
│   └── _evict()        — keep sink[0:4] + window[-512:], drop middle
│
├── Transformer         — N × DecoderLayer → RMSNorm → LM head
├── Sampler             — temperature · top-p · top-k · repetition penalty
├── Engine              — model registry, VRAM tracker, generate loop
└── HTTP Server         — aiohttp, OpenAI-compatible, streaming SSE
```

---

## The KV Cache

Lunference's KV cache is the main engineering differentiator.

**Standard engines** allocate `float16` tensors for every token's Key and Value vectors across all layers. At 128K context on a 7B model that's 14GB — before weights even load.

**Lunference uses Q4 KV cache + attention sink eviction:**

```
Storage:   int8 + per-block float16 scale  →  ~4× smaller than F16
Eviction:  keep tokens[0:4] + tokens[-512:]  →  effective unlimited context
```

The first 4 tokens (attention sinks) are always retained — research shows attention scores spike on early tokens regardless of content, and evicting them degrades quality significantly. Recent tokens stay hot. Everything in between gets dropped when the cache fills.

Result: a 3B model with 36KB/token at Q4 ≈ **18KB/token** → 80,000+ token context in 1.5GB of VRAM.

---

## Dependencies

```
torch     — tensor ops + CUDA backend
numpy     — GGUF binary parsing + dequant
aiohttp   — async HTTP server
```

That's it. No transformers, no llama-cpp-python, no sentencepiece.

---

## Roadmap

- [ ] Q3_K / Q5_K dequant
- [ ] Flash attention kernel
- [ ] Multi-request continuous batching
- [ ] DeepSeek MLA attention
- [ ] MoE routing (Mixtral / Qwen MoE)
- [ ] Speculative decoding
- [ ] WebSocket streaming
- [ ] Browser client via Particle Realms Engine

---

## License

MIT — do whatever you want with it.

---

*Lunference is part of the [Particle Realms](https://particlerealms.online) ecosystem.*
