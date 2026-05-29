# Jarvina — Voice Assistant for JarvisLabs

## What it does

JarvisLabs currently has no voice interface on its website — users must read through docs, pricing pages, and FAQs to find answers. Jarvina is an always-on voice assistant embedded in the JarvisLabs site that lets users ask questions out loud and hear accurate, grounded answers back within seconds. It handles English, Hindi, and Hinglish natively, covering GPU pricing, instance setup, SSH access, framework configuration, and general platform navigation.

---

## Why I built this

I ran into this problem myself. The first time I tried to spin up a GPU instance on JarvisLabs, I spent more time hunting through tabs than actually doing work — I wasn't sure which GPU fit my budget, how storage worked, or how to SSH in correctly. The documentation exists but navigating it takes effort. When I saw this problem again from the outside while thinking about what to build, it felt like an obvious gap: a platform built for developers running AI workloads should have an AI-native way to answer questions about itself. That made this worth building.

---

## How to run it

### Prerequisites
- JarvisLabs A100 40GB instance (or equivalent)
- Python 3.10, CUDA 12+
- ~30GB VRAM free before starting

### 1. Clone and install

```bash
git clone <repo-url> ~/jarvis-assignment
cd ~/jarvis-assignment
pip install -r requirements.txt
pip install kokoro>=0.9.4
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_MODEL=Qwen/Qwen3-32B-AWQ
VLLM_BASE_URL=http://localhost:8001/v1
TTS_BACKEND=kokoro
TTS_VOICE=af_heart
HOST=0.0.0.0
PORT=6006
LLM_MAX_TOKENS=100
TOP_K=3
```

### 3. Start vLLM (LLM backend)

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --port 8001 &
```

Wait for `Application startup complete` before proceeding.

### 4. Build the knowledge base (first run only)

```bash
python scripts/build_kb.py
```

This scrapes `jarvislabs.ai`, chunks and embeds with BGE-M3, and stores in ChromaDB (~702 chunks).

### 5. Start the API server

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 6006
```

### 6. Open in browser

Navigate to `http://<your-instance-ip>:6006` — click the mic button, speak, and hear a response.

> **JarvisLabs port note:** When using JarvisLabs API endpoints, set your endpoint to forward to port `6006`. The public URL will look like `https://<instance-id>.notebooksn.jarvislabs.net/`.

---

## Architecture decisions

### Qwen3-32B-AWQ as the LLM

Most open-weight models that handle Hindi natively are large (30B+). Qwen3-32B-AWQ at 4-bit quantization fits comfortably in 18GB of VRAM, leaving room for ASR and embeddings on the same A100. It also supports English/Hindi/Hinglish without any fine-tuning. I disabled chain-of-thought reasoning (`enable_thinking: false`) because `<think>` tokens sent to TTS produce gibberish audio — a non-obvious failure mode that only shows up at runtime.

### Kokoro-82M for TTS instead of a cloud service

Early versions used Edge TTS (Microsoft cloud). It worked but added ~1 second of network latency per phrase. Kokoro-82M runs locally on the GPU at ~150–200ms per phrase with no network round trip and no API key. The trade-off is English-only voices; Hindi output falls back to English pronunciation, which is acceptable for a Hinglish-heavy user base.

### Sentence-level TTS streaming, not full-response

Waiting for the LLM to finish before sending anything to TTS adds the full generation time to perceived latency. Instead, the orchestrator accumulates LLM tokens into a buffer and fires TTS as soon as a sentence boundary is found or ~100 characters accumulate. The first audio chunk reaches the browser within ~700ms of the LLM producing its first sentence, while generation continues in the background.

### Serial TTS queue instead of concurrent tasks

Kokoro's `KPipeline` is not thread-safe — calling it from multiple threads simultaneously corrupts output. A single async worker drains phrases from a queue one at a time. This also guarantees audio arrives at the browser in the correct order, which concurrent tasks cannot.

### VAD gating before ASR

Parakeet RNNT is always loaded on GPU, but only invoked when silero-VAD confirms a complete utterance (700ms silence after speech). Without VAD, ASR would run on every microphone noise. VAD adds ~0ms overhead since it runs on CPU in under 5ms per chunk.

### BGE-M3 for retrieval embeddings

BGE-M3 supports dense retrieval across multilingual text, which matters because JarvisLabs docs mix English and transliterated Hindi. Alternatives like `text-embedding-ada-002` require an API call per query; BGE-M3 runs locally and returns embeddings in ~32ms warm.

---

## What I used AI for

**Used AI for:**
- Initial instance setup and vLLM launch commands — I had not previously configured cloud GPU servers from scratch and used AI to generate the correct flags and environment variables
- Boilerplate wiring between pipeline stages (VAD → ASR → RAG → LLM → TTS) — the async task structure and WebSocket protocol were AI-generated
- Debugging non-obvious failures: Parakeet returning `Hypothesis` objects instead of strings, Qwen3 emitting `<think>` tokens in voice output, WebSocket requiring `wss://` behind an HTTPS proxy

**Done manually:**
- All architecture decisions: model selection, TTS backend evaluation (ruled out Voxtral, Fish Speech, and Edge TTS through hands-on testing on the actual instance)
- Parameter tuning: `gpu-memory-utilization`, `LLM_MAX_TOKENS`, VAD thresholds, chunk sizes — verified against real VRAM usage and response quality
- API and connection verification: manually tested each stage end-to-end before wiring the next one in
- Knowledge base scope: decided which JarvisLabs pages to scrape and what chunk size preserved enough context
- Override: AI initially suggested concurrent TTS tasks; I identified the thread-safety issue from garbled audio in logs and switched to a serial queue

---

## Measured latency

Measured on a JarvisLabs A100 40GB instance, warm (all models loaded):

| Stage | Latency |
|-------|---------|
| ASR (Parakeet RNNT 1.1B) | 129 ms |
| RAG (BGE-M3 + ChromaDB, 702 chunks) | 32 ms |
| LLM (Qwen3-32B-AWQ, ~100 tokens) | 22,446 ms |
| TTS first chunk (Kokoro-82M) | 200 ms |
| **End-to-end** | **~22,800 ms** |

**What reduced latency:**
- Sentence-level TTS streaming: user hears the first sentence ~700ms after LLM starts generating, not after it finishes
- `LLM_MAX_TOKENS=100` keeps responses short and generation fast
- Kokoro local TTS eliminated the ~1s/phrase Edge TTS network round trip
- BGE-M3 warm retrieval at 32ms vs ~5s cold (lazy load amortized after first query)

The dominant bottleneck is LLM generation at ~22 seconds. This is a function of model size; a smaller or faster model is the primary lever.

---

## Sample interaction

**User says:** *"Tell me about available GPUs I can purchase on JarvisLabs"*

**Expected response:**
> JarvisLabs offers several high-performance NVIDIA GPUs for rent, including H200, H100, A100, A6000, A5000, and L4. These GPUs are available in regions like IN2 and EU1. You can check current pricing and availability via the JarvisLabs dashboard or the SDK. Let me know if you'd like details on a specific GPU!

---

## What I would change with 4 more weeks

**1. Cut LLM latency with a smaller distilled model**
The 22-second LLM time dominates everything else. I would fine-tune or use a smaller model (7B–14B) specifically on JarvisLabs docs and FAQs. A well-tuned 7B model would generate in ~3–4 seconds and stay on topic, making the conversation feel genuinely real-time.

**2. Richer, more verifiable answers**
Right now the LLM can paraphrase or hallucinate pricing and GPU specs. I would add structured data — JSON pricing tables, GPU spec sheets — alongside scraped text in the knowledge base, and instruct the model to cite specific values. This makes wrong answers easier to catch and correct.

**3. Streaming TTS audio output**
Kokoro synthesizes an entire phrase before sending any audio. A streaming-capable TTS model would start sending audio within 50ms of synthesis beginning, cutting per-phrase TTS latency from ~200ms to near-zero perceived gap and making the assistant feel significantly more responsive.

**4. Native Hindi voice**
Kokoro only has English voices, so Hindi and Hinglish responses are pronounced with an English accent. Adding a lightweight Hindi TTS model (a fine-tuned VITS on an Indian English/Hindi corpus) as a fallback when Devanagari or strong Hindi phrasing is detected would make the assistant feel native to Indian users, which is the core JarvisLabs audience.

---

## Stack

| Component | Model / Library |
|-----------|----------------|
| ASR | `nvidia/parakeet-rnnt-1.1b` via NeMo |
| VAD | `silero-vad v5` |
| Embeddings | `BAAI/bge-m3` via FlagEmbedding |
| Vector DB | ChromaDB (persistent, local) |
| LLM | `Qwen/Qwen3-32B-AWQ` via vLLM |
| TTS | `hexgrad/Kokoro-82M`, voice `af_heart` |
| Backend | FastAPI + WebSocket |
| Frontend | Vanilla JS + Web Audio API + AudioWorklet |
