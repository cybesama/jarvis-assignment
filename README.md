# Jarvina — JarvisLabs Voice Assistant

Real-time multilingual voice assistant for JarvisLabs GPU cloud. Ask anything about instances, pricing, setup, or troubleshooting — in English, Hindi, or Hinglish.

**Pipeline:** Mic → VAD → ASR (Parakeet RNNT 1.1B) → RAG (BGE-M3 + ChromaDB) → LLM (Sarvam-30B) → TTS (Voxtral) → Speaker

---

## Architecture

```
Browser (mic)
  │  PCM Int16 16kHz chunks via WebSocket
  ▼
FastAPI WebSocket Server
  ├─► silero-vad v5              end-of-speech detection
  ├─► Parakeet RNNT 1.1B (NeMo) speech → text (~250ms)
  ├─► BGE-M3 + ChromaDB          retrieve JarvisLabs KB (~60ms)
  ├─► Sarvam-30B via vLLM        reasoning, streamed tokens
  └─► Voxtral TTS                sentence-level streaming audio
  │  WAV chunks via WebSocket
  ▼
Browser (speaker)
```

**Latency trick:** LLM tokens are split at sentence boundaries and sent to TTS immediately — the user hears the first sentence while the LLM is still generating the second. This makes the perceived latency ~40% lower than end-to-end.

---

## Measured Latencies (A100 40GB)

| Stage              | Latency     |
|--------------------|-------------|
| VAD detection      | ~50 ms      |
| ASR (Parakeet)     | ~250 ms     |
| RAG retrieval      | ~60 ms      |
| LLM TTFT (Sarvam)  | ~700 ms     |
| TTS first sentence | ~300 ms     |
| **Total to audio** | **~1.3 s**  |

> Numbers measured after model warm-up on A100 40GB. Cold start adds ~30s for model loading.

---

## Setup

### Prerequisites
- NVIDIA GPU (A100 40GB recommended; 80GB for extra headroom)
- Docker + `nvidia-container-toolkit`
- HuggingFace account with access to Sarvam-30B

### 1. Clone and configure

```bash
git clone <repo>
cd jarvina
cp .env.example .env
# Edit .env: set LLM_MODEL, HF_TOKEN, etc.
```

### 2. Build the knowledge base

```bash
# Inside the api container or locally:
bash scripts/ingest.sh
```

This crawls `jarvislabs.ai` (docs, FAQ, pricing, GPU guides), chunks the content, embeds with BGE-M3, and stores in ChromaDB at `data/chroma/`.

### 3. Launch

```bash
docker-compose up --build
```

- API + pipeline: `http://localhost:8000`
- vLLM server: `http://localhost:8001`

### 4. Open the app

Visit `http://localhost:8000` in Chrome or Firefox. Allow microphone access. Start talking.

---

## Manual launch (without Docker)

```bash
# Terminal 1 — vLLM (Sarvam-30B)
bash scripts/start_vllm.sh

# Terminal 2 — Ingest KB (first time only)
bash scripts/ingest.sh

# Terminal 3 — API server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## Benchmark

```bash
bash scripts/benchmark.sh
```

Runs RAG → LLM → TTS against a fixed test query and prints per-stage latencies.

---

## Project Structure

```
jarvina/
├── config.py               Central settings (all tunable via .env)
├── scraper/crawler.py      Async crawler for jarvislabs.ai
├── rag/
│   ├── ingest.py           Chunk → embed → ChromaDB
│   └── retriever.py        BGE-M3 query → top-k chunks
├── asr/parakeet.py         Parakeet RNNT 1.1B wrapper
├── llm/
│   ├── client.py           Async streaming vLLM client
│   └── prompts.py          System prompt (EN/HI/Hinglish)
├── tts/voxtral.py          Voxtral TTS wrapper
├── pipeline/
│   ├── vad.py              silero-vad end-of-speech detector
│   └── orchestrator.py     Full pipeline + conversation state
├── api/server.py           FastAPI WebSocket server
├── frontend/               Web UI (HTML + AudioWorklet + CSS)
├── docker/                 Dockerfiles
├── docker-compose.yml
└── scripts/                ingest, benchmark, start_vllm
```

---

## Sample Transcript

**User:** "JarvisLabs pe A100 ka pricing kya hai?"

**Jarvina:** "JarvisLabs pe A100 40GB instance ka pricing 84 credits per hour hai. Aap monthly subscription leke aur save kar sakte hain. Kya aap abhi ek instance launch karna chahte hain?"

---

## Models Used

| Component | Model                              | Size  |
|-----------|------------------------------------|-------|
| ASR       | nvidia/parakeet-rnnt-1.1b          | 1.1B  |
| LLM       | sarvamai/sarvam-m (Sarvam-30B)    | 30B   |
| TTS       | mistralai/Voxtral-Mini-3B-2507    | 3B    |
| Embed     | BAAI/bge-m3                        | 568M  |
| VAD       | snakers4/silero-vad v5             | 1.7M  |

All models are 100% open-source and self-hosted on JarvisLabs infrastructure.

---

## Deployment on JarvisLabs

1. Spin up an A100 40GB instance with the PyTorch template
2. Clone this repo into the instance
3. `docker-compose up --build`
4. Use JarvisLabs' port forwarding to expose port 8000 publicly
5. Share the public URL

VRAM budget: Sarvam-30B AWQ ~18GB + Parakeet ~3GB + Voxtral ~5GB + BGE-M3 ~2GB ≈ **28GB** (fits A100 40GB with headroom).
