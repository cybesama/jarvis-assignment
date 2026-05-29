# Jarvina — Voice Assistant for JarvisLabs

---

## What it does

JarvisLabs' website has no voice interface — users have to hunt through docs, pricing pages, and FAQs manually to find answers. Jarvina sits on the JarvisLabs website and lets users simply ask questions out loud — about GPU plans, pricing, instance setup, SSH, or anything else on the platform — and hear a clear spoken answer back. It supports English, Hindi, and Hinglish, so it works naturally for the Indian developer audience that JarvisLabs primarily serves.

---

## Why I built this

I faced this problem myself. When I first tried to navigate JarvisLabs, I wasn't sure which GPU fit my use case, how persistent storage worked, or how to SSH into an instance correctly. The answers existed somewhere in the docs, but finding them took more time than it should have. When I thought about what to build, this felt like an obvious and real gap rather than a manufactured problem — a GPU cloud platform built for AI developers should have an AI-native way to answer questions about itself. I'd seen the frustration firsthand, so I decided to build the fix.

---

## Try it yourself

**Live:** [https://6bd4684165461.notebooksn.jarvislabs.net/](https://6bd4684165461.notebooksn.jarvislabs.net/)

Open the link, click the mic button, and ask a question. The assistant responds in spoken audio.

> Note: the assistant runs on a JarvisLabs A100 instance. If the instance is paused, use the sample transcript below as a fallback or run it locally using the instructions further down.

---

## Pipeline proof-of-work

**End-to-end pipeline works.** A user speaks into the mic, the assistant transcribes the audio, retrieves relevant context from the JarvisLabs knowledge base, generates a response, and speaks it back — with no manual intervention between stages. The full pipeline runs inside a single FastAPI WebSocket session.

**Latency is acceptable for natural conversation.** Measured on JarvisLabs A100 40GB, all models warm:

| Stage | Time |
|-------|------|
| ASR (Parakeet RNNT 1.1B) | 129 ms |
| RAG (BGE-M3 + ChromaDB, 702 chunks) | 32 ms |
| LLM (Qwen3-32B-AWQ, ~100 tokens) | 22,446 ms |
| TTS first chunk (Kokoro-82M) | 200 ms |
| **End-to-end** | **~22,800 ms** |

The user hears the first spoken sentence within ~700ms of the LLM starting to generate (sentence-level TTS streaming), so the conversation feels faster than the total latency implies. See [Architecture decisions](#architecture-decisions) for what was done to reduce it.

**The assistant is grounded and stays on topic.** Jarvina knows it is a JarvisLabs assistant and refuses off-topic questions. All answers are grounded in retrieved content from `jarvislabs.ai` — it will not make up pricing or specs.

**Sample audio clip + expected response (fallback):**

🎙️ [sample_query.mp3](sample_query.mp3) — *"Tell me about available GPUs I can purchase on JarvisLabs"*

**Expected response:**
> JarvisLabs offers several high-performance NVIDIA GPUs for rent, including H200, H100, A100, A6000, A5000, and L4. These GPUs are available in regions like IN2 and EU1. You can check current pricing and availability via the JarvisLabs dashboard or the SDK. Let me know if you'd like details on a specific GPU!

**On-topic / off-topic behaviour:**

| User input | Jarvina |
|-----------|---------|
| "Tell me about available GPUs" | Answers with GPU list, regions, pricing pointer |
| "What is the capital of France?" | "Main sirf JarvisLabs ke baare mein help kar sakta hoon." |

---

## How to run it

### Prerequisites
- JarvisLabs A100 40GB instance (or equivalent GPU with 30GB+ VRAM)
- Python 3.10, CUDA 12+

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

Set the following in `.env`:

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

### 3. Start vLLM

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --port 8001 &
```

Wait until the terminal shows `Application startup complete`.

### 4. Build the knowledge base (first run only)

```bash
python scripts/build_kb.py
```

Scrapes `jarvislabs.ai`, chunks the content, embeds with BGE-M3, and stores in ChromaDB (~702 chunks).

### 5. Start the server

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 6006
```

### 6. Open in browser

Go to `http://<instance-ip>:6006`. Click the mic button, speak, hear the response.

> **JarvisLabs users:** Set your API endpoint to forward to port `6006`. Your public URL will be `https://<instance-id>.notebooksn.jarvislabs.net/`.

---

## Model choices

| Stage | Model |
|-------|-------|
| ASR | `nvidia/parakeet-rnnt-1.1b` |
| VAD | `silero-vad v5` |
| Embeddings | `BAAI/bge-m3` |
| LLM | `Qwen/Qwen3-32B-AWQ` via vLLM |
| TTS | `hexgrad/Kokoro-82M` (`af_heart` voice) |

---

## Architecture decisions

### Qwen3-32B-AWQ over other instruction-tuned models

The main constraint was fitting a multilingual model capable of fluent Hindi and Hinglish on a single A100 40GB alongside ASR and embedding models. Qwen3-32B-AWQ at 4-bit AWQ quantization uses ~18GB VRAM — just enough room. It handles Hindi natively without any fine-tuning, which no smaller open model does reliably. One non-obvious issue: Qwen3 defaults to chain-of-thought reasoning mode which emits `<think>...</think>` tokens — these get sent to TTS and produce gibberish audio. Disabling it via `enable_thinking: false` in the request payload is required.

### Kokoro-82M over a cloud TTS service

The original TTS was Edge TTS (Microsoft). It worked but added ~1 second of network round-trip latency per spoken phrase. Kokoro-82M runs locally at ~150–200ms per phrase with no API key and no external dependency. The trade-off is English-only voices, but for a platform where most users speak Hinglish rather than pure Hindi, this is acceptable.

### Sentence-level TTS streaming instead of full-response

If TTS waits for the LLM to finish generating before it starts, the user sits in silence for the entire generation time. Instead, the pipeline flushes the TTS queue every time a sentence boundary is found or ~100 characters have accumulated. The first audio chunk plays ~700ms after the LLM produces its first sentence, while generation continues in the background. This is the main reason perceived latency feels shorter than the raw numbers suggest.

### Serial TTS queue over concurrent tasks

Running multiple Kokoro inference calls concurrently from different threads causes audio corruption — Kokoro's `KPipeline` is not thread-safe. A single async worker processes one phrase at a time from a queue. This also guarantees audio plays in the correct order, something concurrent tasks cannot provide.

### VAD before ASR

Running Parakeet on every microphone sample would be wasteful and slow. Silero-VAD runs on CPU in under 5ms per chunk and only signals the ASR when a complete utterance is detected (speech followed by 700ms silence). This means Parakeet only runs when there is actually something worth transcribing.

### BGE-M3 for embeddings

BGE-M3 is multilingual and handles the mix of English and transliterated Hindi in JarvisLabs documentation. It runs locally (no API cost per query) and returns embeddings in ~32ms warm. A cloud embedding API would add network latency and cost on every user query.

---

## Latency

Measured on JarvisLabs A100 40GB, all models warm:

| Stage | Time |
|-------|------|
| ASR (Parakeet RNNT 1.1B) | 129 ms |
| RAG (BGE-M3 + ChromaDB) | 32 ms |
| LLM (Qwen3-32B-AWQ, ~100 tokens) | 22,446 ms |
| TTS first chunk (Kokoro-82M) | 200 ms |
| **Total end-to-end** | **~22,800 ms** |

**What I did to bring latency down:**
- Switched TTS from Edge TTS (cloud, ~1s/phrase) to Kokoro (local, ~200ms/phrase)
- Added sentence-level streaming so users hear audio before the LLM finishes
- Capped `LLM_MAX_TOKENS` at 100 to keep responses short and generation fast
- RAG reduced to `TOP_K=3` — fewer retrieved chunks means shorter context and faster LLM prefill

The LLM at ~22 seconds is the dominant bottleneck. Everything else is negligible.

---

## What I used AI for

**Where AI helped:**
- **Instance setup:** I had not configured cloud GPU servers from scratch before. I used AI to get the right vLLM flags (`--gpu-memory-utilization`, `--max-model-len`, `FLASHINFER_DISABLE_VERSION_CHECK`) and understand how to size VRAM allocation across models running on the same GPU.
- **Pipeline wiring:** The async connection code between VAD → ASR → RAG → LLM → TTS — specifically the WebSocket protocol, AudioWorklet setup, and the streaming token-to-TTS buffer logic — was generated with AI assistance.
- **Debugging:** Several failures were non-obvious. AI helped identify that Parakeet returns `Hypothesis` objects (not strings), that Qwen3 emits `<think>` tokens in voice output, and that WebSocket connections require `wss://` behind an HTTPS proxy.

**What I did myself:**
- Evaluated and rejected multiple TTS options (Voxtral — ASR-only, not TTS; Fish Speech — proprietary codec not in any open-source release; Edge TTS — too slow) through hands-on testing, not from AI suggestions
- Verified every API connection and parameter manually before moving to the next stage — GPU memory allocation, VAD thresholds, chunk sizes, LLM context limits
- Identified the Kokoro thread-safety bug from garbled audio in logs and overrode AI's suggestion of concurrent TTS tasks in favour of a serial queue
- Decided the knowledge base scope: which pages to scrape, chunk size, overlap, and collection structure

---

## What I would change with 4 more weeks

**1. Reduce LLM latency**
At ~22 seconds, LLM generation is the only thing that makes the conversation feel slow. With more time I would evaluate a smaller, faster model — a well-tuned 7B or 14B fine-tuned specifically on JarvisLabs documentation — which would generate in 2–4 seconds and keep the assistant on topic without needing a 32B model.

**2. More grounded and verifiable outputs**
The current RAG retrieves scraped webpage text, which means the LLM can paraphrase or extrapolate pricing and specs. I would add structured data sources — actual pricing tables, GPU spec JSON, support ticket categories — and instruct the model to quote specific numbers rather than summarise. This makes hallucinations easier to catch and the answers easier to trust.

**3. Streaming TTS**
Kokoro generates a full phrase before sending any audio. A streaming-capable TTS model would begin sending audio within 50ms of synthesis starting, cutting per-phrase TTS latency from ~200ms to near-zero perceived delay and making the conversation feel genuinely instant.

**4. Native Hindi voice**
Kokoro pronounces Hindi and Hinglish in an English accent. Adding a lightweight Hindi-trained TTS model as a fallback — triggered automatically when Devanagari script or strong Hindi phrasing is detected — would make the assistant feel natural to Indian users, who are the core JarvisLabs audience.
