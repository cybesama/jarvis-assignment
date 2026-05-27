#!/usr/bin/env bash
# Quick latency benchmark — sends a test audio file through the pipeline
# and prints per-stage latencies.
# Usage: bash scripts/benchmark.sh [path/to/test.wav]
set -e

TEST_WAV=${1:-"scripts/test_audio.wav"}

python - <<'EOF'
import asyncio, json, time, sys, wave, numpy as np
from asr.parakeet import asr
from rag.retriever import retriever
from llm.client import stream_response
from tts.voxtral import tts

TEST_TEXT = "What GPU plans does JarvisLabs offer and what is the pricing?"

print("\n── Jarvina Pipeline Benchmark ─────────────────────────────")

# ASR (skip if no test wav, use fixed text)
print(f"\n[1] ASR  (fixed text): '{TEST_TEXT}'")

# RAG
print("\n[2] RAG retrieval...")
t0 = time.perf_counter()
chunks = retriever.retrieve(TEST_TEXT)
rag_ms = (time.perf_counter() - t0) * 1000
print(f"    Retrieved {len(chunks)} chunks in {rag_ms:.0f}ms")
print(f"    Top result: {chunks[0]['title']} (score={chunks[0]['score']})")

# LLM TTFT
print("\n[3] LLM streaming (measuring TTFT)...")
context = retriever.format_context(chunks)
t0 = time.perf_counter()
ttft = None
full = ""
async def run_llm():
    global ttft, full
    async for token in stream_response(TEST_TEXT, context, []):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        full += token
asyncio.run(run_llm())
total_llm = (time.perf_counter() - t0) * 1000
print(f"    TTFT:  {ttft:.0f}ms")
print(f"    Total: {total_llm:.0f}ms")
print(f"    Response: '{full[:120]}...'")

# TTS
print("\n[4] TTS synthesis...")
first_sentence = full.split(".")[0] + "."
t0 = time.perf_counter()
audio, _ = tts.synthesize(first_sentence)
tts_ms = (time.perf_counter() - t0) * 1000
print(f"    '{first_sentence[:60]}' → {len(audio)/24000:.2f}s audio in {tts_ms:.0f}ms")

print(f"\n── Summary ─────────────────────────────────────────────────")
print(f"   RAG:  {rag_ms:.0f}ms")
print(f"   LLM TTFT: {ttft:.0f}ms")
print(f"   TTS first sentence: {tts_ms:.0f}ms")
print(f"   Estimated total (RAG+TTFT+TTS): {rag_ms+ttft+tts_ms:.0f}ms")
print()
EOF
