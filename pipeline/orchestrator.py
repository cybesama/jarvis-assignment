"""
Pipeline orchestrator: VAD → ASR → RAG → LLM (streaming) → TTS (streaming).

Key latency trick: LLM tokens are accumulated sentence-by-sentence.
Each complete sentence is sent to TTS immediately — the first audio chunk
reaches the browser before the LLM finishes generating, making the
conversation feel much faster than the total end-to-end latency.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Awaitable

import numpy as np
from loguru import logger

from asr.parakeet import asr
from llm.client import stream_response
from rag.retriever import retriever
from tts.kokoro_wrapper import tts, split_sentences
from pipeline.vad import VADProcessor
from config import settings


class State(str, Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    IDLE = "idle"


@dataclass
class Turn:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class PipelineCallbacks:
    """Async callbacks the WebSocket server plugs in."""
    on_state: Callable[[State], Awaitable[None]]
    on_transcript: Callable[[str, bool], Awaitable[None]]   # (text, is_final)
    on_response_text: Callable[[str], Awaitable[None]]
    on_audio_chunk: Callable[[bytes], Awaitable[None]]
    on_latency: Callable[[dict], Awaitable[None]]


class ConversationSession:
    """
    One active session per connected WebSocket client.
    Holds the VAD processor, conversation history, and pipeline state.
    """

    def __init__(self, callbacks: PipelineCallbacks):
        self.callbacks = callbacks
        self.vad = VADProcessor()
        self.history: list[Turn] = []
        self._state = State.IDLE
        self._processing_lock = asyncio.Lock()
        self._interrupt = False

    # ── State management ──────────────────────────────────────────────────────

    async def _set_state(self, state: State):
        self._state = state
        await self.callbacks.on_state(state)

    # ── Audio ingestion ───────────────────────────────────────────────────────

    async def feed_audio(self, raw_bytes: bytes):
        """
        Called by the WebSocket handler for every incoming audio chunk.
        VAD runs here; triggers the full pipeline when end-of-speech is found.
        """
        # Interrupt: if user speaks while assistant is talking
        if self._state == State.SPEAKING and self.vad._in_speech:
            self._interrupt = True

        await self._set_state(State.LISTENING)

        utterance = self.vad.process(raw_bytes)
        if utterance is None:
            return

        # Don't start a new turn if we're already processing one
        if self._processing_lock.locked():
            logger.debug("Pipeline busy, dropping utterance")
            return

        asyncio.create_task(self._handle_utterance(utterance))

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def _handle_utterance(self, audio: np.ndarray):
        async with self._processing_lock:
            t_total_start = time.perf_counter()
            await self._set_state(State.PROCESSING)
            latencies = {}

            # ── 1. ASR ───────────────────────────────────────────────────────
            t0 = time.perf_counter()
            transcript, asr_lat = await asyncio.get_event_loop().run_in_executor(
                None, asr.transcribe, audio
            )
            latencies["asr_ms"] = round(asr_lat * 1000)

            if not transcript:
                await self._set_state(State.LISTENING)
                return

            await self.callbacks.on_transcript(transcript, is_final=True)

            # ── 2. RAG ───────────────────────────────────────────────────────
            t0 = time.perf_counter()
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, retriever.retrieve, transcript
            )
            context = retriever.format_context(chunks)
            latencies["rag_ms"] = round((time.perf_counter() - t0) * 1000)

            # ── 3. LLM + TTS streaming ───────────────────────────────────────
            history_msgs = [
                {"role": t.role, "content": t.content}
                for t in self.history[-(settings.CONV_HISTORY_TURNS * 2):]
            ]

            full_response = await self._stream_llm_tts(
                transcript, context, history_msgs, latencies
            )

            # ── 4. Update history ────────────────────────────────────────────
            self.history.append(Turn("user", transcript))
            self.history.append(Turn("assistant", full_response))

            latencies["total_ms"] = round((time.perf_counter() - t_total_start) * 1000)
            await self.callbacks.on_latency(latencies)
            logger.info(f"Turn latencies: {latencies}")

            self._interrupt = False
            await self._set_state(State.LISTENING)

    async def _stream_llm_tts(
        self,
        query: str,
        context: str,
        history: list[dict],
        latencies: dict,
    ) -> str:
        """
        Stream LLM tokens → accumulate into sentences → TTS each sentence.
        Returns the full assembled response text.
        """
        await self._set_state(State.SPEAKING)

        token_buffer = ""
        full_text = ""
        first_sentence = True
        t_llm_start = time.perf_counter()

        # Serial TTS queue — Kokoro is not thread-safe; run one at a time
        # but pipeline each phrase immediately as it's ready.
        tts_queue: asyncio.Queue = asyncio.Queue()

        async def tts_worker():
            nonlocal latencies
            while True:
                item = await tts_queue.get()
                if item is None:
                    break
                sentence, is_first = item
                loop = asyncio.get_event_loop()
                audio, tts_lat = await loop.run_in_executor(
                    None, tts.synthesize, sentence
                )
                if is_first:
                    latencies["tts_first_ms"] = round(tts_lat * 1000)
                audio_bytes = tts.audio_to_bytes(audio)
                await self.callbacks.on_audio_chunk(audio_bytes)

        worker_task = asyncio.create_task(tts_worker())

        async for token in stream_response(query, context, history):
            if self._interrupt:
                break

            token_buffer += token
            full_text += token
            await self.callbacks.on_response_text(token)

            # Flush on sentence boundary OR after ~60 chars at a word boundary
            sentences = split_sentences(token_buffer)
            flush_phrases = []
            if len(sentences) >= 2:
                ready = [s for s in sentences[:-1] if len(s) >= 20]
                if ready:
                    flush_phrases = ready
                    token_buffer = sentences[-1]
            elif len(token_buffer) >= 60 and token_buffer[-1] == " ":
                flush_phrases = [token_buffer.strip()]
                token_buffer = ""

            for s in flush_phrases:
                if s.strip():
                    await tts_queue.put((s, first_sentence))
                    first_sentence = False

        latencies["llm_ms"] = round((time.perf_counter() - t_llm_start) * 1000)

        # Flush remaining buffer
        if token_buffer.strip() and not self._interrupt:
            await tts_queue.put((token_buffer.strip(), first_sentence))

        await tts_queue.put(None)  # signal worker to stop
        worker_task_list = [worker_task]

        await asyncio.gather(*worker_task_list)

        return full_text
