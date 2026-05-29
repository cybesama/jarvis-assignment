"""
TTS wrapper using Edge TTS (Microsoft).
Free, no API key, excellent Hindi/English/Hinglish support.
Voices: en-US-JennyNeural (English), hi-IN-SwaraNeural (Hindi/Hinglish)
"""

import asyncio
import io
import re
import time

import numpy as np
import soundfile as sf
from loguru import logger

from config import settings

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

# Hindi character range detection
_HINDI_RE = re.compile(r"[ऀ-ॿ]")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pick_voice(text: str) -> str:
    """Use Hindi voice if text contains Devanagari or common Hinglish cues."""
    if _HINDI_RE.search(text):
        return "hi-IN-SwaraNeural"
    return "en-US-JennyNeural"


class EdgeTTS:

    def __init__(self):
        self._sample_rate = settings.TTS_SAMPLE_RATE

    def _load(self):
        pass  # no model to load

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        voice = _pick_voice(text)

        audio_bytes = asyncio.run(self._synthesize_async(text, voice))

        buf = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self._sample_rate = sr
        latency = time.perf_counter() - t0
        logger.info(
            f"TTS [{voice}]: '{text[:60]}' → {len(audio)/sr:.2f}s ({latency*1000:.0f}ms)"
        )
        return audio, latency

    async def _synthesize_async(self, text: str, voice: str) -> bytes:
        import edge_tts
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice=voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self._sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


tts = EdgeTTS()
