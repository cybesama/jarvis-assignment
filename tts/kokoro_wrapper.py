"""
TTS wrapper for Kokoro-82M (hexgrad/Kokoro-82M).
Self-hosted, ~20-50ms inference on GPU. No network round trip.
Voices: af_heart, af_bella, am_adam, bf_emma — English only.
"""

import io
import re
import time

import numpy as np
import soundfile as sf
from loguru import logger

from config import settings

_SENTENCE_END = re.compile(r"(?<=[.!?।,;])\s+")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class KokoroTTS:

    SAMPLE_RATE = 24000

    def __init__(self):
        self._pipeline = None
        self._voice = settings.TTS_VOICE  # e.g. "af_heart"

    def _load(self):
        if self._pipeline is not None:
            return
        from kokoro import KPipeline
        logger.info(f"Loading Kokoro TTS (voice={self._voice})...")
        # lang_code: 'a' = American English, 'b' = British English
        lang_code = self._voice[0] if self._voice else "a"
        self._pipeline = KPipeline(lang_code=lang_code)
        logger.info("Kokoro TTS ready")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        self._load()
        t0 = time.perf_counter()

        chunks = []
        for _, _, audio in self._pipeline(text, voice=self._voice, speed=1.0):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))

        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        latency = time.perf_counter() - t0
        logger.info(
            f"TTS [{self._voice}]: '{text[:60]}' → {len(audio)/self.SAMPLE_RATE:.2f}s "
            f"({latency*1000:.0f}ms)"
        )
        return audio, latency

    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


tts = KokoroTTS()
