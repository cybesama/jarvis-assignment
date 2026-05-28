"""
TTS wrapper for Fish Speech 1.5 (fishaudio/fish-speech-1.5).
Self-hosted on the JarvisLabs GPU instance.
"""

import io
import re
import time

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from config import settings

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class FishSpeechTTS:

    MODEL_DIR = "/home/models/fish-speech-1.5"

    def __init__(self):
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return
        from fish_speech.inference_engine import TTSInferenceEngine
        logger.info("Loading Fish Speech 1.5...")
        self._engine = TTSInferenceEngine(
            checkpoint_path=self.MODEL_DIR,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision=torch.float16 if torch.cuda.is_available() else torch.float32,
            compile=False,
        )
        logger.info("Fish Speech loaded")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        self._load()
        t0 = time.perf_counter()

        result = self._engine.inference(
            text=text,
            references=[],
        )
        audio = np.array(result.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        latency = time.perf_counter() - t0
        logger.info(f"TTS: '{text[:60]}' → {len(audio)/settings.TTS_SAMPLE_RATE:.2f}s audio ({latency*1000:.0f}ms)")
        return audio, latency

    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, settings.TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


tts = FishSpeechTTS()
