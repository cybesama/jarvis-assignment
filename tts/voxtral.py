"""
TTS wrapper for Voxtral (mistralai/Voxtral-Mini-3B-2507).
Self-hosted via mistral-inference on the JarvisLabs GPU instance.

Falls back to Mistral cloud API if local model isn't loaded
(set TTS_BACKEND=voxtral_api and MISTRAL_API_KEY in .env).
"""

import io
import os
import time
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from config import settings


# Sentence boundary regex — handles Hindi danda (।) too
import re
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    """Split LLM output into speakable sentence chunks."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class VoxtralTTS:
    """
    Self-hosted Voxtral TTS.
    Model: mistralai/Voxtral-Mini-3B-2507
    """

    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return

        backend = settings.TTS_BACKEND

        if backend == "voxtral_api":
            # Cloud API path — no local model needed
            logger.info("Voxtral TTS: using Mistral cloud API")
            self._pipeline = "api"
            return

        # Local model via transformers (Voxtral uses the same HF interface)
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise RuntimeError("transformers not installed: pip install transformers")

        logger.info(f"Loading Voxtral TTS model: {self.MODEL_ID}")
        self._pipeline = hf_pipeline(
            "text-to-speech",
            model=self.MODEL_ID,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        logger.info("Voxtral TTS loaded")

    def _synthesize_local(self, text: str) -> np.ndarray:
        result = self._pipeline(text)
        audio = result["audio"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio.astype(np.float32)

    def _synthesize_api(self, text: str) -> np.ndarray:
        from mistralai import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set for voxtral_api backend")

        client = Mistral(api_key=api_key)
        response = client.audio.speech.create(
            model="voxtral",
            voice=settings.TTS_VOICE,
            input=text,
        )
        # Response is raw WAV bytes
        audio_bytes = response.read()
        buf = io.BytesIO(audio_bytes)
        audio, _ = sf.read(buf, dtype="float32")
        return audio

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """
        Convert text to speech.
        Returns (float32 audio array at TTS_SAMPLE_RATE, latency_seconds).
        """
        self._load()
        t0 = time.perf_counter()

        if self._pipeline == "api":
            audio = self._synthesize_api(text)
        else:
            audio = self._synthesize_local(text)

        latency = time.perf_counter() - t0
        logger.info(f"TTS: '{text[:60]}...' → {len(audio)/settings.TTS_SAMPLE_RATE:.2f}s audio ({latency*1000:.0f}ms)")
        return audio, latency

    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Encode float32 numpy audio as 16-bit PCM WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, audio, settings.TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()


# Module-level singleton
tts = VoxtralTTS()
