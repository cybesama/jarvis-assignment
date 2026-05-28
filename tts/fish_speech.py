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

    LLAMA_CKPT  = "/home/models/fish-speech-1.5"
    DAC_CKPT    = "/home/models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    DAC_CONFIG  = "modded_dac_vq"

    def __init__(self):
        self._engine = None
        self._sample_rate = settings.TTS_SAMPLE_RATE

    def _load(self):
        if self._engine is not None:
            return

        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.dac.inference import load_model
        from fish_speech.inference_engine import TTSInferenceEngine

        device    = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info("Fish Speech: loading LLaMA text2semantic model...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.LLAMA_CKPT,
            device=device,
            precision=precision,
            compile=False,
        )

        logger.info("Fish Speech: loading DAC decoder...")
        decoder = load_model(
            config_name=self.DAC_CONFIG,
            checkpoint_path=self.DAC_CKPT,
            device=device,
        )

        # Detect actual sample rate from decoder
        if hasattr(decoder, "spec_transform"):
            self._sample_rate = decoder.spec_transform.sample_rate
        elif hasattr(decoder, "sample_rate"):
            self._sample_rate = decoder.sample_rate

        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=precision,
            compile=False,
        )
        logger.info(f"Fish Speech loaded — sample_rate={self._sample_rate}")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        self._load()
        t0 = time.perf_counter()

        from fish_speech.utils.schema import ServeTTSRequest

        req = ServeTTSRequest(
            text=text,
            references=[],
            streaming=False,
            normalize=True,
        )

        chunks = []
        for result in self._engine.inference(req):
            if result.code == "error":
                raise RuntimeError(f"Fish Speech inference error: {result.error}")
            if result.code == "header":
                continue
            if result.audio is not None:
                _, audio_chunk = result.audio
                arr = np.asarray(audio_chunk, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.squeeze()
                if arr.size > 0:
                    chunks.append(arr)

        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

        latency = time.perf_counter() - t0
        logger.info(
            f"TTS: '{text[:60]}' → {len(audio)/self._sample_rate:.2f}s "
            f"({latency*1000:.0f}ms)"
        )
        return audio, latency

    def audio_to_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self._sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


tts = FishSpeechTTS()
