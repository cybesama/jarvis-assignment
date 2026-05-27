"""
ASR wrapper for NVIDIA Parakeet RNNT 1.1B (NeMo).
Accepts a float32 numpy array at 16kHz and returns a transcript string.

The RNNT model is batch-transcribe triggered by VAD — no partial streaming
at this layer. Partial transcript display is handled by the orchestrator
sending ASR results to the frontend as soon as they arrive.
"""

import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from config import settings


class ParakeetASR:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise RuntimeError(
                "NeMo is not installed. Run: pip install nemo_toolkit[asr]"
            )

        logger.info(f"Loading ASR model: {settings.ASR_MODEL}")
        self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=settings.ASR_MODEL
        )
        self._model.eval()
        if torch.cuda.is_available():
            self._model.cuda()
            logger.info("Parakeet RNNT loaded on GPU")
        else:
            logger.warning("CUDA not available — ASR running on CPU (slow)")

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe a float32 16kHz audio array.
        Returns (transcript_text, latency_seconds).
        """
        self._load()

        t0 = time.perf_counter()

        # NeMo transcribe expects a file path or list of file paths
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, settings.ASR_SAMPLE_RATE, subtype="PCM_16")
            with torch.no_grad():
                results = self._model.transcribe([tmp.name], batch_size=1)

        latency = time.perf_counter() - t0
        transcript = results[0] if results else ""
        transcript = transcript.strip()

        logger.info(f"ASR: '{transcript}' ({latency*1000:.0f}ms)")
        return transcript, latency


# Module-level singleton
asr = ParakeetASR()
