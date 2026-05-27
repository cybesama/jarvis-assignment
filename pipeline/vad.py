"""
VAD (Voice Activity Detection) using silero-vad v5.
Processes 16kHz Int16 PCM audio chunks and signals end-of-speech
after a configurable silence window.
"""

import numpy as np
import torch
from loguru import logger

from config import settings


class VADProcessor:
    """
    Stateful VAD processor for a single conversation session.
    Feed it raw 16kHz Int16 PCM chunks; it accumulates speech and
    fires when it detects end-of-utterance.

    Usage:
        vad = VADProcessor()
        for chunk in audio_stream:
            result = vad.process(chunk)
            if result is not None:
                # result is a np.float32 array of the full utterance at 16kHz
                transcript = asr.transcribe(result)
    """

    CHUNK_SAMPLES = 512   # silero v5 expects 512 samples per call at 16kHz

    def __init__(self):
        self._model = None
        self._reset_state()

    def _load(self):
        if self._model is None:
            logger.info("Loading silero-vad v5...")
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model.eval()
            logger.info("silero-vad ready")

    def _reset_state(self):
        self._speech_buffer: list[np.ndarray] = []
        self._silence_samples: int = 0
        self._in_speech: bool = False
        self._chunk_carry: bytes = b""   # leftover bytes between calls

    def reset(self):
        self._reset_state()

    @property
    def _silence_threshold_samples(self) -> int:
        return int(settings.VAD_SILENCE_DURATION_MS * settings.VAD_SAMPLE_RATE / 1000)

    @property
    def _min_speech_samples(self) -> int:
        return int(settings.VAD_MIN_SPEECH_MS * settings.VAD_SAMPLE_RATE / 1000)

    def process(self, raw_bytes: bytes) -> np.ndarray | None:
        """
        Feed PCM bytes (Int16 LE, 16kHz).
        Returns float32 numpy array of the complete utterance when
        end-of-speech is detected, otherwise None.
        """
        self._load()

        # Accumulate bytes until we have full 512-sample frames
        data = self._chunk_carry + raw_bytes
        self._chunk_carry = b""

        frame_bytes = self.CHUNK_SAMPLES * 2  # Int16 = 2 bytes per sample
        utterance_ready = None

        i = 0
        while i + frame_bytes <= len(data):
            frame_bytes_slice = data[i : i + frame_bytes]
            i += frame_bytes

            frame = np.frombuffer(frame_bytes_slice, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(frame).unsqueeze(0)   # (1, 512)

            with torch.no_grad():
                prob = self._model(tensor, settings.VAD_SAMPLE_RATE).item()

            is_speech = prob >= settings.VAD_THRESHOLD

            if is_speech:
                self._speech_buffer.append(frame)
                self._silence_samples = 0
                self._in_speech = True
            elif self._in_speech:
                # Pad silence into buffer so ASR hears natural endings
                self._speech_buffer.append(frame)
                self._silence_samples += self.CHUNK_SAMPLES

                if self._silence_samples >= self._silence_threshold_samples:
                    # End of utterance
                    audio = np.concatenate(self._speech_buffer)
                    total_samples = len(audio)

                    if total_samples >= self._min_speech_samples:
                        utterance_ready = audio
                        logger.debug(
                            f"Utterance detected: {total_samples / settings.VAD_SAMPLE_RATE:.2f}s"
                        )
                    else:
                        logger.debug("Noise burst too short, discarding.")

                    self._reset_state()

        # Carry leftover bytes to next call
        self._chunk_carry = data[i:]

        return utterance_ready
