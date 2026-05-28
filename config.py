from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Path(__file__).parent / "data"
    CHROMA_DIR: Path = Path(__file__).parent / "data" / "chroma"
    RAW_DIR: Path = Path(__file__).parent / "data" / "raw"

    # ASR — NVIDIA Parakeet RNNT 1.1B
    ASR_MODEL: str = "nvidia/parakeet-rnnt-1.1b"
    ASR_SAMPLE_RATE: int = 16000
    ASR_CHUNK_DURATION_MS: int = 100

    # LLM — Qwen3-32B-AWQ via vLLM (~18GB VRAM)
    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    LLM_MODEL: str = "Qwen/Qwen3-32B-AWQ"
    LLM_MAX_TOKENS: int = 300
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.9
    CONV_HISTORY_TURNS: int = 6           # rolling window to keep TTFT stable

    # TTS — Voxtral (primary) | Kokoro-82M (fallback)
    TTS_BACKEND: str = "kokoro"           # "voxtral" | "kokoro"
    TTS_VOXTRAL_URL: str = "http://localhost:8002"
    TTS_SAMPLE_RATE: int = 24000
    TTS_VOICE: str = "af_heart"           # Kokoro voice; override for Voxtral

    # VAD — silero-vad v5
    VAD_THRESHOLD: float = 0.45
    VAD_SILENCE_DURATION_MS: int = 700    # silence after speech → end-of-utterance
    VAD_MIN_SPEECH_MS: int = 250          # ignore sub-250ms noise bursts
    VAD_SAMPLE_RATE: int = 16000

    # RAG — BGE-M3 + ChromaDB
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_BATCH_SIZE: int = 32
    COLLECTION_NAME: str = "jarvislabs_kb"
    TOP_K: int = 5
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 60

    # Scraper
    SCRAPE_SEEDS: List[str] = [
        "https://jarvislabs.ai/",
        "https://jarvislabs.ai/docs",
        "https://jarvislabs.ai/faq",
        "https://jarvislabs.ai/pricing",
        "https://jarvislabs.ai/templates",
        "https://jarvislabs.ai/blog",
    ]
    SCRAPE_DOMAIN_ALLOW: str = "jarvislabs.ai"
    SCRAPE_MAX_DEPTH: int = 4
    SCRAPE_CONCURRENCY: int = 8

    # API server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
