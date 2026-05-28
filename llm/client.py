"""
Async streaming LLM client for Qwen3-32B-AWQ served via vLLM.
Uses vLLM's OpenAI-compatible /v1/chat/completions endpoint directly
with httpx so there's no openai SDK dependency.
"""

import json
import time
from typing import AsyncIterator

import httpx
from loguru import logger

from config import settings
from llm.prompts import build_rag_prompt


async def stream_response(
    user_query: str,
    context: str,
    history: list[dict],
) -> AsyncIterator[str]:
    """
    Async generator that yields LLM response tokens as they arrive.
    Measures and logs TTFT (time-to-first-token).
    """
    messages = build_rag_prompt(user_query, context, history)

    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "top_p": settings.LLM_TOP_P,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = f"{settings.VLLM_BASE_URL}/chat/completions"
    t0 = time.perf_counter()
    first_token = True

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"vLLM error {resp.status_code}: {body}")
                yield "[Error: LLM service unavailable. Please try again.]"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")

                if token:
                    if first_token:
                        ttft = time.perf_counter() - t0
                        logger.info(f"LLM TTFT: {ttft*1000:.0f}ms")
                        first_token = False
                    yield token
