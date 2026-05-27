#!/usr/bin/env bash
# Start the vLLM server for Sarvam-30B (run on the JarvisLabs GPU instance).
# Expects HF_TOKEN env var if the model is gated.
set -e

MODEL=${LLM_MODEL:-"sarvamai/sarvam-m"}

echo "Starting vLLM for model: $MODEL"
echo "Port: 8001"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.55 \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name "$MODEL"
