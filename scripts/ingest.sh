#!/usr/bin/env bash
# Run inside the api container (or locally with GPU) to:
#   1. Crawl jarvislabs.ai and save raw JSON
#   2. Chunk, embed, and store in ChromaDB
set -e

echo "==> Step 1: Crawling jarvislabs.ai..."
python -c "
import asyncio
from scraper.crawler import crawl
asyncio.run(crawl())
"

echo "==> Step 2: Ingesting into ChromaDB..."
python -c "
from rag.ingest import ingest
count = ingest()
print(f'Ingested {count} new chunks.')
"

echo "==> Done. Knowledge base is ready."
