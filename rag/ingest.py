"""
RAG ingestion pipeline:
  raw JSON (from crawler) → chunk → embed (BGE-M3) → store in ChromaDB
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import chromadb
from FlagEmbedding import BGEM3FlagModel
from loguru import logger

from config import settings


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks at sentence/paragraph boundaries."""
    # Split on paragraph or sentence boundaries first
    segments = re.split(r"(?<=\.)\s+|\n\n+", text)
    chunks, current, current_len = [], [], 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        words = seg.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # keep overlap words from end of current chunk
            current = current[-overlap:]
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) > 20]  # drop tiny fragments


def _load_embed_model() -> BGEM3FlagModel:
    logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
    return BGEM3FlagModel(settings.EMBED_MODEL, use_fp16=True)


def _get_or_create_collection(client: chromadb.PersistentClient):
    try:
        col = client.get_collection(settings.COLLECTION_NAME)
        logger.info(f"Using existing collection '{settings.COLLECTION_NAME}' ({col.count()} docs)")
        return col
    except Exception:
        logger.info(f"Creating new collection '{settings.COLLECTION_NAME}'")
        return client.create_collection(
            settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def ingest(raw_path: Path = None, reset: bool = False) -> int:
    raw_path = raw_path or (settings.RAW_DIR / "jarvislabs_raw.json")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}. Run the scraper first.")

    docs: List[Dict] = json.loads(raw_path.read_text())
    logger.info(f"Loaded {len(docs)} pages from {raw_path}")

    embed_model = _load_embed_model()

    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

    if reset:
        try:
            chroma.delete_collection(settings.COLLECTION_NAME)
            logger.info("Deleted existing collection (reset=True)")
        except Exception:
            pass

    collection = _get_or_create_collection(chroma)
    existing_ids = set(collection.get()["ids"])

    all_chunks, all_ids, all_metas = [], [], []

    for doc in docs:
        chunks = _chunk_text(doc["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_{i}"
            if chunk_id in existing_ids:
                continue
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "url": doc["url"],
                "title": doc["title"],
                "chunk_index": i,
            })

    if not all_chunks:
        logger.info("No new chunks to ingest.")
        return 0

    logger.info(f"Embedding {len(all_chunks)} new chunks with {settings.EMBED_MODEL}...")

    # Batch embedding
    batch_size = settings.EMBED_BATCH_SIZE
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]

        result = embed_model.encode(
            batch_texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embeddings = result["dense_vecs"].tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        logger.info(f"  ingested {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    total = collection.count()
    logger.info(f"Ingest complete. Collection size: {total} chunks.")
    return len(all_chunks)


if __name__ == "__main__":
    ingest()
