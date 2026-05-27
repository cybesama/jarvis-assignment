"""
RAG retriever — given a query string, returns the top-k relevant chunks
from the JarvisLabs knowledge base.
"""

from typing import List, Dict
import chromadb
from FlagEmbedding import BGEM3FlagModel
from loguru import logger

from config import settings


class Retriever:
    def __init__(self):
        self._embed_model: BGEM3FlagModel | None = None
        self._collection: chromadb.Collection | None = None

    def _load(self):
        if self._embed_model is None:
            logger.info(f"Loading retriever embed model: {settings.EMBED_MODEL}")
            self._embed_model = BGEM3FlagModel(settings.EMBED_MODEL, use_fp16=True)

        if self._collection is None:
            settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            chroma = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
            self._collection = chroma.get_collection(settings.COLLECTION_NAME)
            logger.info(f"Retriever ready ({self._collection.count()} chunks in KB)")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Return top_k chunks relevant to query."""
        self._load()
        top_k = top_k or settings.TOP_K

        result = self._embed_model.encode(
            [query],
            batch_size=1,
            max_length=256,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_vec = result["dense_vecs"][0].tolist()

        hits = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            hits["documents"][0],
            hits["metadatas"][0],
            hits["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
                "score": round(1 - dist, 4),   # cosine similarity
            })

        return chunks

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {c['title']} — {c['url']}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)


# Module-level singleton — loaded once, reused across requests
retriever = Retriever()
