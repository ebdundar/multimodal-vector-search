"""Ingest-related service class.

Provides IngestService which encapsulates single and batch ingest logic.
"""
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image

from app.logger import get_logger
from app.utils import load_image
from app.schemas import IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse, BatchIngestItemResult

log = get_logger("ingest_service")


class IngestService:
    def __init__(self, embedding_service, vector_db):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.log = log

    def _load_image_safe(self, image_str: Optional[str]) -> Optional[Image.Image]:
        if not image_str:
            return None
        return load_image(image_str)

    def ingest(self, request: IngestRequest) -> IngestResponse:
        if not request.text and not request.image:
            raise ValueError("At least one of 'text' or 'image' must be provided")

        image_obj = self._load_image_safe(request.image) if request.image else None
        embeddings_per_item = self.embedding_service.embed(text=request.text, image=image_obj)

        nested_ids = self.vector_db.add_many(
            embeddings_per_item=[embeddings_per_item],
            texts=[request.text],
            image_urls=[request.image],
            metadatas=[request.metadata],
        )

        ids = nested_ids[0] if nested_ids and len(nested_ids) > 0 else []
        return IngestResponse(ids=ids, message="Successfully ingested item")

    def batch_ingest(self, request: BatchIngestRequest) -> BatchIngestResponse:
        if not request.items:
            return BatchIngestResponse(results=[])

        items_to_embed: List[Tuple[Optional[str], Optional[Image.Image]]] = []
        texts: List[Optional[str]] = []
        image_urls: List[Optional[str]] = []
        metadatas: List[Optional[Dict[str, Any]]] = []
        results: List[BatchIngestItemResult] = []
        batched_indices: List[int] = []

        for idx, item in enumerate(request.items):
            results.append(BatchIngestItemResult(index=idx, success=False, message="", ids=None))
            if not item.text and not item.image:
                results[idx].message = "At least one of 'text' or 'image' must be provided"
                continue

            image_obj = None
            if item.image:
                try:
                    image_obj = self._load_image_safe(item.image)
                except Exception as e:
                    results[idx].message = f"Error loading image: {str(e)}"
                    continue

            batched_indices.append(idx)
            items_to_embed.append((item.text, image_obj))
            texts.append(item.text)
            image_urls.append(item.image)
            metadatas.append(item.metadata)

        if not items_to_embed:
            return BatchIngestResponse(results=results)

        embeddings_per_item = self.embedding_service.embed_batch(items_to_embed)

        nested_ids = self.vector_db.add_many(
            embeddings_per_item=embeddings_per_item,
            texts=texts,
            image_urls=image_urls,
            metadatas=metadatas,
        )

        for batch_pos, orig_idx in enumerate(batched_indices):
            results[orig_idx].ids = nested_ids[batch_pos]
            results[orig_idx].success = True
            results[orig_idx].message = "Successfully ingested item"

        return BatchIngestResponse(results=results)
