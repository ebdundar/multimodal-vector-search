"""Search-related service class.

Provides SearchService which encapsulates query embedding and retrieval logic.
"""
from typing import List, Dict, Any
from PIL import Image

from app.logger import get_logger
from app.schemas import SearchRequest, SearchResponse, SearchResult

log = get_logger("search_service")


class SearchService:
    def __init__(self, embedding_service, vector_db):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.log = log

    def _merge_results(self, result_lists: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        if not result_lists:
            return []
        if len(result_lists) == 1:
            return result_lists[0]

        scores_map: Dict[str, List[float]] = {}
        info_map: Dict[str, Dict[str, Any]] = {}

        for rl in result_lists:
            for r in rl:
                rid = r["id"]
                scores_map.setdefault(rid, []).append(r["similarity_score"])
                if rid not in info_map:
                    info_map[rid] = {"metadata": r.get("metadata", {}), "document": r.get("document", "")}

        merged: List[Dict[str, Any]] = []
        for rid, scores in scores_map.items():
            avg_score = sum(scores) / len(scores)
            info = info_map.get(rid, {"metadata": {}, "document": ""})
            merged.append({"id": rid, "similarity_score": avg_score, "metadata": info["metadata"], "document": info["document"]})

        merged.sort(key=lambda x: x["similarity_score"], reverse=True)
        return merged[:top_k]

    def search(self, request: SearchRequest) -> SearchResponse:
        """Search using text queries only (request.query_text must be provided)."""
        if not request.query_text:
            raise ValueError("`query_text` must be provided for text search")
        text_embedding = self.embedding_service.embed_text(request.query_text)
        results = self.vector_db.search(query_embedding=text_embedding, top_k=request.top_k, filter_metadata=request.filter_metadata)
        return SearchResponse(results=[SearchResult(**r) for r in results], query_type="text")

    def search_with_image(self, image: Image.Image, top_k: int = 10, filter_metadata: Dict[str, Any] = None) -> SearchResponse:
        """Search using an already-loaded PIL Image object."""
        image_embedding = self.embedding_service.embed_image(image)
        results = self.vector_db.search(query_embedding=image_embedding, top_k=top_k, filter_metadata=filter_metadata)
        # keep behavior consistent with merged multi-modal results: this returns image-only results
        return SearchResponse(results=[SearchResult(**r) for r in results], query_type="image")
