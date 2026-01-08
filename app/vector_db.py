"""ChromaDB-based vector DB exposed under app.vector_db."""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import uuid
from app.logger import get_logger

log = get_logger("vector_db")


class VectorDB:
    def __init__(self, persist_directory: str = None):
        try:
            from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, COLLECTION_METADATA
        except Exception:
            CHROMA_PERSIST_DIR = "./chroma_db"
            COLLECTION_NAME = "multimodal_embeddings"
            COLLECTION_METADATA = {"hnsw:space": "cosine"}

        pd = persist_directory or CHROMA_PERSIST_DIR

        self.client = chromadb.PersistentClient(
            path=pd,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata=COLLECTION_METADATA
        )
        log.info("chroma_initialized", extra={"path": pd, "collection": COLLECTION_NAME})

    def add(self, embedding: List[float], text: Optional[str] = None, image_url: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        doc_metadata = metadata or {}
        if text:
            doc_metadata["text"] = text
        if image_url:
            doc_metadata["image_url"] = image_url
        doc_metadata["has_text"] = text is not None
        doc_metadata["has_image"] = image_url is not None
        document = text if text else f"Image: {image_url[:50] if image_url else 'N/A'}"
        try:
            self.collection.add(ids=[doc_id], embeddings=[embedding], documents=[document], metadatas=[doc_metadata])
            log.debug("added_vector", extra={"id": doc_id, "has_text": bool(text), "has_image": bool(image_url)})
        except Exception as e:
            log.exception("add_vector_error", extra={"error": str(e)})
            raise
        return doc_id

    def add_many(self, embeddings_per_item: List[List[List[float]]], texts: List[Optional[str]], image_urls: List[Optional[str]], metadatas: List[Optional[Dict[str, Any]]]) -> List[List[str]]:
        flat_ids: List[str] = []
        flat_embeddings: List[List[float]] = []
        flat_documents: List[str] = []
        flat_metas: List[Dict[str, Any]] = []

        entity_ids = [str(uuid.uuid4()) for _ in embeddings_per_item]

        for i, vectors in enumerate(embeddings_per_item):
            text = texts[i] if i < len(texts) else None
            image_url = image_urls[i] if i < len(image_urls) else None
            meta = metadatas[i] if i < len(metadatas) else None
            base_meta = meta.copy() if meta else {}
            base_meta["entity_id"] = entity_ids[i]
            base_meta["has_text"] = text is not None
            base_meta["has_image"] = image_url is not None
            document = text if text else f"Image: {image_url[:50] if image_url else 'N/A'}"
            for sub_idx, emb in enumerate(vectors):
                vid = str(uuid.uuid4())
                per_vector_meta = base_meta.copy()
                per_vector_meta["vector_index"] = sub_idx
                flat_ids.append(vid)
                flat_embeddings.append(emb)
                flat_documents.append(document)
                flat_metas.append(per_vector_meta)

        if flat_ids:
            try:
                self.collection.add(ids=flat_ids, embeddings=flat_embeddings, documents=flat_documents, metadatas=flat_metas)
                log.info("added_many_vectors", extra={"count": len(flat_ids), "entity_count": len(embeddings_per_item)})
            except Exception as e:
                log.exception("add_many_error", extra={"error": str(e)})
                raise

        returned_ids: List[List[str]] = []
        idx = 0
        for vectors in embeddings_per_item:
            count = len(vectors)
            returned_ids.append(flat_ids[idx: idx + count])
            idx += count

        return returned_ids

    def search(self, query_embedding: List[float], top_k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        where = filter_metadata if filter_metadata else None
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        except Exception as e:
            log.exception("query_error", extra={"error": str(e)})
            raise
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "similarity_score": round(1 - results['distances'][0][i], 4), # round to 4 decimal places
                    "metadata": results['metadatas'][0][i],
                    "document": results['documents'][0][i]
                })
        log.debug("search_completed", extra={"top_k": top_k, "returned": len(formatted_results)})
        return formatted_results


    def delete(self, ids: List[str]) -> int:
        if not ids:
            log.debug("delete_called_with_no_ids")
            return 0
        try:
            self.collection.delete(ids=ids)
            log.info("deleted_vectors", extra={"requested": len(ids)})
            return len(ids)
        except Exception as e:
            log.exception("delete_error", extra={"error": str(e)})
            raise

