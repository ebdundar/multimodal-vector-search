"""Dependency container exposed under app.dependencies.

This mirrors the top-level `dependencies.py` but references `app.*` modules to
ensure package-style imports work consistently.
"""
from services.embedding_service import EmbeddingService
from app.vector_db import VectorDB
from services.ingest import IngestService
from services.search import SearchService
from app.config import CHROMA_PERSIST_DIR

_embedding_service = EmbeddingService()
_vector_db = VectorDB(persist_directory=CHROMA_PERSIST_DIR)

ingest_service = IngestService(_embedding_service, _vector_db)
search_service = SearchService(_embedding_service, _vector_db)

embedding_service = _embedding_service
vector_db = _vector_db

