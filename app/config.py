"""Centralized runtime configuration for the project.

This file is a drop-in copy of the top-level `config.py` but placed under
`app` so modules can consistently import `app.config`.
"""
from typing import List, Dict

# Logging configuration
LOG_LEVEL: str = "INFO"


# Embedding model to use (sentence-transformers CLIP model)
MODEL_NAME: str = "clip-ViT-B-32"

# ChromaDB persistence directories in the root directory
CHROMA_PERSIST_DIR: str = "chroma_db"  # relative to project root

# Chroma collection configuration
COLLECTION_NAME: str = "multimodal_embeddings"
COLLECTION_METADATA: Dict[str, str] = {"hnsw:space": "cosine"}

# FastAPI / Uvicorn host and port used when running `python main.py`
HOST: str = "0.0.0.0"
PORT: int = 8000

# Base URL for example clients and scripts
BASE_URL: str = f"http://localhost:{PORT}"

# CORS and other web settings
CORS_ALLOW_ORIGINS: List[str] = ["*"]

