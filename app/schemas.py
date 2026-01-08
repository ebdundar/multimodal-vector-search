"""Pydantic request/response models (schemas) for the API.

This is a copy of the top-level `schemas.py` to allow imports via `app.schemas`.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class IngestRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text content to embed")
    image: Optional[str] = Field(None, description="Image URL or base64 encoded string")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store")


class IngestResponse(BaseModel):
    ids: List[str]
    message: str


class BatchIngestItem(BaseModel):
    text: Optional[str] = Field(None, description="Text content to embed")
    image: Optional[str] = Field(None, description="Image URL or base64 encoded string")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store")


class BatchIngestItemResult(BaseModel):
    index: int
    ids: Optional[List[str]] = None
    success: bool
    message: str


class BatchIngestRequest(BaseModel):
    items: List[BatchIngestItem]


class BatchIngestResponse(BaseModel):
    results: List[BatchIngestItemResult]


class SearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Text query for search")
    query_image: Optional[str] = Field(None, description="Image URL or base64 for search")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for search")


class SearchResult(BaseModel):
    id: str
    similarity_score: float
    metadata: Dict[str, Any]
    document: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_type: str


class DeleteRequest(BaseModel):
    ids: List[str]


class DeleteResponse(BaseModel):
    deleted_count: int
    message: str

