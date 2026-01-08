import pytest
from unittest.mock import Mock
from PIL import Image

from services.search import SearchService
from app.schemas import SearchRequest


def test_text_search_success():
    embedding = Mock()
    embedding.embed_text.return_value = [0.1, 0.2, 0.3]
    vector_db = Mock()
    vector_db.search.return_value = [
        {"id": "doc1", "similarity_score": 0.9, "metadata": {"a": 1}, "document": "doc1 text"}
    ]

    svc = SearchService(embedding, vector_db)
    req = SearchRequest(query_text="find me", query_image=None, top_k=5, filter_metadata=None)

    res = svc.search(req)

    assert res.query_type == "text"
    assert len(res.results) == 1
    assert res.results[0].id == "doc1"
    embedding.embed_text.assert_called_once_with("find me")
    vector_db.search.assert_called_once()


def test_text_search_raises_on_missing_query():
    svc = SearchService(Mock(), Mock())
    req = SearchRequest(query_text=None, query_image=None, top_k=5, filter_metadata=None)
    with pytest.raises(ValueError):
        svc.search(req)


def test_image_search_success():
    embedding = Mock()
    # create a tiny 1x1 image
    image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    embedding.embed_image.return_value = [0.2, 0.3]
    vector_db = Mock()
    vector_db.search.return_value = [
        {"id": "img1", "similarity_score": 0.8, "metadata": {}, "document": ""}
    ]

    svc = SearchService(embedding, vector_db)
    res = svc.search_with_image(image, top_k=3)

    assert res.query_type == "image"
    assert len(res.results) == 1
    assert res.results[0].id == "img1"
    embedding.embed_image.assert_called_once()
    vector_db.search.assert_called_once()
