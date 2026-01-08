import pytest
from unittest.mock import Mock

from services.ingest import IngestService
from app.schemas import IngestRequest, BatchIngestRequest, BatchIngestItem


def test_ingest_text_only_success():
    """Ingest with only text should call embedding and vector_db and return ids."""
    embedding = Mock()
    embedding.embed.return_value = [[0.1, 0.2]]
    vector_db = Mock()
    vector_db.add_many.return_value = [["id1", "id2"]]

    svc = IngestService(embedding, vector_db)
    req = IngestRequest(text="hello world", image=None, metadata={"k": 1})

    res = svc.ingest(req)

    assert res.ids == ["id1", "id2"]
    assert "Successfully ingested" in res.message
    embedding.embed.assert_called_once_with(text="hello world", image=None)
    vector_db.add_many.assert_called_once()


def test_ingest_raises_on_empty():
    """Calling ingest with neither text nor image should raise ValueError."""
    svc = IngestService(Mock(), Mock())
    req = IngestRequest(text=None, image=None, metadata=None)
    with pytest.raises(ValueError):
        svc.ingest(req)


def test_batch_ingest_mixed_items():
    """Batch ingest should handle valid items and report errors for empty ones."""
    embedding = Mock()
    embedding.embed_batch.return_value = [[[0.1]], [[0.2]]]
    vector_db = Mock()
    vector_db.add_many.return_value = [["id1"], ["id2"]]

    svc = IngestService(embedding, vector_db)

    items = [
        BatchIngestItem(text="t1", image=None, metadata=None),
        BatchIngestItem(text=None, image=None, metadata=None),
    ]
    req = BatchIngestRequest(items=items)

    res = svc.batch_ingest(req)

    assert len(res.results) == 2
    assert res.results[0].success is True
    assert res.results[0].ids == ["id1"]
    assert res.results[1].success is False
    assert "At least one of 'text' or 'image'" in res.results[1].message

    embedding.embed_batch.assert_called_once_with([("t1", None)])
    vector_db.add_many.assert_called_once()
