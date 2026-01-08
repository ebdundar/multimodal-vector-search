import sys
import types
from unittest.mock import Mock, patch
import asyncio

# Ensure project root on path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a fake app.dependencies module to avoid heavy initialization on import
fake_deps = types.SimpleNamespace()
fake_deps.vector_db = Mock()
fake_deps.ingest_service = Mock()
fake_deps.search_service = Mock()
sys.modules['app.dependencies'] = fake_deps

from main import delete_items
from app.schemas import DeleteRequest


def test_delete_items_success():
    with patch("main.vector_db") as mock_vector_db:
        mock_vector_db.delete.return_value = 2

        req = DeleteRequest(ids=["id1", "id2"])
        res = asyncio.run(delete_items(req))

        assert res.deleted_count == 2
        assert "Requested deletion" in res.message


def test_delete_items_no_ids():
    req = DeleteRequest(ids=[])
    res = asyncio.run(delete_items(req))
    assert res.deleted_count == 0
    assert res.message == "No ids provided"
