import sys
from pathlib import Path

# Add parent directory to path so we can import from app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from datasets import load_dataset
from PIL import Image
from app.config import BASE_URL
from app.logger import get_logger

log = get_logger("index")


def image_to_data_uri(img_obj: Any) -> Optional[str]:
    """
    Convert various image representations to a data URI string.
    Supports:
      - str (URL or local path). If local path exists, it will be converted to data URI.
      - dict with 'bytes' or 'path' keys (as returned by some datasets Image features).
      - PIL.Image.Image instances.
    Returns None when img_obj is falsy.
    """
    if not img_obj:
        return None

    # If already a string, assume it's a URL or data URI. If it's a local path file, convert it.
    if isinstance(img_obj, str):
        if img_obj.startswith("data:") or img_obj.startswith("http://") or img_obj.startswith("https://"):
            return img_obj
        if os.path.exists(img_obj):
            with open(img_obj, "rb") as f:
                b = f.read()
            mime = "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")
        # fallback: treat as URL
        return img_obj

    # If datasets Image feature gives a dict with bytes or path
    if isinstance(img_obj, dict):
        if "bytes" in img_obj and img_obj["bytes"] is not None:
            b = img_obj["bytes"]
            mime = "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")
        if "path" in img_obj and img_obj["path"] and os.path.exists(img_obj["path"]):
            with open(img_obj["path"], "rb") as f:
                b = f.read()
            mime = "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")
        # if none available, try string conversion
        if "bytes" not in img_obj and "path" not in img_obj:
            # fallback: try str()
            return str(img_obj)

    # If PIL Image
    if isinstance(img_obj, Image.Image):
        with io.BytesIO() as out:
            img_obj.convert("RGB").save(out, format="JPEG")
            b = out.getvalue()
        mime = "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

    # Unknown type: try bytes
    if isinstance(img_obj, (bytes, bytearray)):
        mime = "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(bytes(img_obj)).decode("utf-8")

    return None


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = []
        for _ in range(size):
            try:
                chunk.append(next(it))
            except StopIteration:
                break
        if not chunk:
            break
        yield chunk


def build_items_from_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for ex in examples:
        text = ex.get("productDisplayName") or ex.get("caption") or ex.get("label") or None
        img = ex.get("image")
        image_payload = image_to_data_uri(img)
        item: Dict[str, Any] = {}
        if text:
            item["text"] = text
        if image_payload:
            item["image"] = image_payload
        # include other metadata columns if present (avoid sending large objects)
        metadata = {}
        for k, v in ex.items():
            if k in ("text", "image"):
                continue
            # keep simple metadata types only
            if isinstance(v, (str, int, float, bool, type(None))):
                metadata[k] = v
        if metadata:
            item["metadata"] = metadata
        if not item:
            continue
        items.append(item)
    return items




def batch_ingest():
    # Handle the LocalFileSystem caching issue with HuggingFace datasets
    dataset_name = "ashraq/fashion-product-images-small"
    
    ds = load_dataset(dataset_name)['train']
    

    total = len(ds)
    base_url = BASE_URL.rstrip('/') + '/'
    ingest_url = f"{base_url.rstrip('/')}/ingest/batch"
    headers = {"Content-Type": "application/json"}
    batch_size = 64
    for i, chunk_indices in enumerate(chunked_iterable(range(total), batch_size)):
        batch_examples = [ds[idx] for idx in chunk_indices]
        items = build_items_from_examples(batch_examples)
        if not items:
            print(f"Batch {i}: no valid items, skipping.")
            continue

        payload = {"items": items}
        try:
            resp = requests.post(ingest_url, headers=headers, data=json.dumps(payload), timeout=60)
            if resp.status_code == 200:
                log.info("batch_ingested", extra={"batch": i, "items": len(items)})
            else:
                log.error("batch_failed", extra={"batch": i, "status": resp.status_code, "text": resp.text})
        except Exception as e:
            log.exception("batch_exception", extra={"batch": i, "error": str(e)})


if __name__ == "__main__":
    print("=" * 60)
    print("Batch ingesting examples from fashion-product-images-small dataset...")
    batch_ingest()
    print("Done.")
    print("=" * 60)