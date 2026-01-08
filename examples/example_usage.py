"""Example usage of the Multimodal Vector Search API (moved under examples/).

This is largely the same as the top-level `example_usage.py` but imports
from `app` where applicable.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import base64
from app.config import BASE_URL
from app.logger import get_logger

log = get_logger("example_usage")
BASE_URL = BASE_URL


def ingest_text_example():
    log.info("ingest_text_example_start")
    response = requests.post(
        f"{BASE_URL}/ingest",
        json={
            "text": "A beautiful sunset over the ocean with vibrant orange and pink colors",
            "metadata": {"category": "nature", "type": "description"}
        }
    )
    log.info("ingest_text_example_done", extra={"status": response.status_code})
    return response.json()["ids"]

# ... rest of functions unchanged (omitted here for brevity) ...


def ingest_image_example():
    log.info("ingest_image_example_start")
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
    response = requests.post(
        f"{BASE_URL}/ingest",
        json={
            "image": image_url,
            "metadata": {"source": "unsplash", "category": "nature"}
        }
    )
    log.info("ingest_image_example_done", extra={"status": response.status_code})
    return response.json()["ids"]


def ingest_base64_image_example():
    log.info("ingest_base64_image_example_start")
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
    image_response = requests.get(image_url)
    base64_image = "data:image/jpeg;base64," + base64.b64encode(image_response.content).decode('utf-8')

    response = requests.post(
        f"{BASE_URL}/ingest",
        json={
            "image": base64_image,
            "metadata": {"source": "base64", "category": "test"}
        }
    )
    log.info("ingest_base64_image_example_done", extra={"status": response.status_code})
    return response.json()["ids"]


def ingest_multimodal_example():
    log.info("ingest_multimodal_example_start")
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
    response = requests.post(
        f"{BASE_URL}/ingest",
        json={
            "text": "Mountain landscape at sunset",
            "image": image_url,
            "metadata": {"category": "landscape", "has_both": True}
        }
    )
    log.info("ingest_multimodal_done", extra={"status": response.status_code})
    return response.json()["ids"]


def search_text_example():
    log.info("search_text_example_start")
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query_text": "ocean sunset",
            "top_k": 5
        }
    )
    log.info("search_text_example_done", extra={"status": response.status_code})
    result = response.json()
    log.info("search_text_results", extra={"query_type": result.get('query_type'), "count": len(result.get('results', []))})
    for i, res in enumerate(result['results'], 1):
        print(f"\n  Result {i}:")
        print(f"    ID: {res['id']}")
        print(f"    Similarity: {res['similarity_score']:.4f}")
        print(f"    Metadata: {res['metadata']}")
        print(f"    Document: {res['document'][:100]}...")


def search_with_filter_example():
    log.info("search_with_filter_example_start")
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query_text": "nature",
            "top_k": 5,
            "filter_metadata": {"category": "nature"}
        }
    )
    log.info("search_with_filter_example_done", extra={"status": response.status_code})
    result = response.json()
    log.info("search_with_filter_results", extra={"count": len(result.get('results', []))})
    for i, res in enumerate(result['results'], 1):
        print(f"  {i}. Similarity: {res['similarity_score']:.4f}, Category: {res['metadata'].get('category', 'N/A')}")


def delete_example():
    log.info("delete_example_start")
    response = requests.delete(
        f"{BASE_URL}/items",
        json={"ids": ["id1", "id2"]}
    )
    log.info("delete_example_done", extra={"status": response.status_code})
    return response.json()

if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Vector Search API - Example Usage")
    print("=" * 60)

    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            log.error("api_not_running", extra={"status": health.status_code})
            print("Error: API is not running. Please start the server first.")
            print("Run: python main.py")
            exit(1)

        ingest_text_example()
        ingest_image_example()
        ingest_base64_image_example()
        ingest_multimodal_example()

        import time
        time.sleep(1)

        search_text_example()
        search_with_filter_example()
        delete_example()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        log.error("connection_error")
        print("Error: Could not connect to API.")
        print("Please make sure the server is running:")
        print("  python main.py")
    except Exception as e:
        log.exception("example_usage_error", extra={"error": str(e)})
        print(f"Error: {e}")

