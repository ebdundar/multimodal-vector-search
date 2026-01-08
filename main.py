from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.utils import load_image
from app.config import CORS_ALLOW_ORIGINS, HOST, PORT
from app.logger import get_logger
from app.schemas import (
    IngestRequest,
    IngestResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    SearchRequest,
    SearchResponse,
    DeleteRequest,
    DeleteResponse,
)
from app.dependencies import ingest_service, search_service, vector_db
from app.middleware import request_id_middleware_factory


# structured logger for the service
log = get_logger("main")

app = FastAPI(
    title="Multimodal Vector Search API",
    description="API for ingesting and searching text and images using vector embeddings",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register request lifecycle middleware
app.middleware("http")(request_id_middleware_factory(app))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal Vector Search API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Ingest text and/or images",
            "POST /ingest/batch": "Batch ingest multiple items",
            "POST /search": "Search for similar items",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    log.debug("health_check")
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest text and/or images into the vector database.

    At least one of text or image must be provided.
    """
    try:
        return ingest_service.ingest(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.exception("ingest_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")


# New batch ingest endpoint (multivector support)
@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(request: BatchIngestRequest):
    """Batch ingest multiple items. Stores multiple vectors per entity."""
    if not request.items:
        return BatchIngestResponse(results=[])

    try:
        return ingest_service.batch_ingest(request)
    except Exception as e:
        log.exception("batch_ingest_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error during batch ingestion: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar items using text or image query.

    Exactly one of query_text or query_image must be provided.
    """
    try:
        # If image query provided, load the image and use search_with_image
        if request.query_image:
            img = load_image(request.query_image)
            return search_service.search_with_image(img, top_k=request.top_k, filter_metadata=request.filter_metadata)
        # otherwise do a text search
        return search_service.search(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.exception("search_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")


@app.delete("/items", response_model=DeleteResponse)
async def delete_items(request: DeleteRequest):
    """Delete items by their IDs from the vector database."""
    try:
        if not request.ids:
            return DeleteResponse(deleted_count=0, message="No ids provided")
        deleted = vector_db.delete(request.ids)
        return DeleteResponse(deleted_count=deleted, message=f"Requested deletion of {deleted} ids")
    except Exception as e:
        log.exception("delete_items_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Error deleting items: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    log.info("starting_uvicorn", extra={"host": HOST, "port": PORT})
    uvicorn.run(app, host=HOST, port=PORT)
