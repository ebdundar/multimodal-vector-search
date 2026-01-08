# Multimodal Vector Search API

A backend service for ingesting and searching text and images using vector embeddings. This API allows you to store text and/or images as vector embeddings along with their metadata and perform similarity searches using either text or image queries.

## Features
- Ingest text and/or images with metadata
- Batch ingestion support
- Search using text or image queries
- Delete items by IDs
- Configurable top-K results and metadata filtering
- Cross-modal search (e.g., search images with text queries)
- Persistent vector storage using ChromaDB
- FastAPI-based RESTful API with automatic documentation
- Dockerized for easy deployment
- Extensible design for future enhancements
- Unit tests for core functionality
- Detailed README with setup and usage instructions
- Example scripts for common operations
- Logging and error handling
- Input validation with Pydantic
- Modular code structure for maintainability

### Project Structure
```
multi-modal-vector-search/
├── app/                   # Core application package
│   ├── config.py          # Configuration
│   ├── dependencies.py    # Dependency injection
│   ├── logger.py
│   ├── middleware.py
│   ├── schemas.py         # Pydantic models
│   ├── utils.py
│   └── vector_db.py
├── services/              # Business logic services
├── ├── embedding_service.py
│   ├── ingest.py
│   └── search.py
├── scripts/               # Utility scripts
│   └── batch_ingest.py    # Batch ingestion script
├── examples/              # Example usage
│   └── example_usage.py
├── tests/                 # Test files
│   ├── test_delete_endpoint.py
│   ├── test_ingest_service.py
│   └── test_search_service.py
├── main.py                # FastAPI application entry point
├── requirements.txt
├── README.md
└── Dockerfile
```
## Technology Choices

### Programming Language: Python
- **Why**: Excellent ecosystem for AI/ML with mature libraries for embeddings, image processing, and vector databases
### Web Framework: FastAPI
- **Why**: Modern, fast, async-capable framework with automatic API documentation (OpenAPI/Swagger)

### Embedding Model: CLIP (via sentence-transformers)
- **Why**: CLIP (Contrastive Language-Image Pre-training) is a multimodal model that can embed both text and images into the same vector space, enabling cross-modal search (e.g., search images with text queries)
- **Model**: `clip-ViT-B-32` - Good balance between quality and speed
- **Trade-offs**: 
  - Pros: Single model for both modalities, cross-modal search capability
  - Cons: Larger model size, requires GPU for optimal performance (but works on CPU)

### Vector Database: ChromaDB
- **Why**: 
  - Python-native, easy to set up and use
  - Persistent storage out of the box
  - Good performance for small to medium datasets
  - Simple API
- **Trade-offs**:
  - Pros: Zero configuration, local persistence, simple API
  - Cons: Less scalable than distributed solutions (Qdrant, OpenSearch) for very large datasets

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-modal-vector-search
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The first time you run the application, sentence-transformers will download the CLIP model. This happens automatically.

## Running the Application

Start the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs

## API Endpoints

### POST /ingest

Ingest text and/or images into the vector database.

**Request Body:**
```json
{
  "text": "A beautiful sunset over the ocean",
  "image": "https://example.com/sunset.jpg",
  "metadata": {
    "category": "nature",
    "author": "John Doe"
  }
}
```

**Response:**
```json
{
  "id": "uuid-here",
  "message": "Successfully ingested item"
}
```

**Parameters:**
- `text` (optional): Text string to embed
- `image` (optional): Image URL (http:// or https://) or base64 encoded string
- `metadata` (optional): JSON object with custom metadata

**Note**: At least one of `text` or `image` must be provided.

### POST /search

Search for similar items using text or image query.

**Request Body:**
```json
{
  "query_text": "ocean sunset",
  "top_k": 5,
  "filter_metadata": {
    "category": "nature"
  }
}
```

Or with an image query:
```json
{
  "query_image": "https://example.com/query.jpg",
  "top_k": 10
}
```

**Response:**
```json
{
  "query_type": "text",
  "results": [
    {
      "id": "fc7d98f4-5a70-4b3c-84b8-8108545b5f5b",
      "similarity_score": 0.95,
      "metadata": {
        "text": "A beautiful sunset over the ocean",
        "category": "nature",
        "entity_id": "6da215a0-3254-4cb7-84ee-f551f583c3e0",
        "has_image": true,
        "has_text": true,
        "vector_index": 1 
      },
      "document": "A beautiful sunset over the ocean"
    }
  ]
}
```

**Parameters:**
- `query_text` (optional): Text query string
- `query_image` (optional): Image URL or base64 string
- `top_k` (optional, default: 10): Number of results to return (1-100)
- `filter_metadata` (optional): Metadata filters for search

**Note**: Exactly one of `query_text` or `query_image` must be provided.

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## New Endpoint: DELETE /items

Delete items by their IDs from the vector database.

Request body (JSON):
```json
{
  "ids": ["id1", "id2"]
}
```

Response:
```json
{
  "deleted_count": 2,
  "message": "Requested deletion of 2 ids"
}
```

Example curl:
```bash
curl -X DELETE "http://localhost:8000/items" \
  -H "Content-Type: application/json" \
  -d '{"ids":["id1","id2"]}'
```

## Usage Examples

### Example 1: Ingest Text

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "metadata": {"type": "example"}
  }'
```

### Example 2: Ingest Image from URL

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://example.com/image.jpg",
    "metadata": {"source": "web"}
  }'
```

### Example 3: Search with Text Query

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "brown fox",
    "top_k": 5
  }'
```

### Example 4: Search with Image Query

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "https://example.com/query.jpg",
    "top_k": 10
  }'
```

### Example 5: Delete Items

```bash
curl -X DELETE "http://localhost:8000/items" \
  -H "Content-Type: application/json" \
  -d '{
    "ids": ["uuid-here"]
  }'
```

## Design Decisions and Trade-offs

### 1. CLIP Model Choice
- **Decision**: Use CLIP for both text and image embeddings
- **Rationale**: Enables cross-modal search (text-to-image, image-to-text) in a unified vector space
- **Trade-off**: Single model is simpler but may not be optimal for pure text-only or image-only use cases

### 2. Embedding Combination Strategy
- **Decision**: When both text and image are provided during ingestion, keep them separate in vector database
- **Rationale**: Averaging them would saturate the information.
- **Alternative**: Could concatenate and use a projection layer, but adds complexity

### 3. ChromaDB for Storage
- **Decision**: Use ChromaDB with local persistence
- **Rationale**: Zero-configuration, good for development and small-to-medium datasets
- **Trade-off**: For production at scale, might need distributed solutions (Qdrant, OpenSearch, and so on)

### 4. Similarity Metric
- **Decision**: Use cosine similarity
- **Rationale**: Standard for embeddings, works well with normalized vectors
- **Note**: ChromaDB returns distances, which we convert to similarity scores (1 - distance)

### 5. Image Input Handling
- **Decision**: Support both URLs and base64 strings
- **Rationale**: Flexibility for different use cases (web scraping vs. direct uploads)
- **Trade-off**: Base64 increases payload size but avoids external dependencies


## Future Improvements

Given more time, I would implement:

### 1. **Batch Ingestion** [Done]
- Endpoint to ingest multiple items in one request
- Uses an example dataset for demonstraion purposes
- Update BASE_URL when docker is used

### 2. **Metadata Filtering** [Done]
- Search filtering based on metadata fields

### 3. **Scalability Improvements**
- **Distributed Vector DB**: Migrate to Qdrant or OpenSearch for horizontal scaling
    + Sharding and replication
    + Cloud deployment
    + GPU acceleration for embedding generation
- **Caching**: Add Redis for frequently accessed embeddings
    + Reduce latency for popular queries
    + In-memory storage for hot data
- **Async Processing**: Queue system for batch operations
    + Celery or Redis Queue for background tasks
    + Non-blocking ingestion and search


### 4. **The next steps**
- Fine-tune CLIP on domain-specific data
- Experiment with other multimodal models (e.g., ALIGN, Florence-2)
- Implement weighted embedding combinations
- Use separate models for text and image embeddings if needed
- Dimensionality reduction techniques for large datasets
- Quantization for faster search
- Hybrid search (vector + keyword)
- Re-ranking with more complex models
- Integrate NewRelic or Prometheus for monitoring
- Add CI/CD pipelines
- Comprehensive unit and integration tests

## Assumptions

1. **Image Format**: Assumes common image formats (JPEG, PNG) are supported
2. **Image Size**: Images are reasonably sized (e.g., <224x224 pixels for CLIP)
3. **Token Length**: Text inputs are of moderate length (e.g., <77 tokens for CLIP)
5. **Memory**: CLIP model requires sufficient RAM

## Troubleshooting

### Model Download Issues
If the model fails to download, you can manually download it:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
```

### ChromaDB Permission Issues
Ensure the application has write permissions in the current directory for ChromaDB persistence.

### Image Loading Errors
- Verify image URLs are accessible
- Check base64 encoding is correct
- Ensure image formats are supported (JPEG, PNG)

## Docker

Containerize the FastAPI app using the provided Dockerfile.

### Files
- `Dockerfile`: Multi-stage build with Python 3.11-slim and Gunicorn+Uvicorn
- `.dockerignore`: Excludes unnecessary files from the image

### Build and Run
```bash
# Build the Docker image
docker build -t multimodal-vector-search:local .

# Run the container (exposes port 8080)
docker run --rm -p 8080:8080 multimodal-vector-search:local
```

