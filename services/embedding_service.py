"""Embedding service for generating text and image embeddings using CLIP."""
from typing import Optional, List, Tuple, Any, cast
import torch
import numpy as np
from PIL import Image
from app.logger import get_logger

log = get_logger("embedding_service")


class EmbeddingService:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "Failed to import `sentence_transformers`. This often happens when the installed "
                "version of `huggingface-hub` is incompatible (missing `cached_download`).\n"
                "Recommended fixes:\n"
                "  1) pip install 'huggingface-hub==0.16.4'\n"
                "  2) or upgrade/downgrade `sentence-transformers` to a compatible version.\n"
                f"Original error: {e}"
            ) from e

        try:
            from app.config import MODEL_NAME
        except Exception:
            MODEL_NAME = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        chosen_model = model_name or MODEL_NAME
        self.model = SentenceTransformer(chosen_model, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        log.info("model_loaded", extra={"model": chosen_model, "device": self.device, "dim": self.embedding_dim})

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length (L2 normalization).
        
        Args:
            embedding: NumPy array of shape (dim,) or (batch_size, dim)
            
        Returns:
            Normalized embedding array with the same shape
        """
        # Handle both single vector and batch cases
        if embedding.ndim == 1:
            # Single vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
            return embedding
        else:
            # Batch of vectors
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            # Avoid division by zero - set zero norms to 1 (keeps zero vectors as zero)
            norms = np.where(norms == 0, 1, norms)
            return embedding / norms

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = self._normalize_embedding(embedding)
        log.debug("embed_text", extra={"text_snippet": text[:100]})
        return embedding.tolist()

    def embed_image(self, image: Image.Image) -> List[float]:
        pil_img = image.convert("RGB")
        embedding = self.model.encode(cast(Any, [pil_img]), convert_to_numpy=True)
        embedding = self._normalize_embedding(embedding)
        log.debug("embed_image", extra={"image_size": pil_img.size})
        return embedding[0].tolist()

    def embed(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> List[List[float]]:
        batch_result = self.embed_batch([(text, image)])
        log.debug("embed_called", extra={"has_text": text is not None, "has_image": image is not None})
        return batch_result[0]

    def embed_batch(self, items: List[Tuple[Optional[str], Optional[Image.Image]]]) -> List[List[List[float]]]:
        text_indices = []
        text_values = []
        image_indices = []
        image_values = []

        for i, (text, image) in enumerate(items):
            if text is not None:
                text_indices.append(i)
                text_values.append(text)
            if image is not None:
                image_indices.append(i)
                image_values.append(image.convert("RGB"))

        text_emb_by_index = {}
        image_emb_by_index = {}

        if text_values:
            text_embs = self.model.encode(text_values, convert_to_numpy=True)
            text_embs = self._normalize_embedding(text_embs)
            for idx, emb in zip(text_indices, text_embs):
                text_emb_by_index[idx] = emb

        if image_values:
            image_embs = self.model.encode(cast(Any, image_values), convert_to_numpy=True)
            image_embs = self._normalize_embedding(image_embs)
            for idx, emb in zip(image_indices, image_embs):
                image_emb_by_index[idx] = emb

        log.debug("embed_batch_completed", extra={"items": len(items), "text_count": len(text_values), "image_count": len(image_values)})

        result_embeddings: List[List[List[float]]] = []
        for i, (text, image) in enumerate(items):
            vectors_for_item: List[List[float]] = []
            if i in text_emb_by_index:
                vectors_for_item.append(text_emb_by_index[i].tolist())
            if i in image_emb_by_index:
                vectors_for_item.append(image_emb_by_index[i].tolist())
            if not vectors_for_item:
                raise ValueError(f"Either text or image must be provided for item at index {i}")
            result_embeddings.append(vectors_for_item)

        return result_embeddings

