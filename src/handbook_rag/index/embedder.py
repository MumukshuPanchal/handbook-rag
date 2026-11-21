"""Turn text into vectors.

An embedding model maps a text to a list of numbers so that texts with similar meaning
end up close together. Chunks are embedded once at indexing time, and each question is
embedded when it is asked. Both must use the same model, or their vectors are not
comparable.
"""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Anything that can embed chunks and questions into the same vector space."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """An open-weight embedding model run locally with sentence-transformers.

    Vectors are normalised to length 1, so cosine similarity and dot product agree.
    The model runs on a GPU when one is available and on the CPU otherwise.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        # Imported here because loading PyTorch takes seconds, and commands that never
        # embed anything should not pay for it.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Model {self._name} does not report its embedding size.")
        return int(dimension)

    @property
    def max_tokens(self) -> int:
        """The longest input the model reads. Text beyond it is ignored."""
        limit = self._model.max_seq_length
        if limit is None:
            raise RuntimeError(f"Model {self._name} does not report its input limit.")
        return int(limit)

    @property
    def device(self) -> str:
        return str(self._model.device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts), prompt=None)

    def embed_query(self, text: str) -> list[float]:
        # Some models are trained with a short instruction in front of questions, so a
        # question and the passage that answers it land close together.
        return self._encode([text], prompt=self._query_prefix or None)[0]

    def _encode(self, texts: list[str], prompt: str | None) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            prompt=prompt,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]
