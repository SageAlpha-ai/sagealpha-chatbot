"""
SageAlpha.ai Vector Store
Local vector store with Azure OpenAI embeddings support
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class VectorStore:
    """
    Vector store supporting Azure OpenAI embeddings (production) and
    local dummy embeddings (development).
    """

    def __init__(self, store_dir: str = "vector_store_data") -> None:
        """
        Initialize the vector store.

        Args:
            store_dir: Directory to store embeddings and metadata
        """
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self.emb_path = os.path.join(self.store_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.store_dir, "metadata.json")

        # Azure configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""
        ).strip()

        # Determine mode
        if not self.embedding_deployment or self.embedding_deployment.lower() in (
            "none",
            "null",
        ):
            self.local_mode = True
        else:
            self.local_mode = False

        # Initialize client if not local mode
        self.client: Optional[AzureOpenAI] = None
        if not self.local_mode:
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )
        else:
            print("[VectorStore] LOCAL MODE ENABLED — Azure embeddings disabled.")

        # Document storage
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.temporary_doc_ids: set = set()

        self._load()

    def _load(self) -> None:
        """Load existing embeddings and metadata from disk."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for d in data:
                    self.doc_ids.append(d["doc_id"])
                    self.texts.append(d["text"])
                    self.metas.append(d["meta"])

        if os.path.exists(self.emb_path):
            self.embeddings = np.load(self.emb_path)
        else:
            self.embeddings = None

        print(f"[VectorStore] Loaded {len(self.doc_ids)} documents")

    def _save(self) -> None:
        """Save embeddings and metadata to disk."""
        data = []
        for i, doc_id in enumerate(self.doc_ids):
            data.append(
                {"doc_id": doc_id, "text": self.texts[i], "meta": self.metas[i]}
            )

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if self.embeddings is None:
            np.save(self.emb_path, np.zeros((0, 0)))
        else:
            np.save(self.emb_path, self.embeddings)

    def embed(self, texts: str | List[str]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Local mode: generate deterministic dummy vectors
        if self.local_mode:
            return self._local_embed(texts)

        # Production mode: Azure OpenAI embeddings
        resp = self.client.embeddings.create(
            model=self.embedding_deployment, input=texts
        )

        out = []
        for item in resp.data:
            v = np.array(item.embedding, dtype="float32")
            v /= np.linalg.norm(v) + 1e-10
            out.append(v)

        return np.vstack(out)

    def _local_embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate local dummy embeddings (deterministic based on text hash).

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of embeddings
        """
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.random(512).astype("float32")
            v /= np.linalg.norm(v) + 1e-10
            out.append(v)

        return np.vstack(out)

    def add_document(
        self, doc_id: str, text: str, meta: Dict[str, Any]
    ) -> None:
        """
        Add a document to the store.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            meta: Document metadata
        """
        emb = self.embed(text)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.doc_ids.append(doc_id)
        self.texts.append(text)
        self.metas.append(meta)
        self._save()

    def add_temporary_document(
        self, doc_id: str, text: str, meta: Dict[str, Any]
    ) -> None:
        """
        Add a temporary document (cleared after query).

        Args:
            doc_id: Unique document identifier
            text: Document text content
            meta: Document metadata
        """
        emb = self.embed(text)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.doc_ids.append(doc_id)
        self.texts.append(text)
        self.metas.append(meta)
        self.temporary_doc_ids.add(doc_id)
        self._save()

    def clear_temporary_documents(self) -> None:
        """Remove all temporary documents from the store."""
        keep_ids = []
        keep_texts = []
        keep_metas = []

        for i, doc_id in enumerate(self.doc_ids):
            if doc_id not in self.temporary_doc_ids:
                keep_ids.append(doc_id)
                keep_texts.append(self.texts[i])
                keep_metas.append(self.metas[i])

        if keep_texts:
            self.embeddings = self.embed(keep_texts)
        else:
            self.embeddings = None

        self.doc_ids = keep_ids
        self.texts = keep_texts
        self.metas = keep_metas
        self.temporary_doc_ids = set()
        self._save()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of search results with doc_id, text, meta, and score
        """
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        q = self.embed(query)[0]
        sims = self.embeddings @ q
        idx = np.argsort(-sims)[:k]

        results = []
        for i in idx:
            results.append(
                {
                    "doc_id": self.doc_ids[i],
                    "text": self.texts[i],
                    "meta": self.metas[i],
                    "score": float(sims[i]),
                }
            )

        return results

    def save_index(self) -> None:
        """Explicitly save the index to disk."""
        self._save()

    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.doc_ids)
