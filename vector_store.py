import os
import json
import numpy as np

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

class VectorStore:
    """
    Vector store that supports:
    ✅ Azure OpenAI embeddings (production)
    ✅ Dummy local embeddings (development)
    """

    def __init__(self, store_dir="vector_store_data"):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self.emb_path = os.path.join(self.store_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.store_dir, "metadata.json")

        # Azure config
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        # self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        # # ✅ LOCAL MODE if embedding deployment missing
        # self.local_mode = not bool(self.embedding_deployment)

        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

         # Force local_mode if empty OR "none" OR "null"
        if not self.embedding_deployment or self.embedding_deployment.lower() in ("none", "null"):
               self.local_mode = True
        else:
                self.local_mode = False

        
        if not self.local_mode:
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version
            )
        else:
            print("[VectorStore] LOCAL MODE ENABLED — Azure embeddings disabled.")

        self.doc_ids = []
        self.texts = []
        self.metas = []
        self.embeddings = None
        self.temporary_doc_ids = set()

        self._load()

    def _load(self):
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

    def _save(self):
        data = []
        for i, doc_id in enumerate(self.doc_ids):
            data.append({
                "doc_id": doc_id,
                "text": self.texts[i],
                "meta": self.metas[i]
            })

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        if self.embeddings is None:
            np.save(self.emb_path, np.zeros((0, 0)))
        else:
            np.save(self.emb_path, self.embeddings)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        # ✅ LOCAL MODE — generate dummy vectors
        if self.local_mode:
            print("[VectorStore] Using dummy local embeddings")
            out = []
            for text in texts:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                v = rng.random(512).astype("float32")
                v /= (np.linalg.norm(v) + 1e-10)
                out.append(v)
            return np.vstack(out)

        # ✅ PRODUCTION MODE — real Azure embeddings
        resp = self.client.embeddings.create(
            model=self.embedding_deployment,
            input=texts
        )

        out = []
        for item in resp.data:
            v = np.array(item.embedding, dtype="float32")
            v /= (np.linalg.norm(v) + 1e-10)
            out.append(v)

        return np.vstack(out)
    
        # ------------------------------
    # Local embedding generator
    # ------------------------------
    def _local_embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.random(512).astype("float32")
            v /= (np.linalg.norm(v) + 1e-10)
            out.append(v)

        return np.vstack(out)


    def add_document(self, doc_id, text, meta):
        emb = self.embed(text)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.doc_ids.append(doc_id)
        self.texts.append(text)
        self.metas.append(meta)
        self._save()

    def add_temporary_document(self, doc_id, text, meta):
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

    def clear_temporary_documents(self):
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

    def search(self, query, k=5):
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        q = self.embed(query)[0]

        sims = self.embeddings @ q

        idx = np.argsort(-sims)[:k]

        results = []
        for i in idx:
            results.append({
                "doc_id": self.doc_ids[i],
                "text": self.texts[i],
                "meta": self.metas[i],
                "score": float(sims[i])
            })

        return results
