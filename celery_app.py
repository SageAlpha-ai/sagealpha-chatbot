"""
SageAlpha.ai v3.0 - Celery Configuration
Background task processing with Redis (production) or RabbitMQ (local fallback)
"""

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# ==================== Environment Detection ====================
IS_PRODUCTION = os.getenv("WEBSITE_SITE_NAME") is not None

# ==================== Broker Configuration ====================
# Priority: CELERY_BROKER_URL > AZURE_REDIS_CONNECTION_STRING > REDIS_URL > local
BROKER_URL = (
    os.getenv("CELERY_BROKER_URL") or
    os.getenv("AZURE_REDIS_CONNECTION_STRING") or
    os.getenv("REDIS_URL") or
    "redis://localhost:6379/0"
)

RESULT_BACKEND = (
    os.getenv("CELERY_RESULT_BACKEND") or
    os.getenv("AZURE_REDIS_CONNECTION_STRING") or
    os.getenv("REDIS_URL") or
    "redis://localhost:6379/0"
)


def make_celery(app_name: str = "sagealpha") -> Celery:
    """Create and configure Celery app."""
    celery = Celery(
        app_name,
        broker=BROKER_URL,
        backend=RESULT_BACKEND,
        include=["tasks"],  # Auto-discover tasks module
    )
    
    # Celery configuration
    celery.conf.update(
        # Task settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        
        # Task execution
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_time_limit=600,  # 10 minutes max per task
        task_soft_time_limit=540,  # Soft limit 9 minutes
        
        # Result backend
        result_expires=3600,  # Results expire after 1 hour
        
        # Worker settings
        worker_prefetch_multiplier=1,  # One task at a time for fairness
        worker_concurrency=2 if IS_PRODUCTION else 1,
        
        # Retry settings
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
        
        # Security
        task_always_eager=not IS_PRODUCTION,  # Run tasks synchronously in dev
    )
    
    return celery


# Create the Celery app instance
celery_app = make_celery()


# ==================== Task Definitions ====================
@celery_app.task(bind=True, max_retries=3)
def process_pdf_async(self, file_path: str, session_id: str, filename: str):
    """
    Async task to process uploaded PDF files.
    - Extract text
    - Chunk and embed
    - Index in vector store
    """
    try:
        from extractor import extract_text_from_pdf_file
        from vector_store import VectorStore
        
        # Extract text
        text = extract_text_from_pdf_file(file_path)
        if not text:
            raise ValueError("Could not extract text from PDF")
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=1500, overlap=200)
        
        # Index in vector store
        vs = VectorStore(use_azure=IS_PRODUCTION)
        doc_id = f"upload_{session_id}_{filename}"
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            vs.add_document(
                doc_id=chunk_id,
                text=chunk,
                meta={"source": f"upload:{filename}", "chunk": i, "session_id": session_id}
            )
        
        return {
            "status": "success",
            "filename": filename,
            "chunks": len(chunks),
            "doc_id": doc_id,
        }
        
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(bind=True, max_retries=3)
def index_document_async(self, doc_id: str, text: str, metadata: dict):
    """Async task to index a document in the vector store."""
    try:
        from vector_store import VectorStore
        
        vs = VectorStore(use_azure=IS_PRODUCTION)
        vs.add_document(doc_id=doc_id, text=text, meta=metadata)
        
        return {"status": "success", "doc_id": doc_id}
        
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks


# ==================== Flask Integration ====================
def init_celery(app):
    """Initialize Celery with Flask app context."""
    celery_app.conf.update(app.config)
    
    class ContextTask(celery_app.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery_app.Task = ContextTask
    return celery_app


if __name__ == "__main__":
    # Run worker: python celery_app.py worker --loglevel=info
    celery_app.start()

