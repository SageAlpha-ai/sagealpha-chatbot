"""
SageAlpha.ai v3.0 - 2025 Flask Application
Modern Flask 3.x with Blueprints, SocketIO, and async support
"""

import io
import os
import re
from datetime import datetime
from functools import wraps
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, current_user
from flask_socketio import SocketIO, emit, join_room
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

from blob_utils import BlobReader
from blueprints import auth_bp, chat_bp, pdf_bp
from blueprints.chat import (
    SESSIONS,
    build_session_memory_sections,
    create_session,
    extract_topic,
)
from db_migrate import run_migrations
from extractor import extract_text_from_pdf_bytes, parse_xbrl_file_to_text
from models import Message, User, db
from vector_store import VectorStore

# Load environment variables
load_dotenv()

# ==================== Environment Detection ====================
IS_PRODUCTION = os.getenv("WEBSITE_SITE_NAME") is not None
IS_AZURE = IS_PRODUCTION  # Alias for clarity

# ==================== Configuration ====================
# Flask
FLASK_SECRET = os.getenv("FLASK_SECRET") or os.urandom(24).hex()
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() in ("1", "true", "yes")

# Database - Support both SQLite (local) and PostgreSQL (production)
DATABASE_URL = os.getenv("DATABASE_URL")

# Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Azure OpenAI - Only initialize if credentials present
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")

# Redis/Celery - For background tasks
REDIS_URL = os.getenv("AZURE_REDIS_CONNECTION_STRING") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL") or REDIS_URL

# ==================== Lazy Azure Client Initialization ====================
_openai_client = None
_search_client = None


def get_openai_client():
    """Lazily initialize Azure OpenAI client only when needed."""
    global _openai_client
    if _openai_client is None and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        from openai import AzureOpenAI
        _openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _openai_client


def get_search_client():
    """Lazily initialize Azure Search client only when needed."""
    global _search_client
    if _search_client is None and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        _search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY),
        )
    return _search_client


def create_app(config_name: str = "default") -> Flask:
    """Application factory for Flask app."""
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = FLASK_SECRET

    # ==================== Database Configuration ====================
    # Priority: DATABASE_URL env var > Azure path > local SQLite
    db_path = None  # Only used for SQLite
    
    if DATABASE_URL:
        # Production PostgreSQL (or other DB via DATABASE_URL)
        db_uri = DATABASE_URL
        # Fix Heroku-style postgres:// -> postgresql://
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        print(f"[startup] Using DATABASE_URL: {db_uri.split('@')[-1] if '@' in db_uri else 'configured'}")
    elif IS_PRODUCTION:
        # Azure App Service with SQLite fallback
        db_dir = "/home/data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "sagealpha.db")
        db_uri = f"sqlite:///{db_path}"
        print(f"[startup] Using Azure SQLite: {db_path}")
    else:
        # Local development SQLite
        db_dir = os.path.dirname(__file__) or "."
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "sagealpha.db")
        db_uri = f"sqlite:///{db_path}"
        print(f"[startup] Using local SQLite: {db_path}")

    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Connection pool settings for production
    if IS_PRODUCTION and DATABASE_URL:
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 10,
            "max_overflow": 20,
        }

    # ==================== Session Configuration ====================
    app.config["SESSION_COOKIE_SECURE"] = IS_PRODUCTION
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    
    # Debug mode
    app.config["DEBUG"] = not IS_PRODUCTION

    # ==================== Initialize Extensions ====================
    db.init_app(app)

    # CORS configuration
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=True,
    )

    # Rate limiting - use Redis in production, memory in dev
    rate_limit_storage = REDIS_URL if IS_PRODUCTION else "memory://"
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=rate_limit_storage,
    )

    # Login manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except Exception:
            return None

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(pdf_bp)

    # Context processor for version and environment
    @app.context_processor
    def inject_globals():
        return {
            "APP_VERSION": read_version(),
            "IS_PRODUCTION": IS_PRODUCTION,
        }

    # ==================== Initialize Database ====================
    with app.app_context():
        # Run SQLite migrations only for SQLite databases
        if db_path and os.path.exists(db_path):
            try:
                run_migrations(db_path)
            except Exception as e:
                print(f"[startup][WARN] Migration check failed: {e!r}")
        
        # Create any missing tables (safe with existing tables)
        db.create_all()
        seed_demo_users()

    return app


def read_version() -> str:
    """Get application version."""
    v = os.getenv("SAGEALPHA_VERSION")
    if v:
        return v.strip()
    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r") as f:
            return f.read().strip()
    except Exception:
        return "3.0.0"


def seed_demo_users():
    """Seed demo users if they don't exist."""
    from datetime import datetime, timezone
    
    print("[startup] Ensuring DB tables and demo users...")
    try:
        demo_users = [
            ("demouser", "DemoUser", "DemoPass123!", "demouser@sagealpha.ai"),
            ("devuser", "DevUser", "DevPass123!", "devuser@sagealpha.ai"),
            ("produser", "ProductionUser", "ProdPass123!", "produser@sagealpha.ai"),
        ]
        created_any = False
        updated_any = False
        
        for username, display, pwd, email in demo_users:
            try:
                existing = User.query.filter_by(username=username).first()
            except Exception:
                # Table might have schema issues, try raw query
                existing = None
            
            if not existing:
                u = User(
                    username=username,
                    display_name=display,
                    password_hash=generate_password_hash(pwd),
                    email=email,
                    is_active=True,
                )
                db.session.add(u)
                created_any = True
            else:
                # Update existing users with new fields if they're missing
                needs_update = False
                if not getattr(existing, "email", None):
                    existing.email = email
                    needs_update = True
                if getattr(existing, "is_active", None) is None:
                    existing.is_active = True
                    needs_update = True
                if needs_update:
                    updated_any = True
        
        if created_any or updated_any:
            db.session.commit()
            if created_any:
                print("[startup] Seeded demo users: demouser / devuser / produser")
            if updated_any:
                print("[startup] Updated existing demo users with new fields")
        else:
            print("[startup] Demo users already exist and are up to date.")
    except Exception as e:
        print(f"[startup][ERROR] Failed to seed demo users: {e!r}")
        # Try to rollback any partial changes
        try:
            db.session.rollback()
        except Exception:
            pass


# Create app instance
app = create_app()

# Initialize SocketIO for real-time chat
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="gevent",
    ping_timeout=60,
    ping_interval=25,
)

# Initialize Azure services
search_client = None
if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY and AZURE_SEARCH_INDEX:
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY),
        )
        print(f"[startup] Connected to Azure Search index: {AZURE_SEARCH_INDEX}")
    except Exception as e:
        print(f"[startup] Failed to initialize SearchClient: {e}")
else:
    print("[startup] Azure Search client not initialized.")

# Azure OpenAI client
client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        print("[startup] AzureOpenAI client initialized.")
    except Exception as e:
        print(f"[startup][WARN] Failed to initialize AzureOpenAI client: {e!r}")
else:
    print("[startup][WARN] Azure OpenAI configuration incomplete.")

# BlobReader
BLOB_CONTAINER = "nse-data-raw"
blob_reader = None
if AZURE_BLOB_CONNECTION_STRING:
    try:
        blob_reader = BlobReader(
            conn_str=AZURE_BLOB_CONNECTION_STRING, container=BLOB_CONTAINER
        )
        print("[startup] BlobReader initialized.")
    except Exception as e:
        print(f"[startup][WARN] Failed to initialize BlobReader: {e!r}")
else:
    print("[WARN] Missing AZURE_BLOB_CONNECTION_STRING.")

# Vector store
VECTOR_STORE_DIR = "vector_store_data"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
vs = VectorStore(store_dir=VECTOR_STORE_DIR)

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==================== Helper Functions ====================


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    if not text:
        return text
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(^|\n)#{1,6}\s*", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"(^|\n)[\-\*\+]\s+", r"\1", text)
    text = re.sub(r"\n[-*_]{3,}\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def search_azure(query_text: str, top_k: int = 5) -> list:
    """Search Azure Cognitive Search index."""
    if not search_client or not query_text:
        return []

    search_kwargs = {"search_text": query_text, "top": top_k}

    if AZURE_SEARCH_SEMANTIC_CONFIG:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = AZURE_SEARCH_SEMANTIC_CONFIG

    try:
        results = search_client.search(**search_kwargs)
    except Exception as e:
        print(f"[azure] search error: {e}")
        return []

    output = []
    for r in results:
        content_parts = []
        for field_name in ["merged_content", "content", "imageCaption"]:
            val = r.get(field_name)
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            if val:
                content_parts.append(str(val))
        text = "\n".join(content_parts) or ""
        meta = {
            "source": r.get("metadata_storage_path") or r.get("source"),
            "people": r.get("people"),
            "organizations": r.get("organizations"),
            "locations": r.get("locations"),
        }
        doc_id = r.get("id") or r.get("metadata_storage_path") or ""
        score = float(r.get("@search.score", 0.0))
        output.append({"doc_id": doc_id, "text": text, "meta": meta, "score": score})

    return output


def build_hybrid_messages(
    user_msg: str, retrieved_docs: list, extra_system_msgs: list | None = None
) -> list:
    """Build messages for hybrid RAG."""
    relevance_threshold = 0.35
    relevant_docs = [
        r for r in retrieved_docs if r.get("score", 0.0) >= relevance_threshold
    ]

    if relevant_docs:
        context_chunks = [
            f"Source: {r['meta'].get('source', r['doc_id'])}\n{r['text']}"
            for r in relevant_docs
            if r.get("text")
        ]
        context_text = "\n\n".join(context_chunks)[:6000]
    else:
        context_text = ""

    system_prompt = (
        "You are SageAlpha, a financial assistant powered by SageAlpha.ai.\n"
        "Use this logic:\n"
        "1. If the Context contains useful information, use it to answer.\n"
        "2. If the Context is empty or not relevant, answer using your own knowledge.\n"
        "3. Be precise and financially accurate.\n"
        "4. Respond in clear plain text only. Do not use markdown formatting.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if extra_system_msgs:
        messages.extend(extra_system_msgs)

    messages.append({"role": "system", "content": f"Context:\n{context_text}"})
    messages.append({"role": "user", "content": user_msg})

    return messages


# ==================== Routes ====================


@app.route("/")
def home():
    """Main chat page."""
    if REQUIRE_AUTH and not (
        hasattr(current_user, "is_authenticated") and current_user.is_authenticated
    ):
        return redirect("/login")
    return render_template("index.html", APP_VERSION=read_version())


@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint for AI conversation."""
    if REQUIRE_AUTH:
        if not (
            (
                hasattr(current_user, "is_authenticated")
                and current_user.is_authenticated
            )
            or session.get("user")
        ):
            return jsonify({"error": "Authentication required"}), 401

    if client is None:
        return jsonify({"error": "LLM backend not configured"}), 500

    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Session management
    if "history" not in session:
        session["history"] = [
            {
                "role": "system",
                "content": "I'm SageAlpha, a financial assistant powered by SageAlpha.ai.",
            }
        ]
    if "sections" not in session:
        session["sections"] = []
    if "current_topic" not in session:
        session["current_topic"] = ""

    history = session["history"]
    sections = session["sections"]
    last_topic = session.get("current_topic", "")

    current_topic = extract_topic(user_msg, last_topic)
    session["current_topic"] = current_topic or ""

    history.append({"role": "user", "content": user_msg})

    session_memory_text = build_session_memory_sections(sections, current_topic)
    extra_system_msgs = []
    if session_memory_text:
        extra_system_msgs.append(
            {
                "role": "system",
                "content": f"Session memory (previous Q&A sections):\n{session_memory_text}",
            }
        )

    top_k = int(payload.get("top_k", 5))
    retrieved = search_azure(user_msg, top_k)
    if not retrieved and search_client is None:
        retrieved = vs.search(user_msg, k=top_k)

    messages = build_hybrid_messages(user_msg, retrieved, extra_system_msgs)

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = response.choices[0].message.content

        history.append({"role": "assistant", "content": ai_msg})
        sections.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": user_msg,
                "answer": ai_msg,
            }
        )
        session["history"] = history
        session["sections"] = sections

        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in retrieved
        ]

        msg_id = str(uuid4())
        message_obj = {"id": msg_id, "role": "assistant", "content": ai_msg}

        return jsonify(
            {
                "id": msg_id,
                "response": ai_msg,
                "message": message_obj,
                "data": message_obj,
                "sources": sources,
            }
        )

    except Exception as e:
        error_msg = f"Backend error: {e!s}"
        msg_id = str(uuid4())
        message_obj = {"id": msg_id, "role": "assistant", "content": error_msg}
        print(f"[chat][ERROR] {e!r}")
        return (
            jsonify(
                {
                    "id": msg_id,
                    "response": error_msg,
                    "message": message_obj,
                    "data": message_obj,
                    "sources": [],
                    "error": str(e),
                }
            ),
            500,
        )


@app.route("/chat_session", methods=["POST"])
def chat_session():
    """Chat endpoint with session management."""
    if REQUIRE_AUTH:
        if not (
            hasattr(current_user, "is_authenticated") and current_user.is_authenticated
        ):
            return jsonify({"error": "Authentication required"}), 401

    if client is None:
        return jsonify({"error": "LLM backend not configured"}), 500

    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
        if s.get("owner") != getattr(current_user, "username", None):
            return jsonify({"error": "Session not found"}), 404
    else:
        s = create_session("New chat", owner=getattr(current_user, "username", None))
        session_id = s["id"]

    s["messages"].append({"role": "user", "content": user_msg, "meta": {}})

    last_topic = s.get("current_topic", "")
    current_topic = extract_topic(user_msg, last_topic)
    s["current_topic"] = current_topic or ""

    session_memory_text = build_session_memory_sections(
        s.get("sections", []), current_topic
    )
    extra_system_msgs = []
    if session_memory_text:
        extra_system_msgs.append(
            {
                "role": "system",
                "content": f"Session memory (previous Q&A sections):\n{session_memory_text}",
            }
        )

    top_k = int(payload.get("top_k", 5))
    retrieved = search_azure(user_msg, top_k)
    if not retrieved and search_client is None:
        retrieved = vs.search(user_msg, k=top_k)

    messages = build_hybrid_messages(user_msg, retrieved, extra_system_msgs)

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = resp.choices[0].message.content

        s["messages"].append({"role": "assistant", "content": ai_msg, "meta": {}})
        s["sections"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "query": user_msg,
                "answer": ai_msg,
            }
        )

        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in retrieved
        ]

        return jsonify(
            {"session_id": session_id, "response": ai_msg, "sources": sources}
        )
    except Exception as e:
        print(f"[chat_session][ERROR] {e!r}")
        s["messages"].append(
            {"role": "assistant", "content": f"Backend error: {e}", "meta": {}}
        )
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """Query endpoint for RAG search."""
    if client is None:
        return jsonify({"error": "LLM backend not configured"}), 500

    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Empty query"}), 400

    top_k = int(payload.get("top_k", 5))
    fetch_pdf = bool(payload.get("fetch_pdf", True))

    results = search_azure(q, top_k)
    if not results and search_client is None:
        results = vs.search(q, k=top_k)

    for r in results:
        att = r.get("meta", {}).get("attachment") or ""
        if not fetch_pdf or not att or not blob_reader:
            continue
        try:
            if att.startswith("https://"):
                pdf_bytes = blob_reader.download_blob_url_to_bytes(att)
            else:
                pdf_bytes = blob_reader.download_blob_to_bytes(att)
            extracted = extract_text_from_pdf_bytes(pdf_bytes)
            meta = {"source": f"pdf_temp:{att}", "attachment": att}
            temp_doc_id = f"temp_pdf::{att}"
            try:
                vs.add_temporary_document(doc_id=temp_doc_id, text=extracted, meta=meta)
            except Exception:
                pass
        except Exception as e:
            print(f"[query] failed to download/extract {att}: {e}")

    if search_client:
        final_results = search_azure(q, top_k)
    else:
        final_results = vs.search(q, k=top_k)

    messages = build_hybrid_messages(q, final_results)

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = resp.choices[0].message.content or ""
        ai_msg = strip_markdown(ai_msg)
        try:
            vs.clear_temporary_documents()
        except Exception:
            pass

        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r.get("score", 0.0)),
            }
            for r in final_results
        ]
        return jsonify({"answer": ai_msg, "sources": sources})
    except Exception as e:
        try:
            vs.clear_temporary_documents()
        except Exception:
            pass
        print(f"[query][ERROR] {e!r}")
        return jsonify({"error": str(e)}), 500


@app.route("/report-html", methods=["GET"])
def report_html():
    """Return report HTML fragment for client-side PDF generation."""
    try:
        html = render_template("sagealpha_reports.html")

        wants_fragment = False
        if request.headers.get("X-Requested-With", "").lower() == "xmlhttprequest":
            wants_fragment = True
        else:
            accept_hdr = (request.headers.get("Accept") or "").lower()
            if accept_hdr.startswith("*/*") or "text/html" not in accept_hdr:
                wants_fragment = True

        if not wants_fragment:
            resp = make_response(html)
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp

        head_match = re.search(
            r"<head[^>]*>([\s\S]*?)</head>", html, flags=re.IGNORECASE
        )
        body_match = re.search(
            r"<body[^>]*>([\s\S]*?)</body>", html, flags=re.IGNORECASE
        )

        head_html = head_match.group(1) if head_match else ""
        body_html = body_match.group(1) if body_match else html

        style_blocks = re.findall(
            r"<style[^>]*>([\s\S]*?)</style>", head_html, flags=re.IGNORECASE
        )

        link_tags = re.findall(
            r"<link[^>]*rel=[\"']stylesheet[\"'][^>]*>",
            head_html,
            flags=re.IGNORECASE,
        )

        injected_print_css = """
        <style>
        #sagealpha-report-fragment, #sagealpha-report-fragment * {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            box-sizing: border-box;
        }
        #sagealpha-report-fragment .container { background: white !important; }
        #sagealpha-report-fragment { width: 1200px; max-width: 1200px; margin: 0 auto; }
        #sagealpha-report-fragment a { color: inherit; text-decoration: none; }
        </style>
        """

        combined_styles = ""
        if style_blocks:
            combined_styles = "<style>" + "\n".join(style_blocks) + "</style>"

        links_html = "\n".join(link_tags)

        fragment = (
            f"<div id='sagealpha-report-fragment' style='background:white;'>\n"
            f"{injected_print_css}\n"
            f"{combined_styles}\n"
            f"{links_html}\n"
            f"{body_html}\n"
            f"</div>"
        )

        resp = make_response(fragment)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    except Exception as e:
        print(f"[report-html][ERROR] {e!r}")
        return jsonify({"error": f"Failed to render report HTML: {e!s}"}), 500


@app.route("/refresh", methods=["POST"])
def refresh():
    """Refresh index endpoint."""
    if REQUIRE_AUTH:
        if not (
            (
                hasattr(current_user, "is_authenticated")
                and current_user.is_authenticated
            )
            or session.get("user")
        ):
            return jsonify({"error": "Authentication required"}), 401
    return jsonify({"status": "refreshed"})


@app.route("/test_search")
def test_search():
    """Test Azure Search endpoint."""
    if not search_client:
        return jsonify(
            {"status": "error", "message": "search_client is None."}
        )
    q = request.args.get("q", "cupid")
    try:
        results = search_client.search(search_text=q, top=3)
        items = []
        for r in results:
            items.append(
                {
                    "id": r.get("id"),
                    "score": r.get("@search.score"),
                    "path": r.get("metadata_storage_path"),
                    "content_preview": (
                        r.get("content") or r.get("merged_content") or ""
                    )[:200],
                }
            )
        return jsonify({"status": "ok", "query": q, "results": items})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler."""
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    print(f"[ERROR] {e!r}")
    return jsonify({"error": str(e)}), code


# ==================== File Upload ====================

# Store uploaded documents per session for RAG context
SESSION_DOCUMENTS: dict = {}  # session_id -> [{"doc_id": str, "filename": str, "text": str}]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks for better retrieval."""
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


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle PDF file upload, extract text, and index for RAG.
    Works in both local mode and with Azure services.
    """
    if REQUIRE_AUTH:
        if not (
            (hasattr(current_user, "is_authenticated") and current_user.is_authenticated)
            or session.get("user")
        ):
            return jsonify({"error": "Authentication required"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    session_id = request.form.get("session_id") or str(uuid4())

    # Check file type
    allowed_extensions = {".pdf", ".txt", ".md", ".csv"}
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"File type {file_ext} not supported. Use: {', '.join(allowed_extensions)}"}), 400

    try:
        # Read file content
        file_bytes = file.read()
        file_size = len(file_bytes)

        # Save file locally
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        print(f"[upload] Saved file: {file_path} ({file_size} bytes)")

        # Extract text based on file type
        extracted_text = ""
        if file_ext == ".pdf":
            from extractor import extract_text_from_pdf_bytes
            extracted_text = extract_text_from_pdf_bytes(file_bytes)
        elif file_ext in {".txt", ".md"}:
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif file_ext == ".csv":
            import pandas as pd
            df = pd.read_csv(io.BytesIO(file_bytes))
            extracted_text = df.to_string()

        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from file"}), 400

        print(f"[upload] Extracted {len(extracted_text)} characters from {filename}")

        # Chunk the text for better retrieval
        chunks = chunk_text(extracted_text, chunk_size=1500, overlap=200)
        print(f"[upload] Created {len(chunks)} chunks")

        # Generate unique document ID
        doc_id = f"upload_{session_id}_{timestamp}_{filename}"

        # Index chunks in vector store
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            meta = {
                "source": f"upload:{filename}",
                "filename": filename,
                "session_id": session_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            try:
                vs.add_document(doc_id=chunk_id, text=chunk, meta=meta)
            except Exception as e:
                print(f"[upload] Failed to index chunk {i}: {e}")

        # Store document info for session
        if session_id not in SESSION_DOCUMENTS:
            SESSION_DOCUMENTS[session_id] = []
        
        SESSION_DOCUMENTS[session_id].append({
            "doc_id": doc_id,
            "filename": filename,
            "file_path": file_path,
            "text_preview": extracted_text[:500],
            "chunk_count": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
        })

        # Save vector store
        vs.save_index()

        return jsonify({
            "success": True,
            "filename": filename,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "characters": len(extracted_text),
            "session_id": session_id,
            "url": f"/uploads/{saved_filename}",
            "message": f"Successfully processed {filename} ({len(chunks)} chunks indexed)",
        })

    except Exception as e:
        print(f"[upload][ERROR] {e!r}")
        return jsonify({"error": f"Upload failed: {e!s}"}), 500


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded files."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/session/<session_id>/documents", methods=["GET"])
def get_session_documents(session_id):
    """Get list of documents uploaded to a session."""
    docs = SESSION_DOCUMENTS.get(session_id, [])
    return jsonify({"documents": docs})


# ==================== WebSocket Events ====================


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    print(f"[ws] Client connected: {request.sid}")
    emit("connected", {"status": "ok", "sid": request.sid})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print(f"[ws] Client disconnected: {request.sid}")


@socketio.on("join")
def handle_join(data):
    """Join a chat room."""
    room = data.get("room")
    if room:
        join_room(room)
        emit("joined", {"room": room}, room=room)


@socketio.on("chat_message")
def handle_chat_message(data):
    """Handle real-time chat messages via WebSocket."""
    if client is None:
        emit("error", {"message": "LLM backend not configured"})
        return

    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id")

    if not user_msg:
        emit("error", {"message": "Empty message"})
        return

    # Emit typing indicator
    emit("typing", {"status": True})

    try:
        top_k = int(data.get("top_k", 5))
        retrieved = search_azure(user_msg, top_k)
        if not retrieved and search_client is None:
            retrieved = vs.search(user_msg, k=top_k)

        messages = build_hybrid_messages(user_msg, retrieved)

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = response.choices[0].message.content

        sources = [
            {"doc_id": r["doc_id"], "source": r["meta"].get("source")}
            for r in retrieved
        ]

        emit("typing", {"status": False})
        emit(
            "chat_response",
            {
                "id": str(uuid4()),
                "response": ai_msg,
                "sources": sources,
                "session_id": session_id,
            },
        )

    except Exception as e:
        emit("typing", {"status": False})
        emit("error", {"message": f"Backend error: {e!s}"})
        print(f"[ws][ERROR] {e!r}")


# ==================== Entry Point ====================

def find_available_port(host: str = "0.0.0.0", start_port: int = 8000, max_port: int = 8010) -> int:
    """Find an available port in the given range (development only)."""
    import socket
    import time
    
    for try_port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, try_port))
            sock.close()
            if try_port != start_port:
                print(f"[startup] Port {start_port} busy, using port {try_port}")
            return try_port
        except OSError:
            sock.close()
            print(f"[startup] Port {try_port} is busy, trying next...")
            time.sleep(1)
    
    return None


if __name__ == "__main__":
    host = "0.0.0.0"
    
    if IS_PRODUCTION:
        # ===== PRODUCTION (Azure App Service) =====
        port = int(os.environ.get("PORT", 8000))
        debug = False
        display_host = host
        print(f"[startup] Production mode: PORT={port}, DEBUG=False")
    else:
        # ===== DEVELOPMENT =====
        port = find_available_port(host)
        debug = True
        display_host = "127.0.0.1"
        
        if port is None:
            print("")
            print("=" * 60)
            print("  ERROR: All ports 8000-8010 are in use!")
            print("=" * 60)
            print("")
            print("  To fix this on Windows, run:")
            print("    netstat -ano | findstr :8000")
            print("    taskkill /PID <pid> /F")
            print("")
            print("  Or specify a different port:")
            print("    set PORT=9000 && python app.py")
            print("=" * 60)
            exit(1)
    
    # Print startup banner
    version = read_version()
    print("")
    print("=" * 60)
    print(f"  SageAlpha.ai v{version}")
    print(f"  Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    print("=" * 60)
    print(f"  Server: http://{display_host}:{port}")
    print(f"  Debug: {debug}")
    print("")
    print("  Press CTRL+C to quit")
    print("=" * 60)
    print("")
    
    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            allow_unsafe_werkzeug=not IS_PRODUCTION,
        )
    except KeyboardInterrupt:
        print("\n[shutdown] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Server failed: {e}")
        exit(1)
