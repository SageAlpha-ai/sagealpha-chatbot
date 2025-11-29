import os
import io
import json
import tempfile
from datetime import datetime
from uuid import uuid4

from flask import (
    Flask, request, session, render_template, make_response, jsonify,
    redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import pandas as pd
from functools import wraps

from openai import AzureOpenAI

from blob_utils import BlobReader
from extractor import extract_text_from_pdf_bytes, parse_xbrl_file_to_text
from vector_store import VectorStore
import re

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from werkzeug.exceptions import HTTPException

from flask_login import login_user, logout_user, current_user, LoginManager
from models import db, User, Message
from auth import auth as auth_bp  # your auth blueprint (kept)

load_dotenv()

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
FLASK_SECRET = os.getenv("FLASK_SECRET") or os.urandom(24).hex()

REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")

# Azure Cognitive Search config
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
    print("[WARN] Missing Azure OpenAI configuration – chat may not work correctly.")

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
    print("[WARN] Missing Azure Search configuration – RAG search will be disabled.")

# Azure Cognitive Search client
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

if not AZURE_BLOB_CONNECTION_STRING:
    print("[WARN] Missing AZURE_BLOB_CONNECTION_STRING – blob features disabled.")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = FLASK_SECRET

# ensure DB directory and configure sqlite DB (Azure persistent)
AZURE_DB_DIR = "/home/site/wwwroot"
os.makedirs(AZURE_DB_DIR, exist_ok=True)

db_path = os.path.join(AZURE_DB_DIR, "sagealpha.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# register auth blueprint
try:
    app.register_blueprint(auth_bp)
except Exception as e:
    print("[startup] auth blueprint register failed (maybe already registered):", e)

# ---------------- DB + Login Manager Init ----------------
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


def seed_demo_users():
    """
    Ensure database tables exist and seed demo users:
      - demouser / DemoPass123!
      - devuser  / DevPass123!
      - produser / ProdPass123!
    Only creates them if they don't already exist.
    """
    print("[startup] Ensuring DB tables and demo users...")
    try:
        db.create_all()
        demo_users = [
            ("demouser", "DemoUser", "DemoPass123!"),
            ("devuser", "DevUser", "DevPass123!"),
            ("produser", "ProductionUser", "ProdPass123!"),
        ]
        created_any = False
        for username, display, pwd in demo_users:
            existing = User.query.filter_by(username=username).first()
            if not existing:
                u = User(
                    username=username,
                    display_name=display,
                    password_hash=generate_password_hash(pwd),
                )
                db.session.add(u)
                created_any = True
        if created_any:
            db.session.commit()
            print("[startup] Seeded demo users: demouser / devuser / produser")
        else:
            print("[startup] Demo users already exist.")
    except Exception as e:
        print("[startup][ERROR] Failed to seed demo users:", repr(e))


@app.before_first_request
def initialize():
    """
    Runs once in the app lifetime (per process) before handling the first request.
    Ensures DB exists and demo users are present.
    """
    print("[startup] initialize() called – preparing DB & demo users...")
    seed_demo_users()


@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None


def read_version():
    v = os.getenv("SAGEALPHA_VERSION")
    if v:
        return v.strip()
    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r") as f:
            return f.read().strip()
    except Exception:
        return "0.0.0"


@app.context_processor
def inject_version():
    return {"APP_VERSION": read_version()}


# Azure OpenAI client (wrap errors if missing config)
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

BLOB_CONTAINER = "nse-data-raw"
blob_reader = BlobReader(conn_str=AZURE_BLOB_CONNECTION_STRING, container=BLOB_CONTAINER)

VECTOR_STORE_DIR = "vector_store_data"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
vs = VectorStore(store_dir=VECTOR_STORE_DIR)

INDEX_ON_STARTUP = os.getenv("INDEX_ON_STARTUP", "false").lower() == "true"


def startup_index():
    print("[startup] Listing metadata CSVs...")
    csv_prefix = "metadata/"
    csv_blobs = blob_reader.list_blobs(prefix=csv_prefix)
    total_rows = 0

    for b in csv_blobs:
        if not b.name.lower().endswith(".csv"):
            continue
        print(f"[startup] Loading CSV: {b.name}")
        data = blob_reader.download_blob_to_bytes(b.name)
        df = pd.read_csv(io.BytesIO(data))

        for idx, row in df.iterrows():
            symbol = str(row.get("symbol", "") or "")
            desc = str(row.get("desc", "") or "")
            attfile = str(row.get("attchmntFile", "") or "")
            an_dt = str(row.get("an_dt", "") or row.get("dt", ""))
            doc_text = (
                f"Symbol: {symbol}\nDate: {an_dt}\nDescription: {desc}\nAttachment: {attfile}"
            )
            meta = {
                "source": f"csv:{b.name}",
                "symbol": symbol,
                "csv_row_index": int(idx),
                "attachment": attfile,
                "timestamp": an_dt,
            }
            vs.add_document(
                doc_id=f"csv::{b.name}::row::{idx}",
                text=doc_text,
                meta=meta,
            )
            total_rows += 1

    xbrl_prefix = "cupid_data/"
    xbrl_blobs = blob_reader.list_blobs(prefix=xbrl_prefix)

    for b in xbrl_blobs:
        if not (b.name.lower().endswith(".xml") or b.name.lower().endswith(".xbrl")):
            continue
        print(f"[startup] Parsing XBRL: {b.name}")
        data = blob_reader.download_blob_to_bytes(b.name)
        try:
            text = parse_xbrl_file_to_text(data)
            meta = {"source": f"xbrl:{b.name}", "symbol": "CUPID"}
            vs.add_document(doc_id=f"xbrl::{b.name}", text=text, meta=meta)
        except Exception as e:
            print(f"[startup] Failed parse {b.name}: {e}")

    print(f"[startup] Completed indexing metadata rows: {total_rows}")
    vs.save_index()


if INDEX_ON_STARTUP:
    print("[startup] INDEX_ON_STARTUP=true — starting indexing...")
    try:
        startup_index()
    except Exception as e:
        print("[startup] indexing failed:", e)
    print("[startup] Indexing completed.")
else:
    print("[startup] Skipped indexing on startup (INDEX_ON_STARTUP=false).")

# ---- Simple in-memory session store for chat.js ----
SESSIONS = {}  # session_id -> {id, title, created, messages: [{role, content, meta}]}


def create_session(title="New chat", owner=None):
    sid = str(uuid4())
    now = datetime.utcnow().isoformat()
    SESSIONS[sid] = {
        "id": sid,
        "title": title or "New chat",
        "created": now,
        "owner": owner,
        "messages": [],
    }
    return SESSIONS[sid]


# Directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Helpers ----------
def strip_markdown(text: str) -> str:
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


def search_azure(query_text: str, top_k: int = 5):
    if not search_client:
        print("[azure] search_client is None. Check AZURE_SEARCH_* env vars.")
        return []
    if not query_text:
        print("[azure] Empty query_text.")
        return []
    print(f"[azure] Searching index '{AZURE_SEARCH_INDEX}' for query: {query_text!r}")
    try:
        results = search_client.search(search_text=query_text, top=top_k)
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
    print(f"[azure] Retrieved {len(output)} docs from Azure Search.")
    for i, o in enumerate(output):
        print(
            f"  [azure] {i+1}. doc_id={o['doc_id']!r} "
            f"score={o['score']:.4f} source={o['meta'].get('source')!r}"
        )
    return output


# ---------- Routes ----------
@app.route("/")
def home():
    # If user not authenticated, send to login page.
    if not (hasattr(current_user, "is_authenticated") and current_user.is_authenticated):
        return redirect("/login")
    return render_template("index.html", APP_VERSION=read_version())


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Render login page (GET), accept form POST (username/password) and sign the user in.
    Accepts demo accounts demouser/devuser/produser for local and Azure testing.
    """
    # if already logged-in via flask-login, redirect to home
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        return redirect("/")

    # GET -> render template
    if request.method == "GET":
        return render_template("login.html", APP_VERSION=read_version())

    # POST -> process form-encoded submission
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if not username or not password:
        # re-render with error banner
        return (
            render_template(
                "login.html",
                error="Username and password required.",
                username=username,
            ),
            400,
        )

    # Try DB lookup (if models.User present)
    user = None
    try:
        user = User.query.filter_by(username=username).first()
    except Exception as e:
        print("[login][ERROR] DB lookup failed:", repr(e))
        user = None

    authenticated = False
    if user:
        if hasattr(user, "check_password") and callable(getattr(user, "check_password")):
            try:
                authenticated = user.check_password(password)
            except Exception:
                authenticated = False
        else:
            # dev fallback: plaintext password (only for local testing)
            try:
                if getattr(user, "password", None) == password:
                    authenticated = True
            except Exception:
                authenticated = False

    # Allow demo accounts as fallback (if DB missing or user not found)
    if not authenticated and username in ("demouser", "devuser", "produser"):
        authenticated = True
        if not user:
            # create simple temporary user object compatible with flask-login
            class _TempUser:
                def __init__(self, username):
                    self.id = username
                    self.username = username
                    self.is_active = True
                    self.is_authenticated = True
                    self.is_anonymous = False

                def get_id(self):
                    return str(self.id)

            user = _TempUser(username)

    if not authenticated:
        return (
            render_template(
                "login.html",
                error="Invalid username or password.",
                username=username,
            ),
            401,
        )

    # Perform login (flask-login) and redirect home
    try:
        login_user(user)
    except Exception as e:
        print("[login][WARN] login_user failed:", repr(e))
        # fallback: set session keys
        session["logged_in"] = True
        session["username"] = username

    return redirect("/")


@app.route("/test_search")
def test_search():
    if not search_client:
        return jsonify(
            {
                "status": "error",
                "message": "search_client is None. Check AZURE_SEARCH_* env vars.",
            }
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


@app.route("/chat", methods=["POST"])
def chat():
    if REQUIRE_AUTH:
        if not (
            (hasattr(current_user, "is_authenticated") and current_user.is_authenticated)
            or session.get("user")
        ):
            return jsonify({"error": "Authentication required"}), 401

    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if "history" not in session:
        session["history"] = [
            {
                "role": "system",
                "content": "I’m SageAlpha, a financial assistant powered by SageAlpha.ai to support your financial decisions.",
            }
        ]
    history = session["history"]
    history.append({"role": "user", "content": user_msg})

    top_k = int(payload.get("top_k", 5))
    retrieved = search_azure(user_msg, top_k)
    if not retrieved and search_client is None:
        retrieved = vs.search(user_msg, k=top_k)

    context_chunks = [
        f"Source: {r['meta'].get('source', r.get('doc_id'))}\n{r['text']}\n"
        for r in retrieved
    ]
    context_text = "\n\n".join(context_chunks)[:6000]

    system_prompt = (
        "I’m SageAlpha, a financial assistant powered by SageAlpha.ai to support your financial decisions. "
        "Use the provided context to answer the user's question. "
        "If context is insufficient, be honest and say so. "
        "Respond in clear plain text only. Do not use markdown formatting, "
        "asterisks (*), hash symbols (#), bullet lists, or code blocks."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context_text}"},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = response.choices[0].message.content
        history.append({"role": "assistant", "content": ai_msg})
        session["history"] = history

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
        error_msg = f"Backend error: {str(e)}"
        msg_id = str(uuid4())
        message_obj = {"id": msg_id, "role": "assistant", "content": error_msg}
        print("[chat][ERROR]", repr(e))
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


@app.route("/query", methods=["POST"])
def query():
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
        if not fetch_pdf or not att:
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
                vs.add_temporary_document(
                    doc_id=temp_doc_id, text=extracted, meta=meta
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[query] failed to download/extract {att}: {e}")

    if search_client:
        final_results = search_azure(q, top_k)
    else:
        final_results = vs.search(q, k=top_k)

    context_chunks = [
        f"Source: {r['meta'].get('source', r['doc_id'])}\n{r['text']}"
        for r in final_results
    ]
    context_text = "\n\n".join(context_chunks)[:6000]

    messages = [
        {
            "role": "system",
            "content": "I’m SageAlpha, a financial assistant using smart financial models to guide your decisions.",
        },
        {"role": "system", "content": f"Context:\n{context_text}"},
        {"role": "user", "content": q},
    ]

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
        print("[query][ERROR]", repr(e))
        return jsonify({"error": str(e)}), 500


# --- REPORT HTML endpoint used by client-side PDF generator ---
@app.route("/report-html", methods=["GET"])
def report_html():
    """
    Return a fragment that contains:
      - head style blocks (copied)
      - link stylesheet tags (copied so iframe can load them)
      - body inner HTML
      - injected print CSS to ensure print colors and high-quality capture

    If the request looks like an XHR/fetch (our client), return the fragment.
    Otherwise return the full HTML for normal browsing.
    """
    try:
        html = render_template("sagealpha_reports.html")

        # Determine if caller likely wants the fragment (AJAX/fetch)
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

        # Extract <head> and <body> pieces
        head_match = re.search(
            r"<head[^>]*>([\s\S]*?)</head>", html, flags=re.IGNORECASE
        )
        body_match = re.search(
            r"<body[^>]*>([\s\S]*?)</body>", html, flags=re.IGNORECASE
        )

        head_html = head_match.group(1) if head_match else ""
        body_html = body_match.group(1) if body_match else html  # fallback: entire html

        # Extract inline <style> content from head
        style_blocks = re.findall(
            r"<style[^>]*>([\s\S]*?)</style>", head_html, flags=re.IGNORECASE
        )

        # Extract link rel=stylesheet tags (keep them so iframe can load external css)
        link_tags = re.findall(
            r"<link[^>]*rel=[\"']stylesheet[\"'][^>]*>",
            head_html,
            flags=re.IGNORECASE,
        )

        # IMPORTANT: Inject a small print CSS to ensure colors and layout are preserved during capture
        injected_print_css = """
        <style>
        /* Ensure print/canvas honors background colors */
        #sagealpha-report-fragment, #sagealpha-report-fragment * {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            box-sizing: border-box;
        }
        /* Make sure the main container has white background for PDF */
        #sagealpha-report-fragment .container {
            background: white !important;
        }
        /* Tweak page width for html2canvas capture */
        #sagealpha-report-fragment { width: 1200px; max-width: 1200px; margin: 0 auto; }
        /* Prevent interactive elements from breaking layout */
        #sagealpha-report-fragment a { color: inherit; text-decoration: none; }
        </style>
        """

        # Build combined style string (concat inline styles)
        combined_styles = ""
        if style_blocks:
            combined_styles = "<style>" + "\n".join(style_blocks) + "</style>"

        # Build link tags HTML (convert relative href to absolute URL so iframe can load)
        processed_links = []
        for lt in link_tags:
            href_match = re.search(r'href=[\'"]([^\'"]+)[\'"]', lt)
            if href_match:
                href = href_match.group(1)
                if href.startswith("/"):
                    processed_links.append(lt)
                else:
                    processed_links.append(lt)
            else:
                processed_links.append(lt)
        links_html = "\n".join(processed_links)

        # Compose the fragment: styles (inline + injected), link tags, then the body content
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
        print("[report-html][ERROR]", repr(e))
        return jsonify({"error": "Failed to render report HTML: " + str(e)}), 500


@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        startup_index()
        return jsonify({"status": "refreshed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset_history", methods=["POST"])
def reset_history():
    session.pop("history", None)
    return jsonify({"status": "cleared"})


@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    print("[ERROR]", repr(e))
    return jsonify({"error": str(e)}), code


@app.route("/user", methods=["GET"])
def user():
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        return jsonify(
            {
                "username": current_user.username,
                "email": f"{current_user.username}@local",
                "avatar_url": None,
            }
        )
    return jsonify(
        {"username": "Guest", "email": "guest@gmail.com", "avatar_url": None}
    )


@app.route("/sessions", methods=["GET"])
def list_sessions():
    out = []
    sessions_sorted = sorted(
        SESSIONS.values(), key=lambda x: x["created"], reverse=True
    )
    for s in sessions_sorted:
        if REQUIRE_AUTH:
            if not (
                hasattr(current_user, "is_authenticated")
                and current_user.is_authenticated
            ):
                continue
            if s.get("owner") != current_user.username:
                continue
        out.append(
            {
                "id": s["id"],
                "title": s["title"],
                "created": s["created"],
                "owner": s.get("owner"),
                "message_count": len(s.get("messages", [])),
            }
        )
    return jsonify({"sessions": out})


@app.route("/sessions", methods=["POST"])
def create_session_route():
    data = request.get_json(silent=True) or {}
    title = data.get("title") or "New chat"
    owner = None
    if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
        owner = current_user.username
    if REQUIRE_AUTH and not owner:
        return jsonify({"error": "Authentication required"}), 401
    s = create_session(title, owner=owner)
    return jsonify({"session": s}), 201


@app.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if REQUIRE_AUTH:
        if not (
            hasattr(current_user, "is_authenticated")
            and current_user.is_authenticated
        ):
            return jsonify({"error": "Authentication required"}), 401
        if s.get("owner") != current_user.username:
            return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": s})


@app.route("/sessions/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if REQUIRE_AUTH:
        if not (
            hasattr(current_user, "is_authenticated")
            and current_user.is_authenticated
        ):
            return jsonify({"error": "Authentication required"}), 401
        if s.get("owner") != current_user.username:
            return jsonify({"error": "Session not found"}), 404
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if title:
        s["title"] = title
    return jsonify({"session": s})


@app.route("/chat_session", methods=["POST"])
def chat_session():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
        if REQUIRE_AUTH:
            if not (
                hasattr(current_user, "is_authenticated")
                and current_user.is_authenticated
            ):
                return jsonify({"error": "Authentication required"}), 401
            if s.get("owner") and s.get("owner") != current_user.username:
                return jsonify({"error": "Session not found"}), 404
    else:
        owner = (
            current_user.username
            if (
                hasattr(current_user, "is_authenticated")
                and current_user.is_authenticated
            )
            else None
        )
        if REQUIRE_AUTH and not owner:
            return jsonify({"error": "Authentication required"}), 401
        s = create_session("New chat", owner=owner)
        session_id = s["id"]

    s["messages"].append({"role": "user", "content": user_msg, "meta": {}})

    top_k = int(payload.get("top_k", 5))
    retrieved = search_azure(user_msg, top_k)
    if not retrieved and search_client is None:
        retrieved = vs.search(user_msg, k=top_k)

    context_chunks = [
        f"Source: {r['meta'].get('source', r.get('doc_id'))}\n{r['text']}"
        for r in retrieved
    ]
    context_text = "\n\n".join(context_chunks)[:6000]

    system_prompt = (
        "I’m SageAlpha, a financial assistant using smart financial models to guide your decisions. "
        "Use the provided context to answer the user's question. "
        "If context is insufficient, be honest and say so. "
        "Respond in clear plain text only. Do not use markdown formatting, asterisks (*), hash symbols (#), "
        "bullet lists, or code blocks."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context_text}"},
        {"role": "user", "content": user_msg},
    ]

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

        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in retrieved
        ]

        return jsonify({"session_id": session_id, "response": ai_msg, "sources": sources})
    except Exception as e:
        print("[chat_session][ERROR]", repr(e))
        s["messages"].append(
            {"role": "assistant", "content": f"Backend error: {e}", "meta": {}}
        )
        return jsonify({"error": str(e)}), 500


# ---- start server when run directly ----
if __name__ == "__main__":
    import sys

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug_env = os.getenv("FLASK_DEBUG", "1")
    debug = True if debug_env.lower() in ("1", "true", "yes") else False

    try:
        with app.app_context():
            db.create_all()
            seed_demo_users()
            print("[startup] Database tables ensured (sqlite) and demo users seeded.")
    except Exception as e:
        print("[startup][WARN] Failed to create DB tables or seed users:", repr(e))

    print(f"[startup] Starting Flask app on http://{host}:{port}  (debug={debug})")
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        print("[startup][ERROR] Failed to start Flask:", repr(e))
        sys.exit(1)
