import os
import io
import json
import tempfile
from datetime import datetime

from uuid import uuid4
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd

from openai import AzureOpenAI

from blob_utils import BlobReader
from extractor import extract_text_from_pdf_bytes, parse_xbrl_file_to_text
from vector_store import VectorStore
import re

load_dotenv()

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
FLASK_SECRET = os.getenv("FLASK_SECRET") or os.urandom(24).hex()

if not AZURE_BLOB_CONNECTION_STRING:
    print("[WARN] Missing AZURE_BLOB_CONNECTION_STRING – blob features disabled.")

    
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = FLASK_SECRET

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    # default_headers={"api-key": AZURE_OPENAI_API_KEY}
)

BLOB_CONTAINER = "nse-data-raw"
blob_reader = BlobReader(conn_str=AZURE_BLOB_CONNECTION_STRING, container=BLOB_CONTAINER)

VECTOR_STORE_DIR = "vector_store_data"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
vs = VectorStore(store_dir=VECTOR_STORE_DIR)

# Control indexing with env variable so local vs Azure behave differently
INDEX_ON_STARTUP = os.getenv("INDEX_ON_STARTUP", "false").lower() == "true"

if INDEX_ON_STARTUP:
    print("[startup] INDEX_ON_STARTUP=true — starting indexing...")
    startup_index()
    print("[startup] Indexing completed.")
else:
    print("[startup] Skipped indexing on startup (INDEX_ON_STARTUP=false).")

# ---- Simple in-memory session store for chat.js ----
SESSIONS = {}  # session_id -> {id, title, created, messages: [{role, content, meta}]}

def create_session(title="New chat"):
    sid = str(uuid4())
    now = datetime.utcnow().isoformat()
    SESSIONS[sid] = {
        "id": sid,
        "title": title or "New chat",
        "created": now,
        "messages": [],
    }
    return SESSIONS[sid]

# Directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

    

# build index on startup
# startup_index()
print("[startup] Skipped indexing during local local run.")

def strip_markdown(text: str) -> str:
    """Convert common markdown formatting to plain readable text."""
    if not text:
        return text

    # Remove fenced code blocks ```...```
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove markdown headings like '### Title'
    text = re.sub(r"(^|\n)#{1,6}\s*", r"\1", text)

    # Bold / italics: **text**, *text*, __text__, _text_
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Bullet list markers at start of line: -, *, +
    text = re.sub(r"(^|\n)[\-\*\+]\s+", r"\1", text)

    # Horizontal rules like '---', '***'
    text = re.sub(r"\n[-*_]{3,}\n", "\n", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # Safe JSON parsing
    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if "history" not in session:
        session["history"] = [
            {
                "role": "system",
                "content": "You are SageAlpha, an advanced financial AI assistant."
            }
        ]
    history = session["history"]
    history.append({"role": "user", "content": user_msg})

    top_k = int(payload.get("top_k", 5))
    retrieved = vs.search(user_msg, k=top_k)

    context_chunks = []
    for r in retrieved:
        context_chunks.append(
            f"Source: {r['meta'].get('source', r.get('doc_id'))}\n{r['text']}\n"
        )
    context_text = "\n\n".join(context_chunks)[:6000]

    system_prompt = (
        "You are SageAlpha, an expert financial assistant. "
        "Use the provided context to answer the user's question. "
        "If context is insufficient, be honest and say so."
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

        # ✅ Shape the payload for the frontend
        msg_id = str(uuid4())
        message_obj = {
            "id": msg_id,
            "role": "assistant",
            "content": ai_msg,
        }

        # We return multiple shapes so whatever your friend’s frontend expects, it works:
        # - result.id
        # - result.data.id
        # - result.message.id
        return jsonify({
            "id": msg_id,
            "response": ai_msg,
            "message": message_obj,
            "data": message_obj,
            "sources": sources,
        })

    except Exception as e:
        # Even on error, keep the same structure to avoid `undefined.id` crashes
        error_msg = f"Backend error: {str(e)}"
        msg_id = str(uuid4())
        message_obj = {
            "id": msg_id,
            "role": "assistant",
            "content": error_msg,
        }
        return jsonify({
            "id": msg_id,
            "response": error_msg,
            "message": message_obj,
            "data": message_obj,
            "sources": [],
            "error": str(e),
        }), 500




@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Empty query"}), 400

    top_k = int(payload.get("top_k", 5))
    fetch_pdf = bool(payload.get("fetch_pdf", True))

    results = vs.search(q, k=top_k)

    for r in results:
        att = r["meta"].get("attachment") or ""
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
            vs.add_temporary_document(
                doc_id=temp_doc_id, text=extracted, meta=meta
            )
        except Exception as e:
            print(f"[query] failed to download/extract {att}: {e}")

    final_results = vs.search(q, k=top_k)
    context_chunks = [
        f"Source: {r['meta'].get('source', r['doc_id'])}\n{r['text']}"
        for r in final_results
    ]
    context_text = "\n\n".join(context_chunks)[:6000]

    messages = [
        {
            "role": "system",
            "content": "You are SageAlpha, an expert financial assistant. Use the provided context to answer the user's question.",
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
        vs.clear_temporary_documents()
        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in final_results
        ]
        return jsonify({"answer": ai_msg, "sources": sources})
    except Exception as e:
        vs.clear_temporary_documents()
        return jsonify({"error": str(e)}), 500


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

from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    # If it's an HTTPException, use its code; otherwise 500
    code = 500
    if isinstance(e, HTTPException):
        code = e.code

    # Log error to console for debugging
    print("[ERROR]", repr(e))

    return jsonify({"error": str(e)}), code

@app.route("/user", methods=["GET"])
def user():
    # For now return a guest user; wire to real auth later if needed
    return jsonify({
        "username": "Guest",
        "email": "guest@gmail.com",
        "avatar_url": None,
    })


@app.route("/sessions", methods=["GET"])
def list_sessions():
    # Newest first
    sessions = sorted(
        SESSIONS.values(),
        key=lambda s: s["created"],
        reverse=True,
    )
    return jsonify({"sessions": sessions})


@app.route("/sessions", methods=["POST"])
def create_session_route():
    data = request.get_json(silent=True) or {}
    title = data.get("title") or "New chat"
    s = create_session(title)
    return jsonify({"session": s}), 201

@app.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": s})

@app.route("/sessions/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    s = SESSIONS.get(session_id)
    if not s:
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

    # Ensure session exists
    if session_id and session_id in SESSIONS:
        s = SESSIONS[session_id]
    else:
        s = create_session("New chat")
        session_id = s["id"]

    # Save user message
    s["messages"].append({"role": "user", "content": user_msg, "meta": {}})

    # Retrieve context from vector store (RAG)
    top_k = int(payload.get("top_k", 5))
    retrieved = vs.search(user_msg, k=top_k)

    context_chunks = [
        f"Source: {r['meta'].get('source', r['doc_id'])}\n{r['text']}"
        for r in retrieved
    ]
    context_text = "\n\n".join(context_chunks)[:6000]

    system_prompt = (
        "You are SageAlpha, an expert financial assistant. "
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
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = resp.choices[0].message.content

        # Save assistant reply
        s["messages"].append({"role": "assistant", "content": ai_msg, "meta": {}})

        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in retrieved
        ]

        return jsonify({
            "session_id": session_id,
            "response": ai_msg,
            "sources": sources,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Local URL to serve this file
    url = f"/uploads/{filename}"
    return jsonify({"filename": filename, "url": url})


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")
