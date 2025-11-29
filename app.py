# =====================================================================
#  app.py — Hybrid RAG + LLM Fallback + Session Memory + Topic-Aware
# =====================================================================

import os
import re
from datetime import datetime
from uuid import uuid4

from flask import Flask, request, jsonify, render_template, session, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# -------------------------------------------------------------------
# 1. Environment / Config
# -------------------------------------------------------------------
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Flask secret
FLASK_SECRET = os.getenv("FLASK_SECRET") or os.urandom(24).hex()

# Azure Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")  # e.g. "default"

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
    print("[WARN] Missing Azure OpenAI configuration – chat may not work correctly.")

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
    print("[WARN] Missing Azure Search configuration – RAG search will be disabled.")

# -------------------------------------------------------------------
# 2. Flask app + clients
# -------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = FLASK_SECRET

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Azure Search client
search_client = None
if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY and AZURE_SEARCH_INDEX:
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    print(f"[startup] Connected to Azure Search index: {AZURE_SEARCH_INDEX}")
else:
    print("[startup] Azure Search client not initialized.")

# In-memory multi-session store (for /chat_session)
SESSIONS = {}  # session_id -> {id, title, created, messages, sections, current_topic}


def create_session(title="New chat"):
    """Create a new multi-session chat container."""
    sid = str(uuid4())
    now = datetime.utcnow().isoformat()
    SESSIONS[sid] = {
        "id": sid,
        "title": title or "New chat",
        "created": now,
        "messages": [],
        "sections": [],      # section-wise Q&A memory
        "current_topic": "", # current company/topic for this session
    }
    return SESSIONS[sid]


# Directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 3. Helpers
# -------------------------------------------------------------------
def strip_markdown(text: str) -> str:
    """Convert common markdown formatting to plain readable text."""
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


def extract_topic(user_msg: str, last_topic: str | None = None) -> str | None:
    """
    Very simple topic extractor.

    Goal:
      - When user types 'cupid limited' or 'tata motors' → treat that as a new topic.
      - When user types 'give owner name' or 'who is the owner' → keep using last topic.

    This is heuristic but works well for your pattern of usage.
    """
    if not user_msg:
        return last_topic

    text = user_msg.strip().lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return last_topic

    # Generic question-style prompts – don't change topic
    question_starts = {
        "who",
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "give",
        "tell",
        "show",
        "explain",
        "owner",
        "ceo",
        "chairman",
        "md",
        "director",
    }
    if tokens[0] in question_starts:
        return last_topic

    # Short phrase (1–4 words) that is not just a generic question → assume new topic
    if len(tokens) <= 4:
        return text

    # For longer sentences, keep the last topic (LLM + search will still see full question)
    return last_topic


def search_azure(query_text: str, top_k: int = 5):
    """
    Search Azure Cognitive Search index and return docs in unified shape:
    [
      {
        "doc_id": str,
        "text": str,
        "meta": {...},
        "score": float
      },
      ...
    ]
    Uses semantic search if AZURE_SEARCH_SEMANTIC_CONFIG is set.
    """
    if not search_client:
        print("[azure] search_client is None. Check AZURE_SEARCH_* env vars.")
        return []
    if not query_text:
        print("[azure] Empty query_text.")
        return []

    print(f"[azure] Searching index '{AZURE_SEARCH_INDEX}' for query: {query_text!r}")

    search_kwargs = {
        "search_text": query_text,
        "top": top_k,
    }

    if AZURE_SEARCH_SEMANTIC_CONFIG:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = AZURE_SEARCH_SEMANTIC_CONFIG
        print(f"[azure] Using semantic configuration: {AZURE_SEARCH_SEMANTIC_CONFIG}")

    try:
        results = search_client.search(**search_kwargs)
    except Exception as e:
        print("[azure][ERROR] search failed:", repr(e))
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

        output.append(
            {
                "doc_id": doc_id,
                "text": text,
                "meta": meta,
                "score": score,
            }
        )

    print(f"[azure] Retrieved {len(output)} docs from Azure Search.")
    for i, o in enumerate(output):
        print(
            f"  [azure] {i+1}. doc_id={o['doc_id']!r} "
            f"score={o['score']:.4f} source={o['meta'].get('source')!r}"
        )

    return output


def build_session_memory_sections(sections, current_topic: str | None,
                                  limit: int = 5, max_chars: int = 1500):
    """
    Build a compact 'session memory' string from previous Q&A sections.

    If current_topic is set:
      - Prefer only those Q&A where this topic appears in query or answer.
      - This prevents mixing 'Cupid Limited' memory into 'Tata Motors' queries.

    If no topic or nothing matches:
      - Fall back to last N sections of the chat.
    """
    if not sections:
        return ""

    normalized_topic = (current_topic or "").lower().strip()

    filtered = []
    if normalized_topic:
        for s in sections:
            q = (s.get("query") or "").lower()
            a = (s.get("answer") or "").lower()
            if normalized_topic in q or normalized_topic in a:
                filtered.append(s)

    if not filtered:
        # No topic match → use last N overall
        filtered = sections[-limit:]
    else:
        filtered = filtered[-limit:]

    parts = []
    for s in filtered:
        parts.append(
            f"[{s.get('timestamp','')}] Q: {s.get('query','')}\nA: {s.get('answer','')}"
        )

    memory_text = "\n\n".join(parts)
    return memory_text[:max_chars]


def build_hybrid_messages(user_msg: str, retrieved_docs, extra_system_msgs=None):
    """
    Build messages for hybrid RAG:
    - If there are relevant search results (score above threshold),
      include them as Context.
    - If not, leave Context empty and let the LLM answer from its own knowledge.
    """
    relevance_threshold = 0.35
    relevant_docs = [r for r in retrieved_docs if r.get("score", 0.0) >= relevance_threshold]

    if relevant_docs:
        context_chunks = [
            f"Source: {r['meta'].get('source', r['doc_id'])}\n{r['text']}"
            for r in relevant_docs
            if r.get("text")
        ]
        context_text = "\n\n".join(context_chunks)[:6000]
        print("[hybrid] Using RAG mode with context_length:", len(context_text))
    else:
        context_text = ""
        print("[hybrid] No relevant docs found – using pure LLM mode.")

    system_prompt = (
        "You are SageAlpha, a financial assistant powered by SageAlpha.ai.\n"
        "Use this logic:\n"
        "1. If the Context contains useful information, use it to answer.\n"
        "2. If the Context is empty or not relevant, answer using your own knowledge.\n"
        "3. Be precise and financially accurate.\n"
        "4. Respond in clear plain text only. Do not use markdown formatting, asterisks (*),\n"
        "   hash symbols (#), bullet lists, or code blocks.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if extra_system_msgs:
        messages.extend(extra_system_msgs)

    messages.append({"role": "system", "content": f"Context:\n{context_text}"})
    messages.append({"role": "user", "content": user_msg})

    return messages


# -------------------------------------------------------------------
# 4. Routes
# -------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test_search")
def test_search():
    """
    Simple debug endpoint to verify Azure Search connectivity.
    """
    if not search_client:
        return jsonify(
            {
                "status": "error",
                "message": "search_client is None. Check AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_KEY / AZURE_SEARCH_INDEX.",
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
                    "content_preview": (r.get("content") or r.get("merged_content") or "")[:200],
                }
            )
        return jsonify({"status": "ok", "query": q, "results": items})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------  /chat (single-session)  ----------------------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint used by your UI.
    - Uses Flask session for a single ongoing chat.
    - Azure Search first, then LLM fallback.
    - Topic-aware section-wise memory inside Flask session.
    """
    payload = request.get_json(silent=True) or {}
    user_msg = (payload.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Ensure history + sections + current_topic exist in Flask session
    if "history" not in session:
        session["history"] = [
            {
                "role": "system",
                "content": "I’m SageAlpha, a financial assistant powered by SageAlpha.ai to support your financial decisions.",
            }
        ]
    if "sections" not in session:
        session["sections"] = []
    if "current_topic" not in session:
        session["current_topic"] = ""

    history = session["history"]
    sections = session["sections"]
    last_topic = session.get("current_topic", "")

    # Update topic based on this user message
    current_topic = extract_topic(user_msg, last_topic)
    session["current_topic"] = current_topic or ""

    history.append({"role": "user", "content": user_msg})

    # Build session memory filtered by current topic
    session_memory_text = build_session_memory_sections(sections, current_topic)
    extra_system_msgs = []
    if session_memory_text:
        extra_system_msgs.append(
            {
                "role": "system",
                "content": f"Session memory (previous Q&A sections):\n{session_memory_text}",
            }
        )

    # Search → Hybrid messages
    top_k = int(payload.get("top_k", 5))
    retrieved = search_azure(user_msg, top_k)
    messages = build_hybrid_messages(user_msg, retrieved, extra_system_msgs=extra_system_msgs)

    print(f"[chat] user_msg={user_msg!r}, current_topic={current_topic!r}")

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            top_p=0.95,
        )
        ai_msg = response.choices[0].message.content

        # Save in history + sections
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

        # Prepare sources list (for UI)
        sources = [
            {
                "doc_id": r["doc_id"],
                "source": r["meta"].get("source"),
                "score": float(r["score"]),
            }
            for r in retrieved
        ]

        msg_id = str(uuid4())
        message_obj = {
            "id": msg_id,
            "role": "assistant",
            "content": ai_msg,
        }

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
        message_obj = {
            "id": msg_id,
            "role": "assistant",
            "content": error_msg,
        }
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


# --------------------------  /query  ---------------------------------
@app.route("/query", methods=["POST"])
def query():
    """
    One-shot query endpoint.
    - Uses Azure Search first.
    - Falls back to LLM when nothing relevant is found.
    Returns: answer + sources.
    """
    payload = request.get_json(silent=True) or {}
    q = (payload.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Empty query"}), 400

    top_k = int(payload.get("top_k", 5))
    final_results = search_azure(q, top_k)

    messages = build_hybrid_messages(q, final_results)

    print(f"[query] q={q!r}")

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
        print("[query][ERROR]", repr(e))
        return jsonify({"error": str(e)}), 500


# --------------------------  /refresh  -------------------------------
@app.route("/refresh", methods=["POST"])
def refresh():
    """
    Previously this triggered local VectorStore indexing.
    Now indexing is handled by Azure AI Search indexers in the portal.
    We just return a message.
    """
    return jsonify(
        {
            "status": "noop",
            "message": "Indexing is managed by Azure AI Search indexers. Run them from the Azure portal.",
        }
    )


# -----------------------  /reset_history  ----------------------------
@app.route("/reset_history", methods=["POST"])
def reset_history():
    session.pop("history", None)
    session.pop("sections", None)
    session.pop("current_topic", None)
    return jsonify({"status": "cleared"})


# ---------------------------  /user  ---------------------------------
@app.route("/user", methods=["GET"])
def user():
    return jsonify(
        {
            "username": "Guest",
            "email": "guest@gmail.com",
            "avatar_url": None,
        }
    )


# --------------------------  SESSIONS API  ---------------------------
@app.route("/sessions", methods=["GET"])
def list_sessions():
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


# -----------------------  /chat_session  -----------------------------
@app.route("/chat_session", methods=["POST"])
def chat_session():
    """
    Multi-session chat endpoint.
    - Uses SESSIONS dict for separate chat histories.
    - Azure Search first, LLM fallback.
    - Topic-aware section-wise memory per session.
    """
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

    s["messages"].append({"role": "user", "content": user_msg, "meta": {}})

    # Topic tracking for this multi-session
    last_topic = s.get("current_topic", "")
    current_topic = extract_topic(user_msg, last_topic)
    s["current_topic"] = current_topic or ""

    # Build session memory
    session_memory_text = build_session_memory_sections(s.get("sections", []), current_topic)
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
    messages = build_hybrid_messages(user_msg, retrieved, extra_system_msgs=extra_system_msgs)

    print(f"[chat_session] session_id={session_id}, msg={user_msg!r}, current_topic={current_topic!r}")

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

        # Store section-wise memory (Q&A pair)
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
            {
                "session_id": session_id,
                "response": ai_msg,
                "sources": sources,
            }
        )
    except Exception as e:
        print("[chat_session][ERROR]", repr(e))
        return jsonify({"error": str(e)}), 500


# ---------------------------  Uploads  -------------------------------
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

    url = f"/uploads/{filename}"
    return jsonify({"filename": filename, "url": url})


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ------------------------  Global error handler  ---------------------
@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code

    print("[ERROR]", repr(e))
    return jsonify({"error": str(e)}), code


# -------------------------------------------------------------------
# 5. Run app
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true",
    )
