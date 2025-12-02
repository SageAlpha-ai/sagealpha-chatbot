"""
SageAlpha.ai Chat Blueprint
Real-time chat with WebSocket support via Flask-SocketIO
"""

import os
import re
from datetime import datetime
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, session
from flask_login import current_user

chat_bp = Blueprint("chat", __name__)

# In-memory session store (replace with Redis for production scaling)
SESSIONS: dict = {}


def require_auth(f):
    """Decorator to check authentication."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        require_auth_env = os.getenv("REQUIRE_AUTH", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        if require_auth_env:
            if not (
                (
                    hasattr(current_user, "is_authenticated")
                    and current_user.is_authenticated
                )
                or session.get("user")
            ):
                return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated


def create_session(title: str = "New chat", owner: str | None = None) -> dict:
    """Create a new chat session."""
    sid = str(uuid4())
    now = datetime.utcnow().isoformat()
    SESSIONS[sid] = {
        "id": sid,
        "title": title or "New chat",
        "created": now,
        "owner": owner,
        "messages": [],
        "sections": [],
        "current_topic": "",
    }
    return SESSIONS[sid]


def extract_topic(user_msg: str, last_topic: str | None = None) -> str | None:
    """Extract conversation topic from user message."""
    if not user_msg:
        return last_topic

    text = user_msg.strip().lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return last_topic

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

    if len(tokens) <= 4:
        return text

    return last_topic


def build_session_memory_sections(
    sections: list,
    current_topic: str | None,
    limit: int = 5,
    max_chars: int = 1500,
) -> str:
    """Build session memory from previous Q&A sections."""
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
        filtered = sections[-limit:]
    else:
        filtered = filtered[-limit:]

    parts = []
    for s in filtered:
        parts.append(
            f"[{s.get('timestamp', '')}] Q: {s.get('query', '')}\nA: {s.get('answer', '')}"
        )

    memory_text = "\n\n".join(parts)
    return memory_text[:max_chars]


@chat_bp.route("/sessions", methods=["GET"])
@require_auth
def list_sessions():
    """List all chat sessions for current user."""
    out = []
    sessions_sorted = sorted(
        SESSIONS.values(), key=lambda x: x["created"], reverse=True
    )
    for s in sessions_sorted:
        if s.get("owner") != getattr(current_user, "username", None):
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


@chat_bp.route("/sessions", methods=["POST"])
@require_auth
def create_session_route():
    """Create a new chat session."""
    data = request.get_json(silent=True) or {}
    title = data.get("title") or "New chat"
    owner = getattr(current_user, "username", None)
    s = create_session(title, owner=owner)
    return jsonify({"session": s}), 201


@chat_bp.route("/sessions/<session_id>", methods=["GET"])
@require_auth
def get_session(session_id: str):
    """Get a specific chat session."""
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if s.get("owner") != getattr(current_user, "username", None):
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": s})


@chat_bp.route("/sessions/<session_id>/rename", methods=["POST"])
@require_auth
def rename_session(session_id: str):
    """Rename a chat session."""
    s = SESSIONS.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    if s.get("owner") != getattr(current_user, "username", None):
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if title:
        s["title"] = title
    return jsonify({"session": s})


@chat_bp.route("/reset_history", methods=["POST"])
@require_auth
def reset_history():
    """Clear chat history from session."""
    session.pop("history", None)
    session.pop("sections", None)
    session.pop("current_topic", None)
    return jsonify({"status": "cleared"})

