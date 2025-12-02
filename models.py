"""
SageAlpha.ai Database Models
SQLAlchemy 2.x models for user management and message persistence
"""

from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    User model for authentication.
    Stores username, display name, and bcrypt password hash.
    """

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(120))
    password_hash = db.Column(db.String(256), nullable=False)
    # Note: unique constraint removed to allow multiple NULL values in SQLite
    # Use application-level validation for email uniqueness when not NULL
    email = db.Column(db.String(120), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    messages = db.relationship("Message", backref="user", lazy="dynamic")
    sessions = db.relationship("ChatSession", backref="user", lazy="dynamic")

    def set_password(self, password: str) -> None:
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify the password against the stored hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class ChatSession(db.Model):
    """
    Chat session model for organizing conversations.
    """

    __tablename__ = "chat_sessions"

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(255), default="New chat")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    current_topic = db.Column(db.String(255), default="")

    # Relationships
    messages = db.relationship(
        "Message", backref="session", lazy="dynamic", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.id[:8]}... - {self.title}>"


class Message(db.Model):
    """
    Message model for storing chat history.
    Supports both user and assistant messages with metadata.
    """

    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    session_id = db.Column(
        db.String(36), db.ForeignKey("chat_sessions.id"), nullable=True, index=True
    )
    role = db.Column(db.String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )

    # Optional metadata stored as JSON
    meta_json = db.Column(db.Text, nullable=True)

    def __repr__(self) -> str:
        preview = self.content[:50] if self.content else ""
        return f"<Message {self.role}: {preview}...>"


class Document(db.Model):
    """
    Document model for tracking uploaded files and indexed documents.
    """

    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    doc_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    filename = db.Column(db.String(255))
    file_type = db.Column(db.String(50))
    file_size = db.Column(db.Integer)
    source = db.Column(db.String(512))
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_indexed = db.Column(db.Boolean, default=False)
    index_status = db.Column(db.String(50), default="pending")

    def __repr__(self) -> str:
        return f"<Document {self.filename}>"
