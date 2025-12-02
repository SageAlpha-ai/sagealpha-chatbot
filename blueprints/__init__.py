"""
SageAlpha.ai Blueprints Package
Flask 3.x modular architecture for 2025 stack
"""

from blueprints.auth import auth_bp
from blueprints.chat import chat_bp
from blueprints.pdf import pdf_bp

__all__ = ["auth_bp", "chat_bp", "pdf_bp"]

