"""
SageAlpha.ai v3.0 - Configuration Module
Centralized configuration for all environments
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()


class Config:
    """Base configuration."""
    
    # Flask
    SECRET_KEY: str = os.getenv("FLASK_SECRET") or os.urandom(24).hex()
    
    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    
    # Session
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    
    # App settings
    REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "true").lower() in ("1", "true", "yes")
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB max upload
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Azure Blob Storage
    AZURE_BLOB_CONNECTION_STRING: Optional[str] = (
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or 
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    
    # Azure Cognitive Search
    AZURE_SEARCH_ENDPOINT: Optional[str] = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY: Optional[str] = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "azureblob-index")
    AZURE_SEARCH_SEMANTIC_CONFIG: Optional[str] = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
    
    # Celery / Redis
    REDIS_URL: str = (
        os.getenv("AZURE_REDIS_CONNECTION_STRING") or 
        os.getenv("REDIS_URL") or 
        "redis://localhost:6379/0"
    )
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL") or REDIS_URL
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND") or REDIS_URL


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG: bool = True
    SESSION_COOKIE_SECURE: bool = False
    
    # Local SQLite database
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        db_path = os.path.join(os.path.dirname(__file__) or ".", "sagealpha.db")
        return f"sqlite:///{db_path}"
    
    # Rate limiting in memory
    RATELIMIT_STORAGE_URI: str = "memory://"


class ProductionConfig(Config):
    """Production configuration for Azure App Service."""
    
    DEBUG: bool = False
    SESSION_COOKIE_SECURE: bool = True
    
    # Database - prefer DATABASE_URL (Postgres), fallback to Azure SQLite
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # Fix Heroku-style postgres:// -> postgresql://
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            return db_url
        # Fallback to SQLite on Azure persistent storage
        return "sqlite:////home/data/sagealpha.db"
    
    # Connection pool for production databases
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 10,
        "max_overflow": 20,
    }
    
    # Rate limiting with Redis
    @property
    def RATELIMIT_STORAGE_URI(self) -> str:
        return self.REDIS_URL


class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING: bool = True
    DEBUG: bool = True
    SESSION_COOKIE_SECURE: bool = False
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    RATELIMIT_STORAGE_URI: str = "memory://"
    WTF_CSRF_ENABLED: bool = False


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("FLASK_ENV", "development")
    
    # Auto-detect Azure App Service
    if os.getenv("WEBSITE_SITE_NAME"):
        env = "production"
    
    config_class = config.get(env, DevelopmentConfig)
    return config_class()

