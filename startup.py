#!/usr/bin/env python3
"""
SageAlpha.ai v3.0 - Azure App Service Startup Script
Python-based startup for better cross-platform compatibility and error handling.

Usage:
  python startup.py

Environment Variables:
  PORT              - Server port (default: 8000)
  GUNICORN_WORKERS  - Number of workers (default: 2)
  WEBSITE_SITE_NAME - Azure App Service detection (auto-set by Azure)
  DATABASE_URL      - PostgreSQL connection string (optional)
"""

import os
import subprocess
import sys


def check_dependencies():
    """Verify critical dependencies are installed."""
    required = ["flask", "gunicorn", "gevent"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[startup] Missing packages: {', '.join(missing)}")
        print("[startup] Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)


def install_playwright():
    """Install Playwright browsers for PDF generation."""
    try:
        print("[startup] Installing Playwright browsers...")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
            timeout=300,
            check=False,
        )
        print("[startup] Playwright installed successfully")
    except Exception as e:
        print(f"[startup] Playwright install skipped: {e}")


def run_migrations():
    """Run database migrations on startup."""
    try:
        print("[startup] Running database migrations...")
        from db_migrate import run_migrations as migrate
        
        # Determine database path
        db_url = os.getenv("DATABASE_URL")
        if db_url and db_url.startswith("sqlite"):
            db_path = db_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                migrate(db_path)
        elif not db_url:
            # Default SQLite location for Azure
            db_path = "/home/data/sagealpha.db"
            if os.path.exists(db_path):
                migrate(db_path)
        print("[startup] Migrations complete")
    except Exception as e:
        print(f"[startup] Migration skipped: {e}")


def start_gunicorn():
    """Start Gunicorn with gevent workers for WebSocket support."""
    port = os.getenv("PORT", "8000")
    workers = os.getenv("GUNICORN_WORKERS", "2")
    
    print("")
    print("=" * 60)
    print("  SageAlpha.ai - Production Server")
    print("=" * 60)
    print(f"  Port: {port}")
    print(f"  Workers: {workers}")
    print(f"  Worker Class: geventwebsocket")
    print("=" * 60)
    print("")
    
    # Gunicorn command with gevent for WebSocket support
    cmd = [
        "gunicorn",
        f"--bind=0.0.0.0:{port}",
        "--worker-class=geventwebsocket.gunicorn.workers.GeventWebSocketWorker",
        f"--workers={workers}",
        "--threads=4",
        "--timeout=120",
        "--keep-alive=5",
        "--max-requests=1000",
        "--max-requests-jitter=100",
        "--access-logfile=-",
        "--error-logfile=-",
        "--log-level=info",
        "--capture-output",
        "app:app",
    ]
    
    # Replace current process with gunicorn
    os.execvp("gunicorn", cmd)


def main():
    """Main startup routine."""
    print("")
    print("=" * 60)
    print("  SageAlpha.ai Startup Script")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Azure: {os.getenv('WEBSITE_SITE_NAME', 'N/A')}")
    print("=" * 60)
    print("")
    
    # Check if running on Azure
    is_azure = os.getenv("WEBSITE_SITE_NAME") is not None
    
    if is_azure:
        # Production startup sequence
        check_dependencies()
        install_playwright()
        run_migrations()
        start_gunicorn()
    else:
        # Local development - just run the Flask app directly
        print("[startup] Development mode - running Flask directly")
        os.environ.setdefault("FLASK_DEBUG", "1")
        
        from app import app, socketio
        
        port = int(os.getenv("PORT", 8000))
        socketio.run(app, host="0.0.0.0", port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()

