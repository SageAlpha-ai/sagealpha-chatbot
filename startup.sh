#!/bin/bash
# SageAlpha.ai v3.0 - Azure App Service Startup Script
# Async serving with gevent workers for Flask-SocketIO
#
# This script is called by Azure App Service on startup.
# For more control, use startup.py instead.

set -e

echo ""
echo "============================================================"
echo "  SageAlpha.ai Startup Script (Bash)"
echo "============================================================"
echo "  Date: $(date)"
echo "  Python: $(python --version)"
echo "  Platform: $(uname -a)"
echo "============================================================"
echo ""

# Install dependencies if needed
if [ ! -d "venv" ] && [ ! -d "/home/site/wwwroot/antenv" ]; then
    echo "[startup] Installing Python dependencies..."
    pip install -r requirements.txt --quiet
fi

# Install Playwright browsers if needed (for PDF generation)
echo "[startup] Installing Playwright browsers..."
python -m playwright install --with-deps chromium 2>/dev/null || echo "[startup] Playwright install skipped (not critical)"

# Run database migrations
echo "[startup] Running database migrations..."
python -c "from db_migrate import run_migrations; import os; \
    db_path = '/home/data/sagealpha.db'; \
    run_migrations(db_path) if os.path.exists(db_path) else print('No DB to migrate')" \
    2>/dev/null || echo "[startup] Migration skipped"

# Configuration
PORT=${PORT:-8000}
WORKERS=${GUNICORN_WORKERS:-2}

echo ""
echo "[startup] Configuration:"
echo "  PORT: $PORT"
echo "  WORKERS: $WORKERS"
echo "  WORKER_CLASS: geventwebsocket"
echo ""

# Start Gunicorn with gevent workers for WebSocket support
echo "[startup] Starting Gunicorn server..."
exec gunicorn \
    --bind=0.0.0.0:$PORT \
    --worker-class=geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
    --workers=$WORKERS \
    --threads=4 \
    --timeout=120 \
    --keep-alive=5 \
    --max-requests=1000 \
    --max-requests-jitter=100 \
    --access-logfile=- \
    --error-logfile=- \
    --log-level=info \
    --capture-output \
    --preload \
    app:app
