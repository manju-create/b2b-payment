#!/bin/bash
set -e

PORT="${PORT:-8000}"
echo "Starting Uvicorn server on port ${PORT}..."
exec uvicorn backend.server:app --host 0.0.0.0 --port "${PORT}"
