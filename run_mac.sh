#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: 'venv' directory not found. Please run installation steps first."
    exit 1
fi

# Start Web Server in background
echo "Starting Web Server..."
uvicorn server:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
SERVER_PID=$!

# Wait a bit for server to start
sleep 2

# Open Webpages
echo "Opening Webpages..."
open "http://localhost:8000/public/operator.html"
open "http://localhost:8000/public/audience.html"

# Run Main Program
echo "Starting Main Program..."
echo "Press Ctrl+C to exit."
python main.py

# Cleanup
echo "Stopping Web Server..."
kill $SERVER_PID 2> /dev/null || true
