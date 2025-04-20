#!/bin/bash

# Inventory Optimization System Startup Script
# This script starts the API server for the inventory forecasting system

# Set environment variables
export PYTHONPATH=$(pwd)

# Check if a virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p api/uploads api/results api/static api/templates

# Check command line arguments
HOST="0.0.0.0"
PORT=8000
RELOAD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --port)
        PORT="$2"
        shift
        shift
        ;;
        --reload)
        RELOAD=true
        shift
        ;;
        --host)
        HOST="$2"
        shift
        shift
        ;;
        *)
        # Unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Display startup message
echo "Starting Inventory Optimization System"
echo "--------------------------------------"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Development mode (auto-reload): $RELOAD"
echo "--------------------------------------"
echo "API will be available at http://$HOST:$PORT"
echo "API documentation at http://$HOST:$PORT/docs"
echo "--------------------------------------"

# Start the API server
if [ "$RELOAD" = true ]; then
    python run_api.py --host "$HOST" --port "$PORT" --reload
else
    python run_api.py --host "$HOST" --port "$PORT"
fi 