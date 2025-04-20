#!/usr/bin/env python3
"""
Start API Server

This script starts the Inventory Optimization API server and opens a browser to access it.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pandas
        import matplotlib
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies with:")
        print("pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if data files are available."""
    uploads_dir = Path("api/uploads")
    uploads_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if we have any CSV files
    csv_files = list(uploads_dir.glob("*.csv"))
    
    if not csv_files:
        print("Warning: No data files found in api/uploads directory.")
        print("The API will start, but you'll need to upload data files to use forecasting features.")
        return False
    
    print(f"Found {len(csv_files)} data file(s) in api/uploads directory.")
    return True

def create_output_dirs():
    """Create necessary output directories."""
    Path("api/results").mkdir(exist_ok=True, parents=True)
    print("Created output directories.")

def start_api(port=8000, host="0.0.0.0"):
    """Start the API server."""
    api_dir = Path("api")
    
    if not api_dir.exists() or not (api_dir / "main.py").exists():
        print("Error: API directory or main.py not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {api_dir.absolute()}")
        return False
    
    print(f"Starting API server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Start the server using subprocess
    cmd = [
        sys.executable, 
        "-m", "uvicorn", 
        "api.main:app", 
        "--host", host, 
        "--port", str(port), 
        "--reload"
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open(f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        
        print(f"Browser opened to http://localhost:{port}")
        print("Server is running in the background.")
        print("Run stop_api.py to stop the server when you're done.")
        
        return True
    
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def main():
    """Main function to start the API server."""
    print("=== Inventory Optimization API ===")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create necessary directories
    create_output_dirs()
    
    # Check for data files
    check_data_files()
    
    # Start the API
    port = 8000
    try:
        # Try to get port from command line
        if len(sys.argv) > 1:
            port = int(sys.argv[1])
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}. Using default port 8000.")
    
    start_api(port=port)

if __name__ == "__main__":
    main() 