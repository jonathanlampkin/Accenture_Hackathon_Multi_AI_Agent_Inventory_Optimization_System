#!/usr/bin/env python3
"""
API Launch Script for Inventory Optimization System

This script provides a convenient way to launch the inventory optimization API
with configurable settings for host, port, and other options.
"""

import os
import argparse
import uvicorn
import logging

def setup_logging():
    """Configure logging for the API launcher."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main entry point for the API launcher."""
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch the Inventory Optimization API')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to bind the server to (default: 8000)')
    parser.add_argument('--reload', action='store_true', 
                        help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1, 
                        help='Number of worker processes (default: 1)')
    parser.add_argument('--log-level', type=str, default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Log level (default: info)')
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    os.makedirs('api/uploads', exist_ok=True)
    os.makedirs('api/results', exist_ok=True)
    os.makedirs('api/static', exist_ok=True)
    os.makedirs('api/templates', exist_ok=True)
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Auto-reload: {args.reload}")
    
    # Run the API server using Uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 