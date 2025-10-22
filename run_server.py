#!/usr/bin/env python3
"""
Entry point for the airline customer service API server
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Mock logger to avoid dependency issues
class MockLogger:
    def info(self, msg, **kwargs): print(f"INFO: {msg}")
    def debug(self, msg, **kwargs): print(f"DEBUG: {msg}")
    def error(self, msg, **kwargs): print(f"ERROR: {msg}")
    def warning(self, msg, **kwargs): print(f"WARNING: {msg}")

# Mock the logger import before importing the main module
import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: MockLogger()

# Now import and run the main application
from airline_service.main import app, main

if __name__ == "__main__":
    main()