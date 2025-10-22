#!/usr/bin/env python3
"""
Server runner script for airline customer service
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from airline_service.main import main
    main()