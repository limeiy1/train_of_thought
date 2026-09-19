#!/usr/bin/env python3
"""
Train of Thought - Root Launcher
================================
Runs the Condition Monitoring server located in app/app.py.
"""
import os
import sys

# Change directory to app and run
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, app_dir)

from app import run_server

if __name__ == "__main__":
    run_server()
