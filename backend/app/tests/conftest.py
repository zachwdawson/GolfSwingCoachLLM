"""Pytest configuration and fixtures."""
import os

# Set TESTING environment variable to prevent worker thread from starting
os.environ["TESTING"] = "1"

