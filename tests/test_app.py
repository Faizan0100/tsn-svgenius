"""
This module contains unit tests for the functions in app.py.
"""

from app import create_prompt

def test_create_prompt():
    """
    Test the create_prompt function to ensure it correctly formats the prompt.
    """
    result = create_prompt("test", True, False, "2D")
    assert result == "test, cartoon, 2d"

# Ensure there is a newline character here.
