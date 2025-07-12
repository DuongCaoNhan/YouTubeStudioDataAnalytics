"""
Test Suite for YouTube Analytics
Comprehensive tests for all modules.
"""

import unittest
import pytest

# Import available test modules
# Note: Additional test modules can be added as they are created

__all__ = []

# Test discovery for pytest
def load_tests(loader, tests, pattern):
    """Load tests for unittest discovery."""
    return loader.discover('tests', pattern='test_*.py')

def run_all_tests():
    """Run all tests in the test suite."""
    unittest.main()
