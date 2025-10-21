#!/usr/bin/env python3
"""Test script for improved fetch_url function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from abstractcore.tools.common_tools import fetch_url

def test_fetch_url():
    """Test the improved fetch_url function with various content types."""
    
    print("🧪 Testing improved fetch_url function...\n")
    
    # Test 1: JSON API
    print("=" * 60)
    print("TEST 1: JSON API Response")
    print("=" * 60)
    result = fetch_url("https://httpbin.org/json")
    print(result)
    print()
    
    # Test 2: HTML page
    print("=" * 60)
    print("TEST 2: HTML Page")
    print("=" * 60)
    result = fetch_url("https://httpbin.org/html")
    print(result)
    print()
    
    # Test 3: Custom headers
    print("=" * 60)
    print("TEST 3: Custom Headers")
    print("=" * 60)
    result = fetch_url("https://httpbin.org/headers", headers={"X-Test": "improved-fetch"})
    print(result)
    print()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_fetch_url()
