#!/usr/bin/env python3
"""
Test script to check if the media handler imports are working correctly.
"""

def test_import():
    """Test the import that's failing in Ollama provider."""
    print("Testing import: from abstractcore.media.handlers import LocalMediaHandler")

    try:
        from abstractcore.media.handlers import LocalMediaHandler
        print("✅ Import successful!")

        # Test if the method exists
        if hasattr(LocalMediaHandler, 'create_multimodal_message'):
            print("✅ create_multimodal_message method exists")
        else:
            print("❌ create_multimodal_message method does NOT exist")

        # Check initialization
        try:
            handler = LocalMediaHandler("ollama", {}, model_name="test")
            print("✅ LocalMediaHandler initialization successful")
        except Exception as e:
            print(f"❌ LocalMediaHandler initialization failed: {e}")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_relative_import():
    """Test the relative import that's used in the Ollama provider."""
    print("\nTesting relative import from provider context...")

    import sys
    import os

    # Add the providers directory to path to simulate the provider's context
    providers_dir = "/Users/albou/projects/abstractcore/abstractcore/providers"
    sys.path.insert(0, providers_dir)

    try:
        # Change to providers directory
        original_cwd = os.getcwd()
        os.chdir(providers_dir)

        # Test the relative import as used in ollama_provider.py
        from abstractcore.media.handlers import LocalMediaHandler
        print("✅ Relative import successful from provider context!")

        # Reset working directory
        os.chdir(original_cwd)

    except ImportError as e:
        print(f"❌ Relative import failed: {e}")
        # Reset working directory even on failure
        os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ Unexpected error with relative import: {e}")
        os.chdir(original_cwd)

if __name__ == "__main__":
    test_import()
    test_relative_import()