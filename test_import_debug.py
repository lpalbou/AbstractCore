#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_media_imports():
    """Test the exact imports that are failing in LMStudioProvider"""

    print("🔍 TESTING MEDIA IMPORTS")
    print("=" * 40)

    try:
        print("1. Testing import of ..media from providers/base.py context")
        # Simulate the import as it would happen from providers/base.py

        # Test 1: Import from media module directly
        try:
            from abstractcore.media import AutoMediaHandler
            print("   ✅ SUCCESS: from abstractcore.media import AutoMediaHandler")
        except ImportError as e:
            print(f"   ❌ FAILED: from abstractcore.media import AutoMediaHandler - {e}")
            return False

        # Test 2: Import MediaContent from types
        try:
            from abstractcore.media.types import MediaContent
            print("   ✅ SUCCESS: from abstractcore.media.types import MediaContent")
        except ImportError as e:
            print(f"   ❌ FAILED: from abstractcore.media.types import MediaContent - {e}")
            return False

        # Test 3: Try the relative import as it appears in base.py
        print("\n2. Testing relative imports as they appear in base.py")

        # Change to the providers directory context
        original_path = sys.path[:]
        try:
            # Simulate being in providers/base.py
            providers_dir = '/Users/albou/projects/abstractcore/abstractcore/providers'
            os.chdir(providers_dir)

            # This simulates the relative import from providers/base.py
            import importlib.util
            spec = importlib.util.spec_from_file_location("base", f"{providers_dir}/base.py")

            # Test the imports that base.py is trying to do
            try:
                from abstractcore.media import AutoMediaHandler
                print("   ✅ SUCCESS: Relative import of AutoMediaHandler works")
            except ImportError as e:
                print(f"   ❌ FAILED: Relative import of AutoMediaHandler - {e}")
                return False

            try:
                from abstractcore.media.types import MediaContent
                print("   ✅ SUCCESS: Relative import of MediaContent works")
            except ImportError as e:
                print(f"   ❌ FAILED: Relative import of MediaContent - {e}")
                return False

        finally:
            sys.path[:] = original_path

        # Test 4: Create AutoMediaHandler instance
        print("\n3. Testing AutoMediaHandler instantiation")
        try:
            handler = AutoMediaHandler()
            print("   ✅ SUCCESS: AutoMediaHandler() instance created")
        except Exception as e:
            print(f"   ❌ FAILED: AutoMediaHandler() instantiation - {e}")
            return False

        # Test 5: Test a simple media processing call
        print("\n4. Testing MediaContent creation")
        try:
            # Try to create a MediaContent object
            media_content = MediaContent(
                content="test",
                media_type="image",
                content_format="base64",
                mime_type="image/png",
                metadata={}
            )
            print("   ✅ SUCCESS: MediaContent object created")
        except Exception as e:
            print(f"   ❌ FAILED: MediaContent creation - {e}")
            return False

        print("\n✅ ALL IMPORTS SUCCESSFUL")
        return True

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_media_imports()
    if not success:
        print("\n🚨 Import test failed - this explains the LMStudioProvider issue!")
        sys.exit(1)
    else:
        print("\n🎯 Import test passed - the issue must be elsewhere!")