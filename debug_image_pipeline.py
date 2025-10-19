#!/usr/bin/env python3

import sys
import os
import hashlib
import requests
from PIL import Image

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_image_download_directly():
    """Test downloading the image directly to see what we get"""

    print("🔍 TESTING DIRECT IMAGE DOWNLOAD")
    print("=" * 50)

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    print(f"   URL: {image_url}")

    try:
        # Download like the server does
        req = requests.Request('GET', image_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        req.add_header('Accept', 'image/webp,image/apng,image/*,*/*;q=0.8')
        req.add_header('Accept-Language', 'en-US,en;q=0.9')
        req.add_header('Accept-Encoding', 'gzip, deflate, br')
        req.add_header('DNT', '1')
        req.add_header('Connection', 'keep-alive')
        req.add_header('Upgrade-Insecure-Requests', '1')

        import urllib.request
        urllib_req = urllib.request.Request(image_url)
        for header, value in req.headers.items():
            urllib_req.add_header(header, value)

        with urllib.request.urlopen(urllib_req) as response:
            image_data = response.read()

        print(f"   Downloaded size: {len(image_data)} bytes")

        # Calculate hash
        image_hash = hashlib.md5(image_data).hexdigest()
        print(f"   MD5 hash: {image_hash}")

        # Save to temp file and examine
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            temp_path = tmp_file.name

        # Try to open with PIL to verify it's a valid image
        try:
            img = Image.open(temp_path)
            print(f"   Image format: {img.format}")
            print(f"   Image size: {img.size}")
            print(f"   Image mode: {img.mode}")

            # Get a description of the image by looking at predominant colors
            colors = img.getcolors(maxcolors=1000000)
            if colors:
                # Get most common color
                most_common = max(colors, key=lambda x: x[0])
                print(f"   Most common color: {most_common[1]} (count: {most_common[0]})")

        except Exception as e:
            print(f"   ❌ Invalid image: {e}")
            return None

        # Cleanup
        os.unlink(temp_path)

        return image_hash, len(image_data)

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None

def test_server_media_processing():
    """Test what the server does when processing this specific image"""

    print(f"\n🔍 TESTING SERVER MEDIA PROCESSING")
    print("=" * 50)

    try:
        from abstractcore.server.app import ChatMessage, process_message_content

        # Create the exact message content the server receives
        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]

        print("   Processing message content...")
        message = ChatMessage(role="user", content=content)
        clean_text, media_files = process_message_content(message)

        print(f"   Clean text: '{clean_text}'")
        print(f"   Media files: {len(media_files)}")

        if media_files:
            for i, file_path in enumerate(media_files):
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   File {i}: {file_path}")
                    print(f"     Size: {file_size} bytes")

                    # Calculate hash of the processed file
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    file_hash = hashlib.md5(file_data).hexdigest()
                    print(f"     MD5 hash: {file_hash}")

                    # Try to open and analyze the processed image
                    try:
                        img = Image.open(file_path)
                        print(f"     Image format: {img.format}")
                        print(f"     Image size: {img.size}")
                        print(f"     Image mode: {img.mode}")

                        # Sample a few pixels to characterize the image
                        width, height = img.size
                        center_pixel = img.getpixel((width//2, height//2))
                        top_left = img.getpixel((10, 10))
                        bottom_right = img.getpixel((width-10, height-10))

                        print(f"     Center pixel: {center_pixel}")
                        print(f"     Top-left pixel: {top_left}")
                        print(f"     Bottom-right pixel: {bottom_right}")

                        return file_hash, file_size, file_path

                    except Exception as e:
                        print(f"     ❌ Cannot read processed image: {e}")

                else:
                    print(f"   File {i}: {file_path} (does not exist)")

        return None

    except Exception as e:
        print(f"   ❌ Server processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_images(direct_result, server_result):
    """Compare the direct download vs server processed image"""

    print(f"\n📊 IMAGE COMPARISON")
    print("=" * 50)

    if not direct_result:
        print("   ❌ Direct download failed")
        return

    if not server_result:
        print("   ❌ Server processing failed")
        return

    direct_hash, direct_size = direct_result
    server_hash, server_size, server_path = server_result

    print(f"   Direct download:")
    print(f"     Hash: {direct_hash}")
    print(f"     Size: {direct_size} bytes")

    print(f"   Server processed:")
    print(f"     Hash: {server_hash}")
    print(f"     Size: {server_size} bytes")
    print(f"     Path: {server_path}")

    if direct_hash == server_hash:
        print(f"\n   ✅ IDENTICAL: Server processed same image as direct download")
    else:
        print(f"\n   ❌ DIFFERENT: Server processed a different image!")
        print(f"   This explains why the model sees wrong content.")

        # Size comparison
        if direct_size != server_size:
            print(f"   Size difference: {abs(direct_size - server_size)} bytes")

def main():
    print("Debugging image processing pipeline...")

    # Test 1: Direct download
    direct_result = test_image_download_directly()

    # Test 2: Server processing
    server_result = test_server_media_processing()

    # Test 3: Compare
    compare_images(direct_result, server_result)

    if direct_result and server_result:
        direct_hash, _ = direct_result
        server_hash, _, _ = server_result

        if direct_hash != server_hash:
            print(f"\n🚨 ROOT CAUSE IDENTIFIED:")
            print(f"   The server is downloading/processing a DIFFERENT IMAGE")
            print(f"   than what we expect. This explains the wrong model responses.")
        else:
            print(f"\n🤔 Images are identical - issue must be elsewhere in pipeline")

if __name__ == "__main__":
    main()