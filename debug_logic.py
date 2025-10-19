#!/usr/bin/env python3

def test_logic():
    server_response = """I can't see or interpret images — you'll need to describe the image or provide text from it (e.g., "a cat sitting on a windowsill," or "a red car parked in front of a building"), and I'll help you analyze or explain it.

Alternatively, if you're referring to a specific image you've uploaded or are trying to share, please ensure you're using a platform that supports image uploads (like a chat interface with image support), or describe it in detail.

Let me know how you'd like to proceed! 🖼️"""

    vision_keywords = ["can't see", "don't have access", "can't analyze", "cannot see", "unable to view"]

    print("Testing logic:")
    print(f"Server response: {server_response[:100]}...")

    for keyword in vision_keywords:
        found = keyword in server_response.lower()
        print(f"  '{keyword}' found: {found}")

    server_sees_image = not any(keyword in server_response.lower() for keyword in vision_keywords)
    print(f"server_sees_image = {server_sees_image}")

if __name__ == "__main__":
    test_logic()