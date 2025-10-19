#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def debug_lmstudio_media_handling():
    """Debug what's happening in LMStudio provider media handling"""

    print("🔍 DEBUGGING LMSTUDIO MEDIA HANDLING")
    print("=" * 50)

    # Simulate the server's processed data that gets passed to the provider
    from abstractcore.server.app import ChatMessage, process_message_content

    # Original request content (what server receives)
    original_content = [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
        }
    ]

    print("1. Server Processing:")
    message = ChatMessage(role="user", content=original_content)
    clean_text, media_files = process_message_content(message)

    print(f"   Original content items: {len(original_content)}")
    print(f"   Clean text extracted: '{clean_text}'")
    print(f"   Media files extracted: {len(media_files)}")
    print()

    # Simulate what gets passed to provider
    processed_messages = [{"role": "user", "content": clean_text}]

    print("2. Provider Input:")
    print(f"   Messages: {processed_messages}")
    print(f"   Media files: {media_files}")
    print()

    # Simulate the provider's logic
    print("3. Provider Processing:")

    # Step 1: Provider extracts user_message_text (current buggy logic)
    prompt = ""  # Empty in server context
    chat_messages = processed_messages.copy()

    user_message_text = prompt.strip() if prompt else ""
    if not user_message_text and chat_messages:
        for msg in reversed(chat_messages):
            if msg.get("role") == "user" and msg.get("content"):
                user_message_text = msg["content"]
                break

    print(f"   Extracted user_message_text: '{user_message_text}'")

    # Step 2: Get media handler (simulate without needing real LMStudio server)
    from abstractcore.architectures.detection import supports_vision, get_model_capabilities

    # Simulate the provider's _get_media_handler_for_model logic
    model_name = "qwen/qwen3-vl-4b"
    clean_model_name = model_name.replace("qwen/", "")  # qwen3-vl-4b

    # Get actual model capabilities
    model_capabilities = get_model_capabilities(clean_model_name)
    print(f"   Model capabilities: vision_support = {model_capabilities.get('vision_support', False)}")

    if supports_vision(clean_model_name):
        from abstractcore.media.handlers import OpenAIMediaHandler
        media_handler = OpenAIMediaHandler(model_capabilities, model_name=model_name)
        print(f"   Using OpenAIMediaHandler for vision model")
    else:
        from abstractcore.media.handlers import LocalMediaHandler
        media_handler = LocalMediaHandler("lmstudio", model_capabilities, model_name=model_name)
        print(f"   Using LocalMediaHandler for text-only model")

    print(f"   Media handler type: {type(media_handler).__name__}")

    # Step 3: Create multimodal message
    if media_files:
        # Process media files to MediaContent objects
        from abstractcore.media.auto_handler import AutoMediaHandler
        auto_handler = AutoMediaHandler()
        media_contents = []

        for file_path in media_files:
            result = auto_handler.process_file(file_path)
            if result.success:
                media_contents.append(result.media_content)

        print(f"   Media contents created: {len(media_contents)}")

        # Create multimodal message
        multimodal_message = media_handler.create_multimodal_message(user_message_text, media_contents)

        print(f"   Multimodal message type: {type(multimodal_message)}")
        print(f"   Multimodal message: {multimodal_message}")

        # Step 4: Apply provider's message replacement logic
        if isinstance(multimodal_message, str):
            if chat_messages and chat_messages[-1].get("role") == "user":
                chat_messages[-1]["content"] = multimodal_message
            else:
                chat_messages.append({"role": "user", "content": multimodal_message})
        else:
            if chat_messages and chat_messages[-1].get("role") == "user":
                chat_messages[-1] = multimodal_message  # REPLACE entire message
            else:
                chat_messages.append(multimodal_message)

    # Step 5: Apply the "add prompt as separate message" logic (line 182-186)
    elif prompt and prompt.strip():
        chat_messages.append({"role": "user", "content": prompt})

    print()
    print("4. Final Messages Sent to LMStudio:")
    for i, msg in enumerate(chat_messages):
        print(f"   Message {i}: {msg}")

    print()
    print("🎯 DIAGNOSIS:")

    # Check for issues
    issues = []

    if len(chat_messages) > 1:
        user_messages = [msg for msg in chat_messages if msg.get("role") == "user"]
        if len(user_messages) > 1:
            issues.append("DUPLICATE USER MESSAGES!")

    # Check if any message has proper multimodal content
    has_proper_multimodal = False
    for msg in chat_messages:
        content = msg.get("content")
        if isinstance(content, list):
            has_image = any(item.get("type") == "image_url" for item in content if isinstance(item, dict))
            if has_image:
                has_proper_multimodal = True
                break

    if not has_proper_multimodal:
        issues.append("NO PROPER MULTIMODAL CONTENT!")

    if issues:
        print(f"   ❌ ISSUES FOUND: {', '.join(issues)}")

        # Suggest fixes
        print("\n🔧 SUGGESTED FIXES:")
        if "DUPLICATE USER MESSAGES" in issues:
            print("   • Remove duplicate message creation logic")
            print("   • Don't add prompt as separate message when media is present")
        if "NO PROPER MULTIMODAL CONTENT" in issues:
            print("   • Ensure OpenAIMediaHandler creates proper content array")
            print("   • Verify media files are correctly processed to MediaContent objects")
    else:
        print("   ✅ No obvious issues found")

    return len(issues) == 0

if __name__ == "__main__":
    success = debug_lmstudio_media_handling()

    if not success:
        print("\n🚨 Media handling has issues that need to be fixed!")
    else:
        print("\n✅ Media handling appears to work correctly!")