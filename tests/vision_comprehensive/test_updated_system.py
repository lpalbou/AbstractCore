#!/usr/bin/env python3
"""
Test script to validate the updated vision testing system
with dynamic reference loading.
"""

import sys
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamic_reference_loader import DynamicReferenceLoader, get_test_prompt

def test_dynamic_loading():
    """Test dynamic reference loading for all images."""
    print("🧪 TESTING DYNAMIC REFERENCE LOADING")
    print("=" * 50)

    loader = DynamicReferenceLoader()

    # Test images
    test_images = [
        "mystery1_mp.jpg",
        "mystery2_sc.jpg",
        "mystery3_us.jpg",
        "mystery4_wh.jpg",
        "mystery5_so.jpg"
    ]

    all_passed = True

    for image in test_images:
        print(f"\n📸 Testing {image}:")

        try:
            # Load reference
            reference = loader.load_reference_for_image(image)

            # Validate structure
            required_fields = ["image_name", "description", "keywords", "summary", "structured"]
            missing_fields = [field for field in required_fields if field not in reference]

            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
                all_passed = False
                continue

            # Test convenience functions
            keywords = loader.get_reference_keywords(image)
            summary = loader.get_reference_summary(image)
            structured = loader.get_reference_structured(image)

            # Validate data types and content
            if not isinstance(keywords, list) or len(keywords) == 0:
                print(f"   ❌ Invalid keywords: {type(keywords)}, length {len(keywords)}")
                all_passed = False
                continue

            if not isinstance(summary, str) or len(summary) < 100:
                print(f"   ❌ Invalid summary: {type(summary)}, length {len(summary)}")
                all_passed = False
                continue

            if not isinstance(structured, dict) or len(structured) < 10:
                print(f"   ❌ Invalid structured: {type(structured)}, length {len(structured)}")
                all_passed = False
                continue

            print(f"   ✅ Valid reference loaded")
            print(f"      - Keywords: {len(keywords)} items")
            print(f"      - Summary: {len(summary)} chars")
            print(f"      - Structured: {len(structured)} fields")
            print(f"      - Theme: {structured.get('theme', 'N/A')}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False

    return all_passed

def test_prompts():
    """Test that prompts are accessible."""
    print("\n🧪 TESTING PROMPT SYSTEM")
    print("=" * 50)

    query_types = ["keywords", "summary", "structured"]
    all_passed = True

    for query_type in query_types:
        try:
            prompt = get_test_prompt(query_type)
            if not prompt or len(prompt) < 50:
                print(f"   ❌ Invalid prompt for {query_type}: {len(prompt) if prompt else 0} chars")
                all_passed = False
            else:
                print(f"   ✅ {query_type}: {len(prompt)} chars")
        except Exception as e:
            print(f"   ❌ Error getting {query_type} prompt: {e}")
            all_passed = False

    return all_passed

def test_image_specific_differences():
    """Test that each image has unique reference data."""
    print("\n🧪 TESTING IMAGE-SPECIFIC DIFFERENCES")
    print("=" * 50)

    loader = DynamicReferenceLoader()

    test_images = [
        "mystery1_mp.jpg",
        "mystery2_sc.jpg",
        "mystery3_us.jpg",
        "mystery4_wh.jpg",
        "mystery5_so.jpg"
    ]

    themes = []
    descriptions = []
    keyword_sets = []

    for image in test_images:
        try:
            reference = loader.load_reference_for_image(image)
            themes.append(reference["structured"]["theme"])
            descriptions.append(reference["description"])
            keyword_sets.append(set(reference["keywords"]))
        except Exception as e:
            print(f"   ❌ Error loading {image}: {e}")
            return False

    # Check for uniqueness
    unique_themes = len(set(themes))
    unique_descriptions = len(set(descriptions))

    print(f"   📊 Unique themes: {unique_themes}/{len(themes)}")
    print(f"   📊 Unique descriptions: {unique_descriptions}/{len(descriptions)}")

    # Check keyword overlap (should be low)
    total_overlap = 0
    comparisons = 0

    for i in range(len(keyword_sets)):
        for j in range(i+1, len(keyword_sets)):
            overlap = len(keyword_sets[i].intersection(keyword_sets[j]))
            total_keywords = len(keyword_sets[i].union(keyword_sets[j]))
            overlap_percent = (overlap / total_keywords) * 100 if total_keywords > 0 else 0
            total_overlap += overlap_percent
            comparisons += 1

            print(f"   🔍 {test_images[i]} vs {test_images[j]}: {overlap_percent:.1f}% keyword overlap")

    avg_overlap = total_overlap / comparisons if comparisons > 0 else 0
    print(f"   📊 Average keyword overlap: {avg_overlap:.1f}%")

    # Success criteria
    success = (unique_themes == len(themes) and
              unique_descriptions == len(descriptions) and
              avg_overlap < 30)  # Less than 30% average overlap

    if success:
        print("   ✅ All images have unique, distinct references")
    else:
        print("   ❌ Images may have too similar references")

    return success

def main():
    """Run all validation tests."""
    print("🚀 VALIDATION TESTS FOR UPDATED VISION SYSTEM")
    print("=" * 60)

    # Run all tests
    test1_passed = test_dynamic_loading()
    test2_passed = test_prompts()
    test3_passed = test_image_specific_differences()

    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Dynamic Reference Loading: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Prompt System: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Image-Specific Differences: {'✅ PASS' if test3_passed else '❌ FAIL'}")

    all_passed = test1_passed and test2_passed and test3_passed

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Updated vision system is ready.")
        print("\nNext steps:")
        print("1. Test with a single image: python updated_vision_tester.py --images tests/vision_examples/mystery1_mp.jpg --save-results")
        print("2. Test with all images: python updated_vision_tester.py --images tests/vision_examples/*.jpg --save-results")
        print("3. Replace the old comprehensive_vision_tester.py with updated_vision_tester.py")
    else:
        print("\n❌ SOME TESTS FAILED! Please fix issues before proceeding.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())