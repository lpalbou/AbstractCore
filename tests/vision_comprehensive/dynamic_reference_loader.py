#!/usr/bin/env python3
"""
Dynamic Reference Analysis Loader
Loads image-specific reference analyses from JSON files in vision_examples directory
Replaces the hardcoded reference_analysis.py approach
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class DynamicReferenceLoader:
    """Loads reference analysis dynamically based on image filename."""

    def __init__(self, vision_examples_dir: str = None):
        if vision_examples_dir is None:
            # Default to vision_examples directory relative to this file
            current_dir = Path(__file__).parent
            self.vision_examples_dir = current_dir.parent / "vision_examples"
        else:
            self.vision_examples_dir = Path(vision_examples_dir)

        # Cache for loaded references
        self._reference_cache = {}

    def load_reference_for_image(self, image_path: str) -> Dict[str, Any]:
        """Load reference analysis for a specific image."""
        image_filename = Path(image_path).name
        json_filename = image_filename.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json')
        json_path = self.vision_examples_dir / json_filename

        # Check cache first
        if str(json_path) in self._reference_cache:
            return self._reference_cache[str(json_path)]

        if not json_path.exists():
            raise FileNotFoundError(f"Reference file not found: {json_path}")

        try:
            with open(json_path, 'r') as f:
                reference_data = json.load(f)

            # Cache the loaded reference
            self._reference_cache[str(json_path)] = reference_data
            return reference_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in reference file {json_path}: {e}")

    def get_reference_keywords(self, image_path: str) -> List[str]:
        """Get reference keywords for comparison."""
        reference = self.load_reference_for_image(image_path)
        return reference.get("keywords", [])

    def get_reference_summary(self, image_path: str) -> str:
        """Get reference summary for comparison."""
        reference = self.load_reference_for_image(image_path)
        return reference.get("summary", "")

    def get_reference_structured(self, image_path: str) -> Dict[str, Any]:
        """Get reference structured analysis for comparison."""
        reference = self.load_reference_for_image(image_path)
        return reference.get("structured", {})

    def get_reference_description(self, image_path: str) -> str:
        """Get reference description."""
        reference = self.load_reference_for_image(image_path)
        return reference.get("description", "")

    def list_available_references(self) -> List[str]:
        """List all available reference files."""
        if not self.vision_examples_dir.exists():
            return []

        json_files = []
        for file_path in self.vision_examples_dir.glob("*.json"):
            json_files.append(file_path.name)

        return sorted(json_files)

    def validate_reference_file(self, json_path: str) -> Dict[str, Any]:
        """Validate that a reference file has the required structure."""
        required_fields = ["image_name", "description", "keywords", "summary", "structured"]

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return {"valid": False, "error": str(e)}

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing_fields}"
            }

        # Validate structured field has expected structure
        structured = data.get("structured", {})
        expected_structured_fields = [
            "theme", "mood", "color_tone", "setting", "weather", "time_of_day",
            "composition", "main_objects", "lighting", "suggested_activity",
            "dominant_colors", "visual_elements", "landscape_type", "human_presence"
        ]

        missing_structured = [field for field in expected_structured_fields if field not in structured]

        return {
            "valid": True,
            "missing_structured_fields": missing_structured,
            "total_keywords": len(data.get("keywords", [])),
            "summary_length": len(data.get("summary", "")),
            "structured_fields": len(structured)
        }

# Test prompts - these remain constant across all images
TEST_PROMPTS = {
    "keywords": "Extract keywords from this image. List only single words or short phrases that describe the objects, scenery, colors, lighting, and activities visible. Separate with commas.",

    "summary": "Provide a detailed descriptive summary of this image. Describe what you see including objects, scenery, lighting, composition, and overall atmosphere in 3-4 sentences.",

    "structured": """Analyze this image and provide a structured response in the following format:

Theme: [Main theme/subject]
Mood: [Emotional tone/atmosphere]
Color_tone: [Overall color palette description]
Setting: [Location/environment type]
Weather: [Weather conditions visible]
Time_of_day: [Apparent time based on lighting]
Composition: [Photographic composition elements]
Main_objects: [Key objects visible]
Lighting: [Lighting conditions and quality]
Suggested_activity: [What activity this scene suggests]
Dominant_colors: [Primary colors present]
Visual_elements: [Notable visual features]
Landscape_type: [Type of terrain/landscape]
Human_presence: [Any signs of human activity]

Provide concise but descriptive answers for each field."""
}

def get_test_prompt(query_type: str) -> str:
    """Get the prompt for a specific query type."""
    return TEST_PROMPTS.get(query_type)

def get_all_prompts() -> Dict[str, str]:
    """Get all three test prompts."""
    return TEST_PROMPTS

# Convenience functions for backward compatibility
_default_loader = None

def get_default_loader() -> DynamicReferenceLoader:
    """Get or create the default reference loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = DynamicReferenceLoader()
    return _default_loader

def get_reference_keywords(image_path: str) -> List[str]:
    """Get reference keywords for image (convenience function)."""
    return get_default_loader().get_reference_keywords(image_path)

def get_reference_summary(image_path: str) -> str:
    """Get reference summary for image (convenience function)."""
    return get_default_loader().get_reference_summary(image_path)

def get_reference_structured(image_path: str) -> Dict[str, Any]:
    """Get reference structured analysis for image (convenience function)."""
    return get_default_loader().get_reference_structured(image_path)

if __name__ == "__main__":
    # Test the loader
    loader = DynamicReferenceLoader()

    print("Available reference files:")
    for ref_file in loader.list_available_references():
        print(f"  - {ref_file}")

    # Test loading each reference
    test_images = [
        "mystery1_mp.jpg",
        "mystery2_sc.jpg",
        "mystery3_us.jpg",
        "mystery4_wh.jpg",
        "mystery5_so.jpg"
    ]

    for image in test_images:
        try:
            print(f"\nTesting {image}:")
            keywords = loader.get_reference_keywords(image)
            summary = loader.get_reference_summary(image)
            structured = loader.get_reference_structured(image)

            print(f"  Keywords: {len(keywords)} items")
            print(f"  Summary: {len(summary)} characters")
            print(f"  Structured: {len(structured)} fields")
            print(f"  Theme: {structured.get('theme', 'N/A')}")

        except Exception as e:
            print(f"  ERROR: {e}")