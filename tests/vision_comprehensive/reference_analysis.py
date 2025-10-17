#!/usr/bin/env python3
"""
Reference Analysis for Mystery1 Image
Created by Claude Code for structured vision testing comparison
"""

# My reference analysis as ground truth for mystery1_mp.jpg
REFERENCE_ANALYSIS = {
    "image_name": "mystery1_mp.jpg",
    "description": "Mountain hiking trail with wooden fence and scenic landscape",

    # Query 1: Keywords extraction
    "keywords": [
        "mountain", "hiking", "trail", "fence", "wooden", "railing", "path",
        "dirt", "road", "gravel", "sky", "clouds", "sun", "lens flare", "sunlight",
        "valley", "hills", "landscape", "scenic", "outdoor", "nature", "countryside",
        "grass", "vegetation", "shadows", "perspective", "depth", "vista", "horizon",
        "backlighting", "golden hour"
    ],

    # Query 2: Descriptive summary
    "summary": """A scenic mountain hiking trail captured during daytime with bright sunlight creating a dramatic lens flare effect. The image shows a dirt and gravel path or trail running through mountainous terrain, with a wooden rail fence along the left side of the path. The trail leads toward distant mountains and valleys visible in the background, creating a strong sense of depth and perspective. The sky is bright blue with scattered white clouds, and the sun is positioned in the upper left creating dramatic backlighting and long shadows cast across the path. Green vegetation and grass can be seen along the trail edges and slopes. The composition uses leading lines from both the path and fence to draw the viewer's eye toward the expansive mountain vista in the distance.""",

    # Query 3: Structured analysis
    "structured": {
        "theme": "Outdoor recreation and scenic nature photography",
        "mood": "Peaceful, inspiring, adventurous, serene",
        "color_tone": "Warm golden hour lighting with dominant blue sky",
        "setting": "Mountain trail in rural/natural environment",
        "weather": "Clear sunny day with partial cloud cover",
        "time_of_day": "Late morning or early afternoon",
        "composition": "Leading lines, strong perspective, rule of thirds",
        "main_objects": ["wooden fence", "dirt trail", "mountains", "vegetation"],
        "lighting": "Dramatic backlighting with lens flare and strong contrast",
        "suggested_activity": "Hiking, walking, outdoor recreation",
        "dominant_colors": ["blue", "golden yellow", "green", "brown"],
        "visual_elements": ["lens flare", "shadows", "depth of field", "leading lines"],
        "landscape_type": "Mountainous terrain with rolling hills",
        "human_presence": "None visible, but trail suggests human activity"
    }
}

# Three different prompts for testing
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

def get_reference_keywords():
    """Get reference keywords for comparison."""
    return REFERENCE_ANALYSIS["keywords"]

def get_reference_summary():
    """Get reference summary for comparison."""
    return REFERENCE_ANALYSIS["summary"]

def get_reference_structured():
    """Get reference structured analysis for comparison."""
    return REFERENCE_ANALYSIS["structured"]

def get_test_prompt(query_type):
    """Get the prompt for a specific query type."""
    return TEST_PROMPTS.get(query_type)

def get_all_prompts():
    """Get all three test prompts."""
    return TEST_PROMPTS