"""
config.py

Handles loading and saving of application configuration settings from a JSON file.

This module ensures that the application has a consistent configuration state,
providing default values for any settings that are not explicitly defined in
the config.json file. This prevents errors from missing configuration keys and
makes it easy to add new settings in future updates without breaking existing
installations.
"""

import json
import os
import logging

# Define the name of the configuration file
CONFIG_FILE = "config.json"

def load_config():
    """
    Loads configuration from config.json.

    If the file doesn't exist or is corrupted, it creates a new configuration
    with default values. It also ensures that any new configuration options
    are added to existing files, maintaining backward compatibility.

    Returns:
        dict: A dictionary containing the application's configuration settings.
    """
    # Check if the configuration file exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            logging.error("Configuration file '%s' is corrupted. Loading default config.", CONFIG_FILE)
            config_data = {}
    else:
        # If the file doesn't exist, start with an empty configuration
        config_data = {}

    # Define the default structure and values for the configuration
    default_config = {
        "GEMINI_API_KEY": "",
        "WAVESPEED_AI_KEY": "",
        "GCP_PROJECT_ID": "",
        "GCP_LOCATION": "us-central1",
        "NEWS_API_KEY": "",
        "VIDEO_ENGINE": "WaveSpeed AI",
        "IMAGE_ENGINE": "Gemini API",
        "SPEAKER1": "Kore",
        "SPEAKER2": "Puck",
        "HOST_NAME": "Alex",
        "GUEST_NAME": "Maya",
        "HOST_PERSONA": "A friendly podcast host who loves technology.",
        "GUEST_PERSONA": "An expert on the topic with a calm and informative style.",
        "CHANNEL_NAME": "My AI Channel",
        "SUBSCRIBE_COUNT": 3,
        "SUBSCRIBE_MESSAGE": "Don’t forget to subscribe to {channel} for more awesome content!",
        "SUBSCRIBE_RANDOM": True,
        "PODCAST_STYLE": "Informative News",
        "VIDEO_PROMPT_BASE_STYLE": "An animated and cinematic video. High-quality, 24fps.",
        "IMAGE_PROMPT_STYLE": "A cinematic, photorealistic image representing the podcast topic: {topic}",
        "STORY_ARC": "None",
        "API_DELAY": 2,
        "FACT_CHECK_ENABLED": False,
        "CAPTION_ENABLED": False,
        "GENERATE_METADATA": False,
        "GENERATE_THUMBNAIL": False,
        "GENERATE_TIMED_IMAGES": False,
        "GENERATE_TIMESTAMPS": True, # <-- NEW FEATURE FLAG
        "YOUTUBE_CLIENT_ID": "",
        "YOUTUBE_CLIENT_SECRET": "",
        "FACEBOOK_ACCESS_TOKEN": "",
        "VIDEO_TITLE": "",
        "VIDEO_DESCRIPTION": "",
        "VIDEO_TAGS": "",
        "LANGUAGE_ENABLED": False,
        "PODCAST_LANGUAGE": "English",
        "VIDEO_ASPECT_RATIO": "16:9 (Horizontal)",
        "SCRIPT_LENGTH": "Medium (~5 minutes)",
        "ADD_MUSIC": False,
        "GENERATE_SNIPPETS": False,
        "IMAGE_GENERATION_INTERVAL": 10,
        "TIMED_IMAGES_AS_SLIDESHOW": False,
    }

    # Update the loaded config with any missing default keys
    config_updated = False
    for key, value in default_config.items():
        if key not in config_data:
            config_data[key] = value
            config_updated = True
    
    if config_updated:
        logging.info("Configuration updated with new default values. Saving.")
        save_config(config_data)


    return config_data

def save_config(config_data):
    """
    Saves the provided configuration dictionary to the config.json file.
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        logging.info("Configuration saved successfully to '%s'.", CONFIG_FILE)
    except Exception as e:
        logging.error("Failed to save configuration file: %s", e)

