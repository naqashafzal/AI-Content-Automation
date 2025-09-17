"""
main.py

This is the main entry point for the Content Automation application.
It builds the GUI, handles user interactions, and orchestrates the pipeline.
This version includes a custom logging handler to display console output
directly within the application's UI and a button to check API quotas.
"""

import sys
print(f"--- Running application with Python from: {sys.executable}")
import os
import threading
import logging
import webbrowser
import shutil
import re
import requests
from tkinter import messagebox, filedialog
import pickle
from pathlib import Path

import customtkinter as ctk


# Google Auth Imports
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

# Modular components
from config import load_config, save_config
from pipeline import Pipeline
from api_clients import GoogleClient, NewsApiClient

# --- Constants for GUI Dropdowns ---
# --- UPDATED STEP ORDER ---
PIPELINE_STEPS = [
    "Deep Research",
    "Fact Check Research",
    "Revise Research",
    "Podcast Script",
    "Generate Thumbnail",
    "Analyze Tone",
    "Audio (TTS)",
    "Video Generation",        # Background video now generated BEFORE timed images
    "Generate Timed Images",   # <-- MOVED HERE
    "Add Background Music", 
    "Create Final Video",
    "Generate SEO Metadata",
    "Generate Timestamps",
    "Generate Snippets"
]
# --- END UPDATED STEP ORDER ---

VOICE_OPTIONS = {
    "Achernar": "Clear, mid-range, enthusiastic & approachable", "Achird": "Youthful, breathy, inquisitive tone",
    "Algenib": "Warm, confident, friendly authority", "Alnilam": "Energetic, low pitch, promotional tone",
    "Aoede": "Clear, conversational, thoughtful", "Autonoe": "Mature, resonant, calm and wise",
    "Callirrhoe": "Confident, professional, energetic", "Despina": "Warm, inviting, trustworthy",
    "Erinome": "Professional, articulate, thoughtful", "Gacrux": "Authoritative yet approachable",
    "Iapetus": "Casual, relatable, ‘everyman’ tone", "Kore": "Energetic, youthful, clear & bright",
    "Laomedeia": "Inquisitive, intelligent & engaging", "Leda": "Composed, professional, calm",
    "Orus": "Resonant, authoritative, thoughtful", "Puck": "Confident, informal, trustworthy",
    "Pulcherrima": "Bright, enthusiastic, youthful", "Rasalgethi": "Conversational, thoughtful, quirky",
    "Sadachbia": "Deep, textured, confident, cool", "Sadaltager": "Friendly, enthusiastic, professional",
    "Schedar": "Down-to-earth, approachable", "Sulafat": "Warm, persuasive, articulate",
    "Umbriel": "Authoritative, clear, engaging", "Vindemiatrix": "Calm, mature, smooth, reassuring",
    "Zephyr": "Energetic, bright, perky & enthusiastic", "Zubenelgenubi": "Deep, resonant, powerful authority"
}
PODCAST_STYLES = ["Informative News", "Comedy / Entertaining", "Educational / Explainer", "Motivational / Inspiring", "Casual Conversational", "Serious Debate", "Story Mode", "Documentary", "ASMR"]
STORY_ARCS = ["None", "Hero's Journey", "Three-Act Structure", "Man vs. Nature", "Rags to Riches", "Voyage and Return"]
CONTENT_STYLES = ["Podcast", "ASMR Video", "Documentary", "Product Ad", "Story", "Kids Story", "Horror Story", "Viral Video"]
SCRIPT_LENGTHS = ["Short (~2 minutes)", "Medium (~5 minutes)", "Long (~10 minutes)"]


# --- Custom Logging Handler ---
class TextboxHandler(logging.Handler):
    """A custom logging handler that redirects logs to a CTkTextbox widget."""
    def __init__(self, textbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record):
        """Writes the log message to the textbox."""
        msg = self.format(record)
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg + "\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎙️ Nullpk Ai Content Studio")
        self.geometry("1100x950")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config = load_config()
        self.stop_event = threading.Event()
        self.progress_bars = []
        self.step_status_labels = []

        self._create_widgets()
        self.setup_logging()
        self.load_settings_into_gui()
        self.update_history_tab()

    def setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        handler = TextboxHandler(self.log_box)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(stream_handler)
        logging.info("Logging configured. Application started.")

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        tabs = ctk.CTkTabview(main_frame)
        tabs.pack(fill="both", expand=True)

        self.main_tab = tabs.add("Main")
        self.settings_tab = tabs.add("Settings")
        self.publish_tab = tabs.add("Publish")
        self.history_tab = tabs.add("History")
        self.about_tab = tabs.add("About")

        self._create_main_tab_widgets()
        self._create_settings_tab_widgets()
        self._create_publish_tab_widgets()
        self._create_about_tab_widgets()

    def _create_main_tab_widgets(self):
        self.main_tab.columnconfigure((0, 1), weight=1)
        self.main_tab.rowconfigure(0, weight=1)
        controls_frame = ctk.CTkFrame(self.main_tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(controls_frame, text="Enter Topic", font=("Arial", 16, "bold")).pack(pady=(20, 5))
        self.topic_entry = ctk.CTkEntry(controls_frame, width=400, placeholder_text="e.g., 'The Future of AI'")
        self.topic_entry.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(controls_frame, text="Start From Step:", font=("Arial", 12)).pack(pady=(15, 5))
        self.start_step_combo = ctk.CTkComboBox(controls_frame, values=PIPELINE_STEPS, width=300)
        self.start_step_combo.pack(pady=5)
        ctk.CTkLabel(controls_frame, text="Content Style:", font=("Arial", 12)).pack(pady=(15, 5))
        self.content_style_combo = ctk.CTkComboBox(controls_frame, values=CONTENT_STYLES, width=300, command=self.update_features_based_on_style)
        self.content_style_combo.pack(pady=5)
        
        checkbox_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        checkbox_frame.pack(pady=20, padx=20, fill="x")
        
        self.fact_check_var = ctk.BooleanVar()
        self.fact_check_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Enable Fact-Checking", variable=self.fact_check_var)
        self.fact_check_checkbox.pack(anchor="w", pady=4)
        
        self.metadata_var = ctk.BooleanVar()
        self.metadata_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Generate SEO Metadata", variable=self.metadata_var)
        self.metadata_checkbox.pack(anchor="w", pady=4)

        self.timestamps_var = ctk.BooleanVar()
        self.timestamps_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Generate Timestamps", variable=self.timestamps_var)
        self.timestamps_checkbox.pack(anchor="w", pady=4)
        
        self.caption_var = ctk.BooleanVar()
        self.caption_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Enable Auto-Captioning", variable=self.caption_var)
        self.caption_checkbox.pack(anchor="w", pady=4)
        
        self.add_music_var = ctk.BooleanVar()
        self.add_music_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Add Background Music", variable=self.add_music_var)
        self.add_music_checkbox.pack(anchor="w", pady=4)
        
        self.generate_snippets_var = ctk.BooleanVar()
        self.snippets_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Generate Social Media Snippets", variable=self.generate_snippets_var)
        self.snippets_checkbox.pack(anchor="w", pady=4)
        
        self.generate_thumbnail_var = ctk.BooleanVar()
        self.thumbnail_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Generate Thumbnail", variable=self.generate_thumbnail_var)
        self.thumbnail_checkbox.pack(anchor="w", pady=4)

        self.generate_timed_images_var = ctk.BooleanVar()
        self.timed_images_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Generate Timed Images", variable=self.generate_timed_images_var)
        self.timed_images_checkbox.pack(anchor="w", pady=4)

        self.slideshow_var = ctk.BooleanVar()
        self.slideshow_checkbox = ctk.CTkCheckBox(checkbox_frame, text="Use Timed Images as Slideshow", variable=self.slideshow_var)
        self.slideshow_checkbox.pack(anchor="w", pady=4)

        self.run_button = ctk.CTkButton(controls_frame, text="🚀 Run Pipeline", command=self.start_pipeline_thread, font=("Arial", 16, "bold"), fg_color="#1f6aa5")
        self.run_button.pack(pady=20, ipady=5)
        self.stop_button = ctk.CTkButton(controls_frame, text="⏹️ Stop Pipeline", command=self.stop_pipeline, font=("Arial", 16, "bold"), fg_color="#c42034", hover_color="#851622", state="disabled")
        self.stop_button.pack(pady=10, ipady=5)
        
        log_frame = ctk.CTkFrame(self.main_tab)
        log_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        log_frame.rowconfigure(len(PIPELINE_STEPS) + 1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        ctk.CTkLabel(log_frame, text="Progress Log", font=("Arial", 16, "bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        for i, step_name in enumerate(PIPELINE_STEPS):
            row = ctk.CTkFrame(log_frame, fg_color="transparent")
            row.grid(row=i + 1, column=0, sticky="ew", padx=10, pady=2)
            row.columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=step_name, anchor="w", width=250).pack(side="left", padx=5)
            status_label = ctk.CTkLabel(row, text="⬜", width=60); status_label.pack(side="right", padx=5)
            progress_bar = ctk.CTkProgressBar(row, orientation="horizontal", width=200); progress_bar.set(0); progress_bar.pack(side="right", padx=5)
            self.step_status_labels.append(status_label)
            self.progress_bars.append(progress_bar)
        self.log_box = ctk.CTkTextbox(log_frame, state="disabled")
        self.log_box.grid(row=len(PIPELINE_STEPS) + 1, column=0, padx=10, pady=(10, 10), sticky="nsew")

    def _create_settings_tab_widgets(self):
        settings_tabs = ctk.CTkTabview(self.settings_tab); settings_tabs.pack(fill="both", expand=True, padx=5, pady=5)
        api_tab = settings_tabs.add("API Keys")
        persona_tab = settings_tabs.add("Personalization")
        advanced_tab = settings_tabs.add("Advanced")
        api_tab.columnconfigure(1, weight=1)
        ctk.CTkLabel(api_tab, text="Video Engine").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.video_engine_combo = ctk.CTkComboBox(api_tab, values=["WaveSpeed AI", "Vertex AI (Veo)"], width=300); self.video_engine_combo.grid(row=0, column=1, columnspan=2, sticky="w", padx=10)
        ctk.CTkLabel(api_tab, text="Image Engine").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.image_engine_combo = ctk.CTkComboBox(api_tab, values=["Gemini API", "Vertex AI"], width=300); self.image_engine_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=10)
        ctk.CTkLabel(api_tab, text="Gemini API Key").grid(row=2, column=0, sticky="w", padx=10, pady=8)
        self.gemini_key_entry = ctk.CTkEntry(api_tab, width=400, show="*"); self.gemini_key_entry.grid(row=2, column=1, padx=10, sticky="ew")
        ctk.CTkButton(api_tab, text="Check Quota", command=self.open_quota_page).grid(row=2, column=2, padx=10, sticky="w")
        ctk.CTkLabel(api_tab, text="WaveSpeed AI Key").grid(row=3, column=0, sticky="w", padx=10, pady=8)
        self.wavespeed_key_entry = ctk.CTkEntry(api_tab, width=400, show="*"); self.wavespeed_key_entry.grid(row=3, column=1, columnspan=2, padx=10, sticky="ew")
        ctk.CTkLabel(api_tab, text="Google Cloud Project ID").grid(row=4, column=0, sticky="w", padx=10, pady=8)
        self.gcp_project_entry = ctk.CTkEntry(api_tab, width=400); self.gcp_project_entry.grid(row=4, column=1, columnspan=2, padx=10, sticky="ew")
        ctk.CTkLabel(api_tab, text="Google Cloud Location").grid(row=5, column=0, sticky="w", padx=10, pady=8)
        self.gcp_location_entry = ctk.CTkEntry(api_tab, width=400); self.gcp_location_entry.grid(row=5, column=1, columnspan=2, padx=10, sticky="ew")
        ctk.CTkLabel(api_tab, text="News API Key").grid(row=6, column=0, sticky="w", padx=10, pady=8)
        self.news_api_key_entry = ctk.CTkEntry(api_tab, width=400, show="*"); self.news_api_key_entry.grid(row=6, column=1, columnspan=2, padx=10, sticky="ew")
        voice_list = [f"{v} — {d}" for v, d in VOICE_OPTIONS.items()]
        ctk.CTkLabel(persona_tab, text="TTS Mode").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.tts_mode_combo = ctk.CTkComboBox(persona_tab, values=["Multi-Speaker", "Single-Speaker"], width=300); self.tts_mode_combo.grid(row=0, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Single-Speaker Voice").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.voice_combo = ctk.CTkComboBox(persona_tab, values=voice_list, width=400); self.voice_combo.grid(row=1, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Speaker 1 (Host Voice)").grid(row=2, column=0, sticky="w", padx=10, pady=8)
        self.speaker1_combo = ctk.CTkComboBox(persona_tab, values=voice_list, width=400); self.speaker1_combo.grid(row=2, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Speaker 2 (Guest Voice)").grid(row=3, column=0, sticky="w", padx=10, pady=8)
        self.speaker2_combo = ctk.CTkComboBox(persona_tab, values=voice_list, width=400); self.speaker2_combo.grid(row=3, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Host Name").grid(row=4, column=0, sticky="w", padx=10, pady=8)
        self.host_entry = ctk.CTkEntry(persona_tab, width=300); self.host_entry.grid(row=4, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Guest Name").grid(row=5, column=0, sticky="w", padx=10, pady=8)
        self.guest_entry = ctk.CTkEntry(persona_tab, width=300); self.guest_entry.grid(row=5, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Host Persona").grid(row=6, column=0, sticky="nw", padx=10, pady=8)
        self.host_persona_entry = ctk.CTkTextbox(persona_tab, width=400, height=80); self.host_persona_entry.grid(row=6, column=1, padx=10)
        ctk.CTkLabel(persona_tab, text="Guest Persona").grid(row=7, column=0, sticky="nw", padx=10, pady=8)
        self.guest_persona_entry = ctk.CTkTextbox(persona_tab, width=400, height=80); self.guest_persona_entry.grid(row=7, column=1, padx=10)
        ctk.CTkLabel(advanced_tab, text="Channel Name").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.channel_entry = ctk.CTkEntry(advanced_tab, width=300); self.channel_entry.grid(row=0, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Subscribe Reminder Count").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.sub_count_entry = ctk.CTkEntry(advanced_tab, width=100); self.sub_count_entry.grid(row=1, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Subscribe Message").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.sub_message_entry = ctk.CTkEntry(advanced_tab, width=400); self.sub_message_entry.grid(row=2, column=1, padx=10, sticky="w")
        self.subscribe_random_var = ctk.BooleanVar()
        self.subscribe_random_check = ctk.CTkCheckBox(advanced_tab, text="Random Placement", variable=self.subscribe_random_var); self.subscribe_random_check.grid(row=3, column=0, columnspan=2, pady=5, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Podcast Style").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.style_combo = ctk.CTkComboBox(advanced_tab, values=PODCAST_STYLES, width=300); self.style_combo.grid(row=4, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Story Arc").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.story_arc_combo = ctk.CTkComboBox(advanced_tab, values=STORY_ARCS, width=300); self.story_arc_combo.grid(row=5, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Video Prompt Style Guide").grid(row=6, column=0, sticky="nw", padx=10, pady=5)
        self.video_style_textbox = ctk.CTkTextbox(advanced_tab, width=400, height=80); self.video_style_textbox.grid(row=6, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Image Prompt Template").grid(row=7, column=0, sticky="nw", padx=10, pady=5)
        self.image_style_textbox = ctk.CTkTextbox(advanced_tab, width=400, height=80); self.image_style_textbox.grid(row=7, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Image Interval (s)").grid(row=10, column=0, sticky="w", padx=10, pady=5)
        self.image_interval_entry = ctk.CTkEntry(advanced_tab, width=100); self.image_interval_entry.grid(row=10, column=1, padx=10, sticky="w")
        self.language_enabled_var = ctk.BooleanVar()
        ctk.CTkCheckBox(advanced_tab, text="Enable Language Selection", variable=self.language_enabled_var).grid(row=11, column=0, columnspan=2, pady=5, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Podcast Language").grid(row=12, column=0, sticky="w", padx=10, pady=5)
        self.language_combo = ctk.CTkComboBox(advanced_tab, values=["English", "Spanish", "French", "German", "Urdu"], width=300); self.language_combo.grid(row=12, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Desired Script Length").grid(row=13, column=0, sticky="w", padx=10, pady=5)
        self.script_length_combo = ctk.CTkComboBox(advanced_tab, values=SCRIPT_LENGTHS, width=300); self.script_length_combo.grid(row=13, column=1, padx=10, sticky="w")
        ctk.CTkLabel(advanced_tab, text="Video Aspect Ratio").grid(row=14, column=0, sticky="w", padx=10, pady=5)
        self.aspect_ratio_combo = ctk.CTkComboBox(advanced_tab, values=["16:9 (Horizontal)", "9:16 (Vertical)"], width=300); self.aspect_ratio_combo.grid(row=14, column=1, padx=10, sticky="w")
        ctk.CTkButton(self.settings_tab, text="💾 Save Settings", command=self.save_settings_from_gui).pack(pady=20)
    
    def _create_publish_tab_widgets(self):
        self.publish_tab.columnconfigure(1, weight=1)
        metadata_frame = ctk.CTkFrame(self.publish_tab, fg_color="transparent")
        metadata_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10,5))
        ctk.CTkLabel(metadata_frame, text="Video Metadata", font=("Arial", 16, "bold")).pack(side="left")
        ctk.CTkButton(metadata_frame, text="✨ Generate SEO", command=lambda: self.generate_seo_only()).pack(side="left", padx=10)
        ctk.CTkLabel(self.publish_tab, text="Video Title:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.video_title_entry = ctk.CTkEntry(self.publish_tab, placeholder_text="Enter video title"); self.video_title_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(self.publish_tab, text="Description:").grid(row=2, column=0, sticky="nw", padx=10, pady=5)
        self.video_desc_entry = ctk.CTkTextbox(self.publish_tab, height=120); self.video_desc_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(self.publish_tab, text="Tags:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.video_tags_entry = ctk.CTkEntry(self.publish_tab, placeholder_text="tag1, tag2, tag3"); self.video_tags_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(self.publish_tab, text="--- Upload ---", font=("Arial", 16, "bold")).grid(row=4, column=0, columnspan=3, pady=(20,5), padx=10, sticky="w")
        
        ctk.CTkLabel(self.publish_tab, text="Video File:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.video_path_entry = ctk.CTkEntry(self.publish_tab, placeholder_text="Leave blank to use generated video"); self.video_path_entry.grid(row=5, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(self.publish_tab, text="Browse...", command=lambda: self.browse_file(self.video_path_entry)).grid(row=5, column=2, sticky="w", padx=10, pady=5)
        
        ctk.CTkLabel(self.publish_tab, text="Thumbnail:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        self.thumbnail_path_entry = ctk.CTkEntry(self.publish_tab, placeholder_text="Leave blank to use generated thumbnail"); self.thumbnail_path_entry.grid(row=6, column=1, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(self.publish_tab, text="Browse...", command=lambda: self.browse_file(self.thumbnail_path_entry)).grid(row=6, column=2, sticky="w", padx=10, pady=5)
        
        self.upload_status_label = ctk.CTkLabel(self.publish_tab, text="", font=("Arial", 12))
        self.upload_status_label.grid(row=7, column=0, columnspan=3, pady=(10,0), padx=20, sticky="ew")
        self.upload_progress_bar = ctk.CTkProgressBar(self.publish_tab, orientation="horizontal")
        self.upload_progress_bar.set(0)
        self.upload_progress_bar.grid(row=8, column=0, columnspan=3, pady=(5,10), padx=20, sticky="ew")

        upload_frame = ctk.CTkFrame(self.publish_tab, fg_color="transparent")
        upload_frame.grid(row=9, column=0, columnspan=3, pady=10)
        self.youtube_upload_button = ctk.CTkButton(upload_frame, text="Upload to YouTube", command=self.start_youtube_upload_thread, fg_color="#FF0000", hover_color="#8c0303")
        self.youtube_upload_button.pack(side="left", padx=10)
        self.facebook_upload_button = ctk.CTkButton(upload_frame, text="Upload to Facebook", command=self.start_facebook_upload_thread, fg_color="#1877F2", hover_color="#1456b3")
        self.facebook_upload_button.pack(side="left", padx=10)
        
        ctk.CTkLabel(self.publish_tab, text="--- API Credentials for Publishing ---", font=("Arial", 12, "bold")).grid(row=10, column=0, columnspan=3, pady=(20,5), padx=10, sticky="w")
        ctk.CTkLabel(self.publish_tab, text="Facebook Access Token:").grid(row=11, column=0, sticky="w", padx=10, pady=5)
        self.facebook_token_entry = ctk.CTkEntry(self.publish_tab, width=400, show="*", placeholder_text="User/Page access token"); self.facebook_token_entry.grid(row=11, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
    
    def _create_about_tab_widgets(self):
        ctk.CTkLabel(self.about_tab, text="Nullpk Ai Content Studio", font=("Arial", 20, "bold")).pack(pady=(40, 10))
        ctk.CTkLabel(self.about_tab, text="Version 3.2 (Slideshow Mode)", font=("Arial", 12)).pack(pady=5)
        ctk.CTkLabel(self.about_tab, text="Author: Naqash Afzal", font=("Arial", 12)).pack(pady=5)
        ctk.CTkLabel(self.about_tab, text="This tool automates the creation of YouTube videos using AI.", wraplength=500).pack(pady=20)
        ctk.CTkButton(self.about_tab, text="Donate Now", command=lambda: webbrowser.open_new("https://nullpk.com/donate")).pack(pady=10)
    
    def open_quota_page(self):
        project_id = self.gcp_project_entry.get().strip()
        if project_id:
            url = f"https://console.cloud.google.com/iam-admin/quotas?project={project_id}"
            logging.info(f"Opening quota page for project: {project_id}")
            webbrowser.open_new(url)
        else:
            url = "https://console.cloud.google.com/apis/dashboard"
            logging.warning("GCP Project ID not set. Opening generic API dashboard.")
            messagebox.showinfo("GCP Project ID Missing", "Please enter your Google Cloud Project ID for a direct link.")
            webbrowser.open_new(url)

    def _extract_voice_name(self, val): return val.split(" — ")[0] if " — " in val else val
    
    def update_features_based_on_style(self, selected_style):
        is_podcast = selected_style == "Podcast"
        state = "normal" if is_podcast else "disabled"
        self.fact_check_checkbox.configure(state=state)
        self.style_combo.configure(state=state)
        self.story_arc_combo.configure(state=state)
        self.tts_mode_combo.configure(state=state)
        self.speaker1_combo.configure(state=state)
        self.speaker2_combo.configure(state=state)
        self.host_entry.configure(state=state)
        self.guest_entry.configure(state=state)
        self.host_persona_entry.configure(state=state)
        self.guest_persona_entry.configure(state=state)
    
    def update_status_callback(self, step_index, status, progress):
        if 0 <= step_index < len(self.progress_bars):
            self.progress_bars[step_index].set(progress)
            self.step_status_labels[step_index].configure(text=status)
    
    def start_pipeline_thread(self):
        topic = self.topic_entry.get().strip()
        if not topic: messagebox.showerror("Error", "Please enter a topic."); return
        
        self.update_config_from_all_gui()

        if not self.config.get("GEMINI_API_KEY"): messagebox.showerror("API Key Missing", "Please enter your Gemini API key in Settings."); return
        
        # --- UPDATED CHECK: Ensure Project ID is set, since Vertex is now required for images ---
        if self.config.get("GENERATE_THUMBNAIL") or self.config.get("GENERATE_TIMED_IMAGES"):
             if not self.config.get("GCP_PROJECT_ID") or not self.config.get("GCP_LOCATION"):
                messagebox.showerror("GCP Project Missing", "Image generation now requires Vertex AI. Please enter your GCP Project ID and Location in Settings to proceed.")
                return
        
        self.run_button.configure(state="disabled"); self.stop_button.configure(state="normal")
        self.stop_event.clear()
        for i in range(len(PIPELINE_STEPS)): self.update_status_callback(i, "⬜", 0)
        
        on_finish_callback = lambda success, error: self.after(0, self.on_pipeline_finished, success, topic, error)
        
        pipeline_instance = Pipeline(
            self.config, 
            self.stop_event, 
            self.update_status_callback, 
            self.update_seo_callback,
            on_finish_callback,
            self.update_timestamps_callback
        )
        threading.Thread(target=pipeline_instance.run, args=(topic, self.start_step_combo.get()), daemon=True).start()
    
    def stop_pipeline(self):
        logging.info("🛑 Stop signal received. Finishing current step...")
        self.stop_event.set()
        self.run_button.configure(state="normal"); self.stop_button.configure(state="disabled")

    def on_pipeline_finished(self, success: bool, topic: str, error: Exception):
        """Handles UI updates when the pipeline completes, fails, or is stopped."""
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.update_history_tab()
        if success:
            messagebox.showinfo("Pipeline Complete", f"Successfully generated content for topic:\n'{topic}'")
        else:
            if not self.stop_event.is_set():
                error_message = f"The pipeline stopped due to an error:\n\n{str(error)}\n\nPlease check the logs for details."
                messagebox.showerror("Pipeline Failed", error_message)
    
    def update_seo_callback(self, metadata):
        """Safely updates the publish tab UI with title, description, and tags. THIS RUNS FIRST."""
        def update_ui():
            self.video_title_entry.delete(0, "end")
            self.video_title_entry.insert(0, metadata.get("title", ""))
            self.video_desc_entry.delete("1.0", "end")
            self.video_desc_entry.insert("1.0", metadata.get("description", ""))
            self.video_tags_entry.delete(0, "end")
            self.video_tags_entry.insert(0, metadata.get("tags", ""))
            logging.info("📝 SEO metadata has been pre-filled in the Publish tab.")
        self.after(0, update_ui)

    def update_timestamps_callback(self, timestamps_text: str):
        """Safely appends the generated timestamps to the description box. THIS RUNS SECOND."""
        if not timestamps_text:
            return
        def update_ui():
            self.video_desc_entry.insert("end", timestamps_text) 
            logging.info("⏰ Timestamps have been appended to the description.")
        self.after(0, update_ui)

    def update_config_from_all_gui(self):
        """
        Synchronizes the in-memory self.config dictionary with the current state
        of ALL relevant UI elements.
        """
        logging.info("Syncing all current GUI settings to in-memory config for pipeline run...")
        
        self.config["FACT_CHECK_ENABLED"] = self.fact_check_var.get()
        self.config["GENERATE_METADATA"] = self.metadata_var.get()
        self.config["GENERATE_TIMESTAMPS"] = self.timestamps_var.get()
        self.config["CAPTION_ENABLED"] = self.caption_var.get()
        self.config["ADD_MUSIC"] = self.add_music_var.get()
        self.config["GENERATE_SNIPPETS"] = self.generate_snippets_var.get()
        self.config["GENERATE_THUMBNAIL"] = self.generate_thumbnail_var.get()
        self.config["GENERATE_TIMED_IMAGES"] = self.generate_timed_images_var.get()
        self.config["TIMED_IMAGES_AS_SLIDESHOW"] = self.slideshow_var.get()
        self.config["CONTENT_STYLE"] = self.content_style_combo.get()
        
        self.config["GEMINI_API_KEY"] = self.gemini_key_entry.get().strip()
        self.config["WAVESPEED_AI_KEY"] = self.wavespeed_key_entry.get().strip()
        self.config["GCP_PROJECT_ID"] = self.gcp_project_entry.get().strip()
        self.config["GCP_LOCATION"] = self.gcp_location_entry.get().strip()
        self.config["NEWS_API_KEY"] = self.news_api_key_entry.get().strip()
        self.config["VIDEO_ENGINE"] = self.video_engine_combo.get()
        self.config["IMAGE_ENGINE"] = self.image_engine_combo.get()

        self.config["TTS_MODE"] = self.tts_mode_combo.get()
        self.config["VOICE_NAME"] = self._extract_voice_name(self.voice_combo.get())
        self.config["SPEAKER1"] = self._extract_voice_name(self.speaker1_combo.get())
        self.config["SPEAKER2"] = self._extract_voice_name(self.speaker2_combo.get())
        self.config["HOST_NAME"] = self.host_entry.get().strip() or "Alex"
        self.config["GUEST_NAME"] = self.guest_entry.get().strip() or "Maya"
        self.config["HOST_PERSONA"] = self.host_persona_entry.get("1.0", "end-1c").strip()
        self.config["GUEST_PERSONA"] = self.guest_persona_entry.get("1.0", "end-1c").strip()

        self.config["PODCAST_STYLE"] = self.style_combo.get()
        self.config["STORY_ARC"] = self.story_arc_combo.get()
        self.config["SCRIPT_LENGTH"] = self.script_length_combo.get()
        self.config["VIDEO_ASPECT_RATIO"] = self.aspect_ratio_combo.get()
        self.config["LANGUAGE_ENABLED"] = self.language_enabled_var.get()
        self.config["PODCAST_LANGUAGE"] = self.language_combo.get()
        try: 
            self.config["IMAGE_GENERATION_INTERVAL"] = int(self.image_interval_entry.get())
        except (ValueError, TypeError): 
            self.config["IMAGE_GENERATION_INTERVAL"] = 10
        
        self.config["VIDEO_PROMPT_BASE_STYLE"] = self.video_style_textbox.get("1.0", "end-1c").strip()
        self.config["IMAGE_PROMPT_STYLE"] = self.image_style_textbox.get("1.0", "end-1c").strip()
        
        self.config["CHANNEL_NAME"] = self.channel_entry.get().strip() or "My AI Channel"
        try: 
            self.config["SUBSCRIBE_COUNT"] = int(self.sub_count_entry.get())
        except ValueError: 
            self.config["SUBSCRIBE_COUNT"] = 3
        self.config["SUBSCRIBE_MESSAGE"] = self.sub_message_entry.get().strip()
        self.config["SUBSCRIBE_RANDOM"] = self.subscribe_random_var.get()
    
    def save_settings_from_gui(self):
        """
        Updates the in-memory config from the GUI and then saves it to config.json.
        """
        self.update_config_from_all_gui() 
        self.config["FACEBOOK_ACCESS_TOKEN"] = self.facebook_token_entry.get().strip()
        save_config(self.config)
        messagebox.showinfo("Success", "Settings have been saved successfully.")
    
    def load_settings_into_gui(self):
        cfg = self.config
        self.gemini_key_entry.insert(0, cfg.get("GEMINI_API_KEY", ""))
        self.wavespeed_key_entry.insert(0, cfg.get("WAVESPEED_AI_KEY", ""))
        self.gcp_project_entry.insert(0, cfg.get("GCP_PROJECT_ID", ""))
        self.gcp_location_entry.insert(0, cfg.get("GCP_LOCATION", "us-central1"))
        self.news_api_key_entry.insert(0, cfg.get("NEWS_API_KEY", ""))
        self.facebook_token_entry.insert(0, cfg.get("FACEBOOK_ACCESS_TOKEN", ""))
        self.speaker1_combo.set(f"{cfg['SPEAKER1']} — {VOICE_OPTIONS.get(cfg['SPEAKER1'], '')}")
        self.speaker2_combo.set(f"{cfg['SPEAKER2']} — {VOICE_OPTIONS.get(cfg['SPEAKER2'], '')}")
        self.voice_combo.set(f"{cfg.get('VOICE_NAME', 'Kore')} — {VOICE_OPTIONS.get(cfg.get('VOICE_NAME', 'Kore'), '')}")
        self.tts_mode_combo.set(cfg.get("TTS_MODE", "Multi-Speaker"))
        self.host_entry.insert(0, cfg.get("HOST_NAME", "Alex"))
        self.guest_entry.insert(0, cfg.get("GUEST_NAME", "Maya"))
        self.host_persona_entry.insert("1.0", cfg.get("HOST_PERSONA", ""))
        self.guest_persona_entry.insert("1.0", cfg.get("GUEST_PERSONA", ""))
        self.channel_entry.insert(0, cfg.get("CHANNEL_NAME", "My AI Channel"))
        self.sub_count_entry.insert(0, str(cfg.get("SUBSCRIBE_COUNT", 3)))
        self.sub_message_entry.insert(0, cfg.get("SUBSCRIBE_MESSAGE", ""))
        self.subscribe_random_var.set(cfg.get("SUBSCRIBE_RANDOM", True))
        self.style_combo.set(cfg.get("PODCAST_STYLE", "Informative News"))
        self.story_arc_combo.set(cfg.get("STORY_ARC", "None"))
        self.video_style_textbox.insert("1.0", cfg.get("VIDEO_PROMPT_BASE_STYLE", "An animated and cinematic video. High-quality, 24fps."))
        self.image_style_textbox.insert("1.0", cfg.get("IMAGE_PROMPT_STYLE", "A cinematic, photorealistic image representing the podcast topic: {topic}"))
        self.image_interval_entry.insert(0, str(cfg.get("IMAGE_GENERATION_INTERVAL", 10)))
        self.language_enabled_var.set(cfg.get("LANGUAGE_ENABLED", False))
        self.language_combo.set(cfg.get("PODCAST_LANGUAGE", "English"))
        self.video_engine_combo.set(cfg.get("VIDEO_ENGINE", "WaveSpeed AI"))
        self.image_engine_combo.set(cfg.get("IMAGE_ENGINE", "Gemini API"))
        self.content_style_combo.set(cfg.get("CONTENT_STYLE", "Podcast"))
        self.script_length_combo.set(cfg.get("SCRIPT_LENGTH", "Medium (~5 minutes)"))
        self.aspect_ratio_combo.set(cfg.get("VIDEO_ASPECT_RATIO", "16:9 (Horizontal)"))
        self.fact_check_var.set(cfg.get("FACT_CHECK_ENABLED", False))
        self.metadata_var.set(cfg.get("GENERATE_METADATA", False))
        self.timestamps_var.set(cfg.get("GENERATE_TIMESTAMPS", True))
        self.caption_var.set(cfg.get("CAPTION_ENABLED", False))
        self.add_music_var.set(cfg.get("ADD_MUSIC", False))
        self.generate_snippets_var.set(cfg.get("GENERATE_SNIPPETS", False))
        self.generate_thumbnail_var.set(cfg.get("GENERATE_THUMBNAIL", False))
        self.generate_timed_images_var.set(cfg.get("GENERATE_TIMED_IMAGES", False))
        self.slideshow_var.set(cfg.get("TIMED_IMAGES_AS_SLIDESHOW", False))
        self.video_title_entry.insert(0, cfg.get("VIDEO_TITLE", ""))
        self.video_desc_entry.insert("1.0", cfg.get("VIDEO_DESCRIPTION", ""))
        self.video_tags_entry.insert(0, cfg.get("VIDEO_TAGS", ""))
        self.update_features_based_on_style(self.content_style_combo.get())
    
    def get_history_items(self):
        return [item for item in os.listdir('.') if os.path.isdir(item) and not item.startswith('.')]
    
    def delete_history_item(self, item_name):
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete '{item_name}'?"):
            try:
                shutil.rmtree(item_name); messagebox.showinfo("Success", f"Deleted '{item_name}'.")
                self.update_history_tab()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete '{item_name}': {e}")
    
    def update_history_tab(self):
        for widget in self.history_tab.winfo_children(): widget.destroy()
        ctk.CTkLabel(self.history_tab, text="Generated Content History", font=("Arial", 20, "bold")).pack(pady=(20, 10))
        items = self.get_history_items()
        if not items:
            ctk.CTkLabel(self.history_tab, text="No history found.").pack(pady=20); return
        for item in items:
            row = ctk.CTkFrame(self.history_tab); row.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(row, text=item, anchor="w", font=("Arial", 12)).pack(side="left", padx=10, expand=True, fill="x")
            ctk.CTkButton(row, text="Delete", command=lambda i=item: self.delete_history_item(i), fg_color="#c42034", hover_color="#851622").pack(side="right", padx=10)
    
    def generate_seo_only(self):
        topic = self.topic_entry.get().strip()
        if not topic: messagebox.showerror("Error", "Please enter a topic first."); return
        logging.info("📄 Generating SEO metadata only...")
        try:
            self.config["GEMINI_API_KEY"] = self.gemini_key_entry.get().strip()
            self.config["NEWS_API_KEY"] = self.news_api_key_entry.get().strip()

            google_client = GoogleClient(self.config['GEMINI_API_KEY'], self.config.get("GCP_PROJECT_ID"), self.config.get("GCP_LOCATION"))
            news_client = NewsApiClient(self.config.get("NEWS_API_KEY"))
            
            research = google_client.deep_research(topic, self.config.get("PODCAST_LANGUAGE", "English"), news_client)
            metadata = google_client.generate_seo_metadata(topic, context=research) 
            
            self.video_title_entry.delete(0, ctk.END); self.video_title_entry.insert(0, metadata.get("title", ""))
            self.video_desc_entry.delete("1.0", ctk.END); self.video_desc_entry.insert("1.0", metadata.get("description", ""))
            self.video_tags_entry.delete(0, ctk.END); self.video_tags_entry.insert(0, metadata.get("tags", ""))
            logging.info("✅ SEO metadata generated and pre-filled.")
            messagebox.showinfo("Success", "SEO Title and Description have been generated!")
        except Exception as e:
            logging.error(f"❌ SEO Generation Error: {e}", exc_info=True); messagebox.showerror("Error", f"Failed to generate SEO: {e}")
    
    def browse_file(self, entry_widget):
        filename = filedialog.askopenfilename()
        if filename:
            entry_widget.delete(0, ctk.END); entry_widget.insert(0, filename)
    
    def youtube_auth(self):
        CLIENT_SECRETS_FILE = "client_secrets.json"
        if not os.path.exists(CLIENT_SECRETS_FILE): 
            messagebox.showerror("Authentication Error", f"{CLIENT_SECRETS_FILE} not found. Please place it in the application's root directory.")
            return None
        credentials = None; pickle_file = Path("token.pickle")
        if pickle_file.exists():
            with open(pickle_file, "rb") as token: credentials = pickle.load(token)
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token: 
                try:
                    credentials.refresh(Request())
                except Exception as e:
                    logging.error(f"Failed to refresh token: {e}")
                    pickle_file.unlink()
                    credentials = None
            if not credentials:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=["https://www.googleapis.com/auth/youtube.upload"], redirect_uri='http://localhost:8080/')
                credentials = flow.run_local_server(port=8080)
            with open(pickle_file, "wb") as f: pickle.dump(credentials, f)
        return credentials

    def update_upload_progress(self, status_text, progress_value):
        """Safely updates the upload progress bar and status label from a background thread."""
        def update_ui():
            self.upload_status_label.configure(text=status_text)
            self.upload_progress_bar.set(progress_value)
        self.after(0, update_ui)

    def start_youtube_upload_thread(self):
        """Starts the YouTube upload process in a separate thread to avoid freezing the GUI."""
        safe_topic = re.sub(r'[\\/:*?"<>|]', '', self.topic_entry.get().strip())
        
        video_path = self.video_path_entry.get().strip() or os.path.join(safe_topic, "final_podcast_video.mp4")
        thumbnail_path = self.thumbnail_path_entry.get().strip() or os.path.join(safe_topic, "generated_image.png")

        if not os.path.exists(video_path):
            messagebox.showerror("File Not Found", f"Video file not found at '{video_path}'. Please generate it first or select a valid file.")
            return

        self.youtube_upload_button.configure(state="disabled")
        self.update_upload_progress("Starting YouTube upload...", 0)

        upload_thread = threading.Thread(
            target=self.upload_youtube_logic,
            args=(video_path, thumbnail_path, self.video_title_entry.get(), self.video_desc_entry.get("1.0", ctk.END), self.video_tags_entry.get().split(",")),
            daemon=True
        )
        upload_thread.start()
    
    def upload_youtube_logic(self, video_path, thumbnail_path, title, description, tags):
        """The core logic for uploading to YouTube, designed to be run in a thread."""
        try:
            self.update_upload_progress("Authenticating with YouTube...", 0.05)
            credentials = self.youtube_auth()
            if not credentials: 
                logging.error("❌ YouTube authentication failed."); return

            youtube = build("youtube", "v3", credentials=credentials)
            
            request_body = {
                "snippet": {"title": title, "description": description, "tags": [tag.strip() for tag in tags], "categoryId": "22"},
                "status": {"privacyStatus": "public"}
            }

            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(part=",".join(request_body.keys()), body=request_body, media_body=media)
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    self.update_upload_progress(f"Uploading video... {progress}%", float(progress / 100))
            
            video_id = response.get('id')
            logging.info(f"✅ Video uploaded! Video ID: {video_id}")
            
            if os.path.exists(thumbnail_path):
                youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
                logging.info("✅ Thumbnail uploaded successfully.")
            
            messagebox.showinfo("Upload Complete", f"Successfully uploaded to YouTube!\nURL: https://youtube.com/watch?v={video_id}")

        except Exception as e:
            logging.error(f"❌ YouTube upload failed: {e}", exc_info=True)
            messagebox.showerror("Upload Error", f"An error occurred during upload: {e}")
        finally:
            self.after(0, self.youtube_upload_button.configure, {"state": "normal"})
            self.after(0, self.update_upload_progress, ("", 0.0))

    def start_facebook_upload_thread(self):
        """Starts the Facebook upload process in a separate thread."""
        safe_topic = re.sub(r'[\\/:*?"<>|]', '', self.topic_entry.get().strip())
        video_path = self.video_path_entry.get().strip() or os.path.join(safe_topic, "final_podcast_video.mp4")
        
        if not os.path.exists(video_path): 
            messagebox.showerror("Error", f"Video not found at '{video_path}'."); return
            
        access_token = self.facebook_token_entry.get().strip()
        if not access_token: 
            messagebox.showerror("Error", "Facebook Access Token not found in Settings."); return

        self.facebook_upload_button.configure(state="disabled")
        self.update_upload_progress("Starting Facebook upload...", 0)

        upload_thread = threading.Thread(
            target=self.upload_facebook_logic,
            args=(video_path, access_token, self.video_title_entry.get(), self.video_desc_entry.get("1.0", ctk.END)),
            daemon=True
        )
        upload_thread.start()

    def upload_facebook_logic(self, video_path, access_token, title, description):
        """The core logic for uploading to Facebook, designed to be run in a thread."""
        try:
            logging.info("▶️ Uploading to Facebook...")
            self.update_upload_progress("Initializing Facebook upload...", 0.1)
            with open(video_path, 'rb') as f: video_data = f.read()
            
            init_url = "https://graph-video.facebook.com/v20.0/me/videos"
            init_params = {"access_token": access_token, "upload_phase": "start", "file_size": os.path.getsize(video_path)}
            init_response = requests.post(init_url, params=init_params).json()
            if "error" in init_response: raise RuntimeError(f"API Error: {init_response['error']['message']}")
            
            upload_session_id = init_response["upload_session_id"]
            upload_url = f"https://graph-video.facebook.com/v20.0/{upload_session_id}"
            upload_headers = {"Authorization": f"OAuth {access_token}", "file_offset": "0"}
            
            self.update_upload_progress("Uploading video data to Facebook...", 0.5)
            requests.post(upload_url, headers=upload_headers, data=video_data).json()

            self.update_upload_progress("Finishing Facebook publish...", 0.9)
            finish_url = "https://graph-video.facebook.com/v20.0/me/videos"
            finish_params = {
                "access_token": access_token, "upload_phase": "finish", "upload_session_id": upload_session_id,
                "title": title, "description": description
            }
            requests.post(finish_url, params=finish_params).json()
            
            self.update_upload_progress("Upload complete!", 1.0)
            messagebox.showinfo("Upload Complete", "Successfully published to Facebook!")
            
        except Exception as e:
            logging.error(f"❌ Facebook upload failed: {e}", exc_info=True)
            messagebox.showerror("Upload Error", f"An error occurred: {e}")
        finally:
            self.after(0, self.facebook_upload_button.configure, {"state": "normal"})
            self.after(0, self.update_upload_progress, ("", 0.0))

if __name__ == "__main__":
    app = App()
    app.mainloop()