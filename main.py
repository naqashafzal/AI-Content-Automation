import sys, os, re, unicodedata
import threading
import customtkinter as ctk
import subprocess, requests, time, base64, wave, json
from tkinter import messagebox, filedialog
import google.generativeai as genai
from google.generativeai import types
import webbrowser
import whisper
import srt
import pysubs2
import shutil
import pysubs2
import os
import logging
import numpy as np
import random
from functools import wraps
from google.api_core import exceptions as google_exceptions
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

# ============================
# FORCE UTF-8
# ============================
os.environ["PYTHONIOENCODING"] = "utf-8"
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

# ============================
# LOGGING SETUP
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout)
])

# ============================
# THREADING STOP EVENT
# ============================
stop_event = threading.Event()


# ============================
# CONFIG
# ============================
CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE,"r",encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            logging.error("Config file is corrupted. Loading default config.")
            config_data = {}
    else:
        config_data = {}

    default_config = {
        "GEMINI_API_KEY":"",
        "WAVESPEED_AI_KEY":"",
        "SPEAKER1":"Kore",
        "SPEAKER2":"Puck",
        "HOST_NAME":"Alex",
        "HUMANOID_ENABLED": False,
        "HUMANOID_PROBABILITY": 0.1,
        "GUEST_NAME":"Maya",
        "HOST_PERSONA":"A friendly podcast host who loves technology.",
        "GUEST_PERSONA":"An expert on the topic with a calm and informative style.",
        "CHANNEL_NAME":"My AI Channel",
        "SUBSCRIBE_COUNT":3,
        "SUBSCRIBE_MESSAGE":"Don’t forget to subscribe to {channel} for more awesome content!",
        "SUBSCRIBE_RANDOM":True,
        "PODCAST_STYLE":"Informative News",
        "VIDEO_PROMPT_STYLE": "An animated and cinematic video about the podcast topic: {topic}. High-quality, 24fps.",
        "FACT_CHECK_ENABLED":False,
        "CAPTION_ENABLED":False,
        "GENERATE_METADATA": False,
        "YOUTUBE_CLIENT_ID": "",
        "YOUTUBE_CLIENT_SECRET": "",
        "FACEBOOK_ACCESS_TOKEN": "",
        "VIDEO_TITLE": "",
        "VIDEO_DESCRIPTION": "",
        "VIDEO_TAGS": ""
    }
    # Update config with any missing default keys
    for key, value in default_config.items():
        if key not in config_data:
            config_data[key] = value

    return config_data

def save_config(cfg):
    with open(CONFIG_FILE,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)


config = load_config()
genai_client=None

# ============================
# CONSTANTS
# ============================
GEMINI_TEXT_MODEL  = "gemini-2.5-flash"
GEMINI_TTS_MODEL   = "gemini-2.5-flash-preview-tts"
WAVESPEED_T2V_API_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/t2v-480p-ultra-fast"
WAVESPEED_POLL_URL = "https://api.wavespeed.ai/api/v3/predictions/{}/result"

# Define global file paths, they will be updated dynamically
SUMMARY_FILE="summary.txt"
SCRIPT_FILE="podcast_script.txt"
AUDIO_FILE="podcast.wav"
BACKGROUND_VIDEO_RAW="background.mp4"
FINAL_VIDEO="final_podcast_video.mp4"
CAPTIONS_FILE = "captions.ass" # Changed to .ass format

VOICE_OPTIONS = {
    "Achernar": "Clear, mid-range, enthusiastic & approachable",
    "Achird": "Youthful, breathy, inquisitive tone",
    "Algenib": "Warm, confident, friendly authority",
    "Alnilam": "Energetic, low pitch, promotional tone",
    "Aoede": "Clear, conversational, thoughtful",
    "Autonoe": "Mature, resonant, calm and wise",
    "Callirrhoe": "Confident, professional, energetic",
    "Despina": "Warm, inviting, trustworthy",
    "Erinome": "Professional, articulate, thoughtful",
    "Gacrux": "Authoritative yet approachable",
    "Iapetus": "Casual, relatable, ‘everyman’ tone",
    "Kore": "Energetic, youthful, clear & bright",
    "Laomedeia": "Inquisitive, intelligent & engaging",
    "Leda": "Composed, professional, calm",
    "Orus": "Resonant, authoritative, thoughtful",
    "Puck": "Confident, informal, trustworthy",
    "Pulcherrima": "Bright, enthusiastic, youthful",
    "Rasalgethi": "Conversational, thoughtful, quirky",
    "Sadachbia": "Deep, textured, confident, cool",
    "Sadaltager": "Friendly, enthusiastic, professional",
    "Schedar": "Down-to-earth, approachable",
    "Sulafat": "Warm, persuasive, articulate",
    "Umbriel": "Authoritative, clear, engaging",
    "Vindemiatrix": "Calm, mature, smooth, reassuring",
    "Zephyr": "Energetic, bright, perky & enthusiastic",
    "Zubenelgenubi": "Deep, resonant, powerful authority"
}
PODCAST_STYLES = ["Informative News","Comedy / Entertaining","Educational / Explainer","Motivational / Inspiring","Casual Conversational","Serious Debate"]
VIDEO_PROMPT_STYLES = ["Cinematic Animation", "Documentary Style", "Abstract Visuals", "Futuristic Tech", "Vintage Film Look", "Simple Whiteboard"]
steps=["Deep Research", "Generate SEO Metadata", "Podcast Script","Fact Check", "Revise Script", "Audio (TTS)","Caption Generation", "Video Generation", "Final Video Creation"]
step_rows=[]
progress_bars=[]

# ============================
# HELPERS
# ============================
def log(msg:str):
    safe=msg.encode("utf-8",errors="replace").decode("utf-8")
    log_box.insert("end",safe+"\n")
    log_box.see("end")
    app.update_idletasks()
    logging.info(safe)

def get_audio_length_fast(path):
    import contextlib
    with contextlib.closing(wave.open(path,"rb")) as wf:
        return wf.getnframes()/float(wf.getframerate())

def set_step_status(i:int,status:str, progress=0):
    if i < len(step_rows):
        step_rows[i][1].configure(text=status)
        progress_bars[i].set(progress)
        app.update_idletasks()

def reset_steps(): [set_step_status(i,"⬜",0) for i in range(len(steps))]

def sanitize_for_tts(text:str)->str:
    # A more robust regex to handle various Unicode ranges, including common emojis.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002700-\U000027BF"  # Dingbats
        "]+", re.UNICODE)

    text = emoji_pattern.sub(r'', text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[ \t\r\f\v]+',' ',text).replace('\xa0',' ').strip()
    return text.replace('“','"').replace('”','"').replace('’',"'").replace('—','-').replace('–','-')

def add_fillers_to_script(script: str, probability: float) -> str:
    """Adds conversational fillers and strategic pauses to the script's dialogue."""
    # Fillers that the TTS engine will speak
    spoken_fillers = [
        'umm', 'ah', 'you know', 'I mean', 'hmm', 'like', 'so', 'actually',
        'basically', 'right', 'well', 'anyway', 'sort of', 'kind of', 'I guess'
    ]
    # Non-verbal cues that should remain in parentheses
    non_verbal_fillers = ['(laugh)', '(cough)', '(gasp)', '(sigh)', '(chuckle)']
    # Punctuation fillers for strategic pauses
    punctuation_fillers = [',', '...', '-', ';']

    all_fillers = spoken_fillers + non_verbal_fillers + punctuation_fillers

    lines = script.split('\n')
    new_lines = []

    for line in lines:
        stripped_line = line.strip()
        is_speaker_line = ':**' in stripped_line and len(stripped_line.split(':**', 1)[0]) < 30

        if is_speaker_line and random.random() < probability:
            filler = random.choice(all_fillers)

            try:
                speaker_part, dialogue_part = stripped_line.split(':**', 1)
                speaker_part += ':**'
            except ValueError:
                new_lines.append(line)
                continue

            words = dialogue_part.strip().split()
            if len(words) > 1:
                insert_pos = random.randint(1, len(words) - 1)

                if filler in punctuation_fillers:
                    words[insert_pos - 1] += filler
                    new_line = speaker_part + ' ' + ' '.join(words)
                else:
                    words.insert(insert_pos, filler)
                    new_line = speaker_part + ' ' + ' '.join(words)

                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return '\n'.join(new_lines)

def extract_voice_name(val): return val.split(" — ")[0] if " — " in val else val

def get_video_prompt(style: str, topic: str) -> str:
    """Generates a detailed prompt for WaveSpeed based on a style and topic."""
    prompts = {
        "Cinematic Animation": f"An animated and cinematic video about the podcast topic: {topic}. High-quality, 24fps, dramatic lighting.",
        "Documentary Style": f"A documentary style video about {topic}, featuring realistic visuals and smooth transitions. High-quality, 24fps.",
        "Abstract Visuals": f"An abstract and artistic video representing the concept of {topic}. Fluid motion, vibrant colors, non-representational visuals.",
        "Futuristic Tech": f"A futuristic, high-tech video about {topic}. Holographic interfaces, neon lights, sleek design, cyberpunk aesthetic.",
        "Vintage Film Look": f"A video about {topic} with a vintage film look. 8mm film grain, light leaks, warm color grading, retro aesthetic.",
        "Simple Whiteboard": f"A simple whiteboard explainer animation about {topic}. Clean, minimalist, with clear drawings and text."
    }
    return prompts.get(style, prompts["Cinematic Animation"]) # Default to Cinematic

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def generate_captions(audio_file, captions_file):
    """Transcribes an audio file and generates an ASS caption file for dynamic styling.

    Improvements:
      - Tries word-level timestamps if available.
      - Falls back to segment-level timestamps otherwise.
      - Groups words into readable chunks (no one-word-per-line issue).
      - Splits long lines for readability.
      - Adds detailed logging for debugging.
    """
    log("📝 Transcribing audio for captions (this may take a moment)...")
    try:
        # Load Whisper model
        model = whisper.load_model("base")

        # Try word-level timestamps if supported
        log("🔍 Starting Whisper transcription (word-level if supported)...")
        try:
            result = model.transcribe(audio_file, word_timestamps=True, language="en")
            used_word_timestamps = True
        except TypeError:
            log("⚠️ Whisper version does not support word-level timestamps. Using segment timestamps.")
            result = model.transcribe(audio_file, language="en")
            used_word_timestamps = False

        subs = pysubs2.SSAFile()

        # Caption style
        style = pysubs2.SSAStyle(
            fontname="Arial Black",
            fontsize=36,
            bold=True,
            primarycolor=pysubs2.Color(r=255, g=255, b=0, a=0),  # Yellow
            outlinecolor=pysubs2.Color(r=0, g=0, b=0, a=0),      # Black outline
            outline=2,
            shadow=1,
            alignment=2  # Bottom center
        )
        subs.styles["Default"] = style

        event_count = 0
        segments = result.get("segments", [])
        if not segments:
            log("⚠️ Whisper returned no segments.")
            return None

        def split_into_chunks(text, max_chars=40):
            """Split text into chunks no longer than max_chars (breaking on spaces)."""
            words = text.strip().split()
            if not words:
                return []
            chunks, cur = [], []
            for w in words:
                if cur and len(" ".join(cur + [w])) > max_chars:
                    chunks.append(" ".join(cur))
                    cur = [w]
                else:
                    cur.append(w)
            if cur:
                chunks.append(" ".join(cur))
            return chunks

        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 1.0))
            seg_text = (seg.get("text", "") or "").strip()

            words = seg.get("words") or []
            if words:
                # Normalize words across whisper variants
                processed_words = []
                for w in words:
                    w_text = (w.get("text") or w.get("word") or "").strip()
                    if not w_text:
                        continue
                    w_start = float(w.get("start", seg_start))
                    w_end = float(w.get("end", w_start + 0.3))
                    processed_words.append({"text": w_text, "start": w_start, "end": w_end})

                # Group into readable lines
                MAX_CHARS = 42
                current_words, cur_start, cur_end = [], None, None
                for w in processed_words:
                    if cur_start is None:
                        current_words = [w["text"]]
                        cur_start, cur_end = w["start"], w["end"]
                    else:
                        proposed = " ".join(current_words + [w["text"]])
                        if len(proposed) <= MAX_CHARS and (w["end"] - cur_start) <= 5.0:
                            current_words.append(w["text"])
                            cur_end = w["end"]
                        else:
                            subs.append(pysubs2.SSAEvent(
                                start=pysubs2.make_time(s=cur_start),
                                end=pysubs2.make_time(s=cur_end),
                                text=f"{{\\b1}}{' '.join(current_words)}{{\\b0}}"
                            ))
                            event_count += 1
                            current_words = [w["text"]]
                            cur_start, cur_end = w["start"], w["end"]

                if current_words:
                    subs.append(pysubs2.SSAEvent(
                        start=pysubs2.make_time(s=cur_start),
                        end=pysubs2.make_time(s=cur_end),
                        text=f"{{\\b1}}{' '.join(current_words)}{{\\b0}}"
                    ))
                    event_count += 1
            else:
                # Segment-level fallback
                parts = split_into_chunks(seg_text, max_chars=42)
                if not parts:
                    continue
                seg_duration = max(0.001, seg_end - seg_start)
                for i, part in enumerate(parts):
                    part_start = seg_start + (i / len(parts)) * seg_duration
                    part_end = seg_start + ((i + 1) / len(parts)) * seg_duration
                    subs.append(pysubs2.SSAEvent(
                        start=pysubs2.make_time(s=part_start),
                        end=pysubs2.make_time(s=part_end),
                        text=f"{{\\b1}}{part}{{\\b0}}"
                    ))
                    event_count += 1

        if event_count > 0:
            os.makedirs(os.path.dirname(captions_file), exist_ok=True)
            subs.save(captions_file)
            log(f"✅ Captions generated successfully with {event_count} events (word-level: {used_word_timestamps}).")
            return captions_file
        else:
            log("⚠️ No captions created.")
            return None

    except Exception as e:
        logging.error(f"Failed to generate captions: {e}")
        log(f"❌ Error during caption generation: {e}")
        return None


# ============================
# GEMINI / WAVESPEED
# ============================
def ensure_genai():
    global genai_client
    if genai_client is None:
        key=config.get("GEMINI_API_KEY","").strip()
        if not key: raise RuntimeError("Gemini API key missing (Settings tab).")
        genai.configure(api_key=key)
        genai_client=genai.GenerativeModel(GEMINI_TEXT_MODEL)
def handle_gemini_errors(func):
    """A decorator to catch and handle Gemini API errors, especially for quota."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except google_exceptions.ResourceExhausted as e:
            error_message = "Gemini API quota exceeded (Resource Exhausted). Please check your usage or billing."
            log(f"❌ {error_message}")
            raise RuntimeError(error_message) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                error_message = "Gemini API quota exceeded (HTTP 429). Please check your usage or billing."
                log(f"❌ {error_message}")
                raise RuntimeError(error_message) from e
            # Re-raise other HTTP errors
            raise
    return wrapper




@handle_gemini_errors
def gemini_deep_research(topic:str)->str:
    """Performs deep research using Gemini with live Google Search."""
    key = config.get("GEMINI_API_KEY","").strip()
    if not key: raise RuntimeError("Gemini API key missing (Settings tab).")

    log("🌐 Using Google Search to find fresh, up-to-date information...")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL}:generateContent?key={key}"
    
    language_instruction = ""
    if config.get("LANGUAGE_ENABLED", False):
        language = config.get("PODCAST_LANGUAGE", "English")
        if language.lower() == 'urdu':
            language_instruction = "The summary must be written in Roman Urdu."
        else:
            language_instruction = f"The summary must be written in {language}."

    payload={
        "contents":[{"role":"user","parts":[{"text":(
            f"Use Google Search to find current, fresh, and in-depth information on the topic '{topic}'. "
            "Summarize the background, why it is trending, key facts, controversies, and outlook. "
            f"Provide a well-structured summary based on your search results. {language_instruction}"
        )}]}],
        "tools": [{"google_search": {}}]
    }

    try:
        r = requests.post(api_url, json=payload, timeout=120)
        r.raise_for_status()
        response_json = r.json()

        text_part = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
        if text_part:
            return text_part
        else:
            raise ValueError("No text content found in Gemini response.")

    except Exception as e:
        logging.error(f"Gemini research with search failed: {e}")
        raise RuntimeError("Gemini research with search failed.")

def generate_seo_metadata(topic: str, research: str):
    """Generates an SEO-optimized title, description, and tags for a video."""
    ensure_genai()
    prompt = f"""
    Act as a world-class YouTube SEO expert and content creator. Your task is to generate a highly attractive, SEO-optimized title and a detailed description for a video based on the following topic and research.

    **Video Topic:** {topic}
    **Research:**
    {research}

    **Instructions:**
    1.  **Title:** Create a catchy, click-worthy title that includes the main keywords. The title should be no more than 60 characters.
    2.  **Description:** Write a detailed, SEO-optimized description (200-300 words). It should start with a compelling hook, summarize the video's content, include relevant keywords naturally, and end with a call to action.
    3.  **Tags:** Provide a list of 5-10 highly relevant, comma-separated tags.

    **Format your response as a JSON object:**
    ```json
    {{
      "title": "Generated Title",
      "description": "Generated description...",
      "tags": "tag1, tag2, tag3"
    }}
    ```
    """
    try:
        response = genai_client.generate_content(prompt, safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'},
                                                 generation_config={"response_mime_type": "application/json"})
        metadata = json.loads(response.text)
        return metadata
    except Exception as e:
        logging.error(f"Failed to generate SEO metadata: {e}")
        return {"title": topic, "description": "", "tags": ""}

@handle_gemini_errors
def gemini_podcast_script(topic:str, research:str, host_persona:str, guest_persona:str)->str:
    ensure_genai()
    host, guest = config.get("HOST_NAME","Alex"), config.get("GUEST_NAME","Maya")
    channel=config.get("CHANNEL_NAME","My AI Channel")
    sub_count=config.get("SUBSCRIBE_COUNT",3)
    sub_message=config.get("SUBSCRIBE_MESSAGE","Don’t forget to subscribe to {channel}!").replace("{channel}",channel)
    randomize=config.get("SUBSCRIBE_RANDOM",True)
    style=config.get("PODCAST_STYLE","Informative News")

    language_prompt = ""
    language_instruction = "The entire script, including all dialogue and special instructions, must be in the specified language if provided."
    if config.get("LANGUAGE_ENABLED", False):
        language = config.get("PODCAST_LANGUAGE", "English")
        language_prompt = f"In {language}, "
        if language.lower() == 'urdu':
            language_instruction = "The entire script, including all dialogue and special instructions, must be in Roman Urdu."

    placement_instruction = (
        f"- Insert about {sub_count} reminders randomly and naturally at different points."
        if randomize else
        f"- Insert exactly {sub_count} reminders evenly spaced."
    )

    prompt = f"""
{language_prompt}turn this research into a podcast script.
Podcast Title: {channel} - A Podcast about {topic}
Hosts: {host} (Persona: {host_persona}) and {guest} (Persona: {guest_persona})
Podcast Style: {style}
Format:

{host}: ...
{guest}: ...
Special instructions:
- The script should be conversational, as if two people are talking. Avoid long monologues.
- Use contractions like "don't", "it's", and "you're" to sound more natural.
- Keep sentences relatively short and easy to follow.
- Allow for occasional incomplete sentences or one speaker finishing the other's thought.
{placement_instruction}
- Reminder text: "{sub_message}"
- End with: "{host}: Thanks for listening! {sub_message}"
- Use a conversational and engaging tone.
- {language_instruction}

Topic: {topic}
Research:
{research}
"""
    response = genai_client.generate_content(prompt, safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
    script_content = response.text

    # Check if the humanoid filter is enabled
    if config.get("HUMANOID_ENABLED", False):
        log("✍️ Applying humanoid conversational fillers...")
        humanoid_prob = config.get("HUMANOID_PROBABILITY", 0.1)
        script_content = add_fillers_to_script(script_content, humanoid_prob)

    return script_content
@handle_gemini_errors
def gemini_fact_check(script:str)->str:
    ensure_genai()

    language_instruction = "Your entire response must be in English."
    if config.get("LANGUAGE_ENABLED", False):
        language = config.get("PODCAST_LANGUAGE", "English")
        if language.lower() == 'urdu':
            language_instruction = "Your entire response must be in Roman Urdu."
        else:
            language_instruction = f"Your entire response must be in {language}."

    prompt = f"""
Review the following podcast script for factual accuracy. Identify any claims that are likely incorrect or require more nuance.
Respond with a list of potential issues and suggestions for correction. If the script is generally accurate, state that.
{language_instruction}

Script:
{script}
"""
    response = genai_client.generate_content(prompt, safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
    
    try:
        # This will now only be attempted if the response is valid
        return response.text
    except ValueError:
        # This block will catch the error when response.text is invalid
        logging.warning(f"Fact-check response was blocked or incomplete. Finish Reason: {response.candidates[0].finish_reason}")
        if response.candidates[0].finish_reason.name == "MAX_TOKENS":
            return "Fact-check could not be completed. The script is too long, causing the response to exceed the API's maximum token limit."
        elif response.candidates[0].finish_reason.name == "SAFETY":
            return f"Fact-check failed. The script or the model's response was blocked for safety reasons. Feedback: {response.prompt_feedback}"
        else:
            return f"Fact-check failed due to an unexpected API issue. Finish Reason: {response.candidates[0].finish_reason.name}"

@handle_gemini_errors
def gemini_revise_script(script:str, fact_check_result:str)->str:
    ensure_genai()
    prompt = f"""
    Based on the following fact-check results, please revise the podcast script to be more accurate.
    Only output the revised script, with no extra commentary.

    **Fact-Check Results:**
    {fact_check_result}

    **Original Script:**
    {script}
    """
    response = genai_client.generate_content(prompt, safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
    return response.text

@handle_gemini_errors
def gemini_tts_generate(script: str, output_path: str, mode: str = "Multi-Speaker", single_voice: str = "Kore") -> str:
    key = config.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Gemini API key missing (Settings tab).")

    safe_script = sanitize_for_tts(script)
    host, guest = config.get("HOST_NAME", "Alex"), config.get("GUEST_NAME", "Maya")
    sp1, sp2 = config.get("SPEAKER1", "Kore"), config.get("SPEAKER2", "Puck")
    style = config.get("PODCAST_STYLE", "Informative News")

    payload = {
        "contents": [{"parts": [{"text": f"[Style: {style}]\n\n{safe_script}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {}
        }
    }

    if mode == "Multi-Speaker":
        log(f"🎭 {sp1} ({host}) & {sp2} ({guest}), Style: {style}")
        payload["generationConfig"]["speechConfig"] = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {"speaker": host, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": sp1}}},
                    {"speaker": guest, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": sp2}}}
                ]
            }
        }
    else:
        log(f"🎤 {single_voice}, Style: {style}")
        payload["generationConfig"]["speechConfig"] = {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": single_voice}}
        }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={key}"

    # Retry loop
    for attempt in range(5):
        try:
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            resp_json = response.json()

            # ✅ Defensive parsing
            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"TTS failed: no candidates. Full response: {resp_json}")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError(f"TTS failed: no parts. Full response: {resp_json}")

            audio_data = parts[0].get("inlineData", {}).get("data")
            if not audio_data:
                text_fallback = parts[0].get("text", "")
                raise RuntimeError(f"TTS failed: no audio data. Got text fallback: {text_fallback}")

            # ✅ Save audio file using the provided output_path
            pcm = base64.b64decode(audio_data)
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm)

            return output_path

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            delay = 5 * (attempt + 1)
            log(f"⚠️ Network error ({e}). Retrying in {delay} sec...")
            time.sleep(delay)
            continue
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < 4:
                delay = 2 ** attempt
                log(f"⚠️ Rate limit (429). Retrying in {delay} sec...")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            raise RuntimeError(f"TTS generation failed: {e}")

    # If all retries failed
    raise RuntimeError("Gemini TTS failed after multiple retries.")
def wavespeed_text_to_video(topic: str, video_path) -> str:
    wavespeed_key = config.get("WAVESPEED_AI_KEY", "").strip()
    if not wavespeed_key:
        raise RuntimeError("WaveSpeed AI key missing (Settings tab).")

    # Use the custom prompt template from settings, inserting the current topic
    video_prompt_template = config.get("VIDEO_PROMPT_STYLE", "An animated and cinematic video about {topic}.")
    prompt_text = video_prompt_template.format(topic=topic)
    log(f"ℹ️ Using video prompt: \"{prompt_text}\"")

    headers={"Content-Type":"application/json", "Authorization":f"Bearer {wavespeed_key}"}
    payload={
      "duration": 5,
      "negative_prompt": "",
      "prompt": prompt_text,
      "seed": -1,
      "size": "832*480"
    }

    log("▶️ Sending text to WaveSpeed for video generation...")

    try:
        # 1. Initial POST request to start the video generation
        initial_response = requests.post(WAVESPEED_T2V_API_URL, headers=headers, json=payload, timeout=120)
        initial_response.raise_for_status()

        # The 'id' is inside a 'data' object, as per the official example
        initial_data = initial_response.json().get("data")
        if not initial_data or "id" not in initial_data:
            raise KeyError(f"'id' not found in initial API response. Full response: {initial_response.json()}")

        reqid = initial_data["id"]
        log(f"✅ Task submitted successfully. Request ID: {reqid}")

        # 2. Poll the result URL until the task is complete
        result_url = WAVESPEED_POLL_URL.format(reqid)
        poll_headers = {"Authorization": f"Bearer {wavespeed_key}"}
        start_time = time.time()
        timeout_seconds = 600 # 10 minute timeout

        log("⏱️ Polling for results... (checking every 10 seconds)")
        while time.time() - start_time < timeout_seconds:
            poll_response = requests.get(result_url, headers=poll_headers, timeout=60)

            if poll_response.status_code == 200:
                # The result is always inside a 'data' object
                result_data = poll_response.json().get("data")
                if not result_data:
                    raise ValueError(f"Polling response is missing the 'data' object. Full response: {poll_response.json()}")

                status = result_data.get("status")

                if status == "completed":
                    log("✅ Task completed! Downloading video.")
                    video_url = result_data["outputs"][0]

                    # Download the video
                    video_content = requests.get(video_url).content
                    with open(video_path, "wb") as f:
                        f.write(video_content)
                    log(f"✅ Video successfully saved to {video_path}")
                    return video_path

                elif status == "failed":
                    error_msg = result_data.get('error', 'No details provided.')
                    log(f"❌ Video generation failed: {error_msg}")
                    raise RuntimeError(f"WaveSpeed task failed: {error_msg}")

                else:
                    # e.g., 'queued', 'processing'
                    log(f"⏳ Task status is '{status}'. Waiting...")

            else:
                log(f"⚠️ Polling failed with status {poll_response.status_code}. Retrying...")

            time.sleep(10) # Wait 10 seconds before the next poll

        # If the loop finishes, it's a timeout
        raise TimeoutError("WaveSpeed request timed out after 10 minutes.")

    except Exception as e:
        logging.error(f"Video generation process failed: {e}", exc_info=True)
        raise RuntimeError(f"Video generation process failed: {e}")
def youtube_auth():
    """Handles YouTube API authentication."""
    CLIENT_SECRETS_FILE = "client_secrets.json"
    if not os.path.exists(CLIENT_SECRETS_FILE):
        messagebox.showerror("Error", f"{CLIENT_SECRETS_FILE} not found. Please create one in the Google Cloud Console and place it in the same directory as this script.")
        return None

    credentials = None
    pickle_file = Path("token.pickle")

    if pickle_file.exists():
        with open(pickle_file, "rb") as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE,
                scopes=["https://www.googleapis.com/auth/youtube.upload"],
                redirect_uri='http://localhost:8080/'
            )
            credentials = flow.run_local_server(port=8080)

        with open(pickle_file, "wb") as f:
            pickle.dump(credentials, f)
    
    return credentials

def upload_youtube():
    """Authenticates and uploads a video to YouTube."""
    try:
        # Disable the button to prevent multiple uploads
        youtube_upload_button.configure(state="disabled")

      # Check if a custom video path is provided
        custom_video_path = video_path_entry.get().strip()
        if custom_video_path and os.path.exists(custom_video_path):
            video_path = custom_video_path
        else:
            safe_topic = re.sub(r'[\\/:*?"<>|]', '', topic_entry.get().strip())
            video_path = os.path.join(safe_topic, "final_podcast_video.mp4")

        if not os.path.exists(video_path):
            return messagebox.showerror("Error", "Final video not found. Please run the pipeline or select a valid video file.")


        log("▶️ Authenticating with YouTube...")
        credentials = youtube_auth()
        if not credentials:
            log("❌ YouTube authentication failed.")
            youtube_upload_button.configure(state="normal")
            return

        youtube = build("youtube", "v3", credentials=credentials)
        log("✅ YouTube authentication successful.")

        request_body = {
            "snippet": {
                "title": video_title_entry.get(),
                "description": video_desc_entry.get("1.0", ctk.END),
                "tags": video_tags_entry.get().split(","),
                "categoryId": "22" # People & Blogs category
            },
            "status": {
                "privacyStatus": "public" # or 'private' or 'unlisted'
            }
        }
        
        log(f"📤 Uploading '{video_path}' to YouTube...")
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(
            part=",".join(request_body.keys()),
            body=request_body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log(f"⬆️ Uploaded {int(status.progress() * 100)}%")

        log(f"✅ Video uploaded successfully! Video ID: {response.get('id')}")
        messagebox.showinfo("Upload Complete", f"Successfully uploaded to YouTube! Video ID: {response.get('id')}")

    except Exception as e:
        log(f"❌ YouTube upload failed: {e}")
        messagebox.showerror("Upload Error", f"An error occurred during YouTube upload: {e}")
    finally:
        # Re-enable the button
        youtube_upload_button.configure(state="normal")

def upload_facebook():
    """Authenticates and uploads a video to Facebook."""
    try:
        # Disable the button to prevent multiple uploads
        facebook_upload_button.configure(state="disabled")

         # Check if a custom video path is provided
        custom_video_path = video_path_entry.get().strip()
        if custom_video_path and os.path.exists(custom_video_path):
            video_path = custom_video_path
        else:
            safe_topic = re.sub(r'[\\/:*?"<>|]', '', topic_entry.get().strip())
            video_path = os.path.join(safe_topic, "final_podcast_video.mp4")

        if not os.path.exists(video_path):
            return messagebox.showerror("Error", "Final video not found. Please run the pipeline or select a valid video file.")
        
        log("▶️ Uploading to Facebook...")

        # Step 1: Initialize the upload session
        init_url = f"https://graph-video.facebook.com/v20.0/me/videos"
        init_params = {
            "access_token": access_token,
            "upload_phase": "start"
        }
        init_response = requests.post(init_url, params=init_params).json()
        
        if "error" in init_response:
            raise RuntimeError(f"Facebook API Error: {init_response['error']['message']}")

        upload_session_id = init_response["upload_session_id"]
        video_id = init_response["video_id"]
        
        log(f"✅ Facebook upload session started. Session ID: {upload_session_id}")

        # Step 2: Upload the video file
        upload_url = f"https://graph-video.facebook.com/v20.0/{upload_session_id}"
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()

        upload_headers = {
            "Authorization": f"OAuth {access_token}",
            "file_offset": "0"
        }
        upload_response = requests.post(upload_url, headers=upload_headers, data=video_data)
        
        if "error" in upload_response.json():
            raise RuntimeError(f"Facebook API Error during upload: {upload_response.json()['error']['message']}")
        
        log("✅ Video file uploaded to Facebook.")

        # Step 3: Finish the upload session and publish
        finish_url = f"https://graph-video.facebook.com/v20.0/me/videos"
        finish_params = {
            "access_token": access_token,
            "upload_phase": "finish",
            "upload_session_id": upload_session_id,
            "title": video_title_entry.get(),
            "description": video_desc_entry.get("1.0", ctk.END)
        }
        finish_response = requests.post(finish_url, params=finish_params).json()

        if "error" in finish_response:
            raise RuntimeError(f"Facebook API Error on finishing: {finish_response['error']['message']}")

        log(f"✅ Video published successfully to Facebook! Video ID: {video_id}")
        messagebox.showinfo("Upload Complete", f"Successfully published to Facebook! Video ID: {video_id}")

    except Exception as e:
        log(f"❌ Facebook upload failed: {e}")
        messagebox.showerror("Upload Error", f"An error occurred during Facebook upload: {e}")
    finally:
        # Re-enable the button
        facebook_upload_button.configure(state="normal")

def get_history_items():
    history_items = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in ['__pycache__', 'venv']:
            history_items.append(item)
    return history_items

def delete_history_item(item_name):
    try:
        shutil.rmtree(item_name)
        messagebox.showinfo("Delete", f"Successfully deleted {item_name}.")
        # Refresh the history tab
        update_history_tab()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to delete {item_name}: {e}")

def update_history_tab():
    # Clear existing widgets
    for widget in history_tab_frame.winfo_children():
        widget.destroy()

    ctk.CTkLabel(history_tab_frame, text="History", font=("Arial", 20, "bold")).pack(pady=(20, 10))
    ctk.CTkLabel(history_tab_frame, text="All previously generated podcasts:", font=("Arial", 14)).pack(pady=5)

    items = get_history_items()
    if not items:
        ctk.CTkLabel(history_tab_frame, text="No history found.", font=("Arial", 12)).pack(pady=20)
    else:
        for item in items:
            row = ctk.CTkFrame(history_tab_frame); row.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(row, text=item, anchor="w", font=("Arial", 12, "bold")).pack(side="left", padx=10, pady=5)
            delete_btn = ctk.CTkButton(row, text="Delete", command=lambda i=item: delete_history_item(i), fg_color="#FF0000", hover_color="#8c0303", text_color="white")
            delete_btn.pack(side="right", padx=10, pady=5)
# ============================
# Publish Metadata
# ============================
def generate_seo_only():
    """Generates only the SEO metadata and fills the fields."""
    topic = topic_entry.get().strip()
    if not topic:
        return messagebox.showerror("Error", "Please enter a topic first.")

    safe_topic = re.sub(r'[\\/:*?"<>|]', '', topic)
    summary_file_path = os.path.join(safe_topic, "summary.txt")

    research = ""
    try:
        if os.path.exists(summary_file_path):
            log("➡️ Using existing research for SEO generation.")
            with open(summary_file_path, "r", encoding="utf-8") as f:
                research = f.read()
        else:
            log("🔍 Research not found. Running deep research first...")
            research = gemini_deep_research(topic)
            if not os.path.exists(safe_topic):
                os.makedirs(safe_topic)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(research)
            log("✅ Research complete.")

        log("📄 Generating SEO metadata...")
        metadata = generate_seo_metadata(topic, research)
        
        video_title_entry.delete(0, ctk.END)
        video_title_entry.insert(0, metadata.get("title", topic))
        
        video_desc_entry.delete("1.0", ctk.END)
        video_desc_entry.insert("1.0", metadata.get("description", ""))

        video_tags_entry.delete(0, ctk.END)
        video_tags_entry.insert(0, metadata.get("tags", ""))
        
        log("✅ SEO metadata generated and pre-filled.")
        messagebox.showinfo("Success", "SEO Title and Description have been generated!")

    except Exception as e:
        log(f"❌ Error during SEO generation: {e}")
        messagebox.showerror("Error", f"Failed to generate SEO metadata: {e}")

# ============================
# PIPELINE
# ============================
def run_pipeline():
    # This function is now aware of the stop_event
    topic = topic_entry.get().strip()
    if not topic:
        return messagebox.showerror("Error", "Enter a topic first")
    if not check_ffmpeg():
        return messagebox.showerror("Error", "ffmpeg not found. Please install it and add to PATH.")
    if not config.get("GEMINI_API_KEY", "").strip():
        return messagebox.showerror("API Key Missing", "Please enter your Gemini API key in the 'Branding & Subscribe' tab and save.")
    if not config.get("WAVESPEED_AI_KEY", "").strip():
        return messagebox.showerror("API Key Missing", "Please enter your WaveSpeed AI key in the 'Branding & Subscribe' tab and save.")

    # Create the directory for the current topic if it doesn't exist
    safe_topic = re.sub(r'[\\/:*?"<>|]', '', topic)
    if not os.path.exists(safe_topic):
        os.makedirs(safe_topic)

    SUMMARY_FILE_PATH = os.path.join(safe_topic, "summary.txt")
    SCRIPT_FILE_PATH = os.path.join(safe_topic, "podcast_script.txt")
    AUDIO_FILE_PATH = os.path.join(safe_topic, "podcast.wav")
    BACKGROUND_VIDEO_RAW_PATH = os.path.join(safe_topic, "background.mp4")
    FINAL_VIDEO_PATH = os.path.join(safe_topic, "final_podcast_video.mp4")
    CAPTIONS_FILE_PATH = os.path.join(safe_topic, "captions.ass") # Use .ass format

    start_point = start_step_combo.get()

    reset_steps()

    try:
        # Step 1: Research
        if stop_event.is_set(): return
        research = ""
        if start_point == "Deep Research" or not os.path.exists(SUMMARY_FILE_PATH):
            log(f"🔍 Researching: {topic}"); set_step_status(0,"⏳", 0.1)
            research = gemini_deep_research(topic)
            open(SUMMARY_FILE_PATH, "w", encoding="utf-8").write(research)
            set_step_status(0,"✅", 1.0); log("✅ Research saved")
        else:
            log("➡️ Skipping research step. Using existing summary.")
            research = open(SUMMARY_FILE_PATH, "r", encoding="utf-8").read()
            set_step_status(0,"☑️", 1.0)

        # Step 2: Generate SEO Metadata
        if stop_event.is_set(): return
        if metadata_var.get():
            log("📄 Generating SEO metadata..."); set_step_status(1,"⏳", 0.1)
            metadata = generate_seo_metadata(topic, research)
            video_title_entry.delete(0, ctk.END); video_title_entry.insert(0, metadata.get("title", topic))
            video_desc_entry.delete("1.0", ctk.END); video_desc_entry.insert("1.0", metadata.get("description", ""))
            video_tags_entry.delete(0, ctk.END); video_tags_entry.insert(0, metadata.get("tags", ""))
            set_step_status(1, "✅", 1.0); log("✅ SEO metadata generated and pre-filled.")
        else:
            set_step_status(1, "⏭️", 1.0)

        # Step 3: Script
        if stop_event.is_set(): return
        if start_point == "Podcast Script" or not os.path.exists(SCRIPT_FILE_PATH):
            log("📝 Generating script..."); set_step_status(2,"⏳", 0.1)
            host_p = config.get("HOST_PERSONA"); guest_p = config.get("GUEST_PERSONA")
            script = gemini_podcast_script(topic, research, host_p, guest_p)
            open(SCRIPT_FILE_PATH, "w", encoding="utf-8").write(script)
            set_step_status(2,"✅", 1.0); log("✅ Script saved")
        else:
            log("➡️ Skipping script generation. Using existing script.")
            script = open(SCRIPT_FILE_PATH, "r", encoding="utf-8").read()
            set_step_status(2,"☑️", 1.0)

        # Step 4: Fact Check
        if stop_event.is_set(): return
        if fact_check_var.get():
            log("🧐 Fact checking script..."); set_step_status(3,"⏳", 0.1)
            fact_check_result = gemini_fact_check(script)
            log("--- Fact Check Results ---"); log(fact_check_result)
            set_step_status(3,"✅", 1.0); log("✅ Fact check complete")

            # Step 5: Revise Script
            if stop_event.is_set(): return
            log("✍️ Revising script based on fact-check..."); set_step_status(4,"⏳", 0.1)
            script = gemini_revise_script(script, fact_check_result)
            open(SCRIPT_FILE_PATH, "w", encoding="utf-8").write(script) # Overwrite script with revised version
            set_step_status(4,"✅", 1.0); log("✅ Script revised and saved")
        else:
            set_step_status(3, "⏭️", 1.0)
            set_step_status(4, "⏭️", 1.0) # Skip revise step as well


        # Step 5: TTS
        if stop_event.is_set(): return
        if start_point == "Audio (TTS)" or not os.path.exists(AUDIO_FILE_PATH):
            log("🎙️ Generating audio..."); set_step_status(4,"⏳", 0.1)
            audio_path = gemini_tts_generate(script, AUDIO_FILE_PATH, tts_mode_combo.get(), voice_combo.get())
            audio_len = get_audio_length_fast(audio_path)
            set_step_status(4,f"✅ {audio_len:.1f}s", 1.0); log(f"✅ Audio saved ({audio_len:.1f}s)")
        else:
            log("➡️ Skipping audio generation. Using existing audio file.")
            audio_path = AUDIO_FILE_PATH
            audio_len = get_audio_length_fast(audio_path)
            set_step_status(4,f"☑️ {audio_len:.1f}s", 1.0)

        # Step 6: Caption Generation
        if stop_event.is_set(): return
        captions_path = None
        if caption_var.get():
            log("✍️ Generating captions..."); set_step_status(5,"⏳", 0.1)
            captions_path = generate_captions(audio_path, CAPTIONS_FILE_PATH)
            if captions_path:
                set_step_status(5, "✅", 1.0); log(f"✅ Captions file saved: {captions_path}")
            else:
                set_step_status(5, "❌", 1.0); log("❌ Captions generation failed.")
        else:
            set_step_status(5, "⏭️", 1.0)

        # Step 7: Video Generation
        if stop_event.is_set(): return
        if start_point == "Video Generation" or not os.path.exists(BACKGROUND_VIDEO_RAW_PATH):
            log("🎬 Generating video from text with WaveSpeed..."); set_step_status(6,"⏳", 0.1)
            video_path = wavespeed_text_to_video(topic, BACKGROUND_VIDEO_RAW_PATH)
            set_step_status(6,"✅", 1.0); log("✅ Video created.")
        else:
            log("➡️ Skipping video generation. Using existing video.")
            video_path = BACKGROUND_VIDEO_RAW_PATH
            set_step_status(6,"☑️", 1.0)

        # Step 8: Final Video Creation
        if stop_event.is_set(): return
        log("🎶 Merging video and audio..."); set_step_status(7,"⏳", 0.1)
        captions_file_path_ffmpeg = os.path.normpath(CAPTIONS_FILE_PATH).replace(os.path.sep, '/')
        ffmpeg_cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-t", f"{audio_len:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if caption_var.get() and os.path.exists(CAPTIONS_FILE_PATH):
             ffmpeg_cmd += ["-vf", f"ass={captions_file_path_ffmpeg}"]
        ffmpeg_cmd += [FINAL_VIDEO_PATH]
        subprocess.run(ffmpeg_cmd, check=True)
        set_step_status(7,"✅", 1.0); log(f"✅ Final video saved as {FINAL_VIDEO_PATH}")

        if not stop_event.is_set():
            messagebox.showinfo("Done",f"Video saved: {FINAL_VIDEO_PATH}")

    except Exception as e:
        if not stop_event.is_set():
            log(f"❌ Error: {e}"); messagebox.showerror("Error",str(e))
    finally:
        if stop_event.is_set():
            log("⏹️ Pipeline stopped by user.")
        run_button.configure(state="normal")
        stop_button.configure(state="disabled")
        update_history_tab()

# ============================
# GUI
# ============================
def save_keys():
    config.update({
        "GEMINI_API_KEY": gemini_key_entry.get().strip(),
        "WAVESPEED_AI_KEY": wavespeed_key_entry.get().strip(),
        "SPEAKER1": extract_voice_name(speaker1_combo.get()),
        "SPEAKER2": extract_voice_name(speaker2_combo.get()),
        "HOST_NAME": host_entry.get().strip() or "Alex",
        "GUEST_NAME": guest_entry.get().strip() or "Maya",
        "HOST_PERSONA": host_persona_entry.get("1.0", "end").strip(),
        "GUEST_PERSONA": guest_persona_entry.get("1.0", "end").strip(),
        "CHANNEL_NAME": channel_entry.get().strip() or "My AI Channel",
        "SUBSCRIBE_MESSAGE": sub_message_entry.get().strip(),
        "SUBSCRIBE_RANDOM": subscribe_random_var.get(),
        "PODCAST_STYLE": style_combo.get(),
        "VIDEO_PROMPT_STYLE": video_style_textbox.get("1.0", "end-1c").strip(),
        "FACT_CHECK_ENABLED": fact_check_var.get(),
        "CAPTION_ENABLED": caption_var.get(),
        "GENERATE_METADATA": metadata_var.get(),
        "YOUTUBE_CLIENT_ID": youtube_id_entry.get().strip(),
        "YOUTUBE_CLIENT_SECRET": youtube_secret_entry.get().strip(),
        "FACEBOOK_ACCESS_TOKEN": facebook_token_entry.get().strip(),
        "VIDEO_TITLE": video_title_entry.get().strip(),
        "VIDEO_DESCRIPTION": video_desc_entry.get("1.0", "end").strip(),
        "VIDEO_TAGS": video_tags_entry.get().strip(),
        "HUMANOID_ENABLED": humanoid_enabled_var.get(),
        "LANGUAGE_ENABLED": language_enabled_var.get(),
        "PODCAST_LANGUAGE": language_combo.get(),
        "HUMANOID_PROBABILITY": float(humanoid_prob_entry.get().strip())
        
    })

    try:
        config["SUBSCRIBE_COUNT"] = int(sub_count_entry.get().strip())
    except Exception:
        config["SUBSCRIBE_COUNT"] = 3

    save_config(config)

    global genai_client
    genai_client = None

    messagebox.showinfo("Saved", "✅ Settings saved successfully")

def stop_pipeline():
    """Sets the stop event to gracefully halt the pipeline."""
    log("🛑 Stop signal received. Finishing current step...")
    stop_event.set()

def start_pipeline_thread():
    """Configures buttons and starts the pipeline in a new thread."""
    stop_event.clear()  # Reset the stop event before starting
    run_button.configure(state="disabled")
    stop_button.configure(state="normal")
    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

def load_settings():
    # reload config.json from disk
    cfg = load_config()

    # Apply saved values back into GUI fields
    gemini_key_entry.delete(0, ctk.END); gemini_key_entry.insert(0, cfg.get("GEMINI_API_KEY", ""))
    wavespeed_key_entry.delete(0, ctk.END); wavespeed_key_entry.insert(0, cfg.get("WAVESPEED_AI_KEY", ""))

    speaker1_combo.set(f"{cfg['SPEAKER1']} — {VOICE_OPTIONS.get(cfg['SPEAKER1'], '')}")
    speaker2_combo.set(f"{cfg['SPEAKER2']} — {VOICE_OPTIONS.get(cfg['SPEAKER2'], '')}")

    host_entry.delete(0, ctk.END); host_entry.insert(0, cfg.get("HOST_NAME", "Alex"))
    guest_entry.delete(0, ctk.END); guest_entry.insert(0, cfg.get("GUEST_NAME", "Maya"))

    host_persona_entry.delete("1.0", ctk.END); host_persona_entry.insert("1.0", cfg.get("HOST_PERSONA", ""))
    guest_persona_entry.delete("1.0", ctk.END); guest_persona_entry.insert("1.0", cfg.get("GUEST_PERSONA", ""))

    channel_entry.delete(0, ctk.END); channel_entry.insert(0, cfg.get("CHANNEL_NAME", "My AI Channel"))
    sub_count_entry.delete(0, ctk.END); sub_count_entry.insert(0, str(cfg.get("SUBSCRIBE_COUNT", 3)))
    sub_message_entry.delete(0, ctk.END); sub_message_entry.insert(0, cfg.get("SUBSCRIBE_MESSAGE", ""))

    subscribe_random_var.set(cfg.get("SUBSCRIBE_RANDOM", True))
    style_combo.set(cfg.get("PODCAST_STYLE", "Informative News"))
    video_style_textbox.delete("1.0", ctk.END)
    video_style_textbox.insert("1.0", cfg.get("VIDEO_PROMPT_STYLE", "An animated and cinematic video about {topic}."))
    fact_check_var.set(cfg.get("FACT_CHECK_ENABLED", False))
    caption_var.set(cfg.get("CAPTION_ENABLED", False))
    metadata_var.set(cfg.get("GENERATE_METADATA", False))

    video_title_entry.delete(0, ctk.END); video_title_entry.insert(0, cfg.get("VIDEO_TITLE", ""))
    video_desc_entry.delete("1.0", ctk.END); video_desc_entry.insert("1.0", cfg.get("VIDEO_DESCRIPTION", ""))
    video_tags_entry.delete(0, ctk.END); video_tags_entry.insert(0, cfg.get("VIDEO_TAGS", ""))

    humanoid_enabled_var.set(cfg.get("HUMANOID_ENABLED", False))
    humanoid_prob_entry.delete(0, ctk.END); humanoid_prob_entry.insert(0, str(cfg.get("HUMANOID_PROBABILITY", 0.1)))

    youtube_id_entry.delete(0, ctk.END); youtube_id_entry.insert(0, cfg.get("YOUTUBE_CLIENT_ID", ""))
    youtube_secret_entry.delete(0, ctk.END); youtube_secret_entry.insert(0, cfg.get("YOUTUBE_CLIENT_SECRET", ""))
    facebook_token_entry.delete(0, ctk.END); facebook_token_entry.insert(0, cfg.get("FACEBOOK_ACCESS_TOKEN", ""))
    language_enabled_var.set(cfg.get("LANGUAGE_ENABLED", False))
    language_combo.set(cfg.get("PODCAST_LANGUAGE", "English"))


def browse_file(entry):
    filename = filedialog.askopenfilename(filetypes=[
        ("Video files", "*.mp4;*.mkv;*.avi"),
        ("Image files", "*.png;*.jpg;*.jpeg"), 
        ("Audio files", "*.mp3;*.wav"),
        ("All files", "*.*")
    ])
    if filename:
        entry.delete(0, ctk.END)
        entry.insert(0, filename)

def open_url(url):
    webbrowser.open_new(url)

ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
app=ctk.CTk(); app.title("🎙️ Nullpk Content Automation"); app.geometry("1100x1250")

tabs=ctk.CTkTabview(app,width=1080,height=1150); tabs.pack(padx=10,pady=10,fill="both",expand=True)

# --- HEADER FRAME ---
header_frame = ctk.CTkFrame(app, corner_radius=10, fg_color="transparent")
header_frame.pack(side="top", fill="x", padx=10, pady=(5, 0))

header_label = ctk.CTkLabel(header_frame, text="Nullpk Content Automation", font=("Arial", 24, "bold"), text_color="#1f6aa5")
header_label.pack(side="left", padx=10, pady=5)
author_label = ctk.CTkLabel(header_frame, text="Author: Naqash Afzal", font=("Arial", 14), text_color="#aaaaaa")
author_label.pack(side="right", padx=10, pady=5)

# --- MAIN TAB ---
main=tabs.add("Main")
main.columnconfigure(0, weight=1)
main.columnconfigure(1, weight=1)


# Left side settings panel
settings_frame = ctk.CTkFrame(main, corner_radius=10)
settings_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

ctk.CTkLabel(settings_frame,text="Enter Topic",font=("Arial",16)).pack(pady=(10,0))
topic_entry=ctk.CTkEntry(settings_frame,width=400,placeholder_text="Type custom topic"); topic_entry.pack(pady=5)
ctk.CTkLabel(settings_frame,text="Start Pipeline From:", font=("Arial",12)).pack(pady=5)
start_step_combo = ctk.CTkComboBox(settings_frame, values=["Deep Research", "Generate SEO Metadata", "Podcast Script", "Audio (TTS)", "Caption Generation", "Video Generation", "Final Video Creation"], width=300); start_step_combo.set("Deep Research"); start_step_combo.pack(pady=5)
ctk.CTkLabel(settings_frame,text="TTS Mode", font=("Arial",12)).pack(pady=5)
tts_mode_combo=ctk.CTkComboBox(settings_frame,values=["Multi-Speaker","Single-Speaker"],width=300); tts_mode_combo.set("Multi-Speaker"); tts_mode_combo.pack(pady=5)
ctk.CTkLabel(settings_frame,text="Single-Speaker Voice", font=("Arial",12)).pack(pady=5)
voice_combo=ctk.CTkComboBox(settings_frame,values=[f"{v} — {d}" for v,d in VOICE_OPTIONS.items()],width=300)
voice_combo.set("Kore — Energetic, youthful, clear & bright"); voice_combo.pack(pady=5)

fact_check_var = ctk.BooleanVar(value=config.get("FACT_CHECK_ENABLED", False))
ctk.CTkCheckBox(settings_frame, text="Enable Fact-Checking", variable=fact_check_var).pack(pady=5)
caption_var = ctk.BooleanVar(value=config.get("CAPTION_ENABLED", False))
ctk.CTkCheckBox(settings_frame, text="Enable Auto-Captioning", variable=caption_var).pack(pady=5)
metadata_var = ctk.BooleanVar(value=config.get("GENERATE_METADATA", False))
ctk.CTkCheckBox(settings_frame, text="Generate SEO Metadata", variable=metadata_var).pack(pady=5)

run_button = ctk.CTkButton(settings_frame,text="🚀 Run Pipeline",command=lambda: threading.Thread(target=run_pipeline,daemon=True).start(), font=("Arial", 16, "bold"), fg_color="#1f6aa5"); run_button.pack(pady=20)

stop_button = ctk.CTkButton(settings_frame, text="⏹️ Stop Pipeline", command=stop_pipeline, font=("Arial", 16, "bold"), fg_color="#c42034", hover_color="#851622", state="disabled")
stop_button.pack(pady=(5, 20))


# Right side log console
log_frame = ctk.CTkFrame(main, corner_radius=10)
log_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
log_frame.rowconfigure(7, weight=1)
log_frame.columnconfigure(0, weight=1)

ctk.CTkLabel(log_frame, text="Progress Log", font=("Arial",16)).grid(row=0, column=0, padx=10, pady=(10,5), sticky="w")
steps=["Deep Research", "Generate SEO Metadata", "Podcast Script","Fact Check","Audio (TTS)","Caption Generation", "Video Generation", "Final Video Creation"]
step_rows=[]
progress_bars=[]
for i, s in enumerate(steps):
    row=ctk.CTkFrame(log_frame, fg_color="transparent"); row.grid(row=i+1, column=0, sticky="ew", padx=10, pady=2)
    row.columnconfigure(0, weight=1)
    lbl=ctk.CTkLabel(row,text=s,anchor="w",width=250); lbl.pack(side="left",padx=5)
    mark=ctk.CTkLabel(row,text="⬜",width=60); mark.pack(side="right",padx=5)
    progress = ctk.CTkProgressBar(row, orientation="horizontal", width=200); progress.set(0); progress.pack(side="right", padx=5)
    step_rows.append((lbl,mark))
    progress_bars.append(progress)

log_box=ctk.CTkTextbox(log_frame,width=500,height=400); log_box.grid(row=len(steps)+1, column=0, padx=10, pady=(10,20), sticky="nsew")

# --- VOICES & STYLE ---
voices=tabs.add("Voices & Style")
ctk.CTkLabel(voices,text="Speaker 1 (Host Voice)").grid(row=0,column=0,sticky="w",padx=5,pady=8)
speaker1_combo=ctk.CTkComboBox(voices,values=[f"{v} — {d}" for v,d in VOICE_OPTIONS.items()],width=400)
speaker1_combo.set(f"{config['SPEAKER1']} — {VOICE_OPTIONS.get(config['SPEAKER1'],'')}"); speaker1_combo.grid(row=0,column=1)
ctk.CTkLabel(voices,text="Speaker 2 (Guest Voice)").grid(row=1,column=0,sticky="w",padx=5,pady=8)
speaker2_combo=ctk.CTkComboBox(voices,values=[f"{v} — {d}" for v,d in VOICE_OPTIONS.items()],width=400)
speaker2_combo.set(f"{config['SPEAKER2']} — {VOICE_OPTIONS.get(config['SPEAKER2'],'')}"); speaker2_combo.grid(row=1,column=1)

ctk.CTkLabel(voices,text="Host Name").grid(row=2,column=0,sticky="w",padx=5,pady=8)
host_entry=ctk.CTkEntry(voices,width=300); host_entry.insert(0,config["HOST_NAME"]); host_entry.grid(row=2,column=1)
ctk.CTkLabel(voices,text="Guest Name").grid(row=3,column=0,sticky="w",padx=5,pady=8)
guest_entry=ctk.CTkEntry(voices,width=300); guest_entry.insert(0,config["GUEST_NAME"]); guest_entry.grid(row=3,column=1)

ctk.CTkLabel(voices,text="Host Persona").grid(row=4,column=0,sticky="w",padx=5,pady=8)
host_persona_entry = ctk.CTkTextbox(voices, width=580, height=80); host_persona_entry.insert("1.0", config["HOST_PERSONA"]); host_persona_entry.grid(row=4, column=1, columnspan=3, sticky="w")
ctk.CTkLabel(voices,text="Guest Persona").grid(row=5,column=0,sticky="w",padx=5,pady=8)
guest_persona_entry = ctk.CTkTextbox(voices, width=580, height=80); guest_persona_entry.insert("1.0", config["GUEST_PERSONA"]); guest_persona_entry.grid(row=5, column=1, columnspan=3, sticky="w")

ctk.CTkLabel(voices,text="Podcast Style").grid(row=6,column=0,sticky="w",padx=5,pady=8)
style_combo=ctk.CTkComboBox(voices,values=PODCAST_STYLES,width=300); style_combo.set(config["PODCAST_STYLE"]); style_combo.grid(row=6,column=1)

ctk.CTkLabel(voices,text="Video Prompt Template").grid(row=7,column=0,sticky="nw",padx=5,pady=8)
video_style_textbox = ctk.CTkTextbox(voices, width=580, height=80)
video_style_textbox.grid(row=7, column=1, columnspan=3, sticky="w")
ctk.CTkLabel(voices, text="Use {topic} as a placeholder for the video's topic.", font=("Arial", 10)).grid(row=8, column=1, sticky="w", padx=0, pady=(0, 8))
# --- END OF REPLACEMENT ---

# --- Humanoid Filter Section ---
ctk.CTkLabel(voices, text="Humanoid Filter", font=("Arial", 16, "bold")).grid(row=9, column=0, columnspan=2, sticky="w", padx=5, pady=(20, 5))

humanoid_enabled_var = ctk.BooleanVar(value=config.get("HUMANOID_ENABLED", False))
ctk.CTkCheckBox(voices, text="Enable Humanoid Filter", variable=humanoid_enabled_var).grid(row=10, column=0, columnspan=2, pady=5, sticky="w", padx=10)

ctk.CTkLabel(voices, text="Filler Probability (0.0 - 1.0)").grid(row=11, column=0, sticky="w", padx=5, pady=5)
humanoid_prob_entry = ctk.CTkEntry(voices, width=100)
humanoid_prob_entry.insert(0, str(config.get("HUMANOID_PROBABILITY", 0.1)))
humanoid_prob_entry.grid(row=11, column=1, sticky="w", padx=5, pady=5)

# --- Language Selection Section ---
ctk.CTkLabel(voices, text="Language Settings", font=("Arial", 16, "bold")).grid(row=12, column=0, columnspan=2, sticky="w", padx=5, pady=(20, 5))

language_enabled_var = ctk.BooleanVar(value=config.get("LANGUAGE_ENABLED", False))
ctk.CTkCheckBox(voices, text="Enable Language Selection", variable=language_enabled_var).grid(row=13, column=0, columnspan=2, pady=5, sticky="w", padx=10)

ctk.CTkLabel(voices, text="Podcast Language").grid(row=14, column=0, sticky="w", padx=5, pady=5)
language_combo = ctk.CTkComboBox(voices, values=["English", "Spanish", "French", "German", "Italian", "Portuguese","Urdu","Hindi","Dutch"], width=300)
language_combo.set(config.get("PODCAST_LANGUAGE", "English"))
language_combo.grid(row=14, column=1, sticky="w", padx=5, pady=5)

# --- BRANDING & SUBSCRIBE ---
brand=tabs.add("Branding & Subscribe")
ctk.CTkLabel(brand,text="Channel Name").grid(row=0,column=0,sticky="w",padx=5,pady=8)
channel_entry=ctk.CTkEntry(brand,width=300); channel_entry.insert(0,config["CHANNEL_NAME"]); channel_entry.grid(row=0,column=1)

ctk.CTkLabel(brand,text="Subscribe Reminder Count").grid(row=1,column=0,sticky="w",padx=5,pady=8)
sub_count_entry=ctk.CTkEntry(brand,width=100); sub_count_entry.insert(0,str(config["SUBSCRIBE_COUNT"])); sub_count_entry.grid(row=1,column=1)

ctk.CTkLabel(brand,text="Subscribe Message").grid(row=2,column=0,sticky="w",padx=5,pady=8)
sub_message_entry=ctk.CTkEntry(brand,width=580); sub_message_entry.insert(0,config["SUBSCRIBE_MESSAGE"]); sub_message_entry.grid(row=2,column=1)

subscribe_random_var=ctk.BooleanVar(value=config["SUBSCRIBE_RANDOM"])
subscribe_random_check=ctk.CTkCheckBox(brand,text="Random Placement",variable=subscribe_random_var); subscribe_random_check.grid(row=3,column=0,columnspan=2,pady=5,sticky="w")

ctk.CTkLabel(brand,text="Gemini API Key").grid(row=4,column=0,sticky="w",padx=5,pady=8)
gemini_key_entry=ctk.CTkEntry(brand,width=580,show="*"); gemini_key_entry.insert(0,config["GEMINI_API_KEY"]); gemini_key_entry.grid(row=4,column=1)

ctk.CTkLabel(brand,text="WaveSpeed AI Key").grid(row=5,column=0,sticky="w",padx=5,pady=8)
wavespeed_key_entry=ctk.CTkEntry(brand,width=580,show="*"); wavespeed_key_entry.insert(0,config["WAVESPEED_AI_KEY"]); wavespeed_key_entry.grid(row=5,column=1)


ctk.CTkButton(brand,text="💾 Save Settings",command=save_keys).grid(row=6,column=0,columnspan=3,pady=10)


# --- PUBLISH TAB ---
publish=tabs.add("Publish")
publish.columnconfigure((0, 1, 2, 3), weight=1, uniform="a")

# --- Video Metadata Section ---
metadata_frame = ctk.CTkFrame(publish, fg_color="transparent")
metadata_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=(10,5))
ctk.CTkLabel(metadata_frame, text="Video Metadata", font=("Arial", 16, "bold")).pack(side="left")
ctk.CTkButton(metadata_frame, text="✨ Generate SEO", command=generate_seo_only).pack(side="left", padx=10)

ctk.CTkLabel(publish, text="Video Title:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
video_title_entry = ctk.CTkEntry(publish, width=400, placeholder_text="Enter video title")
video_title_entry.grid(row=1, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

ctk.CTkLabel(publish, text="Description:").grid(row=2, column=0, sticky="nw", padx=10, pady=5)
video_desc_entry = ctk.CTkTextbox(publish, width=400, height=120)
video_desc_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

ctk.CTkLabel(publish, text="Tags (comma-separated):").grid(row=3, column=0, sticky="w", padx=10, pady=5)
video_tags_entry = ctk.CTkEntry(publish, width=400, placeholder_text="tag1, tag2, tag3")
video_tags_entry.grid(row=3, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

# --- Upload Section ---
ctk.CTkLabel(publish, text="--- Upload to Social Media ---", font=("Arial", 16, "bold")).grid(row=4, column=0, columnspan=4, pady=(20,5))

# Video File Selection
ctk.CTkLabel(publish, text="Video File:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
video_path_entry = ctk.CTkEntry(publish, width=300, placeholder_text="Leave blank to use generated video")
video_path_entry.grid(row=5, column=1, columnspan=2, sticky="ew", padx=10, pady=5)
ctk.CTkButton(publish, text="Browse...", command=lambda: browse_file(video_path_entry)).grid(row=5, column=3, sticky="w", padx=10, pady=5)

youtube_upload_button = ctk.CTkButton(publish, text="Upload to YouTube", command=upload_youtube, fg_color="#FF0000", hover_color="#8c0303", text_color="white")
youtube_upload_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
facebook_upload_button = ctk.CTkButton(publish, text="Upload to Facebook", command=upload_facebook, fg_color="#1877F2", hover_color="#1456b3", text_color="white")
facebook_upload_button.grid(row=6, column=2, columnspan=2, padx=10, pady=10)

# --- API Credentials Section ---
ctk.CTkLabel(publish, text="--- API Credentials ---", font=("Arial", 16, "bold")).grid(row=7, column=0, columnspan=4, pady=(20,5))
ctk.CTkLabel(publish, text="YouTube Client ID:").grid(row=8, column=0, sticky="w", padx=10, pady=5)
youtube_id_entry = ctk.CTkEntry(publish, width=400, show="*", placeholder_text="Enter YouTube Client ID")
youtube_id_entry.grid(row=8, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

ctk.CTkLabel(publish, text="YouTube Client Secret:").grid(row=9, column=0, sticky="w", padx=10, pady=5)
youtube_secret_entry = ctk.CTkEntry(publish, width=400, show="*", placeholder_text="Enter YouTube Client Secret")
youtube_secret_entry.grid(row=9, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

ctk.CTkLabel(publish, text="Facebook Access Token:").grid(row=10, column=0, sticky="w", padx=10, pady=5)
facebook_token_entry = ctk.CTkEntry(publish, width=400, show="*", placeholder_text="User/Page access token")
facebook_token_entry.grid(row=10, column=1, columnspan=3, sticky="ew", padx=10, pady=5)

# --- HISTORY TAB ---
history_tab_frame = tabs.add("History")

# --- ABOUT FRAME ---
about=tabs.add("About")
ctk.CTkLabel(about, text="Nullpk Content Automation", font=("Arial", 20, "bold")).pack(pady=(20, 5))
ctk.CTkLabel(about, text="Version 1.0", font=("Arial", 12)).pack(pady=0)
ctk.CTkLabel(about, text="Author: Naqash Afzal", font=("Arial", 12)).pack(pady=5)
ctk.CTkLabel(about, text="""
This tool automates the creation of YouTube videos by combining AI-generated podcast scripts,
text-to-speech audio, and animated video clips. It leverages the power of
the Gemini and WaveSpeed AI APIs to streamline the entire content creation process.
""", wraplength=500, justify="center").pack(pady=20)
ctk.CTkLabel(about, text="Support the project:", font=("Arial", 14, "bold")).pack(pady=5)
donate_button = ctk.CTkButton(about, text="Donate Now", command=lambda: open_url("https://nullpk.com/donate"))
donate_button.pack(pady=10)

# This needs to be called after the GUI is built
app.after(100, load_settings)
app.after(200, update_history_tab)
app.mainloop()
