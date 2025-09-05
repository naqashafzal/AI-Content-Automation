<p align="center">
  <a href="https://naqashafzal.gumroad.com/coffee" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee">
  </a>
</p>

# 🎙️ Nullpk Content Automation

An **AI-powered content automation tool** that generates full podcast-style videos — from deep research to a ready-to-upload MP4 — using **Google Gemini**, **WaveSpeed AI**, **Whisper**, and **ffmpeg**.  
Includes a polished GUI (built with `customtkinter`) for one-click pipelines, voice/persona configuration, captioning, and direct uploads to YouTube & Facebook.

---

## 🔋 How to Use Tutorial:
-<a href="https://youtu.be/ZxYHexaSDwA?si=fhfWIJeohZ23fI12" > Full Video Tutorial </a>

## 🚀 Highlights

- ✅ End-to-end pipeline: Research → Script → TTS → Captions → Video → Final Merge  
- 🎛️ GUI controls for voices, personas, styles, and pipeline steps  
- 🔊 **Humanoid Voice Filter** to make TTS sound natural & conversational  
- 🔁 Save per-topic history and re-run or delete previous projects  
- 📤 Built-in YouTube & Facebook upload support (OAuth + token caching)  
- 📝 Styled `.ass` captions generated from Whisper transcription

---

## 🔥 Table of Contents

- [Features](#-features)  
- [Humanoid Voice Filter (Spotlight)](#-humanoid-voice-filter-spotlight)  
- [Requirements](#-requirements)  
- [Installation](#-installation)  
- [Configuration](#-configuration)  
- [Usage](#-usage)  
- [Output Structure](#-output-structure)  
- [Voices / Personas](#-voices--personas)  
- [Troubleshooting](#-troubleshooting)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Support / Coffee](#-support--coffee)

---

## ✨ Features
- **Multilanguage**
  - "English", "Spanish", "French", "German", "Italian", "Portuguese","Urdu","Hindi","Dutch".
- **Deep Research**  
  - Uses Gemini (with Google Search) to fetch current, in-depth context for topics and saves it as `summary.txt`.

- **SEO Metadata Generator**  
  - Generates YouTube-ready title, description and tags.

- **Podcast Script Generation**  
  - Multi-speaker conversational scripts with host/guest personas and optional subscription reminders.

- **Fact Checking (Optional)**  
  - Script-level fact-checking prompts to Gemini to flag questionable claims.

- **Text-to-Speech (TTS)**  
  - Multi-speaker or single-speaker modes; many pre-configured voice presets; produces `podcast.wav` at 24 kHz.

- **Auto-Captions (Optional)**  
  - Whisper transcription → generates styled `.ass` captions with readable chunks and word/segment timing.

- **AI Video Generation**  
  - WaveSpeed-powered visuals from configurable prompts (Cinematic, Documentary, Abstract, Futuristic, Vintage, Whiteboard).

- **Final Video Assembly**  
  - `ffmpeg` merges background video + audio + optional captions into `final_podcast_video.mp4`.

- **Publishing Tools**  
  - Upload to YouTube & Facebook with OAuth support and token caching.

- **History Management**  
  - Per-topic folders for all generated assets and an in-app delete option.

---

## 🧠 Humanoid Voice Filter (SPOTLIGHT)

> 🔊 **Make your AI-generated podcast sound human.**  
> The Humanoid Voice Filter is designed to add the small imperfections and mannerisms that make speech sound natural.

**What it does**
- Inserts **spoken fillers**: `umm`, `ah`, `you know`, `I mean`, `like`, `so`, etc.  
- Adds **non-verbal cues**: `(laugh)`, `(sigh)`, `(chuckle)`, `(gasp)`.  
- Randomizes filler placement with configurable **probability (0.0 – 1.0)**.  
- Operates at script-level before TTS so audio generation remains deterministic.

**Why use it**
- Great for conversational, casual, and interview-style podcasts.  
- Makes AI voices sound less synthetic while keeping content readable and professional.

---
## 🧠 Examples
-<a href="https://www.youtube.com/watch?v=39w6k_QlpqI&ab_channel=RealAmericaOnline" target="_blank">
    Watch Example Podcast Generated
  </a>

  ---
## ⚙️ Requirements

- **Python 3.9+**  
- System: `ffmpeg` installed and in PATH  
- Python packages (suggested; a `requirements.txt` is recommended):
  ```
  customtkinter
  requests
  google-generativeai
  openai-whisper
  pysubs2
  numpy
  google-auth-oauthlib
  google-api-python-client
  ```
- API keys:
  - **GEMINI_API_KEY** (Google Generative AI)
  - **WAVESPEED_AI_KEY** (WaveSpeed)
  - Optional: YouTube `client_secrets.json` and Facebook access token for uploads

---

## 🛠 Installation

```bash
# clone
git clone https://github.com/yourusername/nullpk-content-automation.git
cd nullpk-content-automation

# (optional) create venv
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
# venv\Scriptsctivate

# install packages (or use pip install -r requirements.txt if provided)
pip install customtkinter requests google-generativeai whisper pysubs2 numpy google-auth-oauthlib google-api-python-client

# verify ffmpeg
ffmpeg -version
```

---

## ⚙️ Configuration

1. Run the app once to generate `config.json`, or create it manually.  
2. Open the **Branding & Subscribe** tab in the GUI and add:
   - Gemini API key
   - WaveSpeed AI key
   - Channel name, subscribe message, and other preferences
3. Optional: add `client_secrets.json` for YouTube OAuth (the app supports the OAuth flow).

Example `config.json` snippet (managed by app UI):
```json
{
  "GEMINI_API_KEY": "YOUR_KEY_HERE",
  "WAVESPEED_AI_KEY": "YOUR_KEY_HERE",
  "SPEAKER1": "Kore",
  "SPEAKER2": "Puck",
  "HOST_NAME": "Alex",
  "GUEST_NAME": "Maya",
  "HUMANOID_ENABLED": false,
  "HUMANOID_PROBABILITY": 0.1
}
```

---

## ▶️ Usage

```bash
python main.py
```

1. Enter your **Topic** in the Main tab.  
2. Choose a start step (or run full pipeline).  
3. Toggle **Fact-check**, **Captions**, and **Generate SEO** options as needed.  
4. Click **Run Pipeline** — the app will run the configured steps and display live logs.  
5. Find outputs in a folder named after your topic (e.g., `./My Topic/`).

Default pipeline:
```
Deep Research -> Generate SEO Metadata -> Podcast Script -> Fact Check -> TTS -> Caption Generation -> Video Generation -> Final Merge
```

---

## 📂 Output Structure

Each topic folder typically contains:
```
/<My Topic>/
 ├── summary.txt                # Research & notes
 ├── podcast_script.txt         # Generated script (with any humanoid fillers)
 ├── podcast.wav                # Generated TTS audio
 ├── background.mp4             # WaveSpeed-generated video
 ├── captions.ass               # Optional Whisper captions (ASS format)
 └── final_podcast_video.mp4    # Final merged video
```

---

## 🎛 Voices & Personas

The app ships with voice presets (e.g., `Kore`, `Puck`, `Despina`, `Sadachbia`) mapped to expressive characteristics. Choose host & guest voices and set short persona descriptions in the GUI.

---

## 🐞 Troubleshooting

- **ffmpeg not found**: install ffmpeg and ensure it’s in PATH.  
- **Gemini/WaveSpeed errors**: re-check keys in Branding tab and verify billing/quota on provider dashboards.  
- **Whisper transcription issues**: ensure audio is valid `.wav` and Whisper package is properly installed.  
- **YouTube upload fails**: verify `client_secrets.json` and complete OAuth flow when prompted.  
- Check `app.log` for detailed error traces.

---

## 🤝 Contributing

Contributions welcome! Suggested workflow:
1. Fork the repo  
2. Create a feature branch  
3. Implement changes & add tests/docs  
4. Open a Pull Request

Please avoid committing API keys or other secrets.

---

## 📝 License

Distributed under the **MIT License** — see `LICENSE` for details.

---

## ☕ Support / Coffee

If you enjoy this tool and want to support continued development:

<p align="center">
  <a href="https://naqashafzal.gumroad.com/coffee" target="_blank">
    <img src="https://img.shields.io/badge/☕-Support%20My%20Work-FFDD00?style=for-the-badge" alt="Buy Me A Coffee">
  </a>
</p>

---

**Author:** Naqash Afzal — *Nullpk Content Automation*
