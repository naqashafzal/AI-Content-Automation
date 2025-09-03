# 🎬 Nullpk YT Automation Tool

A powerful desktop application to **automate the entire workflow of creating engaging YouTube videos**.  
This tool leverages **AI** to handle research, scriptwriting, voice-overs, video creation, and captioning — turning a single topic into a ready-to-publish video.

---

## ☕ Support the Project

If you find this tool useful, please consider supporting its development.  
Your support helps keep the project alive!  

<a href="https://nullpk.com/donate" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="60">
</a>

---

## ▶️ Demo

*(A GIF showing the application running from topic input to final video generation)*

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation & Usage](#-installation--usage)
  - [Method 1: Running from Source (Developer Mode)](#method-1-running-from-source-developer-mode)
  - [Method 2: Building a Secure 1-Click Executable](#method-2-building-a-secure-1-click-executable)
- [User Guide](#-user-guide)
- [License](#-license)

---

## 📖 Overview

This tool is built for **content creators and marketers** looking to streamline their video production pipeline.  
By integrating **Google's Gemini AI** for text/audio and **WaveSpeed AI** for video synthesis, it automates the most time-consuming parts of video creation, producing a **professional-quality video in a fraction of the time**.

---

## ✨ Key Features

- 🤖 **AI-Powered Workflow** – From research to final render, AI manages the heavy lifting.  
- 🔍 **Deep Topic Research** – Uses Google Search via Gemini API for fresh, relevant information.  
- 📝 **SEO & Scriptwriting** – Generates SEO-optimized metadata and a natural, two-person podcast script.  
- 🎤 **Multi-Speaker TTS** – Converts the script into high-quality audio with distinct, professional voices.  
- 🎬 **AI Video Generation** – Creates a dynamic, animated background video relevant to the topic.  
- ✍️ **Automatic Captions** – Transcribes audio and generates styled, word-by-word `.ass` caption files.  
- 🛠️ **Full Customization** – Configure voices, personas, branding, and subscribe messages.  
- 🔒 **Secure & Portable** – Includes scripts to package the app into a single, encrypted executable.  

---

## 🛠️ Prerequisites

- **Python 3.8+** – Ensure Python is installed and added to your system's PATH.  
- **FFmpeg** – Required for merging audio and video.  
  - Download from [ffmpeg.org](https://ffmpeg.org).  
  - Ensure the `ffmpeg` executable location is in your system's PATH.  
- **API Keys**:  
  - Google Gemini API → Get from [Google AI Studio](https://aistudio.google.com).  
  - WaveSpeed AI API → Get from the WaveSpeed AI Platform.  

---

## ⚡ Installation & Usage

### Method 1: Running from Source Code (Developer Mode)

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
