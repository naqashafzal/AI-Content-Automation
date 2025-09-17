"""
api_clients.py

This module centralizes all interactions with external APIs into dedicated classes.
(Updated to use Imagen 2 for video generation instead of Veo 3).
"""
import requests
import time
import base64
import wave
import json
import logging
from functools import wraps
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

# Safely import Google Cloud libraries for Vertex AI
try:
    from google.cloud import aiplatform
    import vertexai
    from vertexai.generative_models import GenerativeModel # Still needed for Gemini text models
    # --- UPDATED IMPORT: Added VideoGenerationModel ---
    from vertexai.preview.vision_models import ImageGenerationModel, VideoGenerationModel
except ImportError:
    logging.warning("Failed to import Google Cloud libraries. Vertex AI functionality will be disabled.")
    aiplatform = None
    vertexai = None
    ImageGenerationModel = None
    VideoGenerationModel = None # <-- ADDED this

# --- API Constants (As specified by user) ---
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
WAVESPEED_T2V_API_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/t2v-480p-ultra-fast"
WAVESPEED_POLL_URL = "https://api.wavespeed.ai/api/v3/predictions/{}/result"
# --- VEO_MODEL_ID constant removed, as we are now using the Imagen 2 model via VideoGenerationModel ---


def handle_api_errors(func):
    """A decorator to catch and handle common API errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except google_exceptions.ResourceExhausted as e:
            if "aiplatform.googleapis.com" in str(e):
                error_message = f"Vertex AI Quota Exceeded: {e.message}. Ensure your project region is set correctly in the app's settings."
            else:
                error_message = f"Gemini API Quota Exceeded: {e.message}. Please check usage or billing."
            logging.error(error_message, exc_info=True)
            raise RuntimeError(error_message) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                error_message = "API rate limit exceeded. Please wait and try again."
                logging.error(error_message, exc_info=True)
                raise RuntimeError(error_message) from e
            raise
    return wrapper

class NewsApiClient:
    """Client for interacting with the NewsAPI."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def get_news(self, topic: str) -> str:
        if not self.api_key:
            logging.warning("News API key is not configured. Skipping news gathering.")
            return ""
        try:
            params = {'q': topic, 'apiKey': self.api_key}
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if articles:
                formatted_news = "\n\n--- Recent News Articles ---\n"
                for i, article in enumerate(articles[:3]):
                    formatted_news += f"Article {i+1}: {article.get('title', '')}\n"
                    formatted_news += f"   - {article.get('description', '')}\n"
                return formatted_news
            return ""
        except Exception as e:
            logging.error(f"Could not retrieve news from NewsAPI: {e}")
            return ""

class GoogleClient:
    """Client for all Google Generative AI interactions."""
    def __init__(self, api_key, project_id=None, location=None):
        if not api_key:
            raise ValueError("Google API key is missing.")
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.text_model = genai.GenerativeModel(GEMINI_TEXT_MODEL, safety_settings=self.safety_settings)
        self.project_id = project_id
        self.location = location

    @handle_api_errors
    def deep_research(self, topic: str, language: str, news_client: NewsApiClient) -> str:
        logging.info(f"Conducting advanced deep research for '{topic}'...")
        external_data = news_client.get_news(topic)
        
        language_instruction = ""
        if language and language.lower() == 'urdu':
            language_instruction = "All output text must be written in Roman Urdu."
        elif language:
            language_instruction = f"All output text must be written in {language}."

        logging.info("Research Step 1: Identifying key facets and sub-topics...")
        facet_prompt = (
            f"Using Google Search, perform a deep analysis of the topic '{topic}'. "
            "Identify: "
            "1. Key sub-topics and foundational concepts. "
            "2. The main individuals, companies, or entities involved. "
            "3. The primary points of controversy, debate, or public questions. "
            f"Format this analysis as a structured brief. {language_instruction}"
        )
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": facet_prompt}]}], "tools": [{"google_search": {}}]}
        
        try:
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            facet_analysis = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            if not facet_analysis: raise ValueError("No text content found in Gemini research (Facet Analysis).")
        except Exception as e:
            logging.error(f"Failed during Research Step 1 (Facet Analysis): {e}")
            raise RuntimeError(f"Failed research step 1: {e}")

        logging.info("Research Step 2: Synthesizing final summary...")
        synthesis_prompt = (
            f"You are a research analyst. Your goal is to create a single, comprehensive, and well-structured summary on the topic '{topic}'. "
            "You must synthesize the following two sources of information: "
            f"\nSOURCE 1: A preliminary analysis of the topic's key facets:\n---START SOURCE 1---\n{facet_analysis}\n---END SOURCE 1---"
            f"\nSOURCE 2: A feed of recent news headlines:\n---START SOURCE 2---\n{external_data}\n---END SOURCE 2---"
            "\nYOUR TASK: "
            "Using ONLY the information provided in the sources above, write a detailed, synthesized summary. This summary must cover the topic's background, why it is trending, key facts, primary controversies/debates, and the future outlook. "
            "Ensure the summary is well-organized, factually dense, and coherent. "
            f"{language_instruction}"
        )
        
        response = self.text_model.generate_content(synthesis_prompt)
        return response.text

    @handle_api_errors
    def generate_seo_metadata(self, topic: str, context: str) -> dict:
        logging.info("Generating expert SEO metadata from context...")
        prompt = f"""
        Act as a world-class YouTube SEO strategist. Your task is to generate a complete, optimized metadata package for a video based on its final script.
        CRITICAL INSTRUCTIONS:
        1.  **Title:** Create a title that is keyword-rich at the beginning, creates intrigue, uses power words/numbers, and is under 70 characters.
        2.  **Description:** Write a 3-paragraph description. The first sentence must be a captivating hook with the main keywords. The rest should summarize the key points discussed in the context provided.
        3.  **Tags:** Generate 10-15 comma-separated tags, mixing broad and specific (long-tail) keywords. The first tag must be the main keyword.
        4.  **Output Format:** Your response MUST be a single, valid JSON object and nothing else. Do not include intros, explanations, or code blocks.
            -   JSON must have keys: "title", "description", "tags".
            -   **DO NOT** include timestamps in this output.
        **VIDEO TOPIC:** {topic}
        **CONTEXT (Full Script or Research Summary):**
        ```
        {context}
        ```
        Generate the complete JSON metadata package now.
        """
        response = self.text_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logging.warning("Initial JSON parsing failed for SEO. Attempting to extract and clean.")
            try:
                text = response.text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON object found in the SEO response.")
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to decode JSON for SEO after fallback: {e}")
                return {"title": topic, "description": "Failed to generate description.", "tags": topic.replace(" ", ",")}

    @handle_api_errors
    def generate_podcast_script(self, topic:str, research:str, config: dict)->str:
        logging.info("Generating podcast script...")
        
        content_style = config.get("CONTENT_STYLE", "Podcast")
        is_podcast_mode = (content_style == "Podcast")
        style = config.get("PODCAST_STYLE", "Informative News")
        script_length = config.get("SCRIPT_LENGTH", "Medium (~5 minutes)")
        story_arc = config.get("STORY_ARC", "None")

        style_instructions = {
            "Informative News": "Adopt a balanced, journalistic tone. Focus on clarity, factual accuracy, and presenting key information concisely.",
            "Comedy / Entertaining": "Inject humor, witty banter, and playful disagreements. Use exaggeration and amusing analogies.",
            "Educational / Explainer": "Break down complex topics into simple, understandable segments. Use analogies and real-world examples.",
            "Motivational / Inspiring": "Use powerful, uplifting language. Build towards an inspiring conclusion. Share personal anecdotes.",
            "Casual Conversational": "Create a relaxed, 'friends chatting' vibe. The dialogue should be natural, with slang and overlapping thoughts.",
            "Serious Debate": "Construct a structured argument with clear points and counterpoints. The tone should be formal and persuasive.",
            "Story Mode": "Narrate a compelling story. Use descriptive language to build atmosphere and suspense.",
            "Documentary": "Follow a formal, narrative structure. Act as a narrator guiding the listener with clarity and authority.",
            "ASMR": "Focus on soft, gentle, and relaxing words. Emphasize crisp consonant sounds and create a calming atmosphere."
        }
        
        language_instruction = "The entire script must be in English."
        if config.get("LANGUAGE_ENABLED", False):
            language = config.get("PODCAST_LANGUAGE", "English")
            if language.lower() == 'urdu': language_instruction = "The entire script must be in Roman Urdu."
            else: language_instruction = f"The entire script must be in {language}."
        
        length_instruction = f"- The total word count must be appropriate for a '{script_length}' spoken podcast."
        story_arc_prompt = f"- Structure the script to follow the '{story_arc}' narrative arc." if story_arc != "None" else ""
        
        if is_podcast_mode:
            logging.info("Generating DUAL-SPEAKER script for Podcast mode.")
            host, guest = config.get("HOST_NAME","Alex"), config.get("GUEST_NAME","Maya")
            host_persona, guest_persona = config.get("HOST_PERSONA"), config.get("GUEST_PERSONA")
            sub_count, sub_message = config.get("SUBSCRIBE_COUNT"), config.get("SUBSCRIBE_MESSAGE").replace("{channel}", config.get("CHANNEL_NAME","My AI Channel"))
            placement_instruction = (f"- Insert about {sub_count} reminders randomly within the host's dialogue." if config.get("SUBSCRIBE_RANDOM") else f"- Insert exactly {sub_count} reminders evenly spaced.")
            
            prompt = f"""
            You are an expert podcast scriptwriter for a show named "{config.get("CHANNEL_NAME")}". Your task is to write a script that sounds 100% natural and unscripted.
            **TOPIC:** {topic}
            **HOSTS & PERSONAS:**
            - **Host ({host}):** {host_persona}
            - **Guest ({guest}):** {guest_persona}
            **TONE & STYLE ({style}):**
            - **Core Instruction:** {style_instructions.get(style)}
            **DIALOGUE DYNAMICS (CRITICAL):**
            - Simulate a real, spontaneous conversation. Use natural fillers ("Right," "Wow," "So...").
            - Keep speaking turns to 2-3 sentences.
            - The host must react, not just ask questions.
            - Script natural interjections and interruptions.
            **REQUIRED STRUCTURE:**
            - A. Cold Open/Hook
            - B. Introduction
            - C. Main Discussion
            - D. Conclusion
            - E. Outro
            **AUDIENCE ENGAGEMENT:**
            - Reminder Message: "{sub_message}"
            - {placement_instruction} (Do not place in intro/outro).
            **FORMATTING & LANGUAGE RULES:**
            - EVERY line must start with either `{host}:` or `{guest}:`.
            - Hosts must take turns speaking.
            - {language_instruction}
            - {length_instruction}
            {story_arc_prompt}
            **RESEARCH MATERIAL TO USE:**
            ```
            {research}
            ```
            Generate the complete script now.
            """
        else: # For Documentary, Story, ASMR, etc.
            logging.info(f"Generating SINGLE-SPEAKER script for {content_style} mode.")
            narrator_persona = config.get("HOST_PERSONA")
            
            prompt = f"""
            You are an expert scriptwriter for a {content_style} video. Your task is to write a compelling narration.
            **TOPIC:** {topic}
            **NARRATOR & PERSONA:**
            - **Narrator:** {narrator_persona}
            **TONE & STYLE ({style}):**
            - **Core Instruction:** {style_instructions.get(style)}
            **SCRIPT REQUIREMENTS:**
            - This is a single-voice narration.
            - Use descriptive language suitable for the content style: '{content_style}'.
            - Use punctuation to guide vocal delivery and rhythm.
            **REQUIRED STRUCTURE:**
            - A. Hook
            - B. Introduction
            - C. Main Body (detailed exploration of research)
            - D. Conclusion
            **FORMATTING & LANGUAGE RULES:**
            - The output must be the raw script text only. DO NOT use any speaker prefixes like "Narrator:".
            - {language_instruction}
            - {length_instruction}
            {story_arc_prompt}
            **RESEARCH MATERIAL TO USE:**
            ```
            {research}
            ```
            Generate the complete narration script now.
            """

        response = self.text_model.generate_content(prompt)
        return response.text

    @handle_api_errors
    def generate_image_prompt_for_segment(self, content_style: str, topic: str, script_segment: str) -> str:
        logging.info(f"Generating image prompt for segment based on style: '{content_style}'...")
        base_prompt = f"A high-quality image for a {content_style} about: {topic}."
        refinement_prompt = (
            f"You are an expert at creating image prompts. Your task is to generate a single, concise, and visually descriptive prompt for an AI image generator. "
            f"The image should visually represent the key idea from the following script segment: '{script_segment}'. "
            f"The overall style of the image must be: '{base_prompt}'. "
            f"**CRITICAL SAFETY RULE:** The generated prompt must be safe for work and adhere to AI safety policies. Do not include direct references to violence, gore, or specific controversial political figures. Instead, focus on creating a visually symbolic or thematic representation. For example, instead of a specific political conflict, describe 'a shattered chess board' or 'two opposing flags on a stormy background'."
            "Output ONLY the final image prompt."
        )
        response = self.text_model.generate_content(refinement_prompt)
        return response.text.strip().replace('"', '')

    @handle_api_errors
    def generate_thumbnail_prompts(self, topic: str, title_text: str) -> dict:
        logging.info("Generating dynamic prompts for split-screen thumbnail (character only)...")
        
        # --- FIX: This prompt asks for ONE prompt for the character photo. ---
        # The text side will be handled by ffmpeg in the pipeline.
        prompt = f"""
        Act as a viral YouTube thumbnail designer. Generate ONE descriptive image prompt for a photorealistic, emotional character relevant to the topic.
        
        Focus on the *emotion* (shock, surprise, thought) and the *theme* of the topic. The character must be safe for work.
        
        VIDEO TOPIC: {topic}
        
        Your entire response MUST be a single, valid JSON object with one key: "character_prompt".
        """
        
        response = self.text_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON for thumbnail prompts. Raw response: {response.text}")
            # Updated the fallback
            return {
                "character_prompt": f"A photorealistic, cinematic close-up of a person looking shocked and amazed, reacting to the theme of '{topic}'."
            }

    @handle_api_errors
    def generate_chapter_titles(self, script: str) -> list:
        logging.info("Identifying logical chapter titles from script...")
        prompt = f"""
        You are a video editor. Read the following podcast script. Your task is to identify 5-10 main logical chapters or topic shifts in the conversation.
        The first chapter MUST be "Intro".
        Return ONLY a valid JSON list of strings and nothing else. Do not add explanations.
        Example: ["Intro", "The Early Days", "A Surprising Discovery", "Conclusion"]
        --- SCRIPT ---
        {script}
        --- END SCRIPT ---
        Generate the JSON list of chapter titles now.
        """
        response = self.text_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        try:
            text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logging.error(f"Failed to parse chapter titles JSON. Raw: {getattr(response, 'text', 'NO TEXT')}")
            return ["Intro"]

    # --- gemini_nanobanana_image function REMOVED to prevent crashes ---

    @handle_api_errors
    def vertex_nanobanana_image(self, prompt: str, output_path: str):
        if not vertexai or not ImageGenerationModel:
            raise RuntimeError("Vertex AI libraries not installed correctly.")
        
        if not self.project_id or not self.location:
            raise ValueError("GCP Project ID and Location must be set in Settings to use Vertex AI Image Generation.")
            
        logging.info(f"Generating image with Vertex AI (Imagen 3): '{prompt}'")
        vertexai.init(project=self.project_id, location=self.location)
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002") # Using a known stable model
        response = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="16:9")
        response[0].save(location=output_path, include_generation_parameters=True)
        logging.info(f"Vertex AI image successfully saved to {output_path}")

    @handle_api_errors
    def fact_check_script(self, script: str, language: str) -> str:
        logging.info("Fact-checking script...")
        language_instruction = "Your entire response must be in English."
        if language and language.lower() == 'urdu': language_instruction = "Your entire response must be in Roman Urdu."
        elif language: language_instruction = f"Your entire response must be in {language}."
        prompt = f"Review the script for factual accuracy. List issues and suggest corrections.\n{language_instruction}\n\nScript:\n{script}"
        response = self.text_model.generate_content(prompt); return response.text

    @handle_api_errors
    def revise_script(self, script: str, fact_check_results: str) -> str:
        logging.info("Revising script based on fact-check...")
        prompt = f"Revise the script based on the fact-check. Output only the revised script.\n\nFact-Check:\n{fact_check_results}\n\nOriginal Script:\n{script}"
        response = self.text_model.generate_content(prompt); return response.text

    @handle_api_errors
    def generate_tts(self, script: str, output_path: str, tts_config: dict):
        logging.info("Generating audio with real Gemini TTS...")
        
        script_for_api = script.split('Text :')[-1].strip() if 'Text :' in script else script
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={self.api_key}"
        
        is_podcast_mode = tts_config.get("CONTENT_STYLE") == "Podcast"
        host_name = tts_config.get("HOST_NAME", "Alex")
        guest_name = tts_config.get("GUEST_NAME", "Maya")
        
        is_multi_speaker_script = is_podcast_mode and f"{host_name}:" in script_for_api and f"{guest_name}:" in script_for_api
        if is_multi_speaker_script:
            logging.info("Multi-speaker script detected. Forcing multi-speaker TTS mode.")
        else:
            logging.info("Single-speaker script detected. Using single voice.")

        CHUNK_SIZE_LIMIT = 4500
        script_lines = script_for_api.split('\n')
        script_chunks = []
        current_chunk = ""

        for line in script_lines:
            if len(current_chunk) + len(line) + 1 > CHUNK_SIZE_LIMIT:
                if current_chunk: script_chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk = f"{current_chunk}\n{line}" if current_chunk else line
        if current_chunk: script_chunks.append(current_chunk)

        if len(script_chunks) > 1:
            logging.info(f"Script is long, splitting into {len(script_chunks)} chunks to ensure quality.")

        all_audio_data = []

        for i, chunk in enumerate(script_chunks):
            logging.info(f"Generating audio for chunk {i+1}/{len(script_chunks)}...")
            payload = {
                "contents": [{"parts": [{"text": chunk}]}],
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            if is_multi_speaker_script:
                 payload["generationConfig"]["speechConfig"] = {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
                    {"speaker": host_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": tts_config["SPEAKER1"]}}},
                    {"speaker": guest_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": tts_config["SPEAKER2"]}}}
                ]}}
            else:
                 payload["generationConfig"]["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": tts_config.get("SPEAKER1", "Kore")}}}

            
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            resp_json = response.json()
            candidates = resp_json.get("candidates", [])
            if not candidates: raise RuntimeError(f"TTS failed: No candidates in response. {resp_json}")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts: raise RuntimeError(f"TTS failed: No content parts. Finish Reason: '{candidates[0].get('finishReason', 'UNKNOWN')}'.")

            audio_data = parts[0].get("inlineData", {}).get("data")
            if not audio_data: raise RuntimeError(f"TTS failed: No audio data. API may have returned text: '{parts[0].get('text', '')}'")
            
            all_audio_data.append(base64.b64decode(audio_data))

        logging.info("All audio chunks generated. Combining into a single file...")
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            for audio_data in all_audio_data:
                wf.writeframes(audio_data)

    @handle_api_errors
    def generate_video_prompt(self, topic: str, research: str, style_guide: str) -> str:
        logging.info("Generating dynamic video prompt...")
        prompt = f"""
        You are an expert prompt engineer for a text-to-video AI model. 
        Your task is to create a SINGLE, highly descriptive video prompt.
        RULES:
        1. Prompt must be concise (under 100 words).
        2. Must visually describe a scene, not just name concepts.
        3. Must incorporate the overall style instructions.
        4. Output ONLY the final video prompt and nothing else.
        TOPIC: {topic}
        STYLE INSTRUCTIONS: {style_guide}
        RESEARCH SUMMARY (for context):
        {research}
        Generate the final, descriptive video prompt now.
        """
        response = self.text_model.generate_content(prompt)
        return response.text.strip().replace('"', '')

    # --- FUNCTION REWRITTEN to use IMAGEN 2 (as requested) ---
    @handle_api_errors
    def vertex_ai_text_to_video(self, prompt: str, output_path: str, aspect_ratio: str):
        if not vertexai or not VideoGenerationModel: 
            raise RuntimeError("Vertex AI libraries (including vision_models.VideoGenerationModel) not installed correctly.")
        
        if not self.project_id or not self.location:
            raise ValueError("GCP Project ID and Location must be set in Settings to use Vertex AI Video Generation.")

        logging.info(f"Generating video with Vertex AI (Imagen 2): '{prompt}'")
        vertexai.init(project=self.project_id, location=self.location)
        
        # This uses the specific Imagen 2 model, fulfilling the "Veo 2" request
        model = VideoGenerationModel.from_pretrained("imagen-2.0-generate-video-002")
        
        # Imagen 2 uses width/height, not an aspect ratio string. Using standard SD sizes.
        width, height = (640, 368) if aspect_ratio == "16:9 (Horizontal)" else (368, 640)

        logging.info(f"Sending video generation request to Vertex AI (Imagen 2) with size {width}x{height}...")
        
        # Call the model
        response = model.generate(
            prompt=prompt,
            generation_config={
                "width": width,
                "height": height,
                "number_of_videos": 1
            }
        )
        
        video_bytes = response[0].video_bytes
        if not video_bytes:
             raise RuntimeError("Vertex AI (Imagen 2) did not return a video.")

        with open(output_path, "wb") as f:
            f.write(video_bytes)
        logging.info(f"Vertex AI video saved to {output_path}")


class WaveSpeedClient:
    """Client for interacting with the WaveSpeed AI video generation API."""
    def __init__(self, api_key):
        self.api_key = api_key

    def text_to_video(self, prompt: str, output_path: str, size: str):
        if not self.api_key: raise ValueError("WaveSpeed AI key is missing.")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "size": size, "duration": 5}
        logging.info(f"Sending request to WaveSpeed AI with size: {size}...")
        initial_response = requests.post(WAVESPEED_T2V_API_URL, headers=headers, json=payload, timeout=120)
        initial_response.raise_for_status()
        req_id = initial_response.json()["data"]["id"]
        logging.info(f"Task submitted. Request ID: {req_id}")
        poll_url = WAVESPEED_POLL_URL.format(req_id)
        start_time = time.time()
        while time.time() - start_time < 600:
            poll_response = requests.get(poll_url, headers=headers, timeout=60)
            poll_response.raise_for_status()
            status = poll_response.json().get("data", {}).get("status")
            if status == "completed":
                logging.info("Video generation completed. Downloading...")
                video_url = poll_response.json()["data"]["outputs"][0]
                with open(output_path, "wb") as f: f.write(requests.get(video_url).content)
                logging.info(f"Video saved to {output_path}"); return
            elif status == "failed":
                raise RuntimeError(f"WaveSpeed task failed: {poll_response.json()['data'].get('error')}")
            else:
                logging.info(f"Task status is '{status}'. Waiting...")
                time.sleep(10)
        raise TimeoutError("WaveSpeed request timed out after 10 minutes.")