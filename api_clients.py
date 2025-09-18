"""
api_clients.py

This module centralizes all interactions with external APIs into dedicated classes.
It handles the construction of API requests, sending them, and processing the
responses, including robust error handling. This keeps the main pipeline logic
clean and focused on orchestration rather than API specifics.
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
from google.api_core import exceptions as google_exceptions

# Safely import Google Cloud libraries for Vertex AI
try:
    from google.cloud import aiplatform
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from vertexai.preview.vision_models import ImageGenerationModel
except ImportError:
    logging.warning("Failed to import Google Cloud libraries. Vertex AI functionality will be disabled.")
    aiplatform = None
    vertexai = None
    ImageGenerationModel = None


# --- API Constants (As specified by user) ---
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
WAVESPEED_T2V_API_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.2/t2v-480p-ultra-fast"
WAVESPEED_POLL_URL = "https://api.wavespeed.ai/api/v3/predictions/{}/result"
VEO_MODEL_ID = "veo-3.0-generate-preview" # For Vertex AI Video
NANO_BANANA_IMAGE_MODEL = "gemini-2.5-flash-image-preview" # For Vertex AI Image


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
        self.text_model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
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
        facet_payload = {"contents": [{"parts": [{"text": facet_prompt}]}], "tools": [{"google_search": {}}]}
        
        try:
            response = requests.post(api_url, json=facet_payload, timeout=120)
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

        final_model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = final_model.generate_content(synthesis_prompt)
        return response.text

    @handle_api_errors
    def generate_seo_metadata(self, topic: str, script: str) -> dict:
        logging.info("Generating expert SEO metadata from final script...")
        prompt = f"""
        Act as a world-class YouTube SEO strategist. Your task is to generate a complete, optimized metadata package for a video based on its final script.

        CRITICAL INSTRUCTIONS:
        1.  **Title:** Create a title that is keyword-rich at the beginning, creates intrigue, uses power words/numbers, and is under 70 characters.
        2.  **Description:** Write a 3-paragraph description. The first sentence must be a captivating hook with the main keywords. The rest should summarize the key points discussed in the script.
        3.  **Tags:** Generate 10-15 comma-separated tags, mixing broad and specific (long-tail) keywords. The first tag must be the main keyword.
        4.  **Output Format:** Your response MUST be a single, valid JSON object and nothing else. Do not include intros, explanations, or code blocks.
            -   JSON must have keys: "title", "description", "tags".
            -   **DO NOT** include timestamps in this output.

        **VIDEO TOPIC:** {topic}
        **FULL SCRIPT (for context):**
        ```
        {script}
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
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in the SEO response.")
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to decode JSON for SEO after fallback: {e}")
                return {"title": topic, "description": "Failed to generate description.", "tags": topic.replace(" ", ",")}

    @handle_api_errors
    def generate_podcast_script(self, topic:str, research:str, config: dict)->str:
        logging.info("Generating podcast script with new 'natural conversation' model...")
        host, guest = config.get("HOST_NAME","Alex"), config.get("GUEST_NAME","Maya")
        host_persona = config.get("HOST_PERSONA", "A friendly podcast host.")
        guest_persona = config.get("GUEST_PERSONA", "An expert on the topic.")
        channel = config.get("CHANNEL_NAME","My AI Channel")
        sub_count, sub_message = config.get("SUBSCRIBE_COUNT", 3), config.get("SUBSCRIBE_MESSAGE","...").replace("{channel}", channel)
        randomize, style = config.get("SUBSCRIBE_RANDOM", True), config.get("PODCAST_STYLE", "Informative News")
        script_length, story_arc = config.get("SCRIPT_LENGTH", "Medium (~5 minutes)"), config.get("STORY_ARC", "None")
        
        style_instructions = {
            "Informative News": "Adopt a balanced, journalistic tone. Focus on clarity, factual accuracy, and presenting key information concisely. The dialogue should be professional and direct.",
            "Comedy / Entertaining": "Inject humor, witty banter, and playful disagreements. Use exaggeration and amusing analogies. The hosts should have great chemistry and sound like they are having fun.",
            "Educational / Explainer": "Break down complex topics into simple, understandable segments. Use analogies and real-world examples. The guest should act as a patient teacher, and the host should ask clarifying questions on behalf of the audience.",
            "Motivational / Inspiring": "Use powerful, uplifting language. The dialogue should build towards an inspiring conclusion. Share personal anecdotes or success stories related to the topic.",
            "Casual Conversational": "Create a relaxed, 'friends chatting over coffee' vibe. The dialogue should be natural, with some slang, interruptions, and overlapping thoughts. It should feel unscripted and authentic.",
            "Serious Debate": "Construct a structured argument with clear points and counterpoints. The hosts should challenge each other's views respectfully but firmly. The tone should be formal, intellectual, and persuasive.",
            "Story Mode": "Narrate a compelling story based on the research. Use descriptive language to build atmosphere and suspense. The dialogue should reveal the story piece by piece.",
            "Documentary": "Follow a formal, narrative structure like a classic documentary. The host acts as a narrator, guiding the listener through the topic with clarity and authority.",
            "ASMR": "The script should focus on soft, gentle, and relaxing words. Emphasize crisp consonant sounds and create a calming, quiet atmosphere. Keep sentences short and paced slowly."
        }
        
        language_instruction = "The entire script must be in English."
        if config.get("LANGUAGE_ENABLED", False):
            language = config.get("PODCAST_LANGUAGE", "English")
            if language.lower() == 'urdu': language_instruction = "The entire script, including dialogue, must be in Roman Urdu."
            else: language_instruction = f"The entire script must be in {language}."
        
        length_instruction = f"- The total word count must be appropriate for a '{script_length}' spoken podcast."
        placement_instruction = (f"- Insert about {sub_count} reminders randomly within the host's dialogue." if randomize else f"- Insert exactly {sub_count} reminders evenly spaced within the host's dialogue.")
        story_arc_prompt = f"- Structure the script to follow the '{story_arc}' narrative arc." if story_arc != "None" else ""

        prompt = f"""
        You are an expert podcast scriptwriter. Your task is to write a script that sounds 100% natural and unscripted, like a real, spontaneous conversation.
        The dialogue MUST NOT sound like two people reading prepared statements. It must sound like the hosts are truly reacting to each other in real-time.

        **1. TOPIC:** {topic}
        **2. HOSTS & PERSONAS (Embody these traits):**
        - **Host ({host}):** {host_persona}
        - **Guest ({guest}):** {guest_persona}

        **3. TONE & STYLE ({style}):**
        - **Core Instruction:** {style_instructions.get(style, "A standard, informative conversation.")}

        **4. DIALOGUE DYNAMICS (MOST CRITICAL RULES):**
        Your primary goal is to simulate a real conversation.
        - **NATURAL FILLERS:** Integrate conversational fillers NATURALLY. People say "Right," "Exactly," "Wow," "So...", "I mean," "You know," and "Hmm."
        - **SHORT TURNS:** Keep speaking turns to 2-3 sentences before the other host interjects.
        - **REACTIONS, NOT JUST QUESTIONS:** The host ({host}) must actively react to the guest ({guest}) with phrases like "Wow, that's incredible," or "Right, and that leads to..."
        - **SCRIPTED INTERRUPTIONS:** Script natural interjections. For example:
            {guest}: So the data shows a 50% increase...
            {host}: Fifty percent? Wow.
            {guest}: Exactly. And that's just in the last quarter.
        - **CADENCE:** Use punctuation (commas, ellipses) to create realistic pauses.

        **5. REQUIRED SHOW STRUCTURE:**
        - **A. Cold Open/Hook:** Start with a provocative question or a surprising fact.
        - **B. Introduction:** The host introduces the show, topic, and guest.
        - **C. Main Discussion:** The highly conversational back-and-forth based on the research.
        - **D. Conclusion:** The host summarizes the key takeaways.
        - **E. Outro:** The host thanks the guest and signs off.

        **6. AUDIENCE ENGAGEMENT:**
        - **Reminder Message:** "{sub_message}"
        - **Placement:** {placement_instruction} Do NOT place reminders in the intro or outro.

        **7. FORMATTING & LANGUAGE RULES:**
        - **Line Prefixes:** EVERY line must start with either `{host}:` or `{guest}:`.
        - **Alternating Speakers:** The hosts must take turns speaking.
        - {language_instruction}
        - {length_instruction}
        {story_arc_prompt}

        **8. RESEARCH MATERIAL TO USE:**
        ```
        {research}
        ```

        Generate the complete, 100% natural-sounding podcast script.
        """
        
        response = self.text_model.generate_content(prompt, safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
        return response.text

    @handle_api_errors
    def generate_image_prompt_for_segment(self, content_style: str, topic: str, script_segment: str) -> str:
        """
        Generates a context-aware image prompt based on the content style and a specific script segment.
        """
        logging.info(f"Generating image prompt for segment based on style: '{content_style}'...")
        
        base_prompt_map = {
            "Podcast": f"A high-quality, professional still image capturing the essence of a podcast discussion. Focus on the core theme of: {topic}.",
            "ASMR Video": f"A serene, calming image suitable for an ASMR video, subtly related to: {topic}. Soft lighting, peaceful atmosphere.",
            "Documentary": f"A powerful, visually rich documentary-style still image, representing a key moment or concept from: {topic}. Dramatic lighting, realistic.",
            "Product Ad": f"A high-end, commercial product advertisement image, highlighting a feature or benefit related to: {topic}. Clean, vibrant, appealing.",
            "Story": f"An illustrative, evocative image depicting a scene from a story about: {topic}. Focus on emotional resonance and narrative elements.",
            "Kids Story": f"A vibrant, friendly, and imaginative illustration for a children's story about: {topic}. Bright colors, whimsical style.",
            "Horror Story": f"A dark, atmospheric, and unsettling image for a horror story about: {topic}. Focus on suspense, shadows, and subtle dread.",
            "Viral Video": f"A dynamic, attention-grabbing, and slightly exaggerated image suitable for a viral video about: {topic}. Energetic, bold, and clear."
        }
        base_prompt = base_prompt_map.get(content_style, f"A high-quality image related to the topic of {topic}.")

        refinement_prompt = (
            f"You are an expert at creating image prompts. Your task is to generate a single, concise, and visually descriptive prompt for an AI image generator. "
            f"The image should visually represent the key idea from the following script segment: '{script_segment}'. "
            f"The overall style of the image must be: '{base_prompt}'. "
            "Do not output any conversational text, explanations, or quotes. Output ONLY the final image prompt."
        )
        
        response = self.text_model.generate_content(refinement_prompt)
        return response.text.strip().replace('"', '')

    @handle_api_errors
    def generate_thumbnail_prompts(self, topic: str, title_text: str) -> dict:
        """
        Generates two separate prompts for a split-screen thumbnail.
        """
        logging.info("Generating dynamic prompts for split-screen thumbnail...")
        
        prompt = f"""
        Act as a viral YouTube thumbnail designer. Your task is to generate two separate image prompts for a split-screen thumbnail based on the video's topic and title.

        The final thumbnail will have a photorealistic character on the left and bold text on the right.

        VIDEO TOPIC: {topic}
        VIDEO TITLE: {title_text}

        CRITICAL INSTRUCTIONS:
        1.  **Character Prompt:** Create a prompt for the left side. It must describe a single character (male, female, child, or animal) that is highly relevant to the topic. The character MUST have a strong, emotional expression like shock, amazement, or excitement. Describe the scene, lighting, and style (e.g., "photorealistic, cinematic lighting, close-up shot...").
        2.  **Text Prompt:** Create a prompt for the right side. It must describe a graphic design with the video title as the main text. Specify background colors (e.g., "dark navy blue with a subtle gradient"), font style (e.g., "large, bold, impactful font"), and text colors (e.g., "main text in white, with key words like 'SHOCKING' or 'FREE' highlighted in bright yellow").

        Your entire response MUST be a single, valid JSON object with two keys: "character_prompt" and "text_prompt". Do not include any other text or formatting.
        """
        
        response = self.text_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON for thumbnail prompts. Raw response: {response.text}")
            # Provide a robust fallback if JSON parsing fails
            return {
                "character_prompt": f"A photorealistic, cinematic close-up of a person looking shocked and amazed, reacting to the topic of '{topic}'.",
                "text_prompt": f"A graphic design for a YouTube thumbnail title card. A dark blue background with the text '{title_text}' in large, bold, yellow and white font."
            }
    
    # --- THIS FUNCTION REPLACES THE OLD, BROKEN ONE ---
    @handle_api_errors
    def generate_chapter_titles(self, script: str) -> list:
        """
        Analyzes the script and returns a JSON list of logical chapter titles.
        This is a fast query that does NOT send timestamp data.
        """
        logging.info("Identifying logical chapter titles from script...")
        prompt = f"""
        You are a video editor. Read the following podcast script. Your task is to identify 5-10 main logical chapters or topic shifts in the conversation.
        The first chapter MUST be "Intro".

        CRITICAL FORMATTING:
        Return ONLY a valid JSON list of strings and nothing else. Do not add explanations.

        Example Output:
        ["Intro", "The Early Days", "A Surprising Discovery", "The Final Confrontation", "Conclusion"]

        --- SCRIPT ---
        {script}
        --- END SCRIPT ---

        Generate the JSON list of chapter titles now.
        """
        
        response = self.text_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse chapter titles JSON. Raw: {response.text}")
            # Fallback to just the intro
            return ["Intro"]


    @handle_api_errors
    def gemini_nanobanana_image(self, prompt: str, output_path: str):
        """
        Generates an image using the gemini-2.5-flash-image-preview model.
        """
        logging.info(f"Generating image with Gemini API (gemini-2.5-flash-image-preview): '{prompt}'")
        model = genai.GenerativeModel(NANO_BANANA_IMAGE_MODEL)
        
        response = model.generate_content(prompt) # Corrected call
        
        image_part = response.candidates[0].content.parts[0]
        if "image" not in image_part.mime_type:
            raise RuntimeError(f"API did not return an image. It may have returned text instead: {response.text}")
        
        image_data = image_part.inline_data.data
        
        with open(output_path, "wb") as f: 
            f.write(base64.b64decode(image_data))
        
        logging.info(f"Image successfully saved to {output_path}")

    @handle_api_errors
    def vertex_nanobanana_image(self, prompt: str, output_path: str):
        """Generates an image using the stable Imagen 3 model on Vertex AI."""
        if not vertexai or not ImageGenerationModel:
            raise RuntimeError("Vertex AI libraries not installed correctly. Please run: pip install --upgrade google-cloud-aiplatform vertexai")
        
        logging.info(f"Generating image with Vertex AI (Imagen 3): '{prompt}'")
        vertexai.init(project=self.project_id, location=self.location)
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9" # This is fine, will be cropped by ffmpeg later
        )
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
        
        logging.info("Sanitizing script for TTS...")
        if 'Text :' in script:
            script_for_api = script.split('Text :')[-1].strip()
        else:
            script_for_api = script
        
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent?key={self.api_key}"
        
        is_multi_speaker_script = False
        host_name = tts_config.get("HOST_NAME", "Alex")
        guest_name = tts_config.get("GUEST_NAME", "Maya")
        
        if f"{host_name}:" in script_for_api and f"{guest_name}:" in script_for_api:
            is_multi_speaker_script = True
            logging.info("Multi-speaker script detected. Forcing multi-speaker TTS mode.")

        CHUNK_SIZE_LIMIT = 4500
        script_lines = script_for_api.split('\n')
        script_chunks = []
        current_chunk = ""

        for line in script_lines:
            if len(current_chunk) + len(line) + 1 > CHUNK_SIZE_LIMIT:
                if current_chunk:
                    script_chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
        
        if current_chunk:
            script_chunks.append(current_chunk)

        if len(script_chunks) > 1:
            logging.info(f"Script is long, intelligently splitting into {len(script_chunks)} chunks to ensure quality.")

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
                 payload["generationConfig"]["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": tts_config.get("VOICE_NAME", "Kore")}}}

            
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            resp_json = response.json()
            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"TTS failed on chunk {i+1}: No candidates in response. Full response: {resp_json}")
            
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
                error_message = (f"TTS failed on chunk {i+1}: API returned no content parts. Finish Reason: '{finish_reason}'.")
                raise RuntimeError(error_message)

            audio_data = parts[0].get("inlineData", {}).get("data")
            if not audio_data:
                text_fallback = parts[0].get("text", "")
                raise RuntimeError(f"TTS failed on chunk {i+1}: No audio data found. API may have returned a text fallback: '{text_fallback}'")
            
            all_audio_data.append(base64.b64decode(audio_data))

        logging.info("All audio chunks generated. Combining into a single file...")
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            for audio_data in all_audio_data:
                wf.writeframes(audio_data)

    @handle_api_errors
    def generate_video_prompt(self, topic: str, research: str, style_guide: str) -> str:
        """Generates a dynamic, context-aware video prompt based on research."""
        logging.info("Generating dynamic video prompt...")
        
        prompt = f"""
        You are an expert prompt engineer for a text-to-video AI model. 
        Your task is to create a SINGLE, highly descriptive video prompt.
        
        RULES:
        1. The prompt must be concise (under 100 words).
        2. It must visually describe a scene, not just name concepts.
        3. It must incorporate the overall style instructions.
        4. Output ONLY the final video prompt and nothing else.

        TOPIC: {topic}
        STYLE INSTRUCTIONS: {style_guide}
        RESEARCH SUMMARY (for context):
        {research}
        
        Generate the final, descriptive video prompt now.
        """
        
        response = self.text_model.generate_content(prompt)
        return response.text.strip().replace('"', '')

    @handle_api_errors
    def vertex_ai_text_to_video(self, prompt: str, output_path: str, aspect_ratio: str):
        if not vertexai: 
            raise RuntimeError("Vertex AI libraries not installed correctly. Please recreate your environment.")
        
        logging.info(f"Generating video with Vertex AI: '{prompt}'")
        vertexai.init(project=self.project_id, location=self.location)
        model = GenerativeModel(VEO_MODEL_ID)
        final_prompt = f"{prompt} The video must be in a {aspect_ratio} aspect ratio."
        
        logging.info("Sending video generation request to Vertex AI (this may take several minutes)...")
        response = model.generate_content(
            [final_prompt],
            generation_config={"response_mime_type": "video/mp4"}
        )

        video_part = response.candidates[0].content.parts[0]
        if "video" not in video_part.mime_type:
            raise RuntimeError(f"Vertex AI did not return a video. Response: {response.text}")
            
        video_bytes = video_part._raw_part.inline_data.data
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
