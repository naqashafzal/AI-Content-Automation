"""
pipeline.py

This module defines the main content creation pipeline, fully implementing
all features from the original script in a structured and robust class.

The Pipeline class orchestrates the entire workflow, from research and scriptwriting
to audio/video generation, captioning, and final processing. It utilizes the
API client classes and the configuration module to customize its behavior.
"""

import os
import re
import time
import logging
import subprocess
import wave
import unicodedata

# Third-party libraries
from pydub import AudioSegment
import whisper
import pysubs2

# Import the API client classes
from api_clients import GoogleClient, WaveSpeedClient, NewsApiClient

# --- Constants for Pipeline Flow ---
PIPELINE_STEPS = [
    "Deep Research",
    "Fact Check Research",
    "Revise Research",
    "Podcast Script",
    "Generate Thumbnail",
    "Analyze Tone",
    "Audio (TTS)",
    "Generate Timed Images",
    "Video Generation",
    "Add Background Music", 
    "Create Final Video",
    "Generate SEO Metadata", # <-- CORRECTED ORDER
    "Generate Timestamps",   # <-- CORRECTED ORDER
    "Generate Snippets"
]

# --- Helper Functions (ported from original script) ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds (float) into MM:SS format."""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

def sanitize_for_tts(text: str) -> str:
    """Removes emojis and normalizes text for TTS processing."""
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
    text = text.replace('*', '')
    text = re.sub(r'[ \t\r\f\v]+', ' ', text).replace('\xa0', ' ').strip()
    return text.replace('“', '"').replace('”', '"').replace('’', "'").replace('—', '-').replace('–', '-')

def mix_audio_with_music(podcast_path, music_path, output_path, music_volume_db=-15):
    """Mixes a podcast audio file with a background music file."""
    logging.info("🎵 Mixing background music...")
    try:
        podcast_audio = AudioSegment.from_file(podcast_path)
        music_audio = AudioSegment.from_file(music_path)
        music_audio = music_audio + music_volume_db
        if len(music_audio) < len(podcast_audio):
            music_audio = music_audio * (len(podcast_audio) // len(music_audio) + 1)
        music_audio = music_audio[:len(podcast_audio)]
        mixed_audio = podcast_audio.overlay(music_audio, position=0)
        mixed_audio.export(output_path, format="mp3")
        logging.info(f"✅ Audio mixed and saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"❌ Error mixing audio: {e}")
        return podcast_path

def generate_captions(audio_file, captions_file, language: str):
    """Transcribes audio and generates a styled ASS caption file, returning word timestamps."""
    logging.info("📝 Transcribing audio for captions (this may take a moment)...")
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, word_timestamps=True, language=language)
        
        all_word_timestamps = []
        for segment in result.get("segments", []):
            all_word_timestamps.extend(segment.get("words", []))

        if captions_file:
            subs = pysubs2.SSAFile()
            style = pysubs2.SSAStyle(fontname="Arial Black", fontsize=36, primarycolor=pysubs2.Color(255, 255, 0), outline=2)
            subs.styles["Default"] = style
            for word in all_word_timestamps:
                start_time = pysubs2.make_time(s=word['start'])
                end_time = pysubs2.make_time(s=word['end'])
                text = f"{{\\b1}}{word['word'].strip().upper()}{{\\b0}}"
                subs.append(pysubs2.SSAEvent(start=start_time, end=end_time, text=text, style="Default"))
            subs.save(captions_file)
            logging.info(f"✅ Captions generated successfully at {captions_file}")
            
        return all_word_timestamps
    except Exception as e:
        logging.error(f"❌ Whisper transcription failed: {e}")
        return []

def sanitize_ffmpeg_path(path: str) -> str:
    """
    Sanitizes a file path for use in an ffmpeg filtergraph, especially for the 'ass' filter on Windows.
    """
    path = os.path.normpath(path)
    path = path.replace('\\', '/')
    if ':' in path:
        path = path.replace(':', '\\:', 1)
    return path


class Pipeline:
    """Orchestrates the entire content creation process."""

    def __init__(self, config, stop_event, status_callback, seo_callback=None, on_finish_callback=None, timestamps_callback=None):
        self.config = config
        self.stop_event = stop_event
        self.update_status = status_callback
        self.update_seo = seo_callback
        self.on_finish = on_finish_callback
        self.update_timestamps = timestamps_callback # New callback
        self.google_client = GoogleClient(config.get("GEMINI_API_KEY"), config.get("GCP_PROJECT_ID"), config.get("GCP_LOCATION"))
        self.wavespeed_client = WaveSpeedClient(config.get("WAVESPEED_AI_KEY"))
        self.news_client = NewsApiClient(config.get("NEWS_API_KEY"))

    def _check_stop(self):
        if self.stop_event.is_set():
            raise InterruptedError("Pipeline execution stopped by user.")

    def _wait(self):
        delay = self.config.get("API_DELAY", 2)
        if delay > 0:
            logging.info(f"Waiting for {delay} seconds...")
            time.sleep(delay)

    def _get_audio_length(self, path):
        try:
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            try:
                audio = AudioSegment.from_file(path)
                return len(audio) / 1000.0
            except Exception:
                return 0
    
    def _get_step_list(self, start_point: str):
        """Returns a list of steps to execute based on the start point."""
        try:
            start_index = PIPELINE_STEPS.index(start_point)
            return PIPELINE_STEPS[start_index:]
        except ValueError:
            logging.error(f"Invalid start point '{start_point}'. Defaulting to start.")
            return PIPELINE_STEPS

    def run(self, topic: str, start_point: str):
        success = False
        self.word_timestamps_for_images = []
        try:
            logging.info(f"Starting pipeline for topic: '{topic}' from step: '{start_point}'")
            safe_topic = re.sub(r'[\\/:*?"<>|]', '', topic)
            output_dir = safe_topic
            os.makedirs(output_dir, exist_ok=True)
            
            # --- File Paths ---
            summary_file = os.path.join(output_dir, "summary.txt")
            script_file = os.path.join(output_dir, "podcast_script.txt")
            audio_file = os.path.join(output_dir, "podcast.wav")
            mixed_audio_file = os.path.join(output_dir, "podcast_with_music.mp3")
            bg_video_file = os.path.join(output_dir, "background.mp4")
            segment_images_dir = os.path.join(output_dir, "segment_images")
            os.makedirs(segment_images_dir, exist_ok=True)
            final_video_file = os.path.join(output_dir, "final_podcast_video.mp4")
            image_file = os.path.join(output_dir, "generated_image.png") # Final thumbnail path
            captions_file = os.path.join(output_dir, "captions.ass")
            
            # --- Prerequisite File Checks ---
            if start_point != "Deep Research" and not os.path.exists(summary_file):
                raise FileNotFoundError(f"Cannot start from '{start_point}' because a required file is missing: {summary_file}")
            if start_point in PIPELINE_STEPS[3:] and not os.path.exists(script_file): # From Podcast Script onward
                 raise FileNotFoundError(f"Cannot start from '{start_point}' because a required file is missing: {script_file}")
            if start_point in PIPELINE_STEPS[6:] and not os.path.exists(audio_file): # From Audio (TTS) onward
                 raise FileNotFoundError(f"Cannot start from '{start_point}' because a required file is missing: {audio_file}")

            steps_to_run = self._get_step_list(start_point)
            
            research, script, seo_title, audio_len = "", "", "", 0
            generated_images_with_times = []

            # --- Research and Scripting ---
            if "Deep Research" in steps_to_run: 
                self.update_status(0, "⏳", 0.2); research = self.google_client.deep_research(topic, self.config.get("PODCAST_LANGUAGE"), self.news_client); open(summary_file, "w", encoding="utf-8").write(research); self.update_status(0, "✅", 1.0)
            elif os.path.exists(summary_file): 
                research = open(summary_file, "r", encoding="utf-8").read(); self.update_status(0, "☑️", 1.0)
            self._check_stop()

            if "Fact Check Research" in steps_to_run:
                if self.config.get("FACT_CHECK_ENABLED", False):
                    self.update_status(1, "⏳", 0.2); logging.info("Fact-checking the core research...")
                    fact_check = self.google_client.fact_check_script(research, self.config.get("PODCAST_LANGUAGE")); self.update_status(1, "✅", 1.0)
                    self._check_stop()
                    if "Revise Research" in steps_to_run:
                        self.update_status(2, "⏳", 0.2); logging.info("Revising research based on fact-check...")
                        research = self.google_client.revise_script(research, fact_check); open(summary_file, "w", encoding="utf-8").write(research); self.update_status(2, "✅", 1.0)
                else:
                    logging.info("Fact-checking is disabled. Skipping."); self.update_status(1, "⏭️", 1.0); self.update_status(2, "⏭️", 1.0)
            self._check_stop()

            if "Podcast Script" in steps_to_run: 
                self.update_status(3, "⏳", 0.2); script = self.google_client.generate_podcast_script(topic, research, self.config); open(script_file, "w", encoding="utf-8").write(script); self.update_status(3, "✅", 1.0)
            elif os.path.exists(script_file): 
                script = open(script_file, "r", encoding="utf-8").read(); self.update_status(3, "☑️", 1.0)
            self._check_stop()

            # --- Visuals and Audio ---
            if "Generate Thumbnail" in steps_to_run:
                if self.config.get("GENERATE_THUMBNAIL", False):
                    self.update_status(4, "⏳", 0.2); title_text = seo_title or topic
                    left_path, right_path = os.path.join(output_dir, "thumb_left.png"), os.path.join(output_dir, "thumb_right.png")
                    try:
                        prompts = self.google_client.generate_thumbnail_prompts(topic, title_text)
                        logging.info(f"Character Prompt: {prompts['character_prompt']}")
                        logging.info(f"Text Prompt: {prompts['text_prompt']}")
                        self.google_client.gemini_nanobanana_image(prompts['character_prompt'], left_path)
                        self._check_stop()
                        self.google_client.gemini_nanobanana_image(prompts['text_prompt'], right_path)
                        self._check_stop()
                        ffmpeg_cmd = ["ffmpeg", "-y", "-i", left_path, "-i", right_path, "-filter_complex", "[0:v]scale=960:1080,setsar=1[left];[1:v]scale=960:1080,setsar=1[right];[left][right]hstack=inputs=2", image_file]
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                        self.update_status(4, "✅", 1.0)
                    except Exception as e: 
                        logging.error(f"Thumbnail generation failed: {e}", exc_info=True); self.update_status(4, "❌", 1.0)
                else: 
                    logging.info("Thumbnail generation is disabled. Skipping."); self.update_status(4, "⏭️", 1.0)
            self._check_stop()

            if "Analyze Tone" in steps_to_run: self.update_status(5, "✅", 1.0)
            
            if "Audio (TTS)" in steps_to_run:
                self.update_status(6, "⏳", 0.2); self.google_client.generate_tts(sanitize_for_tts(script), audio_file, self.config); audio_len = self._get_audio_length(audio_file); self.update_status(6, f"✅ {audio_len:.1f}s", 1.0)
            elif os.path.exists(audio_file):
                audio_len = self._get_audio_length(audio_file); self.update_status(6, f"☑️ {audio_len:.1f}s", 1.0)
            
            # --- Post-Audio Steps ---
            need_timed_images = "Generate Timed Images" in steps_to_run and self.config.get("GENERATE_TIMED_IMAGES", False)
            need_captions = "Create Final Video" in steps_to_run and self.config.get("CAPTION_ENABLED", False)
            need_timestamps = "Generate Timestamps" in steps_to_run and self.config.get("GENERATE_TIMESTAMPS", True)
            
            if (need_timed_images or need_captions or need_timestamps):
                if not self.word_timestamps_for_images: # Only run if it hasn't been run
                    self.word_timestamps_for_images = generate_captions(audio_file, captions_file if need_captions else None, self.config.get("PODCAST_LANGUAGE"))

            if "Generate Timed Images" in steps_to_run:
                if need_timed_images:
                    self.update_status(7, "⏳", 0.2); image_interval, image_count = self.config.get("IMAGE_GENERATION_INTERVAL", 10), 0
                    for i in range(0, int(audio_len), image_interval):
                        segment_text = " ".join([w['word'] for w in self.word_timestamps_for_images if i <= w['start'] < i + image_interval])
                        if not segment_text.strip(): continue
                        image_count += 1; image_output_path = os.path.join(segment_images_dir, f"segment_{image_count:03d}.png")
                        image_prompt = self.google_client.generate_image_prompt_for_segment(self.config.get("CONTENT_STYLE"), topic, segment_text)
                        try: 
                            self.google_client.gemini_nanobanana_image(image_prompt, image_output_path); generated_images_with_times.append((i, image_output_path))
                        except Exception as e: logging.error(f"Failed to generate image {image_count}: {e}")
                        self._wait()
                    self.update_status(7, f"✅ {image_count} images", 1.0)
                else: 
                    logging.info("Timed image generation is disabled. Skipping."); self.update_status(7, "⏭️", 1.0)
            self._check_stop()
            
            if "Video Generation" in steps_to_run:
                if not self.config.get("TIMED_IMAGES_AS_SLIDESHOW", False):
                    self.update_status(8, "⏳", 0.2); aspect = self.config.get("VIDEO_ASPECT_RATIO"); prompt = self.google_client.generate_video_prompt(topic, research, self.config.get("VIDEO_PROMPT_BASE_STYLE"))
                    try:
                        if self.config.get("VIDEO_ENGINE") == "Vertex AI (Veo)": 
                            self.google_client.vertex_ai_text_to_video(prompt, bg_video_file, "16:9" if aspect == "16:9 (Horizontal)" else "9:16")
                        else: 
                            self.wavespeed_client.text_to_video(prompt, bg_video_file, "832*480" if aspect == "16:9 (Horizontal)" else "480*832")
                        self.update_status(8, "✅", 1.0)
                    except Exception as e: 
                        logging.error(f"Background video generation failed: {e}"); self.update_status(8, "❌", 1.0)
                else: 
                    self.update_status(8, "⏭️", 1.0)
            self._check_stop()

            final_audio = audio_file
            if "Add Background Music" in steps_to_run:
                if self.config.get("ADD_MUSIC", False):
                    self.update_status(9, "⏳", 0.2); final_audio = mix_audio_with_music(audio_file, "assets/background_music.mp3", mixed_audio_file); self.update_status(9, "✅", 1.0)
                else: 
                    self.update_status(9, "⏭️", 1.0)
            self._check_stop()

            if "Create Final Video" in steps_to_run:
                self.update_status(10, "⏳", 0.2); output_size = "720x1280" if self.config.get("VIDEO_ASPECT_RATIO") == "9:16 (Vertical)" else "1280x720"
                
                inputs = []            
                filter_segments = []   
                video_input_count = 0  
                final_video_map_tag = "[v_out]" 
                
                if self.config.get("TIMED_IMAGES_AS_SLIDESHOW", False) and generated_images_with_times:
                    logging.info(f"Creating SLIDESHOW video with size {output_size}")
                    concat_file_path = os.path.join(output_dir, "slideshow.txt")
                    with open(concat_file_path, "w") as f:
                        for i, (timestamp, img_path) in enumerate(generated_images_with_times):
                            duration = (generated_images_with_times[i+1][0] - timestamp) if i + 1 < len(generated_images_with_times) else (audio_len - timestamp)
                            f.write(f"file '{os.path.abspath(img_path)}'\n"); f.write(f"duration {duration}\n")
                    
                    inputs.extend(["-f", "concat", "-safe", "0", "-i", concat_file_path])
                    filter_segments.append(f"[{video_input_count}:v]scale={output_size}:force_original_aspect_ratio=decrease,pad={output_size}:(ow-iw)/2:(oh-ih)/2,setsar=1[v_out]")
                    video_input_count += 1
                
                else:
                    logging.info(f"Creating OVERLAY video with size {output_size}")
                    if os.path.exists(bg_video_file): 
                        inputs.extend(["-stream_loop", "-1", "-i", bg_video_file])
                    else: 
                        inputs.extend(["-f", "lavfi", "-i", f"color=c=black:s={output_size}:d={audio_len}"])
                    
                    base_filter_chain = f"[{video_input_count}:v]scale={output_size},setsar=1[base_scaled]"
                    video_input_count += 1
                    last_video_output = "[base_scaled]"
                    
                    overlay_chain_segment = ""
                    if generated_images_with_times:
                        for i, (_, img_path) in enumerate(generated_images_with_times):
                            inputs.extend(["-i", img_path])
                            img_stream = f"[{video_input_count}:v]"
                            video_input_count += 1
                            
                            next_output_tag = f"[v{i+1}]"
                            timestamp, end_time = generated_images_with_times[i][0], (generated_images_with_times[i+1][0] if i + 1 < len(generated_images_with_times) else audio_len)
                            
                            overlay_chain_segment += f";{last_video_output}{img_stream}overlay=(W-w)/2:(H-h)/2:enable='between(t,{timestamp},{end_time})'{next_output_tag}"
                            last_video_output = next_output_tag
                        
                        base_filter_chain += overlay_chain_segment
                    
                    filter_segments.append(base_filter_chain)
                    final_video_map_tag = last_video_output

                if need_captions and os.path.exists(captions_file): 
                    filter_segments.append(f"{final_video_map_tag}ass='{sanitize_ffmpeg_path(os.path.abspath(captions_file))}'[v_final]")
                    final_video_map_tag = "[v_final]"
                
                inputs.extend(["-i", final_audio])
                
                audio_map_index = video_input_count 
                
                ffmpeg_cmd = ["ffmpeg", "-y", *inputs]
                
                if filter_segments: 
                    final_filter_chain = ";".join(filter_segments)
                    logging.debug(f"Final Filter Chain: {final_filter_chain}")
                    ffmpeg_cmd.extend(["-filter_complex", final_filter_chain])
                
                ffmpeg_cmd.extend([
                    "-map", final_video_map_tag,
                    "-map", f"{audio_map_index}:a:0", 
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-t", str(audio_len), final_video_file
                ])
                
                logging.debug(f"Executing FFMPEG command: {' '.join(ffmpeg_cmd)}")
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                self.update_status(10, "✅", 1.0)
            else:
                self.update_status(10, "⏭️", 1.0)
            self._check_stop()

            # --- Final Metadata (RUNS BEFORE TIMESTAMPS) ---
            if "Generate SEO Metadata" in steps_to_run:
                if self.config.get("GENERATE_METADATA", False) and self.update_seo:
                    self.update_status(11, "⏳", 0.2)
                    metadata = self.google_client.generate_seo_metadata(topic, script) # Use script
                    self.update_seo(metadata)
                    self.update_status(11, "✅", 1.0)
                else: 
                    self.update_status(11, "⏭️", 1.0)

            # --- Timestamp Generation (RUNS AFTER SEO, TO APPEND) ---
            if "Generate Timestamps" in steps_to_run:
                if need_timestamps and self.update_timestamps:
                    self.update_status(12, "⏳", 0.2)
                    logging.info("Generating accurate timestamps locally...")
                    
                    # 1. Get chapter titles from AI (fast, small payload)
                    chapter_titles = self.google_client.generate_chapter_titles(script)
                    
                    if not self.word_timestamps_for_images:
                        raise ValueError("Word timestamps are missing, cannot generate chapters.")
                        
                    # 2. Build a quick-lookup dictionary of the *first* time each word appears
                    word_map = {}
                    for word_data in self.word_timestamps_for_images:
                        word = word_data['word'].strip().lower()
                        if word not in word_map:
                            word_map[word] = word_data['start']

                    timestamp_output = "\n\n--- Timestamps ---\n"
                    found_chapters = 0
                    
                    # 3. Match titles locally
                    for title in chapter_titles:
                        # Find the first word of the title (e.g., "The Early Days" -> "the")
                        first_word_of_title = re.sub(r'[^\w]', '', title.split(' ')[0]).strip().lower()
                        
                        if first_word_of_title in word_map:
                            start_seconds = word_map[first_word_of_title]
                            timestamp_output += f"{format_timestamp(start_seconds)} - {title}\n"
                            found_chapters += 1
                        else:
                            logging.warning(f"Could not find timestamp match for chapter: '{title}' (word: '{first_word_of_title}')")

                    if found_chapters > 0:
                        self.update_timestamps(timestamp_output)
                    else:
                        logging.error("Failed to match any chapter titles to timestamp data.")

                    self.update_status(12, "✅", 1.0)
                else: 
                    self.update_status(12, "⏭️", 1.0)

            # --- Snippets ---
            if "Generate Snippets" in steps_to_run:
                if self.config.get("GENERATE_SNIPPETS", False):
                    self.update_status(13, "⏳", 0.2); snippets_dir = os.path.join(output_dir, "snippets"); os.makedirs(snippets_dir, exist_ok=True)
                    is_vertical = self.config.get("VIDEO_ASPECT_RATIO", "16:9 (Horizontal)") == "9:16 (Vertical)"
                    for i in range(int(audio_len // 60)):
                        output_snippet_path = os.path.join(snippets_dir, f"snippet_{i+1}.mp4"); start_time = str(i * 60)
                        if is_vertical: 
                            ffmpeg_cmd = ["ffmpeg", "-y", "-i", final_video_file, "-ss", start_time, "-t", "60", "-c", "copy", output_snippet_path]
                        else: 
                            # FIX for "width not divisible by 2"
                            pan_scan_filter = f"scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
                            ffmpeg_cmd = ["ffmpeg", "-y", "-i", final_video_file, "-ss", start_time, "-t", "60", "-vf", pan_scan_filter, "-c:a", "copy", output_snippet_path]
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    self.update_status(13, "✅", 1.0)
                else: 
                    logging.info("Snippet generation is disabled. Skipping."); self.update_status(13, "⏭️", 1.0)
            
            logging.info("✅✅ Pipeline completed successfully! ✅✅")
            success = True

        except (InterruptedError, FileNotFoundError) as e: logging.warning(str(e))
        except subprocess.CalledProcessError as e: 
            logging.error(f"ffmpeg command failed with exit code {e.returncode}\nffmpeg stderr:\n{e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'No stderr'}")
            self.update_status_on_error()
        except Exception as e: 
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            self.update_status_on_error()
        finally:
            if self.on_finish: self.on_finish(success)

    def update_status_on_error(self):
        """Finds the currently running step and marks it as failed."""
        from main import PIPELINE_STEPS
        for i in range(len(PIPELINE_STEPS)):
            if hasattr(self.update_status, '__self__') and self.update_status.__self__.step_status_labels[i].cget("text") == "⏳":
                self.update_status(i, "❌", 1.0); break

    def _extract_voice_name(self, val): 
        return val.split(" — ")[0] if " — " in val else val
