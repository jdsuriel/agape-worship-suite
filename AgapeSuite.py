import streamlit as st
import os
import glob
import re
import sys
import librosa
import numpy as np
import plotly.graph_objects as go
import whisper
import syncedlyrics
import yt_dlp
import time
import datetime
import shutil
import logging
import json
import traceback
from audio_separator.separator import Separator
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import AudioFileClip, ImageClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
from PIL import Image

# Constants
LOG_FILE = "logs/system_diagnostics.log"
REQUIRED_DIRECTORIES = ["models", "exports", "temp_stems", "logs", "assets/backgrounds"]

# ==============================================================================
# 1. ADVANCED LOGGING & SYSTEM DIAGNOSTICS (PLUS PROGRESS TRACKING)
# ==============================================================================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AgapeWorshipEngine")

# NEW: Progress Bar Class for Streamlit
class StreamlitProgressLogger:
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text

    def __call__(self, **kwargs):
        if 'index' in kwargs and 'total' in kwargs:
            progress = min(1.0, kwargs['index'] / kwargs['total'])
            self.progress_bar.progress(progress)
            self.status_text.text(f"🎬 Rendering Frame: {kwargs['index']} / {kwargs['total']} ({int(progress * 100)}%)")

def deep_clean_workspace():
    """ 
    Performs an exhaustive sanitation of the workspace to prevent memory leaks.
    """
    logger.info("Executing High-Intensity Workspace Sanitation...")
    target_extensions = [
        "*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.webm", 
        "*.srt", "*.lrc", "temp_stems/*", "exports/*.mp4", 
        "yt_*.mp3", "upload_*.mp3", "combined_*.wav"
    ]
    files_removed_count = 0
    for pattern in target_extensions:
        for file_path in glob.glob(pattern, recursive=True):
            if "bg_custom" not in file_path and "models" not in file_path and "cache" not in file_path:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_removed_count += 1
                except Exception as cleaning_error:
                    logger.error(f"Failed to eliminate artifact {file_path}: {cleaning_error}")
    
    logger.info(f"Sanitation complete. {files_removed_count} artifacts purged.")
    return files_removed_count

if 'system_initialized' not in st.session_state:
    logger.info("System Boot Sequence Initiated.")
    deep_clean_workspace()
    st.session_state.system_initialized = True

# ==============================================================================
# 2. CORE ARCHITECTURE & BINARY RESOLUTION
# ==============================================================================
def find_imagemagick_binary():
    """
    Scans the host system for the ImageMagick binary, crucial for MoviePy TextClip.
    """
    logger.info("Locating ImageMagick Binary...")
    search_priorities = [
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-*\magick.exe",
        r"/usr/bin/convert",
        r"/usr/local/bin/convert",
        r"/opt/homebrew/bin/convert",
        r"/bin/convert"
    ]
    for candidate in search_priorities:
        resolved_matches = glob.glob(candidate)
        if resolved_matches:
            magick_binary_path = resolved_matches[0]
            change_settings({"IMAGEMAGICK_BINARY": magick_binary_path})
            logger.info(f"ImageMagick confirmed at: {magick_binary_path}")
            return magick_binary_path
    
    logger.warning("ImageMagick binary not found. Text rendering may fail.")
    return None

SYSTEM_MAGICK_PATH = find_imagemagick_binary()

for dir_path in REQUIRED_DIRECTORIES:
    os.makedirs(dir_path, exist_ok=True)

# ==============================================================================
# 3. HIGH-FIDELITY LYRIC PARSING & VOCAL LEAD-IN ENGINE
# ==============================================================================
def robust_timestamp_converter(raw_timestamp):
    """
    A multi-format timestamp parser that handles standard LRC, SRT, and raw float strings.
    """
    sanitized = str(raw_timestamp).strip().replace('[', '').replace(']', '').replace(',', '.')
    try:
        if ':' in sanitized:
            segments = sanitized.split(':')
            if len(segments) == 3: # HH:MM:SS.ms
                return int(segments[0]) * 3600 + int(segments[1]) * 60 + float(segments[2])
            elif len(segments) == 2: # MM:SS.ms
                return int(segments[0]) * 60 + float(segments[1])
        return float(sanitized)
    except Exception as parse_ex:
        logger.error(f"Critical failure parsing timestamp '{raw_timestamp}': {parse_ex}")
        return 0.0

def process_lyrics_with_lead_in(raw_input_text, enable_cues=True, gap_threshold=3.0):
    """
    Analyzes the gaps between lyrics and injects visual 'Ready' cues (● ● ●) to guide the singer before entries.
    """
    logger.info("Initializing Lyric Processing Engine...")
    final_timed_segments = []
    if not raw_input_text or len(raw_input_text.strip()) < 5:
        logger.warning("No valid lyric input detected for processing.")
        return final_timed_segments

    raw_buffer = []
    # FIX: Standard LRC Format Recognition (Cleaned from KaTeX noise)
    if raw_input_text.strip().startswith('['):
        lines = raw_input_text.strip().split('\n')
        for i, current_line in enumerate(lines):
            # FIXED REGEX: Detects [mm:ss.xx] properly
            pattern_match = re.search(r'\[(\d+):(\d+\.?\d*)\](.*)', current_line)
            if pattern_match:
                start_time = int(pattern_match.group(1)) * 60 + float(pattern_match.group(2))
                lyric_content = pattern_match.group(3).strip()
                if not lyric_content: 
                    continue
                
                # Logic to determine display duration based on next line proximity
                end_time = start_time + 4.5 
                if i + 1 < len(lines):
                    next_line_match = re.search(r'\[(\d+):(\d+\.?\d*)\]', lines[i+1])
                    if next_line_match:
                        end_time = int(next_line_match.group(1)) * 60 + float(next_line_match.group(2))
                raw_buffer.append(((start_time, end_time), lyric_content))
    
    # Standard SRT Format Recognition
    else:
        srt_blocks = re.split(r'\n\s*\n', raw_input_text.strip())
        for block in srt_blocks:
            lines_in_block = block.split('\n')
            if len(lines_in_block) >= 2 and '-->' in lines_in_block[1]:
                timing_parts = lines_in_block[1].split(' --> ')
                start_time = robust_timestamp_converter(timing_parts[0])
                end_time = robust_timestamp_converter(timing_parts[1])
                text_content = " ".join(lines_in_block[2:])
                raw_buffer.append(((start_time, end_time), text_content))

    if not enable_cues:
        return raw_buffer

    # AUTOMATED CUE INJECTION LOGIC
    sequenced_output = []
    for idx, (timing_window, text_string) in enumerate(raw_buffer):
        start_val = timing_window[0]
        
        # Scenario A: Gap between existing lines
        if idx > 0:
            previous_end_val = raw_buffer[idx-1][0][1]
            if (start_val - previous_end_val) >= gap_threshold:
                # Inject a 2.5 second countdown cue
                sequenced_output.append(((start_val - 2.5, start_val - 0.3), "● ● ●"))
        
        # Scenario B: Lead-in for the very first line of the song
        elif start_val > 4.0:
            sequenced_output.append(((start_val - 3.5, start_val - 0.5), "● ● ●"))
        
        sequenced_output.append((timing_window, text_string))
    
    logger.info(f"Lyric processing complete. Generated {len(sequenced_output)} segments.")
    return sequenced_output

def convert_seconds_to_readable(total_seconds):
    """Utility for displaying YouTube video lengths in UI."""
    if not total_seconds: 
        return "0:00"
    try:
        val = int(float(total_seconds))
        minutes, seconds = divmod(val, 60)
        return f"{minutes}:{seconds:02d}"
    except (ValueError, TypeError):
        return "Unknown"

# ==============================================================================
# 4. STREAMLIT INTERFACE & WHITE STUDIO THEME
# ==============================================================================
st.set_page_config(
    page_title="Agape-Worship Master Suite v2.5", 
    page_icon="🙏", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application State Management
if 'search_data' not in st.session_state: 
    st.session_state.search_data = []
if 'active_file' not in st.session_state: 
    st.session_state.active_file = None
if 'lyrics' not in st.session_state: 
    st.session_state.lyrics = ""
if 'stems' not in st.session_state: 
    st.session_state.stems = None
if 'bg_path' not in st.session_state: 
    st.session_state.bg_path = None
if 'last_status' not in st.session_state: 
    st.session_state.last_status = "System Ready"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3504/3504140.png", width=70)
    st.header("🎨 Studio Customization")
    
    # NEW: RESOLUTION SELECTOR (Added per request)
    target_res = st.selectbox("Render Quality", ["1080p (Full HD)", "4K (Ultra HD)", "720p (Standard)"])
    
    # INTERFACE COLOR CONTROLS
    ui_profile = st.radio("System Theme Profile", ["Pure Studio Light", "Custom Brand Palette"])
    
    if ui_profile == "Pure Studio Light":
        ui_bg_color = "#FFFFFF"
        ui_text_color = "#0F172A"
        ui_accent_color = "#00D1B2"
        ui_card_bg = "#F1F5F9"
    else:
        ui_bg_color = st.color_picker("App Background", "#F8FAFC")
        ui_text_color = st.color_picker("Primary Text", "#1E293B")
        ui_accent_color = st.color_picker("Button Accent", "#00D1B2")
        ui_card_bg = "#FFFFFF"
    
    st.divider()
    st.subheader("Rendering Parameters")
    vocal_cue_toggle = st.toggle("Enable Lead-In Indicators (● ● ●)", value=True)
    cue_sensitivity = st.slider("Cue Threshold (Seconds)", 1.5, 6.0, 3.5, help="Minimum gap needed to show countdown dots.")
    
    st.divider()
    st.info(f"**Engine Status:** {st.session_state.last_status}")
    if st.button("🗑️ Full System Reset", use_container_width=True):
        deep_clean_workspace()
        st.session_state.clear()
        st.rerun()

# DYNAMIC CSS INJECTION
st.markdown(f"""
<style>
    .stApp {{ background-color: {ui_bg_color}; color: {ui_text_color}; }}
    h1, h2, h3, h4, p, label, .stMarkdown {{ color: {ui_text_color} !important; }}
    .stButton>button {{ 
        border-radius: 10px; height: 3.2em; font-weight: 800;
        background: linear-gradient(135deg, {ui_accent_color}, #059669);
        color: white !important; border: none; transition: 0.3s all ease;
    }}
    .stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    .search-card {{ 
        background-color: {ui_card_bg}; padding: 20px; border-radius: 12px; 
        border: 1px solid #E2E8F0; margin-bottom: 12px;
        transition: 0.2s transform ease;
    }}
    .search-card:hover {{ transform: translateX(5px); border-left: 4px solid {ui_accent_color}; }}
    .metadata-label {{ font-size: 0.85rem; font-weight: 600; color: {ui_text_color}; opacity: 0.7; }}
    .stTextArea textarea {{ background-color: {ui_card_bg}; border: 1px solid #CBD5E1; color: {ui_text_color}; }}
</style>
""", unsafe_allow_html=True)

st.title("🙏 Agape-Worship Master Suite")
st.write("Professional AI-Powered Video and Stem Processing for Worship Leaders.")

# ==============================================================================
# 5. MODULE 1: MEDIA ACQUISITION & SEARCH (FIXED METADATA)
# ==============================================================================
tab_import, tab_processing, tab_render = st.tabs(["📥 Media Import", "🎚️ AI Separation", "🎬 Final Render"])

with tab_import:
    col_local, col_cloud = st.columns(2)
    
    with col_local:
        with st.container(border=True):
            st.subheader("📁 Local File Hub")
            uploaded_media = st.file_uploader("Upload Worship Track (Audio or Video)", type=["mp3", "wav", "mp4", "m4a", "flac"])
            if uploaded_media:
                timestamp_id = int(time.time())
                file_extension = uploaded_media.name.split('.')[-1]
                save_path = f"upload_{timestamp_id}.{file_extension}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())
                st.session_state.active_file = save_path
                st.success(f"Track Loaded: {uploaded_media.name}")
            
            st.divider()
            uploaded_bg = st.file_uploader("Cinema Background Image", type=["jpg", "png", "jpeg"])
            if uploaded_bg:
                st.session_state.bg_path = "active_bg.jpg"
                with open("active_bg.jpg", "wb") as f:
                    f.write(uploaded_bg.getbuffer())
                st.info("Background asset updated.")

    with col_cloud:
        with st.container(border=True):
            st.subheader("📺 YouTube Cloud Engine")
            format_choice = st.radio("Download Mode:", ["High-Fidelity Audio (MP3)", "Full Production Video (MP4)"], horizontal=True, key="fmt_choice")
            search_query = st.text_input("Enter Song Name, Artist, or URL:", placeholder="e.g. Graves into Gardens Elevation Worship")
            
            sc_1, sc_2 = st.columns(2)
            if sc_1.button("🔍 Search YouTube"):
                if search_query:
                    with st.spinner("Searching global database..."):
                        y_search_opts = {
                            'quiet': True, 
                            'extract_flat': True, 
                            'force_generic_extractor': False,
                            'ignoreerrors': True
                        }
                        try:
                            with yt_dlp.YoutubeDL(y_search_opts) as ydl:
                                search_results = ydl.extract_info(f"ytsearch10:{search_query}", download=False)
                                if search_results and 'entries' in search_results:
                                    st.session_state.search_data = search_results['entries']
                                    st.session_state.last_status = "Search Complete"
                                else:
                                    st.error("No results found for that query.")
                        except Exception as e:
                            st.error(f"Search API Error: {str(e)}")
            
            if sc_2.button("🚀 Fast-Track URL"):
                if search_query:
                    with st.spinner("Processing URL..."):
                        ts_key = int(time.time())
                        is_aud = "Audio" in format_choice
                        ext_type = "mp3" if is_aud else "mp4"
                        output_file = f"yt_{ts_key}.{ext_type}"
                        direct_opts = {
                            'format': 'bestaudio/best' if is_aud else 'bestvideo+bestaudio/best',
                            'outtmpl': f'yt_{ts_key}.%(ext)s',
                            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}] if is_aud else []
                        }
                        with yt_dlp.YoutubeDL(direct_opts) as ydl:
                            ydl.download([search_query])
                        st.session_state.active_file = output_file
                        st.rerun()

    if st.session_state.search_data:
        st.write("### YouTube Search Results")
        for entry in st.session_state.search_data:
            if not entry:
                continue
            
            vid_title = entry.get('title', 'Unknown Title')
            vid_author = entry.get('uploader') or entry.get('channel') or "Official Music Channel"
            vid_time = convert_seconds_to_readable(entry.get('duration'))
            vid_url = entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}"
            vid_thumb = entry['thumbnails'][0]['url'] if entry.get('thumbnails') else ""

            st.markdown(f"""
            <div class="search-card">
                <div style="display: flex; gap: 20px; align-items: flex-start;">
                    <img src="{vid_thumb}" style="width: 160px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="flex-grow: 1;">
                        <h4 style="margin-top: 0; margin-bottom: 8px;">{vid_title}</h4>
                        <div class="metadata-label">🎬 Channel: <span style="font-weight:400;">{vid_author}</span></div>
                        <div class="metadata-label">⏳ Duration: <span style="font-weight:400;">{vid_time}</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Import: {vid_title[:40]}...", key=entry['id']):
                with st.spinner("Downloading and converting track..."):
                    ts_val = int(time.time())
                    final_name = f"yt_{ts_val}.mp3"
                    import_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': f'yt_{ts_val}.%(ext)s',
                        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}]
                    }
                    with yt_dlp.YoutubeDL(import_opts) as ydl:
                        ydl.download([vid_url])
                    st.session_state.active_file = final_name
                    st.session_state.last_status = "Import Successful"
                    st.rerun()

# ==============================================================================
# 6. MODULE 2: WAVEFORM ANALYSIS & AI LYRICS (ROBUST)
# ==============================================================================
if st.session_state.active_file:
    with tab_processing:
        st.subheader("🎤 Song Analysis & Lyric Synchronization")
        
        try:
            waveform_audio, sample_rate = librosa.load(st.session_state.active_file, duration=45)
            waveform_fig = go.Figure(go.Scatter(y=waveform_audio, fill='tozeroy', line=dict(color=ui_accent_color, width=1.5)))
            waveform_fig.update_layout(
                height=150, 
                margin=dict(l=0, r=0, t=0, b=0), 
                template="plotly_white", 
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(waveform_fig, use_container_width=True)
        except Exception as vis_err:
            logger.error(f"Waveform visualization failed: {vis_err}")

        lyric_col_left, lyric_col_right = st.columns([2, 1])
        
        with lyric_col_left:
            song_query_hint = st.text_input("Song Search Hint (Artist - Title):", value=st.session_state.active_file)
            lyric_btn_1, lyric_btn_2 = st.columns(2)
            
            if lyric_btn_1.button("🤖 AI Vocal Transcription (Whisper)"):
                with st.spinner("AI is listening to the track..."):
                    try:
                        whisper_engine = whisper.load_model("base")
                        transcription_result = whisper_engine.transcribe(st.session_state.active_file)
                        srt_formatted_output = ""
                        for i, segment in enumerate(transcription_result['segments'], 1):
                            start_ts = segment['start']
                            end_ts = segment['end']
                            text_val = segment['text'].strip()
                            srt_formatted_output += f"{i}\n{start_ts} --> {end_ts}\n{text_val}\n\n"
                        st.session_state.lyrics = srt_formatted_output
                        st.session_state.last_status = "AI Transcription Complete"
                    except Exception as whisper_err:
                        st.error(f"Whisper Engine Error: {whisper_err}")

            if lyric_btn_2.button("🌍 Search Synced Databases"):
                with st.spinner("Deep-Scanning Global Databases (LRCLib, Musixmatch, Megalobiz)..."):
                    try:
                        # We remove 'allow_search_identifiers' to fix your error
                        # We loop through providers manually to maximize the search area
                        providers_to_try = ["lrclib", "megalobiz", "musixmatch", "netease"]
                        found_lyrics = None
                        
                        for p in providers_to_try:
                            if not found_lyrics:
                                try:
                                    # Attempting search per provider
                                    found_lyrics = syncedlyrics.search(song_query_hint, providers=[p])
                                except:
                                    continue # If a provider is down, move to the next
                        
                        if not found_lyrics:
                            # Final "Hail Mary" search with no restrictions
                            found_lyrics = syncedlyrics.search(song_query_hint)

                        if found_lyrics:
                            st.session_state.lyrics = found_lyrics
                            st.success("High-Fidelity Synced Lyrics located!")
                        else:
                            st.warning("No time-synced lyrics found. The 'AI Vocal Transcription' button is your best backup.")
                                
                    except Exception as db_err:
                        st.error(f"Lyric Database Error: {str(db_err)}")

        with lyric_col_right:
            st.markdown("""
            **Instructional Note:**
            You can manually edit the timestamps below. 
            - Use `[mm:ss.xx]` for LRC.
            - Use `00:00:00 --> 00:00:00` for SRT.
            The engine will automatically detect and format either.
            """)

        st.session_state.lyrics = st.text_area("Lyric Master Editor", value=st.session_state.lyrics, height=350)

# ==============================================================================
# 7. MODULE 3: AI STEM MIXER (HTDEMUCS ARCHITECTURE)
# ==============================================================================
        st.divider()
        st.subheader("🎚️ AI Multi-Stem Mixing Studio")
        
        mixer_info, mixer_action = st.columns([2, 1])
        with mixer_info:
            mixing_preset = st.selectbox(
                "Automation Preset:", 
                ["Full Master Mix", "Performance Ready (Vocals 15%)", "Karaoke/Backing (Vocals 0%)", "Instrumental Focus (Vocals 0%, Bass Boosted)"]
            )
            st.caption("Using Meta's HTDemucs Model for high-fidelity separation.")

        if mixer_action.button("▶️ EXECUTE STEM SEPARATION", use_container_width=True):
            with st.spinner("Splitting audio into Vocals, Drums, Bass, and Melody..."):
                try:
                    separation_engine = Separator(model_file_dir="models", output_format="WAV")
                    separation_engine.load_model(model_filename='htdemucs_ft.yaml')
                    st.session_state.stems = separation_engine.separate(st.session_state.active_file)
                    st.session_state.last_status = "Separation Successful"
                except Exception as sep_err:
                    st.error(f"Stem Separation Failed: {sep_err}")
                    logger.critical(traceback.format_exc())

        if st.session_state.stems:
            st.write("### Channel Level Control")
            stem_categories = ["Vocals", "Drums", "Bass", "Other"]
            processed_mixer_segments = []
            mixing_cols = st.columns(4)
            
            for idx, label in enumerate(stem_categories):
                default_vol = 85
                if "Performance" in mixing_preset: default_vol = 15 if label == "Vocals" else 95
                elif "Karaoke" in mixing_preset: default_vol = 0 if label == "Vocals" else 95
                elif "Instrumental" in mixing_preset: 
                    default_vol = 0 if label == "Vocals" else 100
                
                vol_slider = mixing_cols[idx].slider(f"{label} Volume", 0, 100, default_vol, key=f"mix_vol_{label}")
                
                for file_in_stems in st.session_state.stems:
                    if label.lower() in file_in_stems.lower():
                        audio_segment = AudioSegment.from_file(file_in_stems)
                        gain_level = 20 * (np.log10(vol_slider / 100.0)) if vol_slider > 0 else -120
                        processed_mixer_segments.append(audio_segment.apply_gain(gain_level))

            if st.button("🔊 Bounce Final Master Mix", use_container_width=True):
                with st.spinner("Summing audio channels and normalizing peaks..."):
                    try:
                        summed_audio = processed_mixer_segments[0]
                        for remaining_track in processed_mixer_segments[1:]:
                            summed_audio = summed_audio.overlay(remaining_track)
                        
                        normalized_audio = normalize(summed_audio)
                        output_mix_path = "agape_master_mix.wav"
                        normalized_audio.export(output_mix_path, format="wav")
                        st.audio(output_mix_path)
                        st.success("Master mix generated and ready for render.")
                    except Exception as bounce_err:
                        st.error(f"Mixing Error: {bounce_err}")

# ==============================================================================
# 8. MODULE 4: CINEMATIC GOLD RENDERING ENGINE (4K & PROGRESS FIXED)
# ==============================================================================
    with tab_render:
        st.subheader("🎬 Production Rendering Suite")
        st.write("Merge your master audio with high-definition lyrics and visuals.")
        
        render_col_1, render_col_2 = st.columns(2)
        with render_col_1:
            production_title = st.text_input("Video Filename:", value=f"Agape_Worship_{int(time.time())}")
            video_fps = st.select_slider("Rendering Framerate (FPS):", options=[24, 30, 60], value=24)
        
        if st.button("✨ START CINEMA RENDER", use_container_width=True):
            if not os.path.exists("agape_master_mix.wav"):
                st.warning("Please bounce a Master Mix in the Separation tab first!")
            else:
                # NEW: RESOLUTION MAP (Added per request)
                res_map = {
                    "4K (Ultra HD)": (3840, 2160, 2.0, "25000k"),
                    "1080p (Full HD)": (1920, 1080, 1.0, "10000k"),
                    "720p (Standard)": (1280, 720, 0.65, "5000k")
                }
                target_w, target_h, scale_factor, target_bitrate = res_map[target_res]

                # NEW: PROGRESS BAR UI (Added per request)
                progress_bar = st.progress(0)
                status_text = st.empty()
                render_logger = StreamlitProgressLogger(progress_bar, status_text)

                with st.spinner(f"Initializing Gold Render Engine in {target_res}..."):
                    try:
                        audio_production_clip = AudioFileClip("agape_master_mix.wav")
                        
                        # --- 4K AWARE RESIZE LOGIC ---
                        if st.session_state.bg_path:
                            # Load the image
                            bg_raw = ImageClip(st.session_state.bg_path)
                            
                            # Use fl_image to bypass MoviePy's broken resize() method
                            # This manually resizes every frame using pure Pillow logic with LANCZOS for quality
                            video_background = bg_raw.fl_image(
                                lambda image: np.array(Image.fromarray(image).resize((target_w, target_h), Image.Resampling.LANCZOS))
                            ).set_duration(audio_production_clip.duration)
                        else:
                            # Fallback if no image is uploaded
                            video_background = ColorClip(size=(target_w, target_h), color=(15, 15, 22)).set_duration(audio_production_clip.duration)
                        # --------------------------------------------------

                        def worship_subtitle_stylizer(text_content):
                            is_lead_in = "●" in text_content
                            return TextClip(
                                text_content, 
                                font='Arial-Bold', 
                                # Dynamic Scaling based on Resolution Selection
                                fontsize=int(95 * scale_factor) if not is_lead_in else int(130 * scale_factor),
                                color='#FFFFFF' if is_lead_in else '#FFD700',
                                method='caption', 
                                # Scaled Container Size to prevent clipping
                                size=(target_w * 0.9, target_h * 0.5), 
                                align='center',
                                stroke_color='black', 
                                stroke_width=int(3 * scale_factor)
                            )

                        parsed_lyrics = process_lyrics_with_lead_in(st.session_state.lyrics, vocal_cue_toggle, cue_sensitivity)
                        
                        if parsed_lyrics:
                            sub_clip = SubtitlesClip(parsed_lyrics, worship_subtitle_stylizer)
                            # Positioned relative to canvas (60% down)
                            sub_clip = sub_clip.set_position(('center', 0.60), relative=True)
                            final_production = CompositeVideoClip([video_background, sub_clip]).set_audio(audio_production_clip)
                        else:
                            final_production = video_background.set_audio(audio_production_clip)

                        output_video_path = f"exports/{production_title}.mp4"
                        
                        # Rendering with Progress Logger Hook
                        final_production.write_videofile(
                            output_video_path, 
                            fps=video_fps, 
                            codec="libx264", 
                            audio_codec="aac", 
                            bitrate=target_bitrate,
                            logger=render_logger
                        )
                        
                        st.success(f"Production Finished!")
                        st.video(output_video_path)
                        
                        with open(output_video_path, "rb") as v_file:
                            st.download_button("💾 Download Video", data=v_file, file_name=f"{production_title}.mp4")

                    except Exception as render_ex:
                        st.error(f"Cinema Render Engine Failure: {render_ex}")
                        st.code(traceback.format_exc()) # Shows exactly where the error is