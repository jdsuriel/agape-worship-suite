import streamlit as st
import os
import glob
import re
import sys
import shutil
import logging
import traceback
import time
import json
import numpy as np
import plotly.graph_objects as go
import yt_dlp
import torch
import whisper
import syncedlyrics
from audio_separator.separator import Separator
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import AudioFileClip, ImageClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
from PIL import Image

# ==============================================================================
# 0. CLOUD & SYSTEM CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Agape-Worship Master Suite", 
    page_icon="🙏", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define directories ensuring they exist (Ephemeral storage for Cloud)
REQUIRED_DIRECTORIES = ["models", "exports", "logs", "temp"]
for dir_path in REQUIRED_DIRECTORIES:
    os.makedirs(dir_path, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AgapeWorshipEngine")

# ==============================================================================
# 1. CROSS-PLATFORM BINARY RESOLUTION (LINUX/WINDOWS COMPATIBILITY)
# ==============================================================================
def configure_imagemagick():
    """
    Smart detection for ImageMagick that works on both Windows (Local) and Linux (Cloud).
    """
    # 1. Check for Linux/Cloud standard path (primary for Streamlit Cloud)
    linux_path = shutil.which("convert")
    if linux_path:
        logger.info(f"ImageMagick detected (Linux/Cloud): {linux_path}")
        change_settings({"IMAGEMAGICK_BINARY": linux_path})
        return linux_path

    # 2. Check for Windows local paths (Fallback for local dev)
    windows_search = [
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-*\magick.exe",
    ]
    for candidate in windows_search:
        matches = glob.glob(candidate)
        if matches:
            change_settings({"IMAGEMAGICK_BINARY": matches[0]})
            return matches[0]
            
    return None

configure_imagemagick()

# ==============================================================================
# 2. CACHED MODEL LOADING (PREVENTS RAM CRASHES)
# ==============================================================================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once and cache it to save memory."""
    logger.info("Loading Whisper Model...")
    return whisper.load_model("base")

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def deep_clean_workspace():
    """Sanitize workspace to prevent storage full errors on Cloud."""
    target_extensions = ["*.mp3", "*.wav", "*.mp4", "temp/*", "exports/*"]
    for pattern in target_extensions:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path) and "agape_master" not in file_path:
                    os.remove(file_path)
            except Exception:
                pass

def robust_timestamp_converter(raw_timestamp):
    sanitized = str(raw_timestamp).strip().replace(',', '.')
    try:
        parts = sanitized.split(':')
        if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return int(parts[0])*60 + float(parts[1])
        return float(sanitized)
    except:
        return 0.0

def process_lyrics_with_lead_in(raw_input, enable_cues=True, gap_threshold=3.0):
    if not raw_input or len(raw_input.strip()) < 5: return []
    raw_buffer = []
    
    # Simple Parser Logic
    if raw_input.strip().startswith('['): # LRC
        lines = raw_input.strip().split('\n')
        for i, line in enumerate(lines):
            match = re.search(r'\[(\d+):(\d+\.?\d*)\](.*)', line)
            if match:
                start = int(match.group(1)) * 60 + float(match.group(2))
                text = match.group(3).strip()
                if not text: continue
                # Estimate duration
                end = start + 4.0
                if i + 1 < len(lines):
                    next_match = re.search(r'\[(\d+):(\d+\.?\d*)\]', lines[i+1])
                    if next_match:
                        end = int(next_match.group(1)) * 60 + float(next_match.group(2))
                raw_buffer.append(((start, end), text))
    else: # SRT/Plain
        blocks = re.split(r'\n\s*\n', raw_input.strip())
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 2 and '-->' in lines[1]:
                t = lines[1].split(' --> ')
                raw_buffer.append(((robust_timestamp_converter(t[0]), robust_timestamp_converter(t[1])), " ".join(lines[2:])))

    if not enable_cues: return raw_buffer

    final_seq = []
    for idx, (timing, text) in enumerate(raw_buffer):
        start = timing[0]
        if idx > 0:
            prev_end = raw_buffer[idx-1][0][1]
            if (start - prev_end) >= gap_threshold:
                final_seq.append(((start - 2.5, start - 0.3), "● ● ●"))
        elif start > 4.0:
            final_seq.append(((start - 3.5, start - 0.5), "● ● ●"))
        final_seq.append((timing, text))
    return final_seq

# ==============================================================================
# 4. MAIN UI LOGIC
# ==============================================================================
if 'stems' not in st.session_state: st.session_state.stems = None
if 'active_file' not in st.session_state: st.session_state.active_file = None
if 'lyrics' not in st.session_state: st.session_state.lyrics = ""
if 'bg_path' not in st.session_state: st.session_state.bg_path = None

with st.sidebar:
    st.header("⚙️ Settings")
    ui_mode = st.radio("Theme", ["Light Studio", "Dark Mode"])
    if st.button("🧹 Clear Cache"):
        deep_clean_workspace()
        st.session_state.clear()
        st.rerun()

st.title("☁️ Agape-Worship Cloud Suite")

# TABS
tab_import, tab_process, tab_render = st.tabs(["1. Import", "2. Separation", "3. Render"])

# --- TAB 1: IMPORT ---
with tab_import:
    col_file, col_yt = st.columns(2)
    with col_file:
        st.subheader("Local Upload")
        up_file = st.file_uploader("Audio/Video", type=["mp3", "wav", "mp4"])
        if up_file:
            path = f"temp/{up_file.name}"
            with open(path, "wb") as f: f.write(up_file.getbuffer())
            st.session_state.active_file = path
            st.success(f"Loaded: {up_file.name}")
        
        up_bg = st.file_uploader("Background Image", type=["jpg", "png"])
        if up_bg:
            bg_path = f"temp/{up_bg.name}"
            with open(bg_path, "wb") as f: f.write(up_bg.getbuffer())
            st.session_state.bg_path = bg_path

    with col_yt:
        st.subheader("YouTube (Cloud Mode)")
        st.info("⚠️ Note: YouTube blocks some cloud servers. If this fails, download locally and upload.")
        yt_url = st.text_input("YouTube URL")
        if st.button("Download from YouTube"):
            with st.spinner("Attempting download..."):
                try:
                    ts = int(time.time())
                    out = f"temp/yt_{ts}"
                    # Robust cloud options
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': f'{out}.%(ext)s',
                        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3'}],
                        # Spoofing headers to avoid bot detection
                        'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    st.session_state.active_file = f"{out}.mp3"
                    st.success("Download Complete!")
                except Exception as e:
                    st.error(f"YouTube Blocked Connection: {e}")

# --- TAB 2: SEPARATION & LYRICS ---
with tab_process:
    if st.session_state.active_file:
        st.audio(st.session_state.active_file)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate AI Lyrics (Whisper)"):
                with st.spinner("Transcribing..."):
                    model = load_whisper_model()
                    result = model.transcribe(st.session_state.active_file)
                    txt = ""
                    for seg in result['segments']:
                        txt += f"{int(seg['start']//60)}:{seg['start']%60:05.2f} --> {int(seg['end']//60)}:{seg['end']%60:05.2f}\n{seg['text'].strip()}\n\n"
                    st.session_state.lyrics = txt
            st.session_state.lyrics = st.text_area("Edit Lyrics", st.session_state.lyrics, height=300)

        with c2:
            st.write("### AI Stem Separation")
            # Fallback for CPU
            use_gpu = torch.cuda.is_available()
            st.caption(f"Hardware: {'🚀 NVIDIA GPU' if use_gpu else '🐢 CPU Mode (Slower)'}")
            
            if st.button("Separate Stems"):
                with st.spinner("Separating (This may take 2-5 mins on CPU)..."):
                    try:
                        sep = Separator(model_file_dir="models", output_dir="temp", output_format="WAV")
                        sep.load_model(model_filename='htdemucs_ft.yaml')
                        files = sep.separate(st.session_state.active_file)
                        # Fix paths for Streamlit
                        st.session_state.stems = [os.path.join("temp", f) for f in files]
                        st.success("Separation Done!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.stems:
            st.divider()
            st.write("### Mixer")
            vols = {}
            cols = st.columns(4)
            labels = ["Vocals", "Drums", "Bass", "Other"]
            for i, label in enumerate(labels):
                vols[label] = cols[i].slider(label, 0, 100, 100 if label != "Vocals" else 80)
            
            if st.button("Bounce Mix"):
                mixed = None
                for stem in st.session_state.stems:
                    track_name = os.path.basename(stem).lower()
                    audioseg = AudioSegment.from_file(stem)
                    
                    gain = -100
                    for label in labels:
                        if label.lower() in track_name:
                            gain = 20 * np.log10(vols[label]/100) if vols[label] > 0 else -120
                    
                    audioseg = audioseg.apply_gain(gain)
                    mixed = audioseg if mixed is None else mixed.overlay(audioseg)
                
                mixed = normalize(mixed)
                mixed.export("temp/master_mix.wav", format="wav")
                st.success("Mix Ready!")
                st.audio("temp/master_mix.wav")

# --- TAB 3: RENDER ---
with tab_render:
    if st.button("Render Video"):
        if not os.path.exists("temp/master_mix.wav"):
            st.error("Please bounce a mix first.")
        else:
            with st.spinner("Rendering..."):
                try:
                    audio = AudioFileClip("temp/master_mix.wav")
                    
                    # Background Handling
                    w, h = 1920, 1080
                    if st.session_state.bg_path:
                        img_clip = ImageClip(st.session_state.bg_path)
                        # Resize preserving aspect ratio then crop or pad? 
                        # Simple resize for now to ensure 1080p
                        img_clip = img_clip.resize(newsize=(w, h)) 
                        video_bg = img_clip.set_duration(audio.duration)
                    else:
                        video_bg = ColorClip(size=(w, h), color=(10,10,10)).set_duration(audio.duration)

                    # Subtitles
                    if st.session_state.lyrics:
                        parsed = process_lyrics_with_lead_in(st.session_state.lyrics)
                        if parsed:
                            def style(txt):
                                return TextClip(txt, fontsize=90, color='white', 
                                              font='Arial', stroke_color='black', stroke_width=2,
                                              size=(w*0.9, None), method='caption')
                            subs = SubtitlesClip(parsed, style).set_position(('center', 0.7), relative=True)
                            final = CompositeVideoClip([video_bg, subs])
                        else:
                            final = video_bg
                    else:
                        final = video_bg

                    final = final.set_audio(audio)
                    out_path = "exports/final_video.mp4"
                    
                    # ENCODING SETTINGS (Fixed for Cloud/CPU)
                    # We use 'libx264' (CPU) because 'h264_nvenc' (GPU) won't work on Free Cloud
                    # preset='ultrafast' helps prevent timeouts on cloud
                    final.write_videofile(
                        out_path, 
                        fps=24, 
                        codec="libx264", 
                        audio_codec="aac",
                        preset="ultrafast", 
                        threads=4
                    )
                    
                    st.success("Render Complete!")
                    st.video(out_path)
                    with open(out_path, "rb") as f:
                        st.download_button("Download Video", f, "worship_video.mp4")
                
                except Exception as e:
                    st.error(f"Render Error: {e}")
                    st.write(traceback.format_exc())
