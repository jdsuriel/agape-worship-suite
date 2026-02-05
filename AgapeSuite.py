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
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from audio_separator.separator import Separator
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import AudioFileClip, ImageClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
from PIL import Image
import gc

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class RenderConfig:
    """Rendering configuration"""
    name: str
    width: int
    height: int
    scale_factor: float
    bitrate: str
    
RENDER_CONFIGS = {
    "4K (Ultra HD)": RenderConfig("4K", 3840, 2160, 2.0, "25000k"),
    "1080p (Full HD)": RenderConfig("1080p", 1920, 1080, 1.0, "10000k"),
    "720p (Standard)": RenderConfig("720p", 1280, 720, 0.65, "5000k"),
    "1440p (2K)": RenderConfig("1440p", 2560, 1440, 1.33, "16000k")
}

REQUIRED_DIRECTORIES = ["models", "exports", "temp_stems", "logs", "assets/backgrounds", "cache"]
LOG_FILE = "logs/system_diagnostics.log"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AgapeWorshipEngine")

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def find_imagemagick_binary() -> Optional[str]:
    """Find ImageMagick binary"""
    search_paths = [
        r"C:\Program Files\ImageMagick-*\magick.exe",
        r"/usr/bin/convert",
        r"/usr/local/bin/convert",
        r"/opt/homebrew/bin/convert"
    ]
    
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            logger.info(f"ImageMagick found: {matches[0]}")
            return matches[0]
    
    logger.warning("ImageMagick not found")
    return None

def deep_clean_workspace():
    """Clean temporary files"""
    logger.info("Cleaning workspace...")
    patterns = ["*.mp3", "*.wav", "temp_stems/*", "yt_*.mp3", "upload_*.mp3"]
    excluded = ["bg_custom", "models", "cache", "exports"]
    
    count = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if not any(ex in file_path for ex in excluded):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
    
    gc.collect()
    logger.info(f"Cleaned {count} files")
    return count

def parse_timestamp(raw_timestamp: str) -> float:
    """Parse various timestamp formats"""
    sanitized = str(raw_timestamp).strip().replace('[', '').replace(']', '').replace(',', '.')
    
    try:
        if ':' in sanitized:
            parts = sanitized.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        return float(sanitized)
    except:
        return 0.0

def process_lyrics(text: str, enable_cues: bool = True, gap_threshold: float = 3.0) -> List[Tuple[Tuple[float, float], str]]:
    """Process lyrics with lead-in cues"""
    if not text or len(text.strip()) < 5:
        return []
    
    lyrics = []
    
    # LRC Format
    if text.strip().startswith('['):
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            match = re.search(r'<span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mo stretchy="false">(</mo><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>+</mo><mo stretchy="false">)</mo><mo>:</mo><mo stretchy="false">(</mo><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>+</mo><mover accent="true"><mo stretchy="false">?</mo><mo>˙</mo></mover><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>∗</mo><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">(\d+):(\d+\.?\d*)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">(</span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mord">+</span><span class="mclose">)</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">:</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">(</span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1.1813em;vertical-align:-0.25em;"></span><span class="mord accent"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.9313em;"><span style="top:-3em;"><span class="pstrut" style="height:3em;"></span><span class="mclose">?</span></span><span style="top:-3.2634em;"><span class="pstrut" style="height:3em;"></span><span class="accent-body" style="left:-0.1389em;"><span class="mord">˙</span></span></span></span></span></span></span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mord">∗</span><span class="mclose">)</span></span></span></span></span>(.*)', line)
            if match:
                start = int(match.group(1)) * 60 + float(match.group(2))
                content = match.group(3).strip()
                if not content:
                    continue
                
                end = start + 4.5
                if i + 1 < len(lines):
                    next_match = re.search(r'<span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mo stretchy="false">(</mo><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>+</mo><mo stretchy="false">)</mo><mo>:</mo><mo stretchy="false">(</mo><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>+</mo><mover accent="true"><mo stretchy="false">?</mo><mo>˙</mo></mover><mstyle mathcolor="#cc0000"><mtext>\d</mtext></mstyle><mo>∗</mo><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">(\d+):(\d+\.?\d*)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">(</span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mord">+</span><span class="mclose">)</span><span class="mspace" style="margin-right:0.2778em;"></span><span class="mrel">:</span><span class="mspace" style="margin-right:0.2778em;"></span></span><span class="base"><span class="strut" style="height:1em;vertical-align:-0.25em;"></span><span class="mopen">(</span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1.1813em;vertical-align:-0.25em;"></span><span class="mord accent"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height:0.9313em;"><span style="top:-3em;"><span class="pstrut" style="height:3em;"></span><span class="mclose">?</span></span><span style="top:-3.2634em;"><span class="pstrut" style="height:3em;"></span><span class="accent-body" style="left:-0.1389em;"><span class="mord">˙</span></span></span></span></span></span></span><span class="mord text" style="color:#cc0000;"><span class="mord" style="color:#cc0000;">\d</span></span><span class="mord">∗</span><span class="mclose">)</span></span></span></span></span>', lines[i + 1])
                    if next_match:
                        end = int(next_match.group(1)) * 60 + float(next_match.group(2))
                
                lyrics.append(((start, end), content))
    
    # SRT Format
    else:
        blocks = re.split(r'\n\s*\n', text.strip())
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 2 and '-->' in lines[1]:
                timing = lines[1].split(' --> ')
                start = parse_timestamp(timing[0])
                end = parse_timestamp(timing[1])
                content = " ".join(lines[2:]).strip()
                if content:
                    lyrics.append(((start, end), content))
    
    # Inject cues
    if enable_cues:
        enhanced = []
        for idx, (timing, text) in enumerate(lyrics):
            start = timing[0]
            
            if idx > 0:
                prev_end = lyrics[idx - 1][0][1]
                if (start - prev_end) >= gap_threshold:
                    enhanced.append(((start - 2.5, start - 0.3), "● ● ●"))
            elif start > 4.0:
                enhanced.append(((start - 3.5, start - 0.5), "● ● ●"))
            
            enhanced.append((timing, text))
        
        return enhanced
    
    return lyrics

def format_duration(seconds):
    """Format seconds to MM:SS"""
    try:
        val = int(float(seconds))
        mins, secs = divmod(val, 60)
        return f"{mins}:{secs:02d}"
    except:
        return "0:00"

# ==============================================================================
# INITIALIZATION
# ==============================================================================
for directory in REQUIRED_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

magick_path = find_imagemagick_binary()
if magick_path:
    change_settings({"IMAGEMAGICK_BINARY": magick_path})

if 'system_initialized' not in st.session_state:
    logger.info("System initialization")
    deep_clean_workspace()
    st.session_state.system_initialized = True
    st.session_state.active_file = None
    st.session_state.lyrics = ""
    st.session_state.stems = None
    st.session_state.bg_path = None
    st.session_state.search_results = []
    st.session_state.mixed_audio = None

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(
    page_title="Agape Worship Studio Pro",
    page_icon="🎵",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🎵 Agape Worship Studio")
    st.divider()
    
    resolution = st.selectbox("Video Quality", list(RENDER_CONFIGS.keys()), index=1)
    enable_cues = st.toggle("Enable Lead-In Cues (● ● ●)", value=True)
    cue_threshold = st.slider("Cue Gap Threshold (sec)", 1.5, 6.0, 3.0)
    
    st.divider()
    if st.button("🗑️ Clean Workspace", use_container_width=True):
        count = deep_clean_workspace()
        st.success(f"Cleaned {count} files")
        st.rerun()

# Main UI
st.title("🎵 Agape Worship Studio Pro")
st.caption("Professional worship video production powered by AI")

tab1, tab2, tab3 = st.tabs(["📥 Import", "🎚️ Processing", "🎬 Render"])

# ==============================================================================
# TAB 1: IMPORT
# ==============================================================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Local Upload")
        uploaded = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "mp4", "m4a"])
        if uploaded:
            ts = int(time.time())
            ext = uploaded.name.split('.')[-1]
            path = f"upload_{ts}.{ext}"
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.active_file = path
            st.success(f"✅ Loaded: {uploaded.name}")
        
        st.divider()
        bg_upload = st.file_uploader("Background Image", type=["jpg", "png", "jpeg"])
        if bg_upload:
            with open("active_bg.jpg", "wb") as f:
                f.write(bg_upload.getbuffer())
            st.session_state.bg_path = "active_bg.jpg"
            st.success("✅ Background updated")
    
    with col2:
        st.subheader("🌐 YouTube Download")
        query = st.text_input("Search or paste URL")
        
        c1, c2 = st.columns(2)
        
        if c1.button("🔍 Search"):
            if query:
                with st.spinner("Searching..."):
                    try:
                        opts = {'quiet': True, 'extract_flat': True}
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            results = ydl.extract_info(f"ytsearch10:{query}", download=False)
                            if results and 'entries' in results:
                                st.session_state.search_results = results['entries']
                    except Exception as e:
                        st.error(f"Search failed: {e}")
        
        if c2.button("⬇️ Direct Download"):
            if query:
                with st.spinner("Downloading..."):
                    try:
                        ts = int(time.time())
                        opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': f'yt_{ts}.%(ext)s',
                            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]
                        }
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            ydl.download([query])
                        st.session_state.active_file = f"yt_{ts}.mp3"
                        st.success("✅ Downloaded")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Download failed: {e}")
    
    # Display search results
    if st.session_state.search_results:
        st.divider()
        st.subheader("Search Results")
        for entry in st.session_state.search_results[:5]:
            if not entry:
                continue
            
            title = entry.get('title', 'Unknown')
            duration = format_duration(entry.get('duration'))
            video_id = entry.get('id')
            
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"**{title}** ({duration})")
            
            if col_b.button("Import", key=video_id):
                with st.spinner("Downloading..."):
                    try:
                        ts = int(time.time())
                        url = f"https://youtube.com/watch?v={video_id}"
                        opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': f'yt_{ts}.%(ext)s',
                            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]
                        }
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            ydl.download([url])
                        st.session_state.active_file = f"yt_{ts}.mp3"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

# ==============================================================================
# TAB 2: PROCESSING
# ==============================================================================
with tab2:
    if not st.session_state.active_file:
        st.info("⬅️ Please import a file first")
    else:
        st.subheader("🎤 Lyrics & Analysis")
        
        # Waveform
        try:
            y, sr = librosa.load(st.session_state.active_file, duration=30)
            fig = go.Figure(go.Scatter(y=y, fill='tozeroy', line=dict(color='#00D1B2')))
            fig.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 AI Transcription (Whisper)"):
                with st.spinner("Transcribing..."):
                    try:
                        model = whisper.load_model("base")
                        result = model.transcribe(st.session_state.active_file)
                        srt_output = ""
                        for i, seg in enumerate(result['segments'], 1):
                            srt_output += f"{i}\n{seg['start']} --> {seg['end']}\n{seg['text'].strip()}\n\n"
                        st.session_state.lyrics = srt_output
                        st.success("✅ Transcription complete")
                    except Exception as e:
                        st.error(f"Failed: {e}")
        
        with col2:
            if st.button("🌐 Search Synced Lyrics"):
                with st.spinner("Searching lyric databases..."):
