import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import io
import os
import urllib.parse

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SongID — Audio Fingerprint Matcher",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        border: 1px solid #ffffff15;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #0f3460, #533483);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Stats bar */
    .stat-card {
        background: #1a1a2e;
        border: 1px solid #ffffff15;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 700; color: #e94560; }
    .stat-label  { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }

    /* Match result */
    .match-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 24px 32px;
        margin: 16px 0;
    }
    .match-title { font-size: 0.85rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 2px; }
    .match-song  { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 8px 0; }
    .match-count { font-size: 0.9rem; color: #e94560; }

    /* Song table */
    .song-card {
        background: #1a1a2e;
        border: 1px solid #ffffff10;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .song-name  { color: #ffffff; font-weight: 500; }
    .song-count { color: #e94560; font-size: 0.85rem; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Upload area */
    .stFileUploader > div {
        background: #1a1a2e !important;
        border: 2px dashed #e9456050 !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #c62a47) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }

    /* Radio */
    .stRadio > div { gap: 16px; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Progress */
    .stProgress > div > div { background-color: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "song_fingerprints_.csv_1.gz")
N_FFT             = 2048
PEAK_NEIGHBORHOOD = 20
FAN_OUT           = 15
DT_MIN            = 0.0
DT_MAX            = 5.0
HIST_BIN          = 0.5

# ─────────────────────────────────────────────
# FINGERPRINTING PIPELINE
# ─────────────────────────────────────────────
def load_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    return audio, sr

def compute_spectrogram(audio, sr):
    hop  = sr // 100
    S    = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S, S_db, hop

def get_peaks(S, sr, hop):
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD)
    is_peak   = (S == local_max) & (S > S.mean())
    freq_bins, time_bins = np.where(is_peak)
    times      = librosa.frames_to_time(time_bins, sr=sr, hop_length=hop)
    freqs_hz   = fft_freqs[freq_bins]
    freqs_int  = np.round(freqs_hz).astype(int)
    peaks      = list(zip(times.tolist(), freqs_int.tolist()))
    peaks.sort(key=lambda x: x[0])
    return peaks, times, freqs_hz

def generate_hashes(peaks):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + 1 + FAN_OUT, len(peaks))):
            t2, f2 = peaks[j]
            dt = round(t2 - t1, 2)
            if DT_MIN < dt <= DT_MAX:
                hashes.append((int(f1), int(f2), dt, round(t1, 3)))
    return hashes

@st.cache_data
def load_db():
    df = pd.read_csv(DB_PATH, compression='gzip')
    db = {}
    for row in df.itertuples(index=False):
        key = (int(row.f1), int(row.f2), round(float(row.delta_t), 2))
        if key not in db:
            db[key] = []
        db[key].append((row.song, float(row.anchor_time)))
    return db, df

def match(hashes, db):
    offsets = {}
    for (f1, f2, dt, q_time) in hashes:
        key = (f1, f2, dt)
        if key in db:
            for (song, db_time) in db[key]:
                offset = round(db_time - q_time, 2)
                offsets.setdefault(song, []).append(offset)
    return offsets

def build_histogram(offsets):
    if not offsets:
        return None, None, None
    best_song, best_count = None, 0
    all_hist = {}
    for song, offs in offsets.items():
        offs_arr = np.array(offs)
        mn, mx   = offs_arr.min(), offs_arr.max()
        bins     = np.arange(mn, mx + HIST_BIN, HIST_BIN) if mx - mn >= HIST_BIN else [mn - HIST_BIN, mn + HIST_BIN]
        counts, edges = np.histogram(offs_arr, bins=bins)
        peak     = counts.max()
        all_hist[song] = (counts, edges)
        if peak > best_count:
            best_count = peak
            best_song  = song
    return best_song, best_count, all_hist

def identify(audio_bytes, db):
    audio, sr    = load_audio(audio_bytes)
    S, S_db, hop = compute_spectrogram(audio, sr)
    peaks, peak_times, peak_freqs = get_peaks(S, sr, hop)
    hashes       = generate_hashes(peaks)
    offsets      = match(hashes, db)
    best_song, best_count, all_hist = build_histogram(offsets)
    return {
        "audio": audio, "sr": sr, "hop": hop,
        "S_db": S_db, "peak_times": peak_times, "peak_freqs": peak_freqs,
        "hashes": hashes, "offsets": offsets,
        "best_song": best_song, "best_count": best_count, "all_hist": all_hist,
    }

# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
plt.style.use('dark_background')

def styled_fig():
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0e1117')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ffffff20')
    ax.tick_params(colors='#a0aec0')
    ax.xaxis.label.set_color('#a0aec0')
    ax.yaxis.label.set_color('#a0aec0')
    ax.title.set_color('#ffffff')
    return fig, ax

def plot_spectrogram(S_db, sr, hop):
    fig, ax = styled_fig()
    librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                             x_axis="time", y_axis="hz", ax=ax, cmap="inferno")
    ax.set_title("Spectrogram", fontsize=13, fontweight='bold')
    cb = plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors='#a0aec0')
    plt.tight_layout()
    return fig

def plot_constellation(S_db, sr, hop, peak_times, peak_freqs):
    fig, ax = styled_fig()
    librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                             x_axis="time", y_axis="hz", ax=ax, cmap="inferno", alpha=0.5)
    ax.scatter(peak_times, peak_freqs, color="#e94560", s=6, label="peaks", zorder=5)
    ax.set_title("Constellation Map (Peaks)", fontsize=13, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8, facecolor='#1a1a2e', edgecolor='#ffffff20')
    plt.tight_layout()
    return fig

def plot_histogram(all_hist, best_song):
    if not all_hist or best_song not in all_hist:
        return None
    counts, edges = all_hist[best_song]
    fig, ax = styled_fig()
    ax.bar(edges[:-1], counts, width=np.diff(edges), color="#e94560", align="edge", alpha=0.85)
    ax.set_title(f"Offset Histogram — {best_song}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time Offset (s)")
    ax.set_ylabel("Match Count")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# LOAD DB
# ─────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.error(f"Database not found at: {DB_PATH}")
    st.stop()

with st.spinner("Loading fingerprint database..."):
    db, df_full = load_db()

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎵 SongID</h1>
    <p>Audio Fingerprint Matcher — Identify songs from short clips using acoustic fingerprinting</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATS BAR
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{df_full["song"].nunique()}</div><div class="stat-label">Songs in Database</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{len(db):,}</div><div class="stat-label">Unique Fingerprints</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-number">2</div><div class="stat-label">Identification Modes</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="stat-number">~3s</div><div class="stat-label">Avg. Match Time</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODE TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Single Clip", "📂  Batch Mode", "🎵  Song Database"])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE CLIP
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Upload a Query Clip</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop an audio file here", type=["wav", "mp3", "ogg", "flac"],
                                label_visibility="collapsed")
    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()

        with st.spinner("🔍 Fingerprinting and matching..."):
            result = identify(audio_bytes, db)

        best  = result["best_song"] or "No match found"
        count = result["best_count"] or 0

        # Match result box
        st.markdown(f"""
        <div class="match-box">
            <div class="match-title">🎯 Identified Song</div>
            <div class="match-song">{best}</div>
            <div class="match-count">Peak match count: {count} &nbsp;|&nbsp; Hashes generated: {len(result['hashes']):,}</div>
        </div>
        """, unsafe_allow_html=True)

        # YouTube search link for matched song
        if result["best_song"]:
            yt_query = urllib.parse.quote(result["best_song"])
            st.markdown(f'<a href="https://www.youtube.com/results?search_query={yt_query}" target="_blank" style="color:#e94560; font-weight:600;">▶ Listen on YouTube →</a>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plots
        st.markdown('<div class="section-header">1 — Spectrogram</div>', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(result["S_db"], result["sr"], result["hop"]))

        st.markdown('<div class="section-header">2 — Constellation Map</div>', unsafe_allow_html=True)
        st.pyplot(plot_constellation(result["S_db"], result["sr"], result["hop"],
                                     result["peak_times"], result["peak_freqs"]))

        st.markdown('<div class="section-header">3 — Offset Histogram</div>', unsafe_allow_html=True)
        if result["all_hist"] and result["best_song"]:
            st.pyplot(plot_histogram(result["all_hist"], result["best_song"]))
        else:
            st.warning("No matching hashes found — histogram unavailable.")

        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Rate", f"{result['sr']} Hz")
        c2.metric("Peaks Extracted", f"{len(result['peak_times']):,}")
        c3.metric("Songs in Pool", f"{len(result['offsets'])}")

# ══════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Upload Multiple Clips</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop multiple audio files here",
                                      type=["wav", "mp3", "ogg", "flac"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) ready to process")
        if st.button("🚀 Run Batch Identification"):
            rows     = []
            progress = st.progress(0)
            status   = st.empty()
            for i, f in enumerate(uploaded_files):
                status.text(f"Processing {f.name} ({i+1}/{len(uploaded_files)})...")
                try:
                    result     = identify(f.read(), db)
                    pred       = result["best_song"] if result["best_song"] else "unknown"
                    pred_clean = os.path.splitext(pred)[0]
                except Exception as e:
                    pred_clean = "error"
                    st.error(f"{f.name}: {e}")
                rows.append({"filename": f.name, "prediction": pred_clean})
                progress.progress((i + 1) / len(uploaded_files))

            status.success("✅ Batch complete!")
            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(results_df, use_container_width=True)
            st.download_button("⬇️ Download results.csv",
                               data=results_df.to_csv(index=False),
                               file_name="results.csv", mime="text/csv")

# ══════════════════════════════════════════════
# TAB 3 — SONG DATABASE
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">All Songs in Database</div>', unsafe_allow_html=True)

    song_counts = df_full.groupby('song').size().reset_index(name='fingerprints').sort_values('fingerprints', ascending=False)

    search = st.text_input("🔍 Search songs", placeholder="Type a song name...")
    if search:
        song_counts = song_counts[song_counts['song'].str.contains(search, case=False)]

    for _, row in song_counts.iterrows():
        st.markdown(f"""
        <div class="song-card">
            <span class="song-name">🎵 {row['song']}</span>
            <span class="song-count">{row['fingerprints']:,} fingerprints</span>
        </div>
        """, unsafe_allow_html=True)
