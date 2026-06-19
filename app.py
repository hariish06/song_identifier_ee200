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

st.set_page_config(
    page_title="SONGID — Audio Fingerprint",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Ops+One&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@400;700;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
}

.stApp {
    background: #000000;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── HERO ── */
.hero-wrap {
    position: relative;
    width: 100%;
    min-height: 520px;
    background: radial-gradient(ellipse at 60% 50%, #3a0000 0%, #1a0000 40%, #000000 80%);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 40px 60px;
    overflow: hidden;
    border-bottom: 2px solid #cc000060;
}
.hero-wrap::before {
    content: 'SONGID';
    position: absolute;
    font-family: 'Black Ops One', cursive !important;
    font-size: 22vw;
    color: #ff000008;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    pointer-events: none;
    letter-spacing: -0.02em;
}
.hero-badge {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem;
    color: #cc0000;
    letter-spacing: 6px;
    text-transform: uppercase;
    margin-bottom: 16px;
    border: 1px solid #cc000050;
    padding: 6px 20px;
    border-radius: 2px;
}
.hero-title {
    font-family: 'Black Ops One', cursive !important;
    font-size: clamp(4rem, 10vw, 9rem);
    color: #ffffff;
    line-height: 0.9;
    letter-spacing: -0.02em;
    margin: 0;
    text-shadow: 0 0 80px #cc000060, 0 0 160px #cc000030;
}
.hero-title span { color: #cc0000; }
.hero-sub {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.15rem;
    color: #ffffff70;
    margin-top: 20px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-line {
    width: 80px;
    height: 3px;
    background: #cc0000;
    margin: 24px auto 0;
}

/* ── STATS ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 0;
    background: #0a0a0a;
    border-bottom: 1px solid #cc000030;
}
.stat-item {
    flex: 1;
    max-width: 200px;
    text-align: center;
    padding: 28px 20px;
    border-right: 1px solid #cc000020;
}
.stat-item:last-child { border-right: none; }
.stat-num {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.2rem;
    font-weight: 900;
    color: #cc0000;
    display: block;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #ffffff40;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
    display: block;
}

/* ── CONTENT AREA ── */
.content-area {
    padding: 48px 5% 80px;
    background: #000000;
}

/* ── SECTION TITLE ── */
.sec-title {
    font-family: 'Black Ops One', cursive !important;
    font-size: 1.8rem;
    color: #ffffff;
    letter-spacing: 2px;
    margin: 0 0 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sec-title::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 32px;
    background: #cc0000;
    border-radius: 2px;
}

/* ── MATCH BOX ── */
.match-wrap {
    background: linear-gradient(135deg, #0f0000, #1a0000);
    border: 1px solid #cc0000;
    border-left: 5px solid #cc0000;
    border-radius: 4px;
    padding: 32px 36px;
    margin: 24px 0;
    box-shadow: 0 0 40px #cc000025, inset 0 0 40px #cc000008;
}
.match-label {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem;
    color: #cc0000;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.match-name {
    font-family: 'Black Ops One', cursive !important;
    font-size: 2.8rem;
    color: #ffffff;
    line-height: 1;
    text-shadow: 0 0 30px #cc000060;
}
.match-meta {
    font-size: 0.85rem;
    color: #ffffff40;
    margin-top: 10px;
    letter-spacing: 1px;
}
.match-meta span { color: #cc0000; }

/* ── YT LINK ── */
.yt-btn {
    display: inline-block;
    margin-top: 16px;
    background: #cc0000;
    color: #ffffff !important;
    text-decoration: none !important;
    padding: 10px 28px;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-radius: 2px;
    transition: all 0.2s;
}

/* ── SONG CARDS ── */
.song-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 20px;
    border-bottom: 1px solid #ffffff08;
    transition: background 0.15s;
}
.song-row:hover { background: #cc000010; }
.song-idx {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem;
    color: #cc0000;
    width: 32px;
    flex-shrink: 0;
}
.song-nm { color: #ffffff; font-size: 1rem; font-weight: 600; flex: 1; padding: 0 16px; }
.song-fp { font-family: 'Orbitron', monospace !important; font-size: 0.7rem; color: #ffffff30; letter-spacing: 1px; }

/* ── UPLOAD AREA ── */
.stFileUploader > div {
    background: #0a0a0a !important;
    border: 2px dashed #cc000040 !important;
    border-radius: 4px !important;
}
.stFileUploader label { color: #ffffff60 !important; }

/* ── BUTTON ── */
.stButton > button {
    background: #cc0000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 14px 36px !important;
    width: 100% !important;
    box-shadow: 0 0 30px #cc000040 !important;
}
.stButton > button:hover {
    background: #ff0000 !important;
    box-shadow: 0 0 50px #cc000070 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0a0a !important;
    border-bottom: 2px solid #cc000030 !important;
    gap: 0 !important;
    padding: 0 5% !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    color: #ffffff40 !important;
    padding: 18px 28px !important;
    border-bottom: 3px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #cc0000 !important;
    border-bottom: 3px solid #cc0000 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #000000 !important;
    padding: 0 !important;
}

/* ── TEXT INPUT ── */
.stTextInput input {
    background: #0a0a0a !important;
    border: 1px solid #cc000030 !important;
    color: #ffffff !important;
    border-radius: 2px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #0a0a0a;
    border: 1px solid #cc000020;
    border-top: 3px solid #cc0000;
    border-radius: 2px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #ffffff40 !important; letter-spacing: 2px; font-size: 0.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #ffffff !important;
    font-size: 1.4rem !important;
}

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #cc000030 !important; }

/* ── PROGRESS ── */
.stProgress > div > div { background: #cc0000 !important; }

/* ── HIDE BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── AUDIO ── */
audio { width: 100%; border-radius: 2px; }

/* ── DIVIDER ── */
.red-line { width: 100%; height: 1px; background: linear-gradient(90deg, transparent, #cc0000, transparent); margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "song_fingerprints_.csv.gz")
N_FFT             = 2048
PEAK_NEIGHBORHOOD = 20
FAN_OUT           = 15
DT_MIN            = 0.0
DT_MAX            = 5.0
HIST_BIN          = 0.5

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def load_audio(audio_bytes):
    return librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

def compute_spectrogram(audio, sr):
    hop  = sr // 100
    S    = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S, S_db, hop

def get_peaks(S, sr, hop):
    fft_freqs  = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    local_max  = maximum_filter(S, size=PEAK_NEIGHBORHOOD)
    is_peak    = (S == local_max) & (S > S.mean())
    freq_bins, time_bins = np.where(is_peak)
    times      = librosa.frames_to_time(time_bins, sr=sr, hop_length=hop)
    freqs_int  = np.round(fft_freqs[freq_bins]).astype(int)
    peaks      = sorted(zip(times.tolist(), freqs_int.tolist()), key=lambda x: x[0])
    return peaks, times, fft_freqs[freq_bins]

def generate_hashes(peaks):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i+1, min(i+1+FAN_OUT, len(peaks))):
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
        db.setdefault(key, []).append((row.song, float(row.anchor_time)))
    return db, df

def match(hashes, db):
    offsets = {}
    for (f1, f2, dt, q_time) in hashes:
        for entry in db.get((f1, f2, dt), []):
            song, db_time = entry
            offsets.setdefault(song, []).append(round(db_time - q_time, 2))
    return offsets

def build_histogram(offsets):
    if not offsets:
        return None, None, None
    best_song, best_count, all_hist = None, 0, {}
    for song, offs in offsets.items():
        offs_arr = np.array(offs)
        mn, mx   = offs_arr.min(), offs_arr.max()
        bins     = np.arange(mn, mx+HIST_BIN, HIST_BIN) if mx-mn >= HIST_BIN else [mn-HIST_BIN, mn+HIST_BIN]
        counts, edges = np.histogram(offs_arr, bins=bins)
        all_hist[song] = (counts, edges)
        if counts.max() > best_count:
            best_count = counts.max(); best_song = song
    return best_song, best_count, all_hist

def identify(audio_bytes, db):
    audio, sr    = load_audio(audio_bytes)
    S, S_db, hop = compute_spectrogram(audio, sr)
    peaks, pt, pf = get_peaks(S, sr, hop)
    hashes       = generate_hashes(peaks)
    offsets      = match(hashes, db)
    best, count, hist = build_histogram(offsets)
    return dict(audio=audio, sr=sr, hop=hop, S_db=S_db, peak_times=pt,
                peak_freqs=pf, hashes=hashes, offsets=offsets,
                best_song=best, best_count=count, all_hist=hist)

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
plt.style.use('dark_background')

def dark_fig():
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#050505')
    for sp in ax.spines.values(): sp.set_edgecolor('#cc000030')
    ax.tick_params(colors='#ffffff40')
    ax.xaxis.label.set_color('#ffffff40')
    ax.yaxis.label.set_color('#ffffff40')
    ax.title.set_color('#ffffff')
    return fig, ax

def plot_spectrogram(S_db, sr, hop):
    fig, ax = dark_fig()
    librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap="hot")
    ax.set_title("SPECTROGRAM", fontsize=11, fontweight='bold', letter_spacing=3, color='#ffffff80', fontfamily='monospace')
    cb = plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors='#ffffff40')
    plt.tight_layout()
    return fig

def plot_constellation(S_db, sr, hop, peak_times, peak_freqs):
    fig, ax = dark_fig()
    librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap="hot", alpha=0.4)
    ax.scatter(peak_times, peak_freqs, color="#cc0000", s=5, zorder=5, alpha=0.8)
    ax.set_title("CONSTELLATION MAP", fontsize=11, fontweight='bold', color='#ffffff80', fontfamily='monospace')
    plt.tight_layout()
    return fig

def plot_histogram(all_hist, best_song):
    if not all_hist or best_song not in all_hist: return None
    counts, edges = all_hist[best_song]
    fig, ax = dark_fig()
    ax.bar(edges[:-1], counts, width=np.diff(edges), color="#cc0000", align="edge", alpha=0.9)
    ax.set_title("OFFSET HISTOGRAM", fontsize=11, fontweight='bold', color='#ffffff80', fontfamily='monospace')
    ax.set_xlabel("Time Offset (s)", color='#ffffff40')
    ax.set_ylabel("Match Count", color='#ffffff40')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# LOAD DB
# ─────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

with st.spinner("LOADING DATABASE..."):
    db, df_full = load_db()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-badge">⬡ Audio Fingerprint Technology ⬡</div>
    <div class="hero-title">SONG<span>ID</span></div>
    <div class="hero-sub">Identify any song from a short audio clip</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-num">{df_full['song'].nunique()}</span>
        <span class="stat-lbl">Songs</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">{len(db):,}</span>
        <span class="stat-lbl">Fingerprints</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">2</span>
        <span class="stat-lbl">Modes</span>
    </div>
    <div class="stat-item">
        <span class="stat-num">~3s</span>
        <span class="stat-lbl">Match Time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["◈  SINGLE CLIP", "◈  BATCH MODE", "◈  SONG DATABASE"])

# ══ SINGLE CLIP ══════════════════════════════
with tab1:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">UPLOAD CLIP</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["wav","mp3","ogg","flac"], label_visibility="collapsed")

    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        with st.spinner("ANALYZING..."):
            result = identify(audio_bytes, db)

        best  = result["best_song"] or "NO MATCH"
        count = result["best_count"] or 0

        yt_query = urllib.parse.quote(best) if result["best_song"] else ""
        yt_link  = f'https://www.youtube.com/results?search_query={yt_query}'

        st.markdown(f"""
        <div class="match-wrap">
            <div class="match-label">◈ Identified Song</div>
            <div class="match-name">{best}</div>
            <div class="match-meta">
                Peak Count: <span>{count}</span> &nbsp;|&nbsp;
                Hashes: <span>{len(result['hashes']):,}</span> &nbsp;|&nbsp;
                Sample Rate: <span>{result['sr']} Hz</span>
            </div>
            <a class="yt-btn" href="{yt_link}" target="_blank">▶ Listen on YouTube</a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">ANALYSIS</div>', unsafe_allow_html=True)

        st.pyplot(plot_spectrogram(result["S_db"], result["sr"], result["hop"]))
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(plot_constellation(result["S_db"], result["sr"], result["hop"],
                                     result["peak_times"], result["peak_freqs"]))
        st.markdown("<br>", unsafe_allow_html=True)
        if result["all_hist"] and result["best_song"]:
            fig_h = plot_histogram(result["all_hist"], result["best_song"])
            if fig_h: st.pyplot(fig_h)

        st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("PEAKS", f"{len(result['peak_times']):,}")
        c2.metric("HASHES", f"{len(result['hashes']):,}")
        c3.metric("CANDIDATES", f"{len(result['offsets'])}")

    st.markdown('</div>', unsafe_allow_html=True)

# ══ BATCH ════════════════════════════════════
with tab2:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">BATCH UPLOAD</div>', unsafe_allow_html=True)

    files = st.file_uploader("", type=["wav","mp3","ogg","flac"],
                             accept_multiple_files=True, label_visibility="collapsed")
    if files:
        st.info(f"{len(files)} file(s) loaded")
        if st.button("IDENTIFY ALL"):
            rows     = []
            progress = st.progress(0)
            status   = st.empty()
            for i, f in enumerate(files):
                status.markdown(f'<span style="color:#cc0000;font-family:monospace;font-size:0.8rem;">PROCESSING {f.name} [{i+1}/{len(files)}]</span>', unsafe_allow_html=True)
                try:
                    r          = identify(f.read(), db)
                    pred       = r["best_song"] if r["best_song"] else "unknown"
                    pred_clean = os.path.splitext(pred)[0]
                except Exception as e:
                    pred_clean = "error"
                rows.append({"filename": f.name, "prediction": pred_clean})
                progress.progress((i+1)/len(files))

            status.markdown('<span style="color:#cc0000;font-family:monospace;font-size:0.8rem;">✓ COMPLETE</span>', unsafe_allow_html=True)
            df_res = pd.DataFrame(rows, columns=["filename","prediction"])
            st.dataframe(df_res, use_container_width=True)
            st.download_button("⬇ DOWNLOAD results.csv",
                               data=df_res.to_csv(index=False),
                               file_name="results.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ══ DATABASE ═════════════════════════════════
with tab3:
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">SONG DATABASE</div>', unsafe_allow_html=True)

    search = st.text_input("", placeholder="Search songs...", label_visibility="collapsed")
    song_counts = df_full.groupby('song').size().reset_index(name='fp').sort_values('fp', ascending=False)
    if search:
        song_counts = song_counts[song_counts['song'].str.contains(search, case=False)]

    rows_html = ""
    for i, row in enumerate(song_counts.itertuples(), 1):
        rows_html += f"""
        <div class="song-row">
            <span class="song-idx">{i:02d}</span>
            <span class="song-nm">{row.song}</span>
            <span class="song-fp">{row.fp:,} FP</span>
        </div>"""

    st.markdown(f'<div style="border:1px solid #cc000020;border-radius:4px;">{rows_html}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
