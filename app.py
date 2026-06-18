import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import io
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprint_ .csv")
N_FFT             = 2048
PEAK_NEIGHBORHOOD = 20
FAN_OUT           = 15
DT_MIN            = 0.0
DT_MAX            = 5.0
HIST_BIN          = 0.5

# ─────────────────────────────────────────────
# STEP 1 — Load audio (preserve native sr)
# ─────────────────────────────────────────────
def load_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    return audio, sr

# ─────────────────────────────────────────────
# STEP 2 — Spectrogram
# ─────────────────────────────────────────────
def compute_spectrogram(audio, sr):
    hop = sr // 100          # exactly 10 ms per frame — matches DB
    S   = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S, S_db, hop

# ─────────────────────────────────────────────
# STEP 3 — Constellation peaks (in Hz, not bins)
# ─────────────────────────────────────────────
def get_peaks(S, sr, hop):
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)  # Hz per bin

    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD)
    is_peak   = (S == local_max) & (S > S.mean())

    freq_bins, time_bins = np.where(is_peak)
    times     = librosa.frames_to_time(time_bins, sr=sr, hop_length=hop)
    freqs_hz  = fft_freqs[freq_bins]            # <-- Hz values
    freqs_int = np.round(freqs_hz).astype(int)  # round to integer Hz (matches DB)

    peaks = list(zip(times.tolist(), freqs_int.tolist()))
    peaks.sort(key=lambda x: x[0])
    return peaks, times, freqs_hz

# ─────────────────────────────────────────────
# STEP 4 — Generate hashes
# ─────────────────────────────────────────────
def generate_hashes(peaks):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + 1 + FAN_OUT, len(peaks))):
            t2, f2 = peaks[j]
            dt = round(t2 - t1, 2)
            if DT_MIN < dt <= DT_MAX:
                hashes.append((int(f1), int(f2), dt, round(t1, 3)))
    return hashes

# ─────────────────────────────────────────────
# STEP 5 — Load DB (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_db():
    df = pd.read_csv(DB_PATH)
    db = {}
    for row in df.itertuples(index=False):
        key = (int(row.f1), int(row.f2), round(float(row.delta_t), 2))
        if key not in db:
            db[key] = []
        db[key].append((row.song, float(row.anchor_time)))
    return db

# ─────────────────────────────────────────────
# STEP 6 — Match
# ─────────────────────────────────────────────
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
        mn, mx = offs_arr.min(), offs_arr.max()
        if mx - mn < HIST_BIN:
            bins = [mn - HIST_BIN, mn + HIST_BIN]
        else:
            bins = np.arange(mn, mx + HIST_BIN, HIST_BIN)
        counts, edges = np.histogram(offs_arr, bins=bins)
        peak = counts.max()
        all_hist[song] = (counts, edges)
        if peak > best_count:
            best_count = peak
            best_song  = song
    return best_song, best_count, all_hist

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def identify(audio_bytes, db):
    audio, sr     = load_audio(audio_bytes)
    S, S_db, hop  = compute_spectrogram(audio, sr)
    peaks, peak_times, peak_freqs = get_peaks(S, sr, hop)
    hashes        = generate_hashes(peaks)
    offsets       = match(hashes, db)
    best_song, best_count, all_hist = build_histogram(offsets)
    return {
        "audio": audio, "sr": sr, "hop": hop,
        "S_db": S_db,
        "peak_times": peak_times, "peak_freqs": peak_freqs,
        "hashes": hashes, "offsets": offsets,
        "best_song": best_song, "best_count": best_count,
        "all_hist": all_hist,
    }

# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def plot_spectrogram(S_db, sr, hop):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                             x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_title("Spectrogram")
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

def plot_constellation(S_db, sr, hop, peak_times, peak_freqs):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop,
                             x_axis="time", y_axis="hz", ax=ax, cmap="magma", alpha=0.6)
    ax.scatter(peak_times, peak_freqs, color="cyan", s=6, label="peaks")
    ax.set_title("Constellation Map")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig

def plot_histogram(all_hist, best_song):
    if not all_hist or best_song not in all_hist:
        return None
    counts, edges = all_hist[best_song]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(edges[:-1], counts, width=np.diff(edges), color="steelblue", align="edge")
    ax.set_title(f"Offset Histogram — Best Match: {best_song}")
    ax.set_xlabel("Time Offset (s)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Audio Fingerprinter", layout="wide")
st.title("🎵 Audio Fingerprint Matcher")

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at: {DB_PATH}")
    st.stop()

with st.spinner("Loading fingerprint database..."):
    db = load_db()
st.success(f"Database loaded — {len(db):,} unique hashes")

mode = st.radio("Mode", ["Single Clip", "Batch"], horizontal=True)

# ── SINGLE CLIP ──────────────────────────────
if mode == "Single Clip":
    uploaded = st.file_uploader("Upload a query audio clip", type=["wav", "mp3", "ogg", "flac"])
    if uploaded:
        st.audio(uploaded)
        audio_bytes = uploaded.read()
        with st.spinner("Fingerprinting and matching..."):
            result = identify(audio_bytes, db)

        best  = result["best_song"] or "No match found"
        count = result["best_count"] or 0
        st.subheader(f"🎯 Match: **{best}**  (peak count: {count})")

        st.markdown("### 1 — Spectrogram")
        st.pyplot(plot_spectrogram(result["S_db"], result["sr"], result["hop"]))

        st.markdown("### 2 — Constellation Map")
        st.pyplot(plot_constellation(result["S_db"], result["sr"], result["hop"],
                                     result["peak_times"], result["peak_freqs"]))

        st.markdown("### 3 — Offset Histogram")
        if result["all_hist"] and result["best_song"]:
            fig_h = plot_histogram(result["all_hist"], result["best_song"])
            if fig_h:
                st.pyplot(fig_h)
        else:
            st.warning("No matching hashes found.")

        st.markdown("### Stats")
        st.write(f"- Native sample rate: **{result['sr']} Hz**")
        st.write(f"- Peaks extracted: **{len(result['peak_times'])}**")
        st.write(f"- Hashes generated: **{len(result['hashes'])}**")
        st.write(f"- Songs in offset pool: **{len(result['offsets'])}**")

# ── BATCH MODE ───────────────────────────────
else:
    uploaded_files = st.file_uploader("Upload multiple query clips",
                                      type=["wav", "mp3", "ogg", "flac"],
                                      accept_multiple_files=True)
    if uploaded_files:
        if st.button("Run Batch Identification"):
            rows     = []
            progress = st.progress(0)
            status   = st.empty()
            for i, f in enumerate(uploaded_files):
                status.text(f"Processing {f.name} ({i+1}/{len(uploaded_files)})...")
                try:
                    result = identify(f.read(), db)
                    pred   = result["best_song"] if result["best_song"] else "unknown"
                    pred_clean = os.path.splitext(pred)[0]
                except Exception as e:
                    pred_clean = "error"
                    st.error(f"{f.name}: {e}")
                rows.append({"filename": f.name, "prediction": pred_clean})
                progress.progress((i + 1) / len(uploaded_files))

            status.text("Done!")
            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(results_df)
            st.download_button("⬇️ Download results.csv",
                               data=results_df.to_csv(index=False),
                               file_name="results.csv", mime="text/csv")