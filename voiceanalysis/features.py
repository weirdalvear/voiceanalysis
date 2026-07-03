from __future__ import annotations

from collections.abc import Iterable

import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy import signal, stats

from .config import AnalysisConfig


def _summary(name: str, values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{name}_{k}": np.nan for k in ("mean", "std", "median", "min", "max", "p05", "p95", "iqr", "skew", "kurt")}
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
        f"{name}_p05": float(np.percentile(arr, 5)),
        f"{name}_p95": float(np.percentile(arr, 95)),
        f"{name}_iqr": float(stats.iqr(arr)),
        f"{name}_skew": float(stats.skew(arr, bias=False)) if arr.size > 2 else np.nan,
        f"{name}_kurt": float(stats.kurtosis(arr, bias=False)) if arr.size > 3 else np.nan,
    }


def extract_features(audio: np.ndarray, sr: int, config: AnalysisConfig) -> pd.Series:
    """Extract 100+ acoustic descriptors summarized over time."""

    feats: dict[str, float] = {
        "duration_s": len(audio) / sr,
        "rms_global": float(np.sqrt(np.mean(audio**2))),
        "peak_amplitude": float(np.max(np.abs(audio))),
        "crest_factor": float((np.max(np.abs(audio)) + 1e-12) / (np.sqrt(np.mean(audio**2)) + 1e-12)),
    }

    y = audio.astype(float)
    hop = config.hop_length
    frame = config.frame_length
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc, hop_length=hop)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.n_mels, hop_length=hop)

    for name, values in {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
    }.items():
        feats.update(_summary(name, values))

    for i, row in enumerate(contrast, 1):
        feats.update(_summary(f"spectral_contrast_{i}", row))
    for i, row in enumerate(mfcc, 1):
        feats.update(_summary(f"mfcc_{i}", row))
    for i, row in enumerate(delta, 1):
        feats.update(_summary(f"mfcc_delta_{i}", row))
    for i, row in enumerate(delta2, 1):
        feats.update(_summary(f"mfcc_delta2_{i}", row))
    for i, row in enumerate(chroma, 1):
        feats.update(_summary(f"chroma_{i}", row))
    for i, row in enumerate(librosa.power_to_db(mel + 1e-12), 1):
        feats.update(_summary(f"mel_band_{i}", row))

    freqs, psd = signal.welch(y, sr, nperseg=min(4096, len(y)))
    psd_sum = float(np.sum(psd)) + 1e-12
    feats["spectral_entropy"] = float(stats.entropy(psd / psd_sum))
    for low, high in ((0, 300), (300, 1000), (1000, 3000), (3000, 8000)):
        mask = (freqs >= low) & (freqs < high)
        feats[f"band_energy_{low}_{high}"] = float(np.sum(psd[mask]) / psd_sum)

    snd = parselmouth.Sound(y, sr)
    pitch = snd.to_pitch(time_step=hop / sr, pitch_floor=config.min_pitch, pitch_ceiling=config.max_pitch)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    feats.update(_summary("f0_hz", f0))
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", config.min_pitch, config.max_pitch)
        feats["jitter_local"] = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats["shimmer_local"] = float(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        feats["jitter_local"] = np.nan
        feats["shimmer_local"] = np.nan

    return pd.Series(feats, dtype="float64")
