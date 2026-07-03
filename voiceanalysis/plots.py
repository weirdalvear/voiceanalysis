from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_plots(audio_a: np.ndarray, audio_b: np.ndarray, sr: int, features: pd.DataFrame, outdir: Path, config: AnalysisConfig) -> list[Path]:
    """Generate more than 20 comparison plots for the report."""

    paths: list[Path] = []
    audios = {"A": audio_a, "B": audio_b}
    for label, y in audios.items():
        fig, ax = plt.subplots(figsize=(9, 3)); librosa.display.waveshow(y, sr=sr, ax=ax); ax.set_title(f"Waveform {label}"); paths.append(_save(fig, outdir / f"waveform_{label}.png"))
        fig, ax = plt.subplots(figsize=(9, 4)); D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max); img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax); fig.colorbar(img, ax=ax); ax.set_title(f"Spectrogram {label}"); paths.append(_save(fig, outdir / f"spectrogram_{label}.png"))
        fig, ax = plt.subplots(figsize=(9, 4)); M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.n_mels), ref=np.max); img = librosa.display.specshow(M, sr=sr, x_axis="time", y_axis="mel", ax=ax); fig.colorbar(img, ax=ax); ax.set_title(f"Mel spectrogram {label}"); paths.append(_save(fig, outdir / f"mel_{label}.png"))
        for name, data in {
            "rms": librosa.feature.rms(y=y, hop_length=config.hop_length)[0],
            "zcr": librosa.feature.zero_crossing_rate(y, hop_length=config.hop_length)[0],
            "centroid": librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=config.hop_length)[0],
            "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=config.hop_length)[0],
        }.items():
            fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(data); ax.set_title(f"{name.upper()} over time {label}"); paths.append(_save(fig, outdir / f"{name}_{label}.png"))
        fig, ax = plt.subplots(figsize=(8, 3)); ax.hist(y, bins=80); ax.set_title(f"Amplitude histogram {label}"); paths.append(_save(fig, outdir / f"hist_{label}.png"))
        fig, ax = plt.subplots(figsize=(8, 3)); mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc); img = librosa.display.specshow(mfcc, x_axis="time", ax=ax); fig.colorbar(img, ax=ax); ax.set_title(f"MFCC {label}"); paths.append(_save(fig, outdir / f"mfcc_{label}.png"))

    diff = (features.loc["A"] - features.loc["B"]).abs().sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 8)); diff.sort_values().plot.barh(ax=ax); ax.set_title("Top absolute feature differences"); paths.append(_save(fig, outdir / "top_feature_differences.png"))
    corr = features.T.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4, 4)); im = ax.imshow(corr, vmin=-1, vmax=1); fig.colorbar(im, ax=ax); ax.set_xticks([0, 1], ["A", "B"]); ax.set_yticks([0, 1], ["A", "B"]); ax.set_title("Feature correlation"); paths.append(_save(fig, outdir / "feature_correlation.png"))
    return paths
