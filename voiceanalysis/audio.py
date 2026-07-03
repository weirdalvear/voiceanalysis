from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import AnalysisConfig


def load_audio(path: str | Path, config: AnalysisConfig) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at the configured sample rate."""

    audio, sr = librosa.load(path, sr=config.sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return audio.astype(np.float32), sr


def write_normalized_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write a peak-normalized WAV copy used by external tools/models."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(out, audio / peak * 0.98, sr)
    return out
