from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime configuration for acoustic analysis."""

    sample_rate: int = 16_000
    frame_length: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 7_600.0
    min_pitch: float = 60.0
    max_pitch: float = 500.0
    embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_model_savedir: str = "pretrained_models/spkrec-ecapa-voxceleb"
