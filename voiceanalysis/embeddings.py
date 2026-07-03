from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import write_normalized_wav
from .config import AnalysisConfig


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def compute_ecapa_embedding(audio: np.ndarray, sr: int, workdir: Path, stem: str, config: AnalysisConfig) -> np.ndarray:
    """Compute SpeechBrain ECAPA-TDNN speaker embedding."""

    from speechbrain.inference.speaker import EncoderClassifier

    wav_path = write_normalized_wav(workdir / f"{stem}_normalized.wav", audio, sr)
    classifier = EncoderClassifier.from_hparams(
        source=config.embedding_model_source,
        savedir=str(workdir / config.embedding_model_savedir),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    emb = classifier.encode_file(str(wav_path)).squeeze().detach().cpu().numpy()
    return emb.astype(float)
