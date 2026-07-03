from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .audio import load_audio
from .config import AnalysisConfig
from .embeddings import compute_ecapa_embedding, cosine_similarity
from .features import extract_features
from .plots import generate_plots
from .report import build_pdf


def analyze_pair(file_a: str | Path, file_b: str | Path, output_dir: str | Path, skip_embeddings: bool = False) -> Path:
    """Run the full local analysis pipeline and return the PDF path."""

    config = AnalysisConfig()
    outdir = Path(output_dir)
    plot_dir = outdir / "plots"
    workdir = outdir / "work"
    outdir.mkdir(parents=True, exist_ok=True)
    audio_a, sr = load_audio(file_a, config)
    audio_b, _ = load_audio(file_b, config)
    features = pd.DataFrame({"A": extract_features(audio_a, sr, config), "B": extract_features(audio_b, sr, config)}).T
    stats = pd.DataFrame({
        "A": features.loc["A"],
        "B": features.loc["B"],
        "absolute_difference": (features.loc["A"] - features.loc["B"]).abs(),
        "percent_difference": ((features.loc["A"] - features.loc["B"]).abs() / (features.loc["A"].abs() + 1e-12)) * 100,
    }).sort_values("absolute_difference", ascending=False)
    similarity: dict[str, float | str] = {
        "feature_cosine_similarity": cosine_similarity(features.loc["A"].fillna(0).to_numpy(), features.loc["B"].fillna(0).to_numpy()),
        "feature_correlation": float(features.T.corr(numeric_only=True).iloc[0, 1]),
    }
    if skip_embeddings:
        similarity["ecapa_embedding_cosine"] = "skipped"
    else:
        emb_a = compute_ecapa_embedding(audio_a, sr, workdir, "A", config)
        emb_b = compute_ecapa_embedding(audio_b, sr, workdir, "B", config)
        np.save(outdir / "embedding_A.npy", emb_a); np.save(outdir / "embedding_B.npy", emb_b)
        similarity["ecapa_embedding_cosine"] = cosine_similarity(emb_a, emb_b)
    plots = generate_plots(audio_a, audio_b, sr, features, plot_dir, config)
    return build_pdf(outdir / "voice_comparison_report.pdf", features, stats, plots, similarity)
