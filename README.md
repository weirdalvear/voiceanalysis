# Voice Analysis

A complete local Python pipeline for comparing two WAV recordings. It extracts hundreds of acoustic descriptors, renders 20+ diagnostic graphs, optionally computes SpeechBrain ECAPA-TDNN speaker embeddings, writes CSV statistics, and produces a polished PDF report.

The report is intentionally forensic-style but conservative: it describes acoustic similarity and difference patterns and does **not** make unsupported identity conclusions.

## Features

- Python 3.11/3.12 compatible package.
- Praat/Parselmouth pitch, jitter, and shimmer features.
- librosa spectral, MFCC, delta, chroma, mel-band, energy, and temporal descriptors.
- More than 100 summarized acoustic features per recording.
- 20+ automatically generated plots including waveforms, spectrograms, mel spectrograms, MFCC maps, RMS/ZCR/centroid/rolloff traces, histograms, correlations, and top differences.
- Optional SpeechBrain ECAPA-TDNN neural speaker embeddings with cosine similarity.
- PDF, feature CSV, statistics CSV, embedding `.npy` exports.
- CLI and Tkinter GUI.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

SpeechBrain downloads its pretrained model the first time embeddings are enabled. If you only need acoustic features and graphs, pass `--skip-embeddings`.

## CLI usage

```bash
voiceanalysis recording_a.wav recording_b.wav --output-dir analysis_output
```

Fast acoustic-only run:

```bash
voiceanalysis recording_a.wav recording_b.wav --output-dir analysis_output --skip-embeddings
```

## GUI usage

```bash
voiceanalysis-gui
```

Choose two WAV files, optionally choose an output directory, and press **Analyze**. The GUI runs the same local pipeline as the CLI.

## Outputs

- `voice_comparison_report.pdf` — polished report with caveats, scores, tables, and plots.
- `voice_comparison_report.features.csv` — full feature matrix.
- `voice_comparison_report.stats.csv` — feature differences and percentages.
- `plots/*.png` — all generated visual diagnostics.
- `embedding_A.npy` / `embedding_B.npy` — neural embeddings when enabled.
