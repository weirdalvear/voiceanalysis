from __future__ import annotations

import argparse

from .pipeline import analyze_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and compare two WAV recordings locally.")
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip SpeechBrain model download/inference.")
    args = parser.parse_args()
    pdf = analyze_pair(args.file_a, args.file_b, args.output_dir, args.skip_embeddings)
    print(f"Report written to {pdf}")


if __name__ == "__main__":
    main()
