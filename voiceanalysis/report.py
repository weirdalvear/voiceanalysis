from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf(output_pdf: Path, features: pd.DataFrame, stats: pd.DataFrame, plots: list[Path], similarity: dict[str, float | str]) -> Path:
    """Build a polished PDF report with metrics, plots, and caveats."""

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Local Acoustic Comparison Report", styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph("This report describes acoustic similarity and difference patterns. It does not make identity conclusions.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    sim_rows = [["Metric", "Value"]] + [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in similarity.items()]
    story.append(Table(sim_rows, style=[("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.append(Spacer(1, 12))
    top = stats.head(25).reset_index().rename(columns={"index": "feature"})
    rows = [top.columns.tolist()] + top.round(4).astype(str).values.tolist()
    story.append(Paragraph("Top Feature Differences", styles["Heading2"]))
    story.append(Table(rows, repeatRows=1, style=[("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)], hAlign="LEFT"))
    story.append(PageBreak())
    for i, plot in enumerate(plots):
        story.append(Paragraph(plot.stem.replace("_", " ").title(), styles["Heading3"]))
        story.append(Image(str(plot), width=500, height=260, kind="proportional"))
        story.append(Spacer(1, 8))
        if i and i % 2 == 0:
            story.append(PageBreak())
    features.to_csv(output_pdf.with_suffix(".features.csv"))
    stats.to_csv(output_pdf.with_suffix(".stats.csv"))
    SimpleDocTemplate(str(output_pdf), pagesize=letter).build(story)
    return output_pdf
