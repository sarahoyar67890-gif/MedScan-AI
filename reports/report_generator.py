"""
reports/report_generator.py — Generates a structured PDF analysis report.

Every piece of text in the generated PDF comes directly from either the
actual PredictionResult (label, confidence, probabilities, model metadata)
or the static, reviewed content in knowledge/skin_conditions.py. Nothing
here is written by a model, and nothing is invented — this module only
formats real data into a document.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
from knowledge.skin_conditions import DISCLAIMER as GUIDANCE_DISCLAIMER
from knowledge.skin_conditions import get_guidance_for_result

ACCENT_DEEP = colors.HexColor("#0A5958")
INK_MUTED = colors.HexColor("#5C6B70")
HAIRLINE = colors.HexColor("#DCE3E1")
AMBER_SOFT = colors.HexColor("#FBEEE0")
GREEN_SOFT = colors.HexColor("#E7F5EC")


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("MSTitle", parent=base["Title"], textColor=ACCENT_DEEP,
                                 fontSize=19, leading=22, spaceAfter=2),
        "subtitle": ParagraphStyle("MSSub", parent=base["Normal"], textColor=INK_MUTED,
                                    fontSize=9, spaceAfter=12),
        "h2": ParagraphStyle("MSH2", parent=base["Heading2"], textColor=ACCENT_DEEP,
                              fontSize=12.5, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("MSBody", parent=base["Normal"], fontSize=9.5, leading=14),
        "muted": ParagraphStyle("MSMuted", parent=base["Normal"], fontSize=7.8,
                                 textColor=INK_MUTED, leading=11.5),
        "bullet": ParagraphStyle("MSBullet", parent=base["Normal"], fontSize=9.3,
                                  leading=13.5, leftIndent=10, spaceAfter=2),
    }


def _add_bulleted_section(story: list, styles: dict, title: str, lines: list[str]) -> None:
    story.append(Paragraph(title, styles["h2"]))
    for line in lines:
        story.append(Paragraph(f"•&nbsp;&nbsp;{line}", styles["bullet"]))
    story.append(Spacer(1, 3))


def generate_pdf_report(result, filename: str, original_image_path: Optional[str] = None) -> bytes:
    """Builds the full PDF from a real inference.predictor.PredictionResult
    and returns it as bytes, ready for st.download_button. `result` must
    have: label, confidence, probabilities, is_suspicious, model_stage,
    model_epoch (the same PredictionResult the Streamlit result card uses)."""
    styles = _build_styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title="MedScan AI Analysis Report",
    )

    label = config.CLASS_NAMES[1] if result.is_suspicious else config.CLASS_NAMES[0]
    guidance = get_guidance_for_result(label)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story: list = []
    story.append(Paragraph("MEDSCAN AI — ANALYSIS REPORT", styles["title"]))
    story.append(Paragraph(f"Generated {now} &nbsp;·&nbsp; File: {filename}", styles["subtitle"]))
    story.append(HRFlowable(width="100%", color=HAIRLINE, thickness=1))
    story.append(Spacer(1, 10))

    if original_image_path:
        try:
            story.append(RLImage(original_image_path, width=60 * mm, height=60 * mm, kind="proportional"))
            story.append(Spacer(1, 8))
        except Exception:
            pass  # missing/unreadable image never blocks the rest of the report

    result_rows = [
        ["Screening result", result.label],
        ["Model confidence", f"{result.confidence*100:.1f}%"],
        ["Benign-pattern probability", f"{result.probabilities[config.CLASS_NAMES[0]]*100:.1f}%"],
        ["Suspicious-pattern probability", f"{result.probabilities[config.CLASS_NAMES[1]]*100:.1f}%"],
        ["Model", f"ResNet18 (stage: {result.model_stage or 'n/a'}, epoch: {result.model_epoch or 'n/a'})"],
    ]
    table = Table(result_rows, colWidths=[62 * mm, 98 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.3),
        ("TEXTCOLOR", (0, 0), (0, -1), INK_MUTED),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, HAIRLINE),
        ("BACKGROUND", (1, 0), (1, 0), AMBER_SOFT if result.is_suspicious else GREEN_SOFT),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Why could this happen?", styles["h2"]))
    story.append(Paragraph(guidance["summary"], styles["body"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(guidance["what_happens_in_skin"], styles["body"]))
    story.append(Spacer(1, 4))

    _add_bulleted_section(
        story, styles, "Possible contributing factors",
        [f"{f['factor']} — {f['note']}" for f in guidance["contributing_factors"]],
    )
    _add_bulleted_section(
        story, styles, "What to avoid",
        [f"{i['action']} — Why: {i['why']}" for i in guidance["avoid"]],
    )
    _add_bulleted_section(
        story, styles, "What you can do",
        [f"{i['action']} — Why: {i['why']}" for i in guidance["general_care"]],
    )
    _add_bulleted_section(story, styles, "Common management approaches", guidance["management_approaches"])
    _add_bulleted_section(story, styles, "When to see a dermatologist", guidance["when_to_see_dermatologist"])

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=HAIRLINE, thickness=1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(guidance["reliability_note"], styles["muted"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(GUIDANCE_DISCLAIMER, styles["muted"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "MedScan AI is a research/educational prototype. It is not a medical device, "
        "has not undergone clinical validation, and does not provide a diagnosis.",
        styles["muted"],
    ))

    doc.build(story)
    return buf.getvalue()