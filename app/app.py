"""
app/app.py — MedScan AI Streamlit application (v4 — full dermatology assistant).

Run from the project root:
    streamlit run app/app.py

v4 adds real functionality on top of v3's restructuring (not just visuals):
  - Pre-inference image quality assessment (inference/quality.py) shown
    before analysis, with a required override checkbox for poor-quality images.
  - Uploaded images are now saved (storage/db.py v2) so History/Progress can
    show real thumbnails and a genuine side-by-side scan comparison.
  - A real, structured PDF report (reports/report_generator.py via
    reportlab) replaces the plain-text download.
  - A "Compare two scans" view on the Progress page using saved images.
  - Sidebar identity substantially reworked in app/style_guidance.py (dark
    gradient matching the hero panel, clear active-state nav).

The prediction pipeline itself (ResNet18, Grad-CAM, validation) is
unchanged — this version adds a layer of genuinely new product features
around it, not just a different color scheme.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd
import streamlit as st

import config
from app import style, style_extras, style_guidance
from inference.predictor import InferenceError, ModelNotReadyError, get_predictor
from inference.quality import assess_quality
from inference.validation import ImageValidationError, validate_image_bytes
from knowledge.skin_conditions import (
    DISCLAIMER as GUIDANCE_DISCLAIMER,
    INGREDIENT_EXPLAINER,
    SKIN_KNOWLEDGE_CENTER,
    get_guidance_for_result,
)
from storage import db

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="MedScan AI — AI Dermatology Intelligence",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="expanded",
)
style.inject(st)
style_extras.inject_extras(st)
style_guidance.inject_guidance(st)

db.init_db()

predictor = get_predictor()
MODEL_READY = predictor.is_ready
model_status = predictor.status()


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def render_state_card(kind: str, icon: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="state-card {kind}">
          <div class="state-icon">{icon}</div>
          <div class="state-title">{title}</div>
          <div class="state-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_badges(report) -> None:
    """NEW in v4. Shows the pre-inference quality report as badges + any
    warnings, before the user commits to running the model."""
    badge_class = report.overall.lower()
    st.markdown(
        f"""
        <div class="quality-badge-row">
          <span class="quality-badge {badge_class}">Image Quality: {report.overall}</span>
          <span class="quality-badge {badge_class}">{report.width}×{report.height}px</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if report.warnings:
        warn_html = "<div class='quality-warning-list'>" + "".join(
            f"<div>⚠ {w}</div>" for w in report.warnings
        ) + "</div>"
        st.markdown(warn_html, unsafe_allow_html=True)


def render_result_card(result) -> None:
    badge_class = "badge-amber" if result.is_suspicious else "badge-green"
    badge_text = "Suspicious-pattern" if result.is_suspicious else "Benign-pattern"
    probs = result.probabilities
    benign_pct = probs[config.CLASS_NAMES[0]] * 100
    suspicious_pct = probs[config.CLASS_NAMES[1]] * 100

    st.markdown(
        f"""
        <div class="result-card result-reveal">
          <span class="result-badge {badge_class}">{badge_text} prediction</span>
          <div class="result-headline">Preliminary screening signal</div>
          <div style="display:flex; align-items:baseline; gap:0.6rem; margin: 0.9rem 0 1.6rem 0;">
            <div class="result-confidence-num">{result.confidence*100:.1f}%</div>
            <div class="result-confidence-label">model confidence</div>
          </div>

          <div class="prob-row">
            <div class="prob-row-label"><span>Benign-pattern</span><span>{benign_pct:.1f}%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{benign_pct:.1f}%; background:var(--signal-green);"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-row-label"><span>Suspicious-pattern</span><span>{suspicious_pct:.1f}%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{suspicious_pct:.1f}%; background:var(--signal-amber);"></div></div>
          </div>
          <div class="metric-sub" style="margin-top:0.9rem;">Inference time: {result.inference_ms:.0f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">What did the AI notice?</span>', unsafe_allow_html=True)
    st.markdown("#### Model attention · Grad-CAM")

    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown('<div class="compare-label">Original image</div>', unsafe_allow_html=True)
        st.markdown('<div class="compare-frame">', unsafe_allow_html=True)
        st.image(result.original_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with gc2:
        st.markdown('<div class="compare-label">Grad-CAM overlay</div>', unsafe_allow_html=True)
        st.markdown('<div class="compare-frame">', unsafe_allow_html=True)
        st.image(result.gradcam_overlay, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class="msc-muted" style="font-size:0.85rem; margin-top:0.9rem;">
          Grad-CAM highlights image regions that influenced the model's prediction.
          It is an interpretability aid, not a medical explanation, and does not
          indicate which regions a clinician would consider diagnostically relevant.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_guidance_sections(is_suspicious: bool) -> None:
    """Renders the educational dermatology-assistant sections using the
    static, reviewed content in knowledge/skin_conditions.py — nothing here
    is generated by a model."""
    label = config.CLASS_NAMES[1] if is_suspicious else config.CLASS_NAMES[0]
    guidance = get_guidance_for_result(label)

    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Understanding this result</span>', unsafe_allow_html=True)
    st.markdown("### Why could this happen?")
    st.markdown(f"<p>{guidance['summary']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='msc-muted'>{guidance['what_happens_in_skin']}</p>", unsafe_allow_html=True)

    st.markdown('<div class="guidance-section-title">🧬 Possible contributing factors</div>', unsafe_allow_html=True)
    chips_html = '<div class="factor-chip-row">'
    for f in guidance["contributing_factors"]:
        chips_html += f'<span class="factor-chip">{f["factor"]}</span>'
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)
    with st.expander("Why these factors are commonly mentioned"):
        for f in guidance["contributing_factors"]:
            st.markdown(f"**{f['factor']}** — {f['note']}")

    st.markdown('<div class="guidance-section-title">🚫 What to avoid</div>', unsafe_allow_html=True)
    for item in guidance["avoid"]:
        st.markdown(
            f"""
            <div class="why-card avoid-card">
              <div class="what">{item['action']}</div>
              <div class="why-label">Why</div>
              <div class="why-text">{item['why']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="guidance-section-title">🌿 What you can do</div>', unsafe_allow_html=True)
    for item in guidance["general_care"]:
        st.markdown(
            f"""
            <div class="why-card care-card">
              <div class="what">{item['action']}</div>
              <div class="why-label">Why</div>
              <div class="why-text">{item['why']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="guidance-section-title">💊 Common management approaches</div>', unsafe_allow_html=True)
    for line in guidance["management_approaches"]:
        st.markdown(f"- {line}")

    warning_items = "".join(f"<li>{w}</li>" for w in guidance["when_to_see_dermatologist"])
    st.markdown(
        f"""
        <div class="derm-warning-card">
          <h4>🩺 When to seek professional evaluation</h4>
          <ul>{warning_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p class='msc-muted' style='font-size:0.82rem; margin-top:1rem;'>{guidance['reliability_note']}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p class='msc-muted' style='font-size:0.78rem;'>{GUIDANCE_DISCLAIMER}</p>",
        unsafe_allow_html=True,
    )


def run_prediction(raw_bytes: bytes, filename: str, content_type: str, source: str = "streamlit"):
    """Validate + predict + record history (now including the saved image).
    Returns (result, record_id, error_message)."""
    try:
        validated = validate_image_bytes(raw_bytes, filename=filename, content_type=content_type)
    except ImageValidationError as e:
        return None, None, str(e)

    try:
        result = predictor.predict(validated)
    except ModelNotReadyError as e:
        return None, None, str(e)
    except InferenceError as e:
        return None, None, str(e)

    record_id = db.record_analysis(
        filename=filename,
        predicted_label=result.label,
        is_suspicious=result.is_suspicious,
        confidence=result.confidence,
        prob_benign=result.probabilities[config.CLASS_NAMES[0]],
        prob_suspicious=result.probabilities[config.CLASS_NAMES[1]],
        image_width=validated.width,
        image_height=validated.height,
        inference_ms=result.inference_ms,
        model_stage=result.model_stage,
        model_epoch=result.model_epoch,
        source=source,
        image_bytes=raw_bytes,
    )
    return result, record_id, None


def offer_pdf_report(result, filename: str, record_id) -> None:
    """Builds the real PDF report and offers it as a download. Fails soft —
    if reportlab isn't installed or generation errors out, shows a small
    caption instead of crashing the whole result view."""
    try:
        from reports.report_generator import generate_pdf_report
        image_path = None
        if record_id:
            rec = db.get_by_id(record_id)
            if rec and rec.image_path:
                image_path = rec.image_path
        pdf_bytes = generate_pdf_report(result, filename, original_image_path=image_path)
        st.download_button(
            "📄 Download analysis report (PDF)",
            data=pdf_bytes,
            file_name=f"medscan_report_{Path(filename).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception:
        log.exception("PDF report generation failed")
        st.caption("Report generation is unavailable right now — check that `reportlab` is installed "
                     "(`pip install reportlab`).")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("🏠", "Dashboard"),
    ("🔬", "Skin Scan"),
    ("📊", "My Analysis"),
    ("🧠", "Skin Guidance"),
    ("📚", "Skin Knowledge"),
    ("📈", "Progress"),
    ("⚙️", "Settings"),
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown(
        """
        <div class="msc-sidebar-brand">
          <div class="glyph">M</div>
          <div class="brand-text">
            <div class="name">MedScan AI</div>
            <div class="tagline">AI Dermatology Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="msc-sidebar-section-label">Navigate</div>', unsafe_allow_html=True)
    for icon, name in NAV_ITEMS:
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                      type="primary" if st.session_state.page == name else "secondary"):
            st.session_state.page = name
            st.rerun()

    model_line = "ResNet18 · Ready" if MODEL_READY else "No checkpoint"
    st.markdown(
        f"""
        <div class="msc-sidebar-footer">
          MODEL &nbsp;·&nbsp; {model_line}<br>
          Research Prototype · v4
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page

if not MODEL_READY and page == "Skin Scan":
    if model_status.error:
        render_state_card(
            "error", "⚠", "Checkpoint found but couldn't be loaded",
            f"{model_status.error} — image analysis is disabled until this is fixed.",
        )
    else:
        render_state_card(
            "empty", "◐", "No trained checkpoint found",
            "checkpoints/medscan_resnet18_best.pt doesn't exist yet, so image analysis "
            "is disabled. The interface still renders so you can review the design — "
            "train the model first (see README) to enable live predictions.",
        )
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE: Dashboard
# =============================================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="msc-hero">
          <div>
            <span class="eyebrow">AI-Assisted Screening · Research Prototype</span>
            <h1>Understand your skin.<br>With AI-powered<br>skin intelligence.</h1>
            <p class="msc-sub">
              Analyze visible skin concerns, understand possible contributing factors,
              and get evidence-informed general skincare guidance — with a clear
              explanation of what the AI actually looked at.
            </p>
            <div class="msc-note">
              This is a research/educational tool, not a medical device. It does not
              diagnose disease and cannot replace evaluation by a qualified clinician.
            </div>
          </div>
          <div class="scan-panel">
            <div class="scan-corner tl"></div>
            <div class="scan-corner tr"></div>
            <div class="scan-corner bl"></div>
            <div class="scan-corner br"></div>
            <div class="scan-line"></div>
            <svg class="scan-glyph" width="120" height="120" viewBox="0 0 120 120" fill="none">
              <circle cx="60" cy="60" r="42" stroke="#5FE8D6" stroke-opacity="0.35" stroke-width="1.5"/>
              <circle cx="60" cy="60" r="28" stroke="#5FE8D6" stroke-opacity="0.5" stroke-width="1.5"/>
              <circle cx="60" cy="60" r="3" fill="#5FE8D6"/>
            </svg>
            <div class="scan-label">Model &nbsp;<b>ResNet18</b>&nbsp;·&nbsp; Mode &nbsp;<b>Screening</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="knowledge-grid">', unsafe_allow_html=True)
    feature_cards = [
        ("🔬", "AI Skin Analysis", "Upload a lesion photo and get a probability-based screening signal in seconds."),
        ("🧠", "Explainable Results", "Every prediction ships with a Grad-CAM visual explanation of what the model attended to."),
        ("🌿", "Skin Guidance", "Understand possible contributing factors, what to avoid, and general care — with the why."),
        ("📈", "Progress Tracking", "Your analyses are saved so you can compare scans and see patterns over time."),
    ]
    cards_html = ""
    for icon, title, desc in feature_cards:
        cards_html += f"""
        <div class="knowledge-card">
          <div class="title">{icon} {title}</div>
          <div class="desc">{desc}</div>
        </div>"""
    st.markdown(cards_html + "</div>", unsafe_allow_html=True)

    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Overview</span>', unsafe_allow_html=True)
    st.markdown("### Your analysis dashboard")

    stats = db.get_stats()
    total = stats["total_analyses"]

    if total == 0:
        render_state_card("empty", "◐", "No analyses yet",
                           "Run a screening to start populating your dashboard.")
    else:
        avg_conf = stats["avg_confidence"] or 0.0
        avg_ms = stats["avg_inference_ms"] or 0.0
        suspicious_pct = (stats["suspicious_count"] / total * 100) if total else 0

        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-label">Total analyses</div>
                <div class="metric-value">{total}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Suspicious-pattern</div>
                <div class="metric-value">{stats['suspicious_count']}</div>
                <div class="metric-sub">{suspicious_pct:.0f}% of total</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Avg. confidence</div>
                <div class="metric-value">{avg_conf*100:.1f}%</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Avg. inference time</div>
                <div class="metric-value">{avg_ms:.0f}<span style="font-size:1rem;">ms</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_path = config.METRICS_JSON_PATH
        if metrics_path.exists():
            with open(metrics_path) as fh:
                test_metrics = json.load(fh)
            st.markdown('<hr class="hairline">', unsafe_allow_html=True)
            st.markdown("#### Held-out test-set metrics")
            st.markdown(
                '<p class="msc-muted" style="font-size:0.85rem;">From the last run of '
                '<code>evaluation/evaluate.py</code> against the held-out test split — '
                "not this dashboard's live traffic.</p>",
                unsafe_allow_html=True,
            )
            metric_defs = [("accuracy", "Accuracy"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1_score", "F1")]
            metrics_html = '<div class="metric-grid">'
            for key, label in metric_defs:
                val = test_metrics.get(key)
                val_display = f"{val*100:.1f}%" if val is not None else "—"
                metrics_html += f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val_display}</div>
                </div>"""
            metrics_html += "</div>"
            st.markdown(metrics_html, unsafe_allow_html=True)


# =============================================================================
# PAGE: Skin Scan
# =============================================================================
elif page == "Skin Scan":
    st.markdown('<span class="eyebrow">Pipeline</span>', unsafe_allow_html=True)
    st.markdown("### How MedScan AI works")

    scan_steps = [
        ("01", "Image", "Upload a clear photo of a skin lesion (JPG or PNG)."),
        ("02", "Quality Check", "Resolution, blur, brightness, and contrast are assessed before analysis."),
        ("03", "Analysis", "A transfer-learned ResNet18 analyzes the validated image."),
        ("04", "Understanding", "Grad-CAM shows what the model attended to."),
        ("05", "Guidance", "Contributing factors, what to avoid, and general care — each with a why."),
        ("06", "Report", "Download a structured PDF summary of the full analysis."),
    ]
    progress_html = '<div class="scan-progress">'
    for i, (num, title, _) in enumerate(scan_steps):
        progress_html += f'<span class="scan-progress-step">{num} {title}</span>'
        if i < len(scan_steps) - 1:
            progress_html += '<span class="scan-progress-arrow">→</span>'
    progress_html += "</div>"
    st.markdown(progress_html, unsafe_allow_html=True)

    step_html = '<div class="step-grid">'
    for num, title, desc in scan_steps:
        step_html += f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div class="step-title">{title}</div>
          <p class="step-desc">{desc}</p>
        </div>"""
    step_html += "</div>"
    st.markdown(step_html, unsafe_allow_html=True)

    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Screening</span>', unsafe_allow_html=True)
    st.markdown("### Upload an image")

    uploaded_file = st.file_uploader(
        "Upload a skin lesion photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        disabled=not MODEL_READY,
        key="single_upload",
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        try:
            preview = validate_image_bytes(raw_bytes, uploaded_file.name, uploaded_file.type)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(preview.image, caption=f"{preview.width} × {preview.height} px",
                          use_container_width=True)
            with col2:
                st.markdown(
                    f"""
                    <div class="msc-card msc-card-tight" style="margin-top:0.5rem;">
                      <div class="spec-key">File</div>
                      <div class="spec-val" style="font-size:0.85rem; word-break:break-all;">{uploaded_file.name}</div>
                      <div class="spec-key" style="margin-top:0.9rem;">Dimensions</div>
                      <div class="spec-val">{preview.width} × {preview.height} px</div>
                      <div class="spec-key" style="margin-top:0.9rem;">Size</div>
                      <div class="spec-val">{preview.size_bytes / 1024:.0f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                quality_report = assess_quality(preview.image)
                render_quality_badges(quality_report)

                proceed_ok = True
                if quality_report.should_block:
                    proceed_ok = st.checkbox(
                        "I understand the image quality may reduce reliability — analyze anyway",
                        key="quality_override",
                    )

                analyze_clicked = st.button(
                    "Analyze image", use_container_width=True,
                    disabled=(not MODEL_READY) or (not proceed_ok), key="analyze_single",
                )

            if analyze_clicked:
                with st.spinner("Analyzing…"):
                    result, record_id, error = run_prediction(
                        raw_bytes, uploaded_file.name, uploaded_file.type, source="streamlit"
                    )
                if error:
                    render_state_card("error", "⚠", "Analysis failed", error)
                else:
                    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
                    st.markdown('<span class="eyebrow">Result</span>', unsafe_allow_html=True)
                    render_result_card(result)
                    render_guidance_sections(result.is_suspicious)
                    offer_pdf_report(result, uploaded_file.name, record_id)

        except ImageValidationError as e:
            render_state_card("error", "⚠", "This file can't be used", str(e))

    with st.expander("Batch analysis — analyze multiple images at once"):
        st.markdown(
            '<p class="msc-muted" style="font-size:0.88rem;">Upload several lesion photos '
            "to screen them in one pass. Each is validated and analyzed independently.</p>",
            unsafe_allow_html=True,
        )
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            disabled=not MODEL_READY,
            label_visibility="collapsed",
            key="batch_upload",
        )
        if batch_files:
            run_batch = st.button(f"Analyze {len(batch_files)} image(s)",
                                   disabled=not MODEL_READY, key="analyze_batch")
            if run_batch:
                progress = st.progress(0.0, text="Starting…")
                batch_html = '<div class="batch-grid">'
                n = len(batch_files)
                for i, f in enumerate(batch_files):
                    progress.progress(i / n, text=f"Analyzing {f.name} ({i+1}/{n})…")
                    raw = f.getvalue()
                    result, _, error = run_prediction(raw, f.name, f.type, source="streamlit-batch")
                    if error:
                        batch_html += f"""
                        <div class="batch-item">
                          <span class="hist-file">{f.name}</span>
                          <div class="state-desc" style="text-align:left; margin-top:0.3rem;">{error}</div>
                        </div>"""
                    else:
                        badge_class = "badge-amber" if result.is_suspicious else "badge-green"
                        badge_text = "Suspicious" if result.is_suspicious else "Benign"
                        batch_html += f"""
                        <div class="batch-item">
                          <span class="result-badge {badge_class}">{badge_text}</span>
                          <div class="hist-file" style="margin-top:0.4rem;">{f.name}</div>
                          <div class="metric-sub">{result.confidence*100:.1f}% confidence</div>
                        </div>"""
                batch_html += "</div>"
                progress.progress(1.0, text="Done.")
                st.markdown(batch_html, unsafe_allow_html=True)
                st.markdown(
                    '<p class="msc-muted" style="font-size:0.82rem; margin-top:1rem;">'
                    "Batch results are also saved to your analysis history and dashboard.</p>",
                    unsafe_allow_html=True,
                )
        elif not MODEL_READY:
            render_state_card("empty", "◐", "Batch analysis disabled",
                               "Train a model checkpoint first to enable batch screening.")


# =============================================================================
# PAGE: My Analysis
# =============================================================================
elif page == "My Analysis":
    st.markdown('<span class="eyebrow">Log</span>', unsafe_allow_html=True)
    st.markdown("### Recent analyses")

    recent = db.get_recent(limit=25)
    if not recent:
        render_state_card("empty", "◐", "No history yet",
                           "Analyses you run in this app are logged here, locally.")
    else:
        col_clear, _ = st.columns([1, 3])
        with col_clear:
            if st.button("Clear history", key="clear_history"):
                db.clear_history()
                st.rerun()

        rows_html = '<div class="msc-card msc-card-tight">'
        for r in recent:
            badge_class = "badge-amber" if r.is_suspicious else "badge-green"
            badge_text = "Suspicious" if r.is_suspicious else "Benign"
            rows_html += f"""
            <div class="hist-row">
              <span class="hist-file">{r.filename}</span>
              <span class="result-badge {badge_class}">{badge_text}</span>
              <span>{r.confidence*100:.1f}%</span>
              <span class="hist-time">{r.created_at[:19].replace('T', ' ')}</span>
            </div>"""
        rows_html += "</div>"
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown(
            '<p class="msc-muted" style="font-size:0.8rem; margin-top:0.75rem;">'
            "History is stored locally in a SQLite file under outputs/ — nothing "
            "leaves this machine. Saved images live alongside it under outputs/scan_images/.</p>",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: Skin Guidance
# =============================================================================
elif page == "Skin Guidance":
    st.markdown('<span class="eyebrow">Guidance</span>', unsafe_allow_html=True)
    st.markdown("### Guidance from your most recent scan")
    st.markdown(
        '<p class="msc-muted" style="font-size:0.88rem;">Full guidance (with the image and '
        "Grad-CAM) is also shown immediately after each scan on the Skin Scan page. This view "
        "is a quick way to revisit it without re-running a scan.</p>",
        unsafe_allow_html=True,
    )
    recent = db.get_recent(limit=1)
    if not recent:
        render_state_card("empty", "◐", "No scan yet",
                           "Run a scan from the Skin Scan page to see personalized guidance here.")
    else:
        latest = recent[0]
        if latest.image_path and Path(latest.image_path).exists():
            st.image(latest.image_path, width=220)
        st.markdown(
            f"<p class='msc-muted' style='font-size:0.82rem;'>Based on: <b>{latest.filename}</b> "
            f"(analyzed {latest.created_at[:19].replace('T', ' ')})</p>",
            unsafe_allow_html=True,
        )
        render_guidance_sections(latest.is_suspicious)


# =============================================================================
# PAGE: Skin Knowledge
# =============================================================================
elif page == "Skin Knowledge":
    st.markdown('<span class="eyebrow">Education</span>', unsafe_allow_html=True)
    st.markdown("### Skin Knowledge Center")
    st.markdown(
        f"<p class='msc-muted' style='font-size:0.85rem;'>{GUIDANCE_DISCLAIMER}</p>",
        unsafe_allow_html=True,
    )

    topic_labels = {k: k.replace("_", " ").title() for k in SKIN_KNOWLEDGE_CENTER}
    selected_topic_label = st.selectbox("Browse a topic", list(topic_labels.values()), key="knowledge_topic")
    selected_key = next(k for k, v in topic_labels.items() if v == selected_topic_label)
    topic = SKIN_KNOWLEDGE_CENTER[selected_key]

    st.markdown(f"#### {selected_topic_label}")
    st.markdown(f"**What:** {topic['what']}")
    if topic["common_causes"]:
        st.markdown("**Common causes / contributors:**")
        for c in topic["common_causes"]:
            st.markdown(f"- {c}")
    if topic["may_worsen_it"]:
        st.markdown('<div class="guidance-section-title">🚫 What may worsen it</div>', unsafe_allow_html=True)
        for item in topic["may_worsen_it"]:
            st.markdown(
                f"""<div class="why-card avoid-card"><div class="what">{item['action']}</div>
                <div class="why-label">Why</div><div class="why-text">{item['why']}</div></div>""",
                unsafe_allow_html=True,
            )
    if topic["general_care"]:
        st.markdown('<div class="guidance-section-title">🌿 General care</div>', unsafe_allow_html=True)
        for item in topic["general_care"]:
            st.markdown(
                f"""<div class="why-card care-card"><div class="what">{item['action']}</div>
                <div class="why-label">Why</div><div class="why-text">{item['why']}</div></div>""",
                unsafe_allow_html=True,
            )
    if topic["when_to_see_doctor"]:
        warning_items = "".join(f"<li>{w}</li>" for w in topic["when_to_see_doctor"])
        st.markdown(
            f"""<div class="derm-warning-card"><h4>🩺 When to see a dermatologist</h4>
            <ul>{warning_items}</ul></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
    st.markdown("### Ingredient Explainer")
    ingredient_labels = {k: k.replace("_", " ").title() for k in INGREDIENT_EXPLAINER}
    selected_ing_label = st.selectbox("Search an ingredient", list(ingredient_labels.values()), key="ingredient_select")
    selected_ing_key = next(k for k, v in ingredient_labels.items() if v == selected_ing_label)
    ing = INGREDIENT_EXPLAINER[selected_ing_key]

    st.markdown(
        f"""
        <div class="ingredient-card">
          <div class="ingredient-field"><div class="label">What is it?</div><div class="value">{ing['what_is_it']}</div></div>
          <div class="ingredient-field"><div class="label">Commonly used for</div><div class="value">{ing['commonly_used_for']}</div></div>
          <div class="ingredient-field"><div class="label">Why it's used</div><div class="value">{ing['why_used']}</div></div>
          <div class="ingredient-field"><div class="label">Potential benefits</div><div class="value">{', '.join(ing['potential_benefits'])}</div></div>
          <div class="ingredient-field"><div class="label">Potential irritation</div><div class="value">{ing['potential_irritation']}</div></div>
          <div class="ingredient-field"><div class="label">Who should be cautious</div><div class="value">{ing['who_should_be_cautious']}</div></div>
          <div class="ingredient-field"><div class="label">Commonly combined with</div><div class="value">{ing['commonly_combined_with']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='msc-muted' style='font-size:0.78rem; margin-top:0.6rem;'>Educational information only — "
        "not a personalized product or prescribing recommendation.</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: Progress — trend + real scan-to-scan comparison
# =============================================================================
elif page == "Progress":
    st.markdown('<span class="eyebrow">Trend</span>', unsafe_allow_html=True)
    st.markdown("### AI analysis history over time")
    st.markdown(
        '<p class="msc-muted" style="font-size:0.85rem;">This reflects the analyses you\'ve run '
        "in this app over time — it does not track the same lesion across scans, and confidence "
        "changes here are NOT evidence of disease progression.</p>",
        unsafe_allow_html=True,
    )

    recent = db.get_recent(limit=200)
    if not recent:
        render_state_card("empty", "◐", "No history yet",
                           "Run a few scans to see your analysis trend here.")
    else:
        df = pd.DataFrame([
            {"created_at": r.created_at, "confidence": r.confidence, "suspicious": int(r.is_suspicious)}
            for r in recent
        ])
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at").set_index("created_at")

        st.markdown("#### Model confidence per analysis")
        st.line_chart(df["confidence"])

        st.markdown('<hr class="hairline">', unsafe_allow_html=True)
        st.markdown("#### Suspicious-pattern rate over time (rolling)")
        rolling_window = min(10, max(2, len(df) // 3))
        df["suspicious_rate_rolling"] = df["suspicious"].rolling(rolling_window, min_periods=1).mean()
        st.line_chart(df["suspicious_rate_rolling"])
        st.markdown(
            f"<p class='msc-muted' style='font-size:0.78rem;'>Rolling average over the last "
            f"{rolling_window} analyses.</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="hairline">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Compare</span>', unsafe_allow_html=True)
        st.markdown("#### Compare two scans")

        if len(recent) >= 2:
            options = {f"#{r.id} · {r.filename} · {r.created_at[:19].replace('T', ' ')}": r.id for r in recent}
            labels = list(options.keys())
            col_a, col_b = st.columns(2)
            with col_a:
                pick_a = st.selectbox("Scan A", labels, index=0, key="compare_a")
            with col_b:
                pick_b = st.selectbox("Scan B", labels, index=1, key="compare_b")

            rec_a = db.get_by_id(options[pick_a])
            rec_b = db.get_by_id(options[pick_b])

            if rec_a and rec_b:
                st.markdown('<div class="compare-panel">', unsafe_allow_html=True)
                for rec in (rec_a, rec_b):
                    badge_class = "badge-amber" if rec.is_suspicious else "badge-green"
                    badge_text = "Suspicious" if rec.is_suspicious else "Benign"
                    st.markdown(
                        f'<div class="compare-panel-card"><div class="compare-date">'
                        f'{rec.created_at[:19].replace("T", " ")}</div>',
                        unsafe_allow_html=True,
                    )
                    if rec.image_path and Path(rec.image_path).exists():
                        st.image(rec.image_path, use_container_width=True)
                    st.markdown(
                        f'<span class="result-badge {badge_class}">{badge_text}</span> '
                        f'&nbsp; {rec.confidence*100:.1f}% confidence</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                delta = (rec_b.confidence - rec_a.confidence) * 100
                st.markdown(
                    f"<p class='msc-muted' style='font-size:0.82rem; margin-top:0.7rem;'>"
                    f"Model confidence changed by {delta:+.1f} percentage points between these two "
                    f"scans. This reflects the model's output on these two specific images — it is "
                    f"not evidence of a medically confirmed change, and does not confirm these are "
                    f"even the same lesion.</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Run at least 2 scans to compare them here.")


# =============================================================================
# PAGE: Settings
# =============================================================================
elif page == "Settings":
    st.markdown('<span class="eyebrow">Under the hood</span>', unsafe_allow_html=True)
    st.markdown("### Model information")

    checkpoint_status = "Loaded" if MODEL_READY else "Not trained yet"
    val_f1_display = f"{model_status.val_f1:.3f}" if MODEL_READY and model_status.val_f1 else "Not yet evaluated"

    st.markdown(
        f"""
        <div class="spec-grid">
          <div class="spec-item"><div class="spec-key">Architecture</div><div class="spec-val">ResNet18</div></div>
          <div class="spec-item"><div class="spec-key">Framework</div><div class="spec-val">PyTorch</div></div>
          <div class="spec-item"><div class="spec-key">Learning approach</div><div class="spec-val">Transfer learning</div></div>
          <div class="spec-item"><div class="spec-key">Dataset</div><div class="spec-val">HAM10000</div></div>
          <div class="spec-item"><div class="spec-key">Explainability</div><div class="spec-val">Grad-CAM</div></div>
          <div class="spec-item"><div class="spec-key">Interface</div><div class="spec-val">Streamlit + FastAPI</div></div>
          <div class="spec-item"><div class="spec-key">Checkpoint status</div><div class="spec-val">{checkpoint_status}</div></div>
          <div class="spec-item"><div class="spec-key">Validation F1 (best epoch)</div><div class="spec-val">{val_f1_display}</div></div>
          <div class="spec-item"><div class="spec-key">Device</div><div class="spec-val">{model_status.device.upper()}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Refresh model status", key="refresh_model"):
        predictor.reload()
        st.rerun()

    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Responsible use</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer-card">
          <h4>Important medical disclaimer</h4>
          <ul>
            <li>MedScan AI is an educational / research portfolio project, not a medical device.</li>
            <li>It does not provide a medical diagnosis and should never be treated as one.</li>
            <li>Predictions can be wrong in both directions — false positives and false negatives are possible.</li>
            <li>The underlying model has not undergone clinical validation or regulatory review.</li>
            <li>Training data (HAM10000) consists mostly of dermatoscopic images, so performance on
                ordinary smartphone photos may differ meaningfully.</li>
            <li>Dataset and labeling limitations mean real-world performance may not match test-set metrics.</li>
            <li>The Skin Knowledge Center and Ingredient Explainer contain general, static educational
                content — not personalized medical advice, and not generated by the AI model.</li>
            <li>Uploaded images are stored locally (outputs/scan_images/) only so you can review and
                compare your own past scans — they are not sent anywhere else by this app.</li>
            <li>If you have a lesion that concerns you, please consult a qualified dermatologist or physician —
                regardless of what this tool predicts.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="msc-footer">
      <div>MedScan AI — AI-assisted screening research prototype</div>
      <div class="mono">ResNet18 · HAM10000 · Grad-CAM · PyTorch · Streamlit · FastAPI</div>
    </div>
    """,
    unsafe_allow_html=True,
)
