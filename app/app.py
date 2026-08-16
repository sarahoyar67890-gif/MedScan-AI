"""
app/app.py — MedScan AI Streamlit application (v2).

Run from the project root:
    streamlit run app/app.py

Architecture note: this app calls `inference.predictor` directly (in-process)
rather than over HTTP, so the demo works as a single process with no extra
moving parts. The FastAPI service in `backend/main.py` wraps the *same*
predictor module for API consumers — both share one source of truth for
model loading and inference logic.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import config
from app import style, style_extras
from inference.predictor import InferenceError, ModelNotReadyError, get_predictor
from inference.validation import ImageValidationError, validate_image_bytes
from storage import db

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="MedScan AI — Skin Lesion Screening",
    page_icon="◐",
    layout="centered",
    initial_sidebar_state="collapsed",
)
style.inject(st)
style_extras.inject_extras(st)
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


def render_result_card(result) -> None:
    badge_class = "badge-amber" if result.is_suspicious else "badge-green"
    badge_text = "Suspicious-pattern" if result.is_suspicious else "Benign-pattern"
    probs = result.probabilities
    benign_pct = probs[config.CLASS_NAMES[0]] * 100
    suspicious_pct = probs[config.CLASS_NAMES[1]] * 100

    st.markdown(
        f"""
        <div class="result-card">
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
    st.markdown('<span class="eyebrow">Explainability</span>', unsafe_allow_html=True)
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


def run_prediction(raw_bytes: bytes, filename: str, content_type: str, source: str = "streamlit"):
    """Validate + predict + record history. Returns (result, error_message)."""
    try:
        validated = validate_image_bytes(raw_bytes, filename=filename, content_type=content_type)
    except ImageValidationError as e:
        return None, str(e)

    try:
        result = predictor.predict(validated)
    except ModelNotReadyError as e:
        return None, str(e)
    except InferenceError as e:
        return None, str(e)

    db.record_analysis(
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
    )
    return result, None


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="msc-hero">
      <div>
        <span class="eyebrow">AI-Assisted Screening · Research Prototype</span>
        <h1>Preliminary, explainable<br>skin&nbsp;lesion screening.</h1>
        <p class="msc-sub">
          MedScan AI is an experimental computer-vision system that reads an uploaded
          lesion photo and returns a probability-based screening signal, together with
          a visual explanation of what the model looked at.
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
    <hr class="hairline">
    """,
    unsafe_allow_html=True,
)

if not MODEL_READY:
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

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_batch, tab_dashboard, tab_history, tab_about = st.tabs(
    ["Screening", "Batch analysis", "Dashboard", "History", "About & model info"]
)

# ---- Screening tab ---------------------------------------------------------
with tab_screen:
    st.markdown('<span class="eyebrow">Pipeline</span>', unsafe_allow_html=True)
    st.markdown("### How MedScan AI works")

    steps = [
        ("01", "Upload", "Provide a clear photo of a skin lesion (JPG or PNG)."),
        ("02", "Validate", "Size, format, and dimensions are checked before anything runs."),
        ("03", "Preprocess", "The image is resized and normalized to match training."),
        ("04", "Analyze", "A transfer-learned ResNet18 extracts and weighs visual patterns."),
        ("05", "Explain", "Grad-CAM highlights the regions that most influenced the prediction."),
        ("06", "Next step", "You're encouraged to seek a professional evaluation if concerned."),
    ]
    step_html = '<div class="step-grid">'
    for num, title, desc in steps:
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
                analyze_clicked = st.button("Analyze image", use_container_width=True,
                                             disabled=not MODEL_READY, key="analyze_single")

            if analyze_clicked:
                with st.spinner("Analyzing…"):
                    result, error = run_prediction(
                        raw_bytes, uploaded_file.name, uploaded_file.type, source="streamlit"
                    )
                if error:
                    render_state_card("error", "⚠", "Analysis failed", error)
                else:
                    st.markdown('<hr class="hairline">', unsafe_allow_html=True)
                    st.markdown('<span class="eyebrow">Result</span>', unsafe_allow_html=True)
                    render_result_card(result)

        except ImageValidationError as e:
            render_state_card("error", "⚠", "This file can't be used", str(e))

# ---- Batch analysis tab ----------------------------------------------------
with tab_batch:
    st.markdown('<span class="eyebrow">Batch</span>', unsafe_allow_html=True)
    st.markdown("### Analyze multiple images at once")
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
                result, error = run_prediction(raw, f.name, f.type, source="streamlit-batch")
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

# ---- Dashboard tab ----------------------------------------------------------
with tab_dashboard:
    st.markdown('<span class="eyebrow">Overview</span>', unsafe_allow_html=True)
    st.markdown("### Analysis dashboard")

    stats = db.get_stats()
    total = stats["total_analyses"]

    if total == 0:
        render_state_card("empty", "◐", "No analyses yet",
                           "Run a screening or batch analysis to start populating your dashboard.")
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

        st.markdown('<hr class="hairline">', unsafe_allow_html=True)
        st.markdown("#### Benign vs. suspicious-pattern")
        st.bar_chart(
            {"benign-pattern": [stats["benign_count"]], "suspicious-pattern": [stats["suspicious_count"]]},
            horizontal=True,
        )

        metrics_path = config.METRICS_JSON_PATH
        if metrics_path.exists():
            import json
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
            mcols = st.columns(4)
            for col, key, label in zip(
                mcols,
                ["accuracy", "precision", "recall", "f1_score"],
                ["Accuracy", "Precision", "Recall", "F1"],
            ):
                val = test_metrics.get(key)
                col.metric(label, f"{val*100:.1f}%" if val is not None else "—")

# ---- History tab -------------------------------------------------------------
with tab_history:
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
            "leaves this machine.</p>",
            unsafe_allow_html=True,
        )

# ---- About / model info tab --------------------------------------------------
with tab_about:
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
