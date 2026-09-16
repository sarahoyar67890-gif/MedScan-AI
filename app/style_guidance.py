"""
app/style_guidance.py (v2) — Design tokens for the dermatology-assistant
layer: a genuinely distinct sidebar identity (dark gradient, matching the
existing scan-panel aesthetic already used on the hero), a clearly visible
active-nav state, guidance sections, Skin Knowledge Center, Ingredient
Explainer, image-quality badges, scan-comparison layout, and a subtle
background texture.

v2 change from the first version: the sidebar is no longer a plain white
panel with a hairline border — it now uses the same dark teal gradient as
the hero's scan-panel (--accent-deep → near-black), so the brand identity
reads as one deliberate product instead of "default Streamlit + a few CSS
tweaks". The nav buttons (rendered as real st.button widgets in app.py) are
styled entirely from here — this file is the single source for that CSS,
replacing the inline block that used to live in app.py.

Still reuses style.py's tokens (--accent, --radius-*, --hairline) for
everything outside the sidebar, so the rest of the app stays visually
consistent with the original design system rather than introducing a
second competing palette.
"""

CSS_GUIDANCE = """
<style>
:root {
  --lavender: #6B5FA8;
  --lavender-soft: #EFECF8;
  --clinical-blue: #2B6CA3;
  --clinical-blue-soft: #E7F0F8;
  --sidebar-bg-top: #0F211F;
  --sidebar-bg-bottom: #081413;
  --sidebar-ink: #E7F3F1;
  --sidebar-ink-muted: #8FA8A4;
}

/* ---------- subtle cellular/medical background texture (main content) ---------- */
.stApp {
  background-color: var(--bg);
  background-image:
    radial-gradient(circle at 8% 12%, rgba(14, 124, 123, 0.05) 0, transparent 9%),
    radial-gradient(circle at 92% 8%, rgba(107, 95, 168, 0.045) 0, transparent 11%),
    radial-gradient(circle at 15% 85%, rgba(43, 108, 163, 0.04) 0, transparent 10%),
    radial-gradient(circle at 85% 92%, rgba(14, 124, 123, 0.04) 0, transparent 12%),
    radial-gradient(circle at 50% 50%, rgba(107, 95, 168, 0.025) 0, transparent 22%);
  background-attachment: fixed;
}

/* ---------- premium sidebar: dark gradient, matches the hero scan-panel ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--sidebar-bg-top) 0%, var(--sidebar-bg-bottom) 100%);
  border-right: none;
}
[data-testid="stSidebar"] * { color: var(--sidebar-ink); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }

.msc-sidebar-brand {
  padding: 0 1.2rem 1.3rem 1.2rem;
  border-bottom: 1px solid rgba(255,255,255,0.09);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.msc-sidebar-brand .glyph {
  width: 34px; height: 34px;
  border-radius: 9px;
  background: linear-gradient(135deg, #5FE8D6 0%, var(--accent) 100%);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  color: #0A1817;
  font-size: 1rem;
  flex-shrink: 0;
}
.msc-sidebar-brand .brand-text .name {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.08rem;
  letter-spacing: -0.01em;
  color: #FFFFFF;
  line-height: 1.2;
}
.msc-sidebar-brand .brand-text .tagline {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #5FE8D6;
  margin-top: 0.15rem;
}

.msc-sidebar-section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--sidebar-ink-muted);
  padding: 0 1.2rem;
  margin: 0.6rem 0 0.35rem 0;
}

/* nav buttons are real st.button widgets, styled here */
[data-testid="stSidebar"] .stButton { margin-bottom: 0.15rem; }
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  color: var(--sidebar-ink-muted) !important;
  border: none !important;
  border-radius: 10px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  padding: 0.6rem 0.9rem !important;
  width: 100% !important;
  box-shadow: none !important;
  transition: background 0.15s ease, color 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,255,255,0.06) !important;
  color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(90deg, rgba(95,232,214,0.16), rgba(95,232,214,0.05)) !important;
  color: #5FE8D6 !important;
  font-weight: 600 !important;
  box-shadow: inset 3px 0 0 0 #5FE8D6 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: linear-gradient(90deg, rgba(95,232,214,0.22), rgba(95,232,214,0.08)) !important;
  color: #5FE8D6 !important;
}

.msc-sidebar-footer {
  position: sticky;
  bottom: 0;
  padding: 1rem 1.2rem;
  margin-top: 1.5rem;
  border-top: 1px solid rgba(255,255,255,0.09);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem;
  color: var(--sidebar-ink-muted);
  line-height: 1.5;
}

/* ---------- guidance sections (result page) ---------- */
.guidance-section { margin: 2rem 0; }
.guidance-section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* "what/why" cards used for contributing factors, avoid, and care sections */
.why-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
  box-shadow: 0 2px 8px -4px rgba(10, 30, 28, 0.08);
}
.why-card .what {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--ink);
}
.why-card .why-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-top: 0.5rem;
}
.why-card .why-text {
  font-size: 0.88rem;
  color: var(--ink-muted);
  line-height: 1.5;
  margin-top: 0.15rem;
}
.avoid-card { border-left: 3px solid var(--signal-amber); }
.care-card { border-left: 3px solid var(--accent); }

.factor-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.factor-chip {
  background: var(--lavender-soft);
  color: var(--lavender);
  border-radius: 999px;
  padding: 0.4rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 500;
}

.derm-warning-card {
  background: var(--clinical-blue-soft);
  border: 1px solid #C7DCEC;
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
}
.derm-warning-card h4 { margin-top: 0; color: var(--clinical-blue); }
.derm-warning-card li { margin-bottom: 0.4rem; line-height: 1.5; font-size: 0.92rem; }

/* ---------- Skin Knowledge Center ---------- */
.knowledge-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1rem;
}
@media (max-width: 900px) { .knowledge-grid { grid-template-columns: 1fr 1fr; } }
.knowledge-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 1.2rem 1.3rem;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.knowledge-card:hover {
  box-shadow: 0 10px 24px -12px rgba(10, 30, 28, 0.18);
  transform: translateY(-1px);
}
.knowledge-card .title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 1rem;
}
.knowledge-card .desc {
  font-size: 0.84rem;
  color: var(--ink-muted);
  margin-top: 0.35rem;
  line-height: 1.5;
}

/* ---------- Ingredient Explainer ---------- */
.ingredient-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
}
.ingredient-field { margin-bottom: 0.9rem; }
.ingredient-field .label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-muted);
}
.ingredient-field .value { font-size: 0.92rem; margin-top: 0.2rem; line-height: 1.5; }

/* ---------- scan-flow progress indicator ---------- */
.scan-progress {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 1.2rem 0 1.6rem 0;
  flex-wrap: wrap;
}
.scan-progress-step {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  color: var(--ink-muted);
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: var(--surface-2);
}
.scan-progress-step.done { background: var(--accent-soft); color: var(--accent-deep); }
.scan-progress-step.current { background: var(--accent); color: white; }
.scan-progress-arrow { color: var(--hairline); font-size: 0.8rem; }

/* ---------- image quality badges (pre-analysis feedback) ---------- */
.quality-badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.8rem 0; }
.quality-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  font-weight: 600;
}
.quality-badge.good { background: var(--signal-green-soft); color: var(--signal-green); }
.quality-badge.acceptable { background: var(--signal-amber-soft); color: var(--signal-amber); }
.quality-badge.poor { background: #FDEEEC; color: #C0392B; }
.quality-warning-list { font-size: 0.85rem; color: var(--ink-muted); margin-top: 0.4rem; line-height: 1.6; }

/* ---------- scan comparison ---------- */
.compare-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
}
@media (max-width: 800px) { .compare-panel { grid-template-columns: 1fr; } }
.compare-panel-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 1.1rem 1.2rem;
}
.compare-panel-card .compare-date {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: var(--ink-muted);
  margin-bottom: 0.5rem;
}

/* ---------- micro-interactions (subtle only, per project's healthcare tone) ---------- */
.msc-card, .knowledge-card, .step-card, .why-card { transition: box-shadow 0.15s ease; }
.result-reveal { animation: result-fade-up 0.4s ease both; }
@keyframes result-fade-up {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


def inject_guidance(st) -> None:
    st.markdown(CSS_GUIDANCE, unsafe_allow_html=True)
