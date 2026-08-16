"""
app/style.py — MedScan AI visual identity.

Design tokens (documented here so they're easy to tune as a whole):

Color
  --bg           #F5F7F6   cool paper background (not warm cream, not pure white)
  --surface      #FFFFFF   card surfaces
  --surface-2    #EFF3F2   recessed panels (upload well, footer)
  --ink          #10161B   primary text — near-black, slight blue cast
  --ink-muted    #5C6B70   secondary text
  --hairline     #DCE3E1   borders / dividers
  --accent       #0E7C7B   "scan teal" — primary brand accent
  --accent-deep  #0A5958   pressed / dark accent
  --accent-soft  #E4F2F1   tint backgrounds behind accent content
  --signal-amber #B45309   "suspicious-pattern" indicator only
  --signal-green #15803D   "benign-pattern" indicator only

Type
  Display : "Space Grotesk"   — headings, hero, big numbers (used with restraint)
  Body    : "IBM Plex Sans"   — paragraph text, UI labels
  Mono    : "IBM Plex Mono"   — eyebrows, technical readouts, metrics, dataset codes

Signature element: the "scan line" — a horizontal light band that sweeps
top-to-bottom inside a bracketed viewfinder frame. Used once, large, in the
hero, and reused (small) as the live "analyzing" state — the one motif the
product is remembered by.
"""

FONT_IMPORT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
<style>
:root {
  --bg: #F5F7F6;
  --surface: #FFFFFF;
  --surface-2: #EFF3F2;
  --ink: #10161B;
  --ink-muted: #5C6B70;
  --hairline: #DCE3E1;
  --accent: #0E7C7B;
  --accent-deep: #0A5958;
  --accent-soft: #E4F2F1;
  --signal-amber: #B45309;
  --signal-amber-soft: #FBEEE0;
  --signal-green: #15803D;
  --signal-green-soft: #E7F5EC;
  --radius-lg: 20px;
  --radius-md: 14px;
  --radius-sm: 8px;
}

/* ---------- base ---------- */
html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
  color: var(--ink);
}
.stApp {
  background: var(--bg);
}
#MainMenu, header[data-testid="stHeader"], footer {visibility: hidden; height:0;}
.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1080px;
}
h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: -0.01em;
  color: var(--ink);
}
p, span, div, label { color: var(--ink); }
.msc-muted { color: var(--ink-muted); }

.mono {
  font-family: 'IBM Plex Mono', monospace;
}
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent-deep);
  font-weight: 500;
}
.hairline {
  border: none;
  border-top: 1px solid var(--hairline);
  margin: 2.25rem 0;
}

/* ---------- hero ---------- */
.msc-hero {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 2.5rem;
  align-items: center;
  padding: 1.5rem 0 1rem 0;
}
@media (max-width: 900px) {
  .msc-hero { grid-template-columns: 1fr; }
}
.msc-hero h1 {
  font-size: 2.6rem;
  line-height: 1.08;
  margin: 0.6rem 0 0.9rem 0;
  font-weight: 600;
}
.msc-hero .msc-sub {
  font-size: 1.02rem;
  color: var(--ink-muted);
  max-width: 46ch;
  line-height: 1.55;
  margin-bottom: 0;
}
.msc-note {
  margin-top: 1.4rem;
  font-size: 0.82rem;
  color: var(--ink-muted);
  border-left: 2px solid var(--accent);
  padding-left: 0.75rem;
  max-width: 44ch;
  line-height: 1.5;
}

/* signature scan panel */
.scan-panel {
  position: relative;
  aspect-ratio: 4 / 3.4;
  background: linear-gradient(180deg, #12201F 0%, #0B1615 100%);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: 0 24px 48px -20px rgba(10, 30, 28, 0.35);
}
.scan-panel::before {
  content: "";
  position: absolute;
  inset: 22px;
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 10px;
}
.scan-corner {
  position: absolute;
  width: 22px; height: 22px;
  border-color: var(--accent);
  border-style: solid;
  border-width: 0;
  opacity: 0.9;
}
.scan-corner.tl { top: 22px; left: 22px; border-top-width: 2px; border-left-width: 2px; border-top-left-radius: 6px;}
.scan-corner.tr { top: 22px; right: 22px; border-top-width: 2px; border-right-width: 2px; border-top-right-radius: 6px;}
.scan-corner.bl { bottom: 22px; left: 22px; border-bottom-width: 2px; border-left-width: 2px; border-bottom-left-radius: 6px;}
.scan-corner.br { bottom: 22px; right: 22px; border-bottom-width: 2px; border-right-width: 2px; border-bottom-right-radius: 6px;}
.scan-line {
  position: absolute;
  left: 22px; right: 22px;
  height: 2px;
  background: linear-gradient(90deg, transparent, #5FE8D6, transparent);
  box-shadow: 0 0 14px 2px rgba(95, 232, 214, 0.55);
  animation: scan-sweep 3.6s ease-in-out infinite;
}
@keyframes scan-sweep {
  0%   { top: 26px; opacity: 0; }
  8%   { opacity: 1; }
  50%  { top: calc(100% - 30px); opacity: 1; }
  58%  { opacity: 0; }
  100% { top: 26px; opacity: 0; }
}
.scan-label {
  position: absolute;
  bottom: 34px; left: 34px; right: 34px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  color: rgba(255,255,255,0.55);
  text-transform: uppercase;
}
.scan-label b { color: #5FE8D6; font-weight: 500; }
.scan-glyph {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -55%);
  opacity: 0.5;
}

/* ---------- cards ---------- */
.msc-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
}
.msc-card-tight { padding: 1.2rem 1.4rem; }

/* ---------- process steps ---------- */
.step-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.1rem;
  margin-top: 1rem;
}
@media (max-width: 900px) { .step-grid { grid-template-columns: 1fr 1fr; } }
.step-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 1.25rem 1.3rem;
}
.step-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.75rem;
  color: var(--accent);
  font-weight: 600;
  letter-spacing: 0.04em;
}
.step-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.02rem;
  font-weight: 600;
  margin: 0.35rem 0 0.3rem 0;
}
.step-desc { font-size: 0.86rem; color: var(--ink-muted); line-height: 1.5; margin:0; }

/* ---------- result card ---------- */
.result-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-lg);
  padding: 2rem;
}
.result-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  font-weight: 600;
}
.badge-amber { background: var(--signal-amber-soft); color: var(--signal-amber); }
.badge-green { background: var(--signal-green-soft); color: var(--signal-green); }

.result-headline {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.7rem;
  font-weight: 600;
  margin: 0.75rem 0 0.15rem 0;
}
.result-confidence-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2.6rem;
  font-weight: 600;
  color: var(--accent-deep);
  line-height: 1;
}
.result-confidence-label {
  font-size: 0.78rem;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-family: 'IBM Plex Mono', monospace;
}

.prob-row { margin-bottom: 0.6rem; }
.prob-row-label {
  display: flex; justify-content: space-between;
  font-size: 0.82rem; margin-bottom: 0.3rem;
  font-family: 'IBM Plex Mono', monospace;
}
.prob-bar-track {
  height: 8px; background: var(--surface-2); border-radius: 999px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; }

/* ---------- gradcam compare ---------- */
.compare-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 0.5rem;
  text-align: center;
}
.compare-frame {
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  overflow: hidden;
}

/* ---------- disclaimer ---------- */
.disclaimer-card {
  background: #FBF7EF;
  border: 1px solid #E9DCBB;
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
}
.disclaimer-card h4 { margin-top: 0; color: #7A5B12; }
.disclaimer-card li { margin-bottom: 0.4rem; line-height: 1.5; font-size: 0.92rem; }

/* ---------- model info ---------- */
.spec-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1rem;
}
@media (max-width: 900px) { .spec-grid { grid-template-columns: 1fr 1fr; } }
.spec-item {
  border-left: 2px solid var(--hairline);
  padding-left: 0.85rem;
}
.spec-key {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
}
.spec-val { font-size: 0.95rem; font-weight: 500; margin-top: 0.15rem; }

/* ---------- footer ---------- */
.msc-footer {
  margin-top: 3rem;
  padding-top: 1.6rem;
  border-top: 1px solid var(--hairline);
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: var(--ink-muted);
}

/* ---------- streamlit widget overrides ---------- */
[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--hairline);
  border-radius: var(--radius-md);
  background: var(--surface-2);
  padding: 0.5rem;
}
.stButton > button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: var(--radius-sm);
  padding: 0.55rem 1.4rem;
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 500;
  transition: background 0.15s ease;
}
.stButton > button:hover { background: var(--accent-deep); color: white; }
.stButton > button:focus-visible { outline: 2px solid var(--accent-deep); outline-offset: 2px; }

[data-testid="stExpander"] {
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  background: var(--surface);
}
</style>
"""


def inject(st) -> None:
    st.markdown(FONT_IMPORT, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)
