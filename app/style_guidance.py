"""
app/style_guidance.py — Design tokens for the dermatology-assistant layer
(premium sidebar, guidance sections, Skin Knowledge Center, Ingredient
Explainer, subtle background texture).

Kept separate from style.py and style_extras.py, following this project's
existing convention: each stylesheet layers on top of the last without
editing it in place, so the original design system stays intact and
reviewable on its own. Reuses style.py's existing tokens (--surface,
--hairline, --accent, --ink-muted, --radius-*) rather than inventing a
second palette, so everything still reads as one coherent product.
"""

CSS_GUIDANCE = """
<style>
:root {
  --lavender: #6B5FA8;
  --lavender-soft: #EFECF8;
  --clinical-blue: #2B6CA3;
  --clinical-blue-soft: #E7F0F8;
}

/* ---------- subtle cellular/medical background texture ----------
   Very low-opacity layered radial gradients behind the existing --bg
   color, meant to read as "medical/cellular" at a glance without
   competing with foreground content. No images, no network requests. */
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

/* ---------- premium sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--hairline);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }

.msc-sidebar-brand {
  padding: 0 1.1rem 1.2rem 1.1rem;
  border-bottom: 1px solid var(--hairline);
  margin-bottom: 0.9rem;
}
.msc-sidebar-brand .name {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.15rem;
  letter-spacing: -0.01em;
  color: var(--ink);
}
.msc-sidebar-brand .tagline {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-deep);
  margin-top: 0.2rem;
}

.msc-nav-item {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.6rem 1.1rem;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 0.92rem;
  font-weight: 500;
  color: var(--ink-muted);
  border-left: 2px solid transparent;
  cursor: default;
}
.msc-nav-item.active {
  color: var(--accent-deep);
  background: var(--accent-soft);
  border-left: 2px solid var(--accent);
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

/* "avoid" cards get a warm-amber left border to visually distinguish
   from "do" cards, reusing the existing signal-amber token */
.avoid-card { border-left: 3px solid var(--signal-amber); }
.care-card { border-left: 3px solid var(--accent); }

/* contributing-factor chips */
.factor-chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.factor-chip {
  background: var(--lavender-soft);
  color: var(--lavender);
  border-radius: 999px;
  padding: 0.4rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 500;
}

/* prominent "when to see a dermatologist" card */
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

/* ---------- micro-interactions (subtle only, per project's healthcare tone) ---------- */
.msc-card, .knowledge-card, .step-card, .why-card {
  transition: box-shadow 0.15s ease;
}
.result-reveal { animation: result-fade-up 0.4s ease both; }
@keyframes result-fade-up {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


def inject_guidance(st) -> None:
    st.markdown(CSS_GUIDANCE, unsafe_allow_html=True)
