"""
app/style_extras.py — Additional design tokens layered on top of style.py
for the v2 multi-tab interface (dashboard, history, batch, empty/error states).

Kept separate from style.py rather than edited in place, so the original
design system stays intact and reviewable on its own.
"""

CSS_EXTRAS = """
<style>
/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--hairline);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 500;
  font-size: 0.92rem;
  color: var(--ink-muted);
  padding: 0.6rem 1rem;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-deep) !important;
}

/* ---------- metric cards (dashboard) ---------- */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1rem 0 0.5rem 0;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: 1fr 1fr; } }
.metric-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 1.1rem 1.3rem;
}
.metric-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-muted);
}
.metric-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.9rem;
  font-weight: 600;
  margin-top: 0.25rem;
  color: var(--ink);
}
.metric-sub { font-size: 0.78rem; color: var(--ink-muted); margin-top: 0.15rem; }

/* ---------- empty / error / success states ---------- */
.state-card {
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.8rem;
  border: 1px solid var(--hairline);
  background: var(--surface);
  text-align: center;
}
.state-card.empty { background: var(--surface-2); }
.state-card.error { background: #FDEEEC; border-color: #F3C6BE; }
.state-card.success { background: var(--signal-green-soft); border-color: #BFE3CC; }
.state-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.state-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; }
.state-desc { color: var(--ink-muted); font-size: 0.88rem; margin-top: 0.3rem; }

/* ---------- history table rows ---------- */
.hist-row {
  display: grid;
  grid-template-columns: 1.3fr 1fr 0.9fr 0.9fr;
  gap: 0.75rem;
  align-items: center;
  padding: 0.75rem 0.9rem;
  border-bottom: 1px solid var(--hairline);
  font-size: 0.86rem;
}
.hist-row:last-child { border-bottom: none; }
.hist-file { font-weight: 500; word-break: break-all; }
.hist-time { color: var(--ink-muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }

/* ---------- batch grid ---------- */
.batch-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.9rem;
  margin-top: 0.75rem;
}
.batch-item {
  border: 1px solid var(--hairline);
  border-radius: var(--radius-md);
  padding: 0.6rem;
  background: var(--surface);
}
.batch-item .badge-amber, .batch-item .badge-green {
  font-size: 0.62rem;
  padding: 0.2rem 0.5rem;
}
</style>
"""


def inject_extras(st) -> None:
    st.markdown(CSS_EXTRAS, unsafe_allow_html=True)
