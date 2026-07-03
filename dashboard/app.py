import streamlit as st
import requests
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Free-Verify Cascade · Routing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS — glassmorphism + gradient accents + micro-animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Global Background ───────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #080614 0%, #0d0b1e 35%, #120e24 65%, #080614 100%);
        color: #cdd6f4;
    }

    /* Subtle animated grain overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 20%, rgba(168,85,247,0.05) 0%, transparent 50%),
                    radial-gradient(ellipse at 50% 80%, rgba(59,130,246,0.04) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        position: relative;
    }
    .hero-header h1 {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #c084fc 60%, #f0abfc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        line-height: 1.15;
    }
    .hero-header p {
        color: #7c85a6;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.02em;
    }

    /* ── Glassmorphism Card ───────────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        box-shadow: 0 8px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    }

    /* ── Stat Cards ──────────────────────────────────────────────────── */
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(129, 140, 248, 0.3);
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .stat-sub {
        font-size: 0.75rem;
        color: #6b7280;
    }

    /* Value colors */
    .val-indigo { color: #818cf8; }
    .val-emerald { color: #34d399; }
    .val-amber { color: #fbbf24; }
    .val-rose { color: #fb7185; }
    .val-sky { color: #38bdf8; }

    /* ── Tier Badges ─────────────────────────────────────────────────── */
    .tier-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .tier-cache {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
    }
    .tier-local {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
    }
    .tier-escalated {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
    }

    /* ── Pipeline Flow Visualization ─────────────────────────────────── */
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .pipeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        text-align: center;
        min-width: 80px;
        transition: all 0.3s ease;
        position: relative;
    }
    .pipeline-node .node-icon {
        font-size: 1.3rem;
        margin-bottom: 0.2rem;
    }
    .node-active {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
    }
    .node-passed {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }
    .node-skipped {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #4b5563;
    }
    .node-failed {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
    }
    .pipeline-arrow {
        color: #4b5563;
        font-size: 1.1rem;
        margin: 0 0.15rem;
        flex-shrink: 0;
    }
    .pipeline-arrow-active {
        color: #818cf8;
    }

    /* ── Confidence Gauge ────────────────────────────────────────────── */
    .gauge-outer {
        width: 100%;
        height: 10px;
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        overflow: hidden;
        position: relative;
        margin: 0.5rem 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.6s ease;
    }
    .gauge-marker {
        position: absolute;
        top: -3px;
        height: 16px;
        width: 2px;
        background: #fbbf24;
        border-radius: 1px;
    }

    /* ── Section Labels ──────────────────────────────────────────────── */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Example Prompt Chips ────────────────────────────────────────── */
    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ── History Table ────────────────────────────────────────────────── */
    .history-row {
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .history-row:hover {
        background: rgba(99,102,241,0.06);
        border-color: rgba(99,102,241,0.2);
    }
    .history-prompt {
        font-size: 0.85rem;
        color: #cdd6f4;
        font-weight: 500;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 0.5rem;
    }
    .history-meta {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-shrink: 0;
    }

    /* ── Misc ─────────────────────────────────────────────────────────── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 50%, transparent 100%);
        margin: 1rem 0;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .text-muted { color: #6b7280; font-size: 0.8rem; }

    /* Fix Streamlit elements to match dark theme */
    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: #cdd6f4 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(129,140,248,0.5) !important;
        box-shadow: 0 0 0 2px rgba(129,140,248,0.15) !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.8rem;
    }
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8080")
HISTORY_FILE = "dashboard_history.json"
BASELINE_TOKENS_PER_QUERY = 600  # Assumed all-remote baseline cost

EXAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Solve for x: 3x + 5 = 20",
    "Write a Python function to check if a number is prime",
    "Explain quantum computing in one sentence",
    "Calculate the eigenvalues of a 4×4 matrix",
    "Design a lock-free queue in C++",
]

# ─────────────────────────────────────────────────────────────────────────────
# History persistence
# ─────────────────────────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def tier_badge(tier: str) -> str:
    tier = tier.lower()
    if tier == "cache":
        return "<span class='tier-badge tier-cache'>⚡ Cache Hit</span>"
    elif tier == "local":
        return "<span class='tier-badge tier-local'>🟢 Local Pass</span>"
    else:
        return "<span class='tier-badge tier-escalated'>🔴 Escalated</span>"

def confidence_gauge(value: float, threshold: float) -> str:
    pct = max(0, min(100, value * 100))
    thr_pct = max(0, min(100, threshold * 100))
    # Color: green if above threshold, red if below
    color = "linear-gradient(90deg, #059669, #34d399)" if value >= threshold else "linear-gradient(90deg, #dc2626, #f87171)"
    return f"""
    <div class='gauge-outer'>
        <div class='gauge-fill' style='width:{pct}%; background:{color};'></div>
        <div class='gauge-marker' style='left:{thr_pct}%;' title='Threshold: {threshold:.2f}'></div>
    </div>
    """

def pipeline_flow_html(trace: dict, tier: str) -> str:
    """Build an animated pipeline flow visualization based on the routing trace."""
    stages = [
        ("🗃️", "Cache", "cache"),
        ("📊", "Classify", "classify"),
        ("🔀", "CISC ×3", "verify"),
        ("✍️", "Refine", "refine"),
        ("⚖️", "Judge", "judge"),
        ("☁️", "Escalate", "escalate"),
    ]

    is_cache_hit = trace.get("stage") == "cache_hit"
    was_escalated = trace.get("escalated", False)

    nodes = []
    for i, (icon, label, stage_key) in enumerate(stages):
        if is_cache_hit:
            cls = "node-passed" if stage_key == "cache" else "node-skipped"
        elif stage_key == "cache":
            cls = "node-skipped"  # cache miss
        elif stage_key == "escalate":
            cls = "node-failed" if was_escalated else "node-skipped"
        else:
            cls = "node-passed"

        nodes.append(f"<div class='pipeline-node {cls}'><span class='node-icon'>{icon}</span>{label}</div>")

        if i < len(stages) - 1:
            arrow_cls = "pipeline-arrow-active" if not is_cache_hit and stage_key != "cache" else "pipeline-arrow"
            nodes.append(f"<span class='{arrow_cls} pipeline-arrow'>→</span>")

    return f"<div class='pipeline-flow'>{''.join(nodes)}</div>"

def compute_stats(history):
    total = len(history)
    cache = sum(1 for x in history if x["tier_used"] == "cache")
    local = sum(1 for x in history if x["tier_used"] == "local")
    escalated = sum(1 for x in history if x["tier_used"] == "escalated")
    tokens = sum(x["tokens_paid"] for x in history)
    baseline = total * BASELINE_TOKENS_PER_QUERY
    savings_pct = 100.0 * (1 - tokens / baseline) if baseline > 0 else 0.0
    return dict(total=total, cache=cache, local=local, escalated=escalated,
                tokens=tokens, baseline=baseline, savings_pct=savings_pct)

def send_query(prompt: str, task_id: str):
    """Post a query to the API gateway and return the response dict or None."""
    try:
        resp = requests.post(
            f"{API_URL}/solve",
            json={"task_id": task_id, "prompt": prompt, "context": {}},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        st.error(f"Connection failed ({API_URL}): {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Load state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = load_history()

history = st.session_state.history

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:2rem;'>⚡</div>
        <div style='font-size:1.1rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#c084fc);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Free-Verify Cascade</div>
        <div style='font-size:0.7rem; color:#6b7280; margin-top:0.2rem;'>AMD Hackathon · Track 1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # API Status check
    st.markdown("<div class='section-label'>🔌 API Status</div>", unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "ok" and health.get("pipeline_initialized"):
            st.success("Pipeline Online", icon="✅")
        else:
            st.warning("Pipeline Degraded", icon="⚠️")
    except Exception:
        st.error("API Offline", icon="🔴")

    st.caption(f"`{API_URL}`")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Models
    st.markdown("<div class='section-label'>🤖 Models</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;'>
        <div style='margin-bottom:0.5rem;'>
            <span style='color:#34d399; font-weight:700;'>LOCAL</span>
            <span style='color:#6b7280;'> · </span>
            <span class='mono' style='color:#a5b4fc;'>Gemma 3 4B IT</span>
            <br><span style='color:#4b5563; font-size:0.72rem;'>vLLM on ROCm · Free tokens</span>
        </div>
        <div>
            <span style='color:#fb7185; font-weight:700;'>REMOTE</span>
            <span style='color:#6b7280;'> · </span>
            <span class='mono' style='color:#a5b4fc;'>Gemma 3 27B IT</span>
            <br><span style='color:#4b5563; font-size:0.72rem;'>Fireworks API · Paid tokens</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Pipeline settings display
    st.markdown("<div class='section-label'>⚙️ Settings</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem; line-height:1.8;'>
        <span style='color:#6b7280;'>Cache Threshold:</span> <span class='mono' style='color:#cdd6f4;'>0.96</span><br>
        <span style='color:#6b7280;'>Escalation Threshold:</span> <span class='mono' style='color:#cdd6f4;'>0.72</span><br>
        <span style='color:#6b7280;'>CISC Samples:</span> <span class='mono' style='color:#cdd6f4;'>3</span><br>
        <span style='color:#6b7280;'>Refine Rounds:</span> <span class='mono' style='color:#cdd6f4;'>1</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Actions
    st.markdown("<div class='section-label'>🛠️ Actions</div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.history = []
        save_history([])
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <h1>⚡ Free-Verify Cascade</h1>
    <p>Token-Efficient Hybrid Routing Agent · AMD ROCm + Fireworks Gemma</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Live Stat Cards Row
# ─────────────────────────────────────────────────────────────────────────────
stats = compute_stats(history)

c1, c2, c3, c4, c5 = st.columns(5)
stat_items = [
    (c1, "Total Queries", str(stats["total"]), "queries routed", "val-indigo"),
    (c2, "Token Savings", f"{stats['savings_pct']:.1f}%", "vs all-remote", "val-emerald"),
    (c3, "Cache Hits", str(stats["cache"]), f"{100*stats['cache']/stats['total']:.0f}% of total" if stats["total"] else "—", "val-sky"),
    (c4, "Local Pass", str(stats["local"]), f"{100*stats['local']/stats['total']:.0f}% of total" if stats["total"] else "—", "val-amber"),
    (c5, "Escalated", str(stats["escalated"]), f"{stats['tokens']} tokens paid", "val-rose"),
]
for col, label, value, sub, cls in stat_items:
    with col:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value {cls}'>{value}</div>
            <div class='stat-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout: Input / Output
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_history = st.tabs(["🚀 Single Query", "📦 Batch Mode", "📜 History"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Single Query
# ═══════════════════════════════════════════════════════════════════════════
with tab_single:
    col_input, col_output = st.columns([5, 5])

    with col_input:
        st.markdown("<div class='section-label'>📥 Input</div>", unsafe_allow_html=True)

        # Example prompt chips
        st.markdown("**Try an example:**")
        chip_cols = st.columns(3)
        for i, example in enumerate(EXAMPLE_PROMPTS):
            with chip_cols[i % 3]:
                if st.button(example[:35] + ("…" if len(example) > 35 else ""), key=f"ex_{i}", use_container_width=True):
                    st.session_state["prompt_input"] = example

        prompt = st.text_area(
            "Enter your task prompt:",
            value=st.session_state.get("prompt_input", ""),
            placeholder="e.g. Explain the difference between TCP and UDP",
            height=140,
            key="prompt_area",
        )

        run_clicked = st.button("⚡ Route & Solve", type="primary", use_container_width=True)

        if run_clicked and prompt.strip():
            task_id = f"t_{len(history) + 1:04d}"
            start_t = time.time()

            with st.spinner("Processing through verification cascade…"):
                res = send_query(prompt, task_id)

            elapsed = time.time() - start_t

            if res:
                item = {
                    "prompt": prompt,
                    "answer": res["answer"],
                    "tier_used": res["tier_used"],
                    "tokens_paid": res["tokens_paid"],
                    "confidence": res["confidence"],
                    "trace": res["trace"],
                    "latency_ms": round(elapsed * 1000),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                history.insert(0, item)
                st.session_state.history = history
                save_history(history)
                st.session_state["prompt_input"] = ""
                st.rerun()

        elif run_clicked:
            st.warning("Please enter a prompt first.")

    with col_output:
        st.markdown("<div class='section-label'>📤 Result</div>", unsafe_allow_html=True)

        if history:
            latest = history[0]
            tier = latest["tier_used"]
            trace = latest["trace"]
            threshold = 0.72

            # Tier badge
            st.markdown(tier_badge(tier), unsafe_allow_html=True)

            # Latency tag
            latency = latest.get("latency_ms", 0)
            st.caption(f"⏱ {latency} ms" + (f"  ·  🪙 {latest['tokens_paid']} tokens paid" if latest["tokens_paid"] > 0 else "  ·  🪙 0 tokens (free)"))

            # Pipeline flow
            st.markdown(pipeline_flow_html(trace, tier), unsafe_allow_html=True)

            # Answer
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("**Answer**")
            st.code(latest["answer"], language="markdown")
            st.markdown("</div>", unsafe_allow_html=True)

            # Confidence gauge + trace detail
            if trace.get("stage") == "cache_hit":
                st.info("⚡ Direct semantic cache hit — cosine similarity ≥ 0.96. All model stages skipped.", icon="⚡")
            else:
                # Confidence gauge
                combined = trace.get("combined", 0.0)
                st.markdown(f"**Confidence** `{combined:.2f}` / threshold `{threshold:.2f}`")
                st.markdown(confidence_gauge(combined, threshold), unsafe_allow_html=True)

                # Signal breakdown
                sig_cols = st.columns(3)
                signals = [
                    ("📊 P(easy)", trace.get("p_easy", 0.5), "×0.3"),
                    ("🔀 Agreement", trace.get("agreement", 0.0), "×0.3"),
                    ("⚖️ Judge", trace.get("judge", 0.0), "×0.4"),
                ]
                for col, (label, val, weight) in zip(sig_cols, signals):
                    with col:
                        st.metric(label, f"{val:.2f}", delta=weight, delta_color="off")

                # Escalation detail
                if trace.get("escalated"):
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.markdown("**🔴 Escalation Breakdown**")
                    esc_cols = st.columns(3)
                    with esc_cols[0]:
                        st.metric("Prompt Tokens", trace.get("prompt_tokens", 0))
                    with esc_cols[1]:
                        st.metric("Cached Tokens", trace.get("cached_tokens", 0))
                    with esc_cols[2]:
                        st.metric("Completion Tokens", trace.get("completion_tokens", 0))
                else:
                    st.success(f"✅ Local answer accepted — confidence {combined:.2f} ≥ threshold {threshold:.2f}")
        else:
            st.markdown("""
            <div class='glass-card' style='text-align:center; padding:3rem;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>🧪</div>
                <div style='font-size:1.1rem; font-weight:600; color:#a5b4fc;'>No queries yet</div>
                <div style='color:#6b7280; margin-top:0.5rem;'>Enter a prompt and click Route & Solve to begin</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Batch Mode
# ═══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div class='section-label'>📦 Batch Query Mode</div>", unsafe_allow_html=True)
    st.markdown("Run multiple prompts in sequence to stress-test routing and measure aggregate savings.")

    batch_input = st.text_area(
        "Enter one prompt per line:",
        placeholder="What is 2 + 2?\nExplain photosynthesis\nDerive the quadratic formula\nWho painted the Mona Lisa?",
        height=180,
        key="batch_input",
    )

    batch_clicked = st.button("🚀 Run Batch", type="primary", use_container_width=True, key="batch_btn")

    if batch_clicked:
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Enter at least one prompt.")
        else:
            progress = st.progress(0, text="Starting batch…")
            batch_results = []

            for i, line in enumerate(lines):
                progress.progress((i + 1) / len(lines), text=f"Processing {i+1}/{len(lines)}: {line[:40]}…")
                task_id = f"batch_{len(history) + 1:04d}"
                start_t = time.time()
                res = send_query(line, task_id)
                elapsed = time.time() - start_t

                if res:
                    item = {
                        "prompt": line,
                        "answer": res["answer"],
                        "tier_used": res["tier_used"],
                        "tokens_paid": res["tokens_paid"],
                        "confidence": res["confidence"],
                        "trace": res["trace"],
                        "latency_ms": round(elapsed * 1000),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    history.insert(0, item)
                    batch_results.append(item)

            st.session_state.history = history
            save_history(history)
            progress.empty()

            if batch_results:
                st.success(f"✅ Batch complete — {len(batch_results)} queries processed.")

                # Batch summary
                b_stats = compute_stats(batch_results)
                b_cols = st.columns(4)
                with b_cols[0]:
                    st.metric("Queries", b_stats["total"])
                with b_cols[1]:
                    st.metric("Token Savings", f"{b_stats['savings_pct']:.1f}%")
                with b_cols[2]:
                    st.metric("Tokens Paid", b_stats["tokens"])
                with b_cols[3]:
                    avg_lat = np.mean([r["latency_ms"] for r in batch_results])
                    st.metric("Avg Latency", f"{avg_lat:.0f} ms")

                # Batch detail table
                batch_df = pd.DataFrame([{
                    "Prompt": r["prompt"][:50],
                    "Tier": r["tier_used"].upper(),
                    "Tokens": r["tokens_paid"],
                    "Confidence": f"{r['confidence']:.2f}",
                    "Latency": f"{r['latency_ms']} ms",
                } for r in batch_results])
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: History
# ═══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-label'>📜 Query History</div>", unsafe_allow_html=True)

    if not history:
        st.info("No queries in history yet. Run a query to see results here.")
    else:
        # Filters
        filter_cols = st.columns([2, 2, 2])
        with filter_cols[0]:
            tier_filter = st.multiselect("Filter by tier", ["cache", "local", "escalated"], default=["cache", "local", "escalated"])
        with filter_cols[1]:
            sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Highest confidence", "Most tokens"])
        with filter_cols[2]:
            search_q = st.text_input("Search prompts", placeholder="keyword…")

        filtered = [h for h in history if h["tier_used"] in tier_filter]
        if search_q:
            filtered = [h for h in filtered if search_q.lower() in h["prompt"].lower()]

        if sort_by == "Oldest first":
            filtered = list(reversed(filtered))
        elif sort_by == "Highest confidence":
            filtered = sorted(filtered, key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "Most tokens":
            filtered = sorted(filtered, key=lambda x: x["tokens_paid"], reverse=True)

        st.caption(f"Showing {len(filtered)} of {len(history)} entries")

        # Routing distribution chart
        if filtered:
            chart_data = pd.DataFrame({
                "Tier": ["Cache Hit", "Local Pass", "Escalated"],
                "Count": [
                    sum(1 for x in filtered if x["tier_used"] == "cache"),
                    sum(1 for x in filtered if x["tier_used"] == "local"),
                    sum(1 for x in filtered if x["tier_used"] == "escalated"),
                ],
            })
            st.bar_chart(chart_data, x="Tier", y="Count", color="#818cf8", height=200)

        # History entries
        for i, entry in enumerate(filtered):
            with st.expander(f"{tier_badge(entry['tier_used'])}  {entry['prompt'][:60]}{'…' if len(entry['prompt']) > 60 else ''}", expanded=False):
                st.markdown(tier_badge(entry["tier_used"]), unsafe_allow_html=True)

                detail_cols = st.columns([3, 1, 1, 1])
                with detail_cols[0]:
                    st.markdown(f"**Prompt:** {entry['prompt']}")
                with detail_cols[1]:
                    st.metric("Tokens", entry["tokens_paid"])
                with detail_cols[2]:
                    st.metric("Confidence", f"{entry['confidence']:.2f}")
                with detail_cols[3]:
                    st.metric("Latency", f"{entry.get('latency_ms', '—')} ms")

                st.markdown("**Answer:**")
                st.code(entry["answer"], language="markdown")

                # Trace
                trace = entry["trace"]
                if trace.get("stage") != "cache_hit":
                    st.markdown(pipeline_flow_html(trace, entry["tier_used"]), unsafe_allow_html=True)
                    t_cols = st.columns(4)
                    with t_cols[0]:
                        st.metric("P(easy)", f"{trace.get('p_easy', 0.5):.2f}")
                    with t_cols[1]:
                        st.metric("Agreement", f"{trace.get('agreement', 0.0):.2f}")
                    with t_cols[2]:
                        st.metric("Judge", f"{trace.get('judge', 0.0):.2f}")
                    with t_cols[3]:
                        st.metric("Combined", f"{trace.get('combined', 0.0):.2f}")

                st.caption(entry.get("timestamp", ""))
