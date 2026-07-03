import streamlit as st
import requests
import json
import os
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Free-Verify Cascade Routing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown CSS injection
st.markdown("""
<style>
    /* Fonts and global colors */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #06050e 100%);
        color: #e2e8f0;
    }
    
    /* Header styling */
    .title-container {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .custom-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    
    /* Routing badges */
    .badge-cache {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-local {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-escalated {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint config
API_URL = os.getenv("API_URL", "http://localhost:8080")
HISTORY_FILE = "dashboard_history.json"

# Load history
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Save history
def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

# App header
st.markdown("<div class='title-container'><h1>⚡ Free-Verify Cascade</h1><h3>Hybrid Token-Efficient Routing Agent Dashboard</h3></div>", unsafe_allow_html=True)

history = load_history()

# Sidebar for configuration stats & controls
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.info(f"API Target: `{API_URL}`")
    
    # Model info
    st.markdown("### 🤖 Models in Cascade")
    st.markdown("- **Local (Free):** `Gemma 3 4B IT`")
    st.markdown("- **Remote (Paid):** `Gemma 3 27B IT`")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        history = []
        save_history(history)
        st.success("History cleared!")
        st.rerun()

# Layout: Main columns
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📥 Run Query")
    
    # Input prompt
    prompt = st.text_area("Enter a task prompt:", placeholder="e.g. Solve the equation: 5x + 3 = 18", height=120)
    
    if st.button("🚀 Route & Solve", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt first!")
        else:
            with st.spinner("Processing through verification cascade..."):
                try:
                    # Request target API
                    resp = requests.post(
                        f"{API_URL}/solve",
                        json={"task_id": f"t_{len(history) + 1:03d}", "prompt": prompt, "context": {}},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        res_data = resp.json()
                        
                        # Save to history
                        new_item = {
                            "prompt": prompt,
                            "answer": res_data["answer"],
                            "tier_used": res_data["tier_used"],
                            "tokens_paid": res_data["tokens_paid"],
                            "confidence": res_data["confidence"],
                            "trace": res_data["trace"]
                        }
                        history.insert(0, new_item)
                        save_history(history)
                        st.success("Query solved!")
                    else:
                        st.error(f"API returned error code {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API Gateway at {API_URL}: {str(e)}")

    # Main output showing the latest result
    if history:
        latest = history[0]
        st.markdown("---")
        st.markdown("### 📤 Output & Trace Details")
        
        # Tier badge selector
        tier = latest["tier_used"].lower()
        badge_html = ""
        if tier == "cache":
            badge_html = "<span class='badge-cache'>⚡ SEMANTIC CACHE HIT</span>"
        elif tier == "local":
            badge_html = "<span class='badge-local'>🟢 LOCAL GENERATION PASS</span>"
        else:
            badge_html = "<span class='badge-escalated'>🔴 ESCALATED TO FIREWORKS</span>"
            
        st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)
        
        # Answer display
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("**Answer:**")
        st.code(latest["answer"], language="markdown")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Trace flow chart
        st.markdown("#### 🔍 Verification Trace")
        trace = latest["trace"]
        
        if "stage" in trace and trace["stage"] == "cache_hit":
            st.info("Direct hit on Semantic Cache (Cosine similarity >= 0.96). All model runs skipped.")
        else:
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            
            with t_col1:
                st.metric("Classifier (P-easy)", f"{trace.get('p_easy', 0.5):.2f}")
            with t_col2:
                st.metric("CISC Agreement", f"{trace.get('agreement', 0.0):.2f}")
            with t_col3:
                st.metric("LLM Judge Score", f"{trace.get('judge', 0.0):.2f}")
            with t_col4:
                st.metric("Combined Confidence", f"{trace.get('combined', 0.0):.2f}")
                
            # Escalation visualization
            threshold_val = 0.72  # Loaded from env
            combined_val = trace.get('combined', 0.0)
            
            st.markdown(f"**Escalation Decision Progress Bar** (Threshold: `{threshold_val:.2f}`):")
            st.progress(min(1.0, combined_val))
            
            if combined_val >= threshold_val:
                st.success(f"Combined Confidence ({combined_val:.2f}) >= Threshold ({threshold_val:.2f}) -> Trusting Local Answer.")
            else:
                st.warning(f"Combined Confidence ({combined_val:.2f}) < Threshold ({threshold_val:.2f}) -> Escalated to remote Gemma-3 27B. paid {latest['tokens_paid']} tokens.")
                if "prompt_tokens" in trace:
                    st.caption(f"Fireworks breakdown: Prompt: {trace['prompt_tokens']} (Cached: {trace.get('cached_tokens', 0)}) | Completion: {trace['completion_tokens']}")

with col_right:
    st.markdown("### 📊 Metrics Summary")
    
    # Calculate running statistics
    total_runs = len(history)
    cache_hits = sum(1 for x in history if x["tier_used"] == "cache")
    local_runs = sum(1 for x in history if x["tier_used"] == "local")
    escalated_runs = sum(1 for x in history if x["tier_used"] == "escalated")
    
    total_tokens = sum(x["tokens_paid"] for x in history)
    
    # Baseline: if we called Fireworks on 100% of these queries (assume ~600 tokens average per query input+output)
    baseline_tokens = total_runs * 600
    token_savings = 100.0 * (1 - (total_tokens / baseline_tokens)) if baseline_tokens > 0 else 0.0
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("💰 Total Runs")
        st.markdown(f"<div class='metric-val'>{total_runs}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("📈 Token Savings")
        st.markdown(f"<div class='metric-val'>{token_savings:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("vs all-remote baseline")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Segment stats
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Routing Distribution")
    dist_df = pd.DataFrame({
        "Tier": ["Semantic Cache", "Local Gemma 4B", "Escalated to 27B"],
        "Count": [cache_hits, local_runs, escalated_runs]
    })
    st.bar_chart(dist_df, x="Tier", y="Count", color="#818cf8")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # History logs
    st.markdown("#### 📜 History Log")
    if history:
        hist_data = []
        for x in history:
            hist_data.append({
                "Prompt Snippet": x["prompt"][:40] + "...",
                "Tier": x["tier_used"].upper(),
                "Tokens": x["tokens_paid"],
                "Confidence": f"{x['confidence']:.2f}"
            })
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True)
    else:
        st.info("No queries run yet.")
