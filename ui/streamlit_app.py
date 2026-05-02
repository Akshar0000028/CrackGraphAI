import streamlit as st
import requests
import base64
import io
import json
import os
import sys
from PIL import Image
from typing import Dict, Any, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.si_scoring import classify_risk, SIRiskThresholds

# Professional Engineering Configuration
st.set_page_config(
    page_title="CrackGraphAI - Structural Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8fafc; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .title-text {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .subtitle-text {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Severity Badges */
    .severity-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .severity-critical { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .severity-high { background: #fff7ed; color: #ea580c; border: 1px solid #fed7aa; }
    .severity-moderate { background: #fefce8; color: #ca8a04; border: 1px solid #fef08a; }
    .severity-low { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    
    /* Score Card */
    .score-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
    .score-excellent { color: #16a34a; }
    .score-good { color: #22c55e; }
    .score-fair { color: #eab308; }
    .score-poor { color: #f97316; }
    .score-critical { color: #dc2626; }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    .metric-unit {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Image Container */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .image-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    /* Upload Zone */
    .upload-zone {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.2s;
    }
    .upload-zone:hover {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    
    /* Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Analysis Report Card */
    .report-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg { background-color: #1e293b; }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Core Logic (Unchanged) ---

# --- Configuration ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# --- Helper Functions ---

def call_predict_api(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    files = {"image": (filename, image_bytes, "image/png")}
    response = requests.post(API_URL, files=files, timeout=120)
    response.raise_for_status()
    return response.json()

def b64_to_pil(data: str) -> Optional[Image.Image]:
    try:
        raw = base64.b64decode(data.encode("utf-8"))
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None

def get_severity_level(si_score: float) -> Tuple[str, str, str]:
    """Returns (level, class, description) based on SI score.
    
    Uses the damage-centric SI scoring system from features.si_scoring.
    
    Score Interpretation:
    - 0.80 - 1.00: Low Risk (Good condition)
    - 0.60 - 0.80: Moderate Risk (Minor concerns)
    - 0.40 - 0.60: High Risk (Significant concerns)
    - 0.20 - 0.40: Critical Risk (Severe damage)
    - 0.00 - 0.20: Failure Imminent (Emergency)
    """
    return classify_risk(si_score)

def get_score_color_class(si_score: float) -> str:
    """Returns CSS class for score coloring. Thresholds match severity levels."""
    if si_score >= 0.80:
        return "score-excellent"  # Green
    elif si_score >= 0.60:
        return "score-good"       # Light green
    elif si_score >= 0.40:
        return "score-fair"       # Yellow
    elif si_score >= 0.20:
        return "score-poor"       # Orange
    else:
        return "score-critical"   # Red

def format_number(val: float, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")
    
    api_url_input = st.text_input("API Endpoint", value=API_URL)
    if api_url_input != API_URL:
        API_URL = api_url_input
    
    st.markdown("### 📋 About")
    st.info("""
    **CrackGraphAI** uses advanced graph-based neural networks 
    to analyze structural cracks in infrastructure imagery.
    
    **Key Features:**
    - Semantic segmentation
    - Skeleton extraction
    - Graph topology analysis
    - Structural integrity scoring
    """)
    
    st.markdown("### 🔗 Links")
    st.markdown("- [API Docs](http://127.0.0.1:8000/docs)")
    st.markdown("- [Health Check](http://127.0.0.1:8000/health)")

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <div class="title-text">🏗️ CrackGraphAI</div>
    <div class="subtitle-text">Advanced Structural Integrity Analysis System</div>
</div>
""", unsafe_allow_html=True)

# --- Upload Section ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload infrastructure image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear photo of the structural surface for analysis",
        label_visibility="collapsed"
    )

if not uploaded_file:
    st.markdown("""
    <div style="background: white; padding: 4rem 2rem; border-radius: 16px; text-align: center; border: 2px dashed #cbd5e1; margin: 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;">
            Upload an Image to Begin Analysis
        </div>
        <div style="color: #64748b; font-size: 0.95rem;">
            Supported formats: PNG, JPG, JPEG (max 10MB)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Store image
img = Image.open(uploaded_file)
img_bytes = uploaded_file.getvalue()

# Show preview
st.markdown('<div class="section-header">📷 Image Preview</div>', unsafe_allow_html=True)
preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
with preview_col2:
    st.image(img, caption="Uploaded Image", use_container_width=True)

# Analysis Button
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    analyze_clicked = st.button("🔍 Start Structural Analysis", use_container_width=True)

if analyze_clicked:
    with st.spinner("🔬 Running AI analysis... This may take 10-30 seconds"):
        try:
            result = call_predict_api(img_bytes, uploaded_file.name)
            
            # Extract values
            si_score = result.get("si_score", 0)
            connectivity = result.get("connectivity_score", 0)
            latency = result.get("latency_seconds", 0)
            request_id = result.get("request_id", "unknown")
            features = result.get("graph_features", {})
            damage = result.get("damage_metrics", {})
            
            severity_level, severity_class, severity_desc = get_severity_level(si_score)
            score_color = get_score_color_class(si_score)
            
            # --- ANALYSIS REPORT HEADER ---
            st.markdown('<div class="section-header">📊 Analysis Report</div>', unsafe_allow_html=True)
            
            # Main Score Card
            st.markdown(f"""
            <div class="report-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                    <div>
                        <span class="severity-badge {severity_class}">{severity_level} Risk</span>
                    </div>
                    <div style="color: #64748b; font-size: 0.875rem;">
                        Request ID: <code>{request_id}</code>
                    </div>
                </div>
                <div class="score-card" style="margin-bottom: 1.5rem;">
                    <div class="score-value {score_color}">{si_score:.3f}</div>
                    <div class="score-label">Structural Integrity Score</div>
                </div>
                <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid {'#16a34a' if si_score >= 0.85 else '#eab308' if si_score >= 0.50 else '#dc2626'};">
                    <strong>Assessment:</strong> {severity_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- DAMAGE BREAKDOWN (if available) ---
            if damage:
                st.markdown('<div class="section-header">⚠️ Damage Analysis</div>', unsafe_allow_html=True)
                total_dmg = damage.get('total_damage', 0)
                dmg_color = '#dc2626' if total_dmg > 0.6 else '#eab308' if total_dmg > 0.3 else '#16a34a'
                
                damage_html = f"""
                <div class="report-card" style="margin-bottom: 1.5rem;">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                        <div style="text-align: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#dc2626' if damage.get('density_damage', 0) > 0.5 else '#64748b'};">{damage.get('density_damage', 0):.2f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Crack Density</div>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#dc2626' if damage.get('connectivity_damage', 0) > 0.5 else '#64748b'};">{damage.get('connectivity_damage', 0):.2f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Connectivity</div>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: 600; color: {'#dc2626' if damage.get('complexity_damage', 0) > 0.5 else '#64748b'};">{damage.get('complexity_damage', 0):.2f}</div>
                            <div style="font-size: 0.75rem; color: #64748b;">Complexity</div>
                        </div>
                    </div>
                    <div style="text-align: center; padding: 1rem; background: #fef2f2; border-radius: 8px; border: 2px solid {dmg_color};">
                        <div style="font-size: 2rem; font-weight: 700; color: {dmg_color};">{total_dmg:.2f}</div>
                        <div style="font-size: 0.875rem; color: #64748b;">Total Damage Score (0=Good, 1=Critical)</div>
                    </div>
                </div>
                """
                st.markdown(damage_html, unsafe_allow_html=True)
            
            # --- METRICS GRID ---
            st.markdown('<div class="section-header">📈 Structural Metrics</div>', unsafe_allow_html=True)
            
            metrics_html = f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Connectivity Score</div>
                    <div class="metric-value">{format_number(connectivity, 3)} <span class="metric-unit">/ 1.0</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Crack Length</div>
                    <div class="metric-value">{format_number(features.get('total_crack_length', 0))} <span class="metric-unit">px</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Network Branches</div>
                    <div class="metric-value">{int(features.get('num_branches', 0))} <span class="metric-unit">branches</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Graph Diameter</div>
                    <div class="metric-value">{format_number(features.get('graph_diameter', 0))} <span class="metric-unit">px</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Longest Path</div>
                    <div class="metric-value">{format_number(features.get('longest_path', 0))} <span class="metric-unit">px</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Node Degree</div>
                    <div class="metric-value">{format_number(features.get('mean_node_degree', 0), 2)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Endpoints</div>
                    <div class="metric-value">{int(features.get('endpoints', 0))}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Junctions</div>
                    <div class="metric-value">{int(features.get('junctions', 0))}</div>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # --- VISUALIZATIONS ---
            st.markdown('<div class="section-header">🎨 AI Visualizations</div>', unsafe_allow_html=True)
            
            vis_col1, vis_col2, vis_col3, vis_col4 = st.columns(4)
            
            with vis_col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown('<div class="image-label">Original Image</div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with vis_col2:
                mask_img = b64_to_pil(result.get("segmentation_mask_png_b64", ""))
                if mask_img:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.markdown('<div class="image-label">🔍 Crack Segmentation</div>', unsafe_allow_html=True)
                    st.image(mask_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with vis_col3:
                skel_img = b64_to_pil(result.get("skeleton_png_b64", ""))
                if skel_img:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.markdown('<div class="image-label">🦴 Structural Skeleton</div>', unsafe_allow_html=True)
                    st.image(skel_img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with vis_col4:
                kp_b64 = result.get("keypoints_overlay_png_b64")
                if kp_b64:
                    kp_img = b64_to_pil(kp_b64)
                    if kp_img:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown('<div class="image-label">📍 Endpoints & Junctions</div>', unsafe_allow_html=True)
                        st.image(kp_img, use_container_width=True)
                        st.markdown(
                            '<div style="font-size:0.75rem; color:#64748b; text-align:center; margin-top:0.5rem;">'
                            '🔴 Endpoints &nbsp;|&nbsp; 🟡 Junctions'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # --- DOWNLOAD & INFO ---
            st.markdown('<div class="section-header">💾 Export & Details</div>', unsafe_allow_html=True)
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                report_json = json.dumps(result, indent=2)
                st.download_button(
                    label="📥 Download Full JSON Report",
                    data=report_json,
                    file_name=f"crack_analysis_{request_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with dl_col2:
                st.metric("Processing Time", f"{latency:.2f}s", delta=None)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ **Connection Error:** Cannot reach the API server. Please ensure the backend is running on port 8000.")
        except Exception as e:
            st.error(f"❌ **Analysis Error:** {str(e)}")
