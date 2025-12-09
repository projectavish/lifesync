"""
LifeSync Dashboard - Main Runner
This script launches the LifeSync Dashboard with both modules - Dashboard and Simulator.
"""

import streamlit as st
import sys
import os

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="LifeSync: Predictive Dashboard and Simulator for Personalized Wellness Forecasting",
    page_icon="💫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from modules instead of whole modules
# This avoids the set_page_config conflict
from dashboard.app_dashboard import main as dashboard_main
from dashboard.app_simulator import main as simulator_main

# Initialize session state for tab management
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
    
# Initialize simulator session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'inputs' not in st.session_state:
    st.session_state.inputs = None
if 'processed_inputs' not in st.session_state:
    st.session_state.processed_inputs = None

# Bootstrap CSS integration
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
""", unsafe_allow_html=True)

# Enhanced CSS with Custom Header and Zero Top Space
st.markdown("""
<style>
    /* Remove ALL Streamlit default spacing and padding */
    .main {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Hide default Streamlit header */
    .stApp > header {
        height: 0px !important;
        display: none !important;
    }
    
    /* Remove top padding from main content */
    .main > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Remove spacing from all elements */
    h1, h2, h3 {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove default Streamlit toolbar */
    .stApp > div:first-child {
        display: none;
    }
    
    /* Custom Header with Logo and Tabs */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        margin: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        color: white;
    }
    
    .logo-section h1 {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0 0 0 0.5rem;
    }
    
    .logo-section p {
        font-size: 1.85rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .custom-tabs {
        display: flex;
        gap: 0.5rem;
    }
    
    .custom-tab {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .custom-tab:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-1px);
        color: white;
        text-decoration: none;
    }
    
    .custom-tab.active {
        background: white;
        color: #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Hide default Streamlit tabs */
    .stTabs {
        display: none !important;
    }
    
    /* Content area styling */
    .content-area {
        padding: 1rem;
        margin: 0;
    }
    
    /* Remove excessive padding from containers */
    div[data-testid="stVerticalBlock"] > div {
        padding-bottom: 0.25rem;
    }
    
    /* Bootstrap-enhanced card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }
    
    .metric-card.primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card.success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .metric-card.warning {
        background: linear-gradient(135deg, #fcb045 0%, #fd1d1d 100%);
    }
    
    .metric-card.info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .overview-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Custom Header with Logo and Tabs
st.markdown("""
<div class="custom-header">    <div class="logo-section">
        <i class="fas fa-heart-pulse" style="font-size: 1.8rem; margin-right: 0.5rem;"></i>
        <div>
            <h1>LifeSync</h1>
            <p>A Predictive Dashboard and Simulator for Personalized Wellness Forecasting</p>
        </div>
    </div>
    
</div>
""", unsafe_allow_html=True)

# Initialize session state for tab management
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Check for tab switching using query params (only check, don't modify session state)
query_params = st.query_params
requested_tab = None
if 'tab' in query_params:
    try:
        tab_index = int(query_params['tab'])
        if tab_index in [0, 1] and tab_index != st.session_state.active_tab:
            requested_tab = tab_index
    except Exception as e:
        pass  # Silently handle any errors

# Add navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 Dashboard", key="dash_btn", use_container_width=True):
        if st.session_state.active_tab != 0:
            st.session_state.active_tab = 0
            st.rerun()
        
with col2:
    if st.button("🧪 Simulator", key="sim_btn", use_container_width=True):
        if st.session_state.active_tab != 1:
            st.session_state.active_tab = 1
            st.rerun()

# Handle query param tab switching after buttons are created
if requested_tab is not None:
    st.session_state.active_tab = requested_tab
    st.rerun()

# JavaScript to trigger the Streamlit buttons when header tabs are clicked
st.markdown(f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    const dashboardTab = document.getElementById('tab-0');
    const simulatorTab = document.getElementById('tab-1');
    
    if (dashboardTab) {{
        dashboardTab.addEventListener('click', function() {{
            // Find and click the Streamlit dashboard button
            const dashBtn = document.querySelector('[data-testid="baseButton-secondary"]:has([title*="Dashboard"]), button:contains("Dashboard")');
            if (dashBtn) {{
                dashBtn.click();
            }}
        }});
    }}
    
    if (simulatorTab) {{
        simulatorTab.addEventListener('click', function() {{
            // Find and click the Streamlit simulator button
            const simBtn = document.querySelector('[data-testid="baseButton-secondary"]:has([title*="Simulator"]), button:contains("Simulator")');
            if (simBtn) {{
                simBtn.click();
            }}
        }});
    }}
    
    // Set active tab styling
    const tabs = document.querySelectorAll('.custom-tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const activeTabIndex = {st.session_state.active_tab};
    if (activeTabIndex === 0 && dashboardTab) {{
        dashboardTab.classList.add('active');
    }} else if (activeTabIndex === 1 && simulatorTab) {{
        simulatorTab.classList.add('active');
    }}
}});
</script>
""", unsafe_allow_html=True)

# Content area with conditional rendering based on active tab
st.markdown('<div class="content-area">', unsafe_allow_html=True)

# Render content based on active tab
if st.session_state.active_tab == 0:
    # Dashboard Content
    st.markdown('<div id="dashboard-content">', unsafe_allow_html=True)
    dashboard_main()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Hide simulator content
    st.markdown('<div id="simulator-content" style="display: none;"></div>', unsafe_allow_html=True)
else:
    # Hide dashboard content
    st.markdown('<div id="dashboard-content" style="display: none;"></div>', unsafe_allow_html=True)
    
    # Simulator Content
    st.markdown('<div id="simulator-content">', unsafe_allow_html=True)
    simulator_main()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("LifeSync | A Predictive Dashboard and Simulator for Personalized Wellness Forecasting")
