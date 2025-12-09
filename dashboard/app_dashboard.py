"""
LifeSync Dashboard - App Dashboard (Reorganized)
This Streamlit app displays visualizations and insights from models trained on lifestyle and mental health data.
Organized as per user requirements:
1. Filters Section
2. Top Overview Section (4 Metrics)
3. Lifestyle Factor Distributions (12 Charts)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
import shap
from PIL import Image

# Set matplotlib and seaborn style for better appearance
plt.style.use('default')
sns.set_palette("husl")

# Bootstrap and Font Awesome integration
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
""", unsafe_allow_html=True)

# Enhanced CSS with Bootstrap integration
st.markdown("""
<style>
    /* Compact main container */
    .main {
        padding: 0.25rem 0.5rem;
    }
    
    /* Compact headers */
    h1, h2, h3 {
        padding-top: 0.25rem;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    
    /* Remove excessive spacing from containers */
    div[data-testid="stVerticalBlock"] > div {
        padding-bottom: 0.1rem;
    }
    
    /* Bootstrap-enhanced metric box styling */
    .metric-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-2px);
    }
    
    /* Overview metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.25rem;
        color: white;
        box-shadow: 0 4px 20px rgba(31, 38, 135, 0.25);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 0.75rem;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(31, 38, 135, 0.4);
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
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .overview-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chart section styling */
    .chart-title {
        font-size: 1rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 8px;
        border-bottom: 2px solid #4e8df5;
        padding-bottom: 5px;
    }
    
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 18px;
        border: 1px solid #eaeaea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Filter container styling */
    .filter-container {
        background: rgba(248, 249, 250, 0.8);
        backdrop-filter: blur(10px);
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .compact-filter {
        background: rgba(240, 242, 246, 0.8);
        backdrop-filter: blur(5px);
        padding: 6px 10px;
        border-radius: 8px;
        margin-bottom: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .compact-filter .stSelectbox, 
    .compact-filter .stMultiSelect,
    .compact-filter .stSlider,
    .compact-filter .stSelectSlider {
        padding-bottom: 0;
        margin-bottom: 0;
    }
    
    /* Make sliders more compact */
    .stSlider {
        padding-top: 0;
        padding-bottom: 0.5rem;
    }
    
    .stSelectSlider {
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Path configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Mental_Health_Lifestyle_Dataset.csv")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Load feature importance data
@st.cache_data
def load_feature_importance():
    fi_path = os.path.join(MODELS_PATH, "feature_importance.csv")
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None

# Load SHAP images
def load_shap_images():
    images = {}
    shap_files = {
        "Happiness Summary": "shap_summary_happiness.png",
        "Happiness Dot": "shap_dot_happiness.png",
        "Stress Summary": "shap_summary_stress.png",
        "Stress Dot": "shap_dot_stress.png"
    }
    
    for name, filename in shap_files.items():
        path = os.path.join(MODELS_PATH, filename)
        if os.path.exists(path):
            images[name] = Image.open(path)
    
    return images

# Utility function to create charts with automatic cleanup
def create_and_display_chart(chart_func, title, container_class='chart-container'):
    """Helper function to create matplotlib charts with automatic cleanup"""
    st.markdown(f"<div class='{container_class}'>", unsafe_allow_html=True)
    st.markdown(f"<p class='chart-title'>{title}</p>", unsafe_allow_html=True)
    
    fig = chart_func()
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Clean up memory
    
    st.markdown("</div>", unsafe_allow_html=True)

def reset_filters_callback():
    """Callback function to reset all filters"""
    # Clear all filter-related keys from session state
    filter_keys = [key for key in st.session_state.keys() if key.startswith("country_filter_") or
                   key.startswith("gender_filter_") or
                   key.startswith("exercise_filter_") or
                   key.startswith("diet_filter_") or
                   key.startswith("mh_filter_") or
                   key.startswith("age_filter_") or
                   key.startswith("sleep_filter_")]
    for key in filter_keys:
        del st.session_state[key]

    # Use timestamp to ensure truly unique keys for each reset
    import time
    st.session_state.reset_key = str(int(time.time() * 1000))

def main():
    # Load data
    df = load_data()
    
    # --- 1. FILTERS SECTION ---
    
    st.markdown("<h3 style='margin-bottom: 8px; font-size: 1.2rem;'>🔍 Filters</h3>", unsafe_allow_html=True)
    
    # Create filter columns
    filter_container = st.container()
    filter_cols = filter_container.columns([1, 1, 1, 1, 1])
      # Check if reset was triggered - use timestamp-based unique key
    reset_key = st.session_state.get('reset_key', 'default')
    
    # Country filter
    with filter_cols[0]:
        countries = sorted([str(x) for x in df["Country"].unique()])
        filter_key = f"country_filter_{reset_key}"
        selected_countries = st.multiselect("Country", countries, default=[], key=filter_key, 
                                            label_visibility="collapsed", placeholder="Country")
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Country</p>", unsafe_allow_html=True)
    
    # Gender filter  
    with filter_cols[1]:
        genders = sorted([str(x) for x in df["Gender"].unique()])
        filter_key = f"gender_filter_{reset_key}"
        selected_genders = st.multiselect("Gender", genders, default=[], key=filter_key, 
                                          label_visibility="collapsed", placeholder="Gender")
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Gender</p>", unsafe_allow_html=True)
    
    # Exercise filter
    with filter_cols[2]:
        exercise_levels = sorted([str(x) for x in df["Exercise Level"].unique()])
        filter_key = f"exercise_filter_{reset_key}"
        selected_exercise_levels = st.multiselect("Exercise", exercise_levels, default=[], key=filter_key, 
                                                  label_visibility="collapsed", placeholder="Exercise")
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Exercise</p>", unsafe_allow_html=True)
    
    # Diet filter
    with filter_cols[3]:
        diet_types = sorted([str(x) for x in df["Diet Type"].unique()])
        filter_key = f"diet_filter_{reset_key}"
        selected_diet_types = st.multiselect("Diet", diet_types, default=[], key=filter_key, 
                                             label_visibility="collapsed", placeholder="Diet")
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Diet</p>", unsafe_allow_html=True)
    
    # Mental Health filter
    with filter_cols[4]:
        mh_unique = df["Mental Health Condition"].dropna().unique()
        mh_conditions = sorted([str(x) for x in mh_unique])
        filter_key = f"mh_filter_{reset_key}"
        selected_mh_conditions = st.multiselect("Mental Health", mh_conditions, default=[], key=filter_key, 
                                                label_visibility="collapsed", placeholder="Mental Health")
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Mental Health</p>", unsafe_allow_html=True)
    
    # Second row with range sliders
    slider_cols = st.columns([1, 1, 1])
      # Age slider
    with slider_cols[0]:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Age Range</p>", unsafe_allow_html=True)
        filter_key = f"age_filter_{reset_key}"
        selected_age_range = st.slider("Age", age_min, age_max, (age_min, age_max), key=filter_key, 
                                      label_visibility="collapsed")      # Sleep slider
    with slider_cols[1]:
        sleep_min, sleep_max = float(df["Sleep Hours"].min()), float(df["Sleep Hours"].max())
        st.markdown("<p style='font-size:0.8rem; margin:0; padding:0'>Sleep Hours</p>", unsafe_allow_html=True)
        filter_key = f"sleep_filter_{reset_key}"
        selected_sleep_range = st.slider("Sleep Hours", sleep_min, sleep_max, (sleep_min, sleep_max), 
                                        step=0.1, key=filter_key, label_visibility="collapsed")# Reset button
    with slider_cols[2]:
        st.button("Reset Filters", use_container_width=True, on_click=reset_filters_callback)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close filter container
      # Apply filters to data
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
    if selected_genders:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
    if selected_exercise_levels:
        filtered_df = filtered_df[filtered_df["Exercise Level"].isin(selected_exercise_levels)]
    if selected_diet_types:
        filtered_df = filtered_df[filtered_df["Diet Type"].isin(selected_diet_types)]
    if selected_mh_conditions:
        # Only filter when specific conditions are selected
        # This excludes entries with null Mental Health values when filtering
        filtered_df = filtered_df[filtered_df["Mental Health Condition"].isin(selected_mh_conditions)]
    # When no Mental Health conditions are selected, include all entries (including nulls)
    filtered_df = filtered_df[filtered_df["Age"].between(selected_age_range[0], selected_age_range[1])]
    filtered_df = filtered_df[filtered_df["Sleep Hours"].between(selected_sleep_range[0], selected_sleep_range[1])]
    
    # --- 2. TOP OVERVIEW SECTION (4 METRICS) ---
    st.markdown("""
    <div class="overview-container mb-4">
        <h4 class="text-center mb-3" style="color: #2c3e50;">
            <i class="fas fa-chart-line"></i> Dataset Overview
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    total_entries = len(df)
    selected_entries = len(filtered_df)
    avg_happiness = filtered_df["Happiness Score"].mean() if not filtered_df.empty else 0
    
    # Handle stress calculation
    try:
        pd.to_numeric(filtered_df["Stress Level"])
        avg_stress = filtered_df["Stress Level"].mean() if not filtered_df.empty else 0
        stress_scale = "/10"
    except:
        stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
        stress_numeric = filtered_df["Stress Level"].map(stress_mapping) if not filtered_df.empty else pd.Series([0])
        avg_stress = stress_numeric.mean()
        stress_scale = "/3"
    
    # Display 4 metric cards in a row
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card primary">
            <div class="metric-number">{total_entries:,}</div>
            <div class="metric-label">
                <i class="fas fa-database"></i> Total Entries
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card info">
            <div class="metric-number">{selected_entries:,}</div>
            <div class="metric-label">
                <i class="fas fa-filter"></i> Selected Entries
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-number">{avg_happiness:.1f}/10</div>
            <div class="metric-label">
                <i class="fas fa-smile"></i> Avg Happiness Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown(f"""
        <div class="metric-card warning">
            <div class="metric-number">{avg_stress:.1f}{stress_scale}</div>
            <div class="metric-label">
                <i class="fas fa-brain"></i> Avg Stress Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- 3. LIFESTYLE FACTOR DISTRIBUTIONS (12 CHARTS) ---
   
    st.markdown("<h2 style='text-align:center; color:#2c3e50; margin-bottom:20px;'>📊 Lifestyle Factor Distributions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#555; margin-bottom:25px;'>12 filter-responsive charts showing distribution of key lifestyle factors</p>", unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Row 1: Charts 2.1-2.4 (Demographics & Personal Factors)
        st.markdown("### 🌍 Demographics & Personal Factors")
        row1_cols = st.columns(4)
        
        # 2.1 Country Distribution (Bar Chart)
        with row1_cols[0]:
            
            st.markdown("<p class='chart-title'>🌍 Country Distribution</p>", unsafe_allow_html=True)
            country_counts = filtered_df['Country'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(range(len(country_counts)), country_counts.values, color='#667eea')
            ax.set_xticks(range(len(country_counts)))
            ax.set_xticklabels(country_counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.2 Age Group Distribution (Histogram)
        with row1_cols[1]:
            
            st.markdown("<p class='chart-title'>📊 Age Distribution</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(20, len(filtered_df['Age'].unique()))
            ax.hist(filtered_df['Age'], bins=bins, color='#11998e', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Age', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.3 Gender (Pie Chart)
        with row1_cols[2]:
            
            st.markdown("<p class='chart-title'>⚧ Gender Distribution</p>", unsafe_allow_html=True)
            gender_counts = filtered_df['Gender'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                              autopct='%1.1f%%', colors=colors[:len(gender_counts)])
            
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.4 Exercise Level (Bar Chart)
        with row1_cols[3]:
           
            st.markdown("<p class='chart-title'>💪 Exercise Level</p>", unsafe_allow_html=True)
            exercise_counts = filtered_df['Exercise Level'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(exercise_counts.index, exercise_counts.values, color='#e74c3c')
            ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 2: Charts 2.5-2.8 (Health & Lifestyle Patterns)
        st.markdown("### 🍎 Health & Lifestyle Patterns")
        row2_cols = st.columns(4)
        
        # 2.5 Diet Type (Pie Chart)
        with row2_cols[0]:
            
            st.markdown("<p class='chart-title'>🥗 Diet Type</p>", unsafe_allow_html=True)
            diet_counts = filtered_df['Diet Type'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            wedges, texts, autotexts = ax.pie(diet_counts.values, labels=diet_counts.index, 
                                              autopct='%1.1f%%', colors=colors[:len(diet_counts)])
            
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.6 Sleep Hours (Histogram)
        with row2_cols[1]:
            
            st.markdown("<p class='chart-title'>💤 Sleep Hours</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(15, len(filtered_df['Sleep Hours'].unique()))
            ax.hist(filtered_df['Sleep Hours'], bins=bins, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Sleep Hours', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            mean_sleep = filtered_df['Sleep Hours'].mean()
            ax.axvline(mean_sleep, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_sleep:.1f}h')
            ax.legend(fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.7 Stress Score (Histogram)
        with row2_cols[2]:
            
            st.markdown("<p class='chart-title'>😰 Stress Level</p>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(4, 3))
            try:
                pd.to_numeric(filtered_df['Stress Level'])
                bins = min(10, len(filtered_df['Stress Level'].unique()))
                ax.hist(filtered_df['Stress Level'], bins=bins, color='#e67e22', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Stress Level', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                
                mean_stress = filtered_df['Stress Level'].mean()
                ax.axvline(mean_stress, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_stress:.1f}')
                ax.legend(fontsize=7)
            except:
                stress_counts = filtered_df['Stress Level'].value_counts()
                bars = ax.bar(stress_counts.index, stress_counts.values, color='#e67e22')
                ax.set_ylabel('Count', fontsize=8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=7)
            
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.8 Mental Health Condition (Count Plot)
        with row2_cols[3]:
            
            st.markdown("<p class='chart-title'>🧠 Mental Health</p>", unsafe_allow_html=True)
            mh_counts = filtered_df['Mental Health Condition'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(range(len(mh_counts)), mh_counts.values, color='#2ecc71')
            ax.set_xticks(range(len(mh_counts)))
            ax.set_xticklabels(mh_counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Row 3: Charts 2.9-2.12 (Behavioral & Emotional Metrics)
        st.markdown("### ⚡ Behavioral & Emotional Metrics")
        row3_cols = st.columns(4)
        
        # 2.9 Work Hours per Week (Histogram)
        with row3_cols[0]:
            
            st.markdown("<p class='chart-title'>💼 Work Hours/Week</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(15, len(filtered_df['Work Hours per Week'].unique()))
            ax.hist(filtered_df['Work Hours per Week'], bins=bins, color='#34495e', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Work Hours/Week', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            mean_work = filtered_df['Work Hours per Week'].mean()
            ax.axvline(mean_work, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_work:.1f}h')
            ax.legend(fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 2.10 Screen Time (Histogram)
        with row3_cols[1]:
            
            st.markdown("<p class='chart-title'>📱 Screen Time/Day</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(15, len(filtered_df['Screen Time per Day (Hours)'].unique()))
            ax.hist(filtered_df['Screen Time per Day (Hours)'], bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Screen Time (Hours)', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            mean_screen = filtered_df['Screen Time per Day (Hours)'].mean()
            ax.axvline(mean_screen, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_screen:.1f}h')
            ax.legend(fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
          # 2.11 Social Interaction Score (Histogram)
        with row3_cols[2]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("<p class='chart-title'>👥 Social Interaction</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(10, len(filtered_df['Social Interaction Score'].unique()))
            ax.hist(filtered_df['Social Interaction Score'], bins=bins, color='#e91e63', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Social Interaction Score', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            mean_social = filtered_df['Social Interaction Score'].mean()
            ax.axvline(mean_social, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_social:.1f}')
            ax.legend(fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
          # 2.12 Happiness Score (Histogram)
        with row3_cols[3]:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.markdown("<p class='chart-title'>😊 Happiness Score</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            bins = min(10, len(filtered_df['Happiness Score'].unique()))
            ax.hist(filtered_df['Happiness Score'], bins=bins, color='#f39c12', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Happiness Score', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            
            mean_happiness = filtered_df['Happiness Score'].mean()
            ax.axvline(mean_happiness, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_happiness:.1f}')
            ax.legend(fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters. Please adjust your filter criteria.")
    
    st.markdown("</div>", unsafe_allow_html=True)
      # --- 4. CORRELATION ANALYSIS SECTION ---
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#2c3e50; margin-bottom:20px;'>🔗 Correlation Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d; margin-bottom:30px;'>How Lifestyle Factors Relate to Each Other</p>", unsafe_allow_html=True)
      # Create two-column layout for correlation analysis (equal size)
    corr_col1, corr_col2 = st.columns([1, 1])
    
    # Left column: Correlation Heatmap
    with corr_col1:
        
        st.markdown("<p class='chart-title'>🔗 Correlation Heatmap</p>", unsafe_allow_html=True)
        
        # Select only numeric columns for correlation
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 1:
            # Calculate correlation matrix
            corr_matrix = df[numeric_columns].corr()
            
            # Create correlation heatmap with smaller size
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create mask for upper triangle to show only lower half
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8},
                       ax=ax,
                       annot_kws={'size': 8})
            
            ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Not enough numeric columns for correlation analysis.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Right column: Key Correlation Insights (Minimized)
    with corr_col2:

        st.markdown("### 📊 Key Insights")
        
        if len(numeric_columns) > 1:
            # Find strongest positive and negative correlations
            corr_flat = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
            corr_pairs = corr_flat.stack().reset_index()
            corr_pairs.columns = ['Factor1', 'Factor2', 'Correlation']
            corr_pairs = corr_pairs.sort_values('Correlation', key=abs, ascending=False)
            
            # Display top correlation (positive)
            st.markdown("#### 🔴 Strongest Positive")
            if len(corr_pairs[corr_pairs['Correlation'] > 0]) > 0:
                top_pos = corr_pairs[corr_pairs['Correlation'] > 0].iloc[0]
                st.markdown(f"**{top_pos['Factor1']}** ↔ **{top_pos['Factor2']}**")
                st.markdown(f"<span style='color: #e74c3c; font-weight: bold;'>r = {top_pos['Correlation']:.3f}</span>", unsafe_allow_html=True)
            else:
                st.markdown("No strong positive correlations found")
            
            st.markdown("---")
            
            # Display top correlation (negative)
            st.markdown("#### 🔵 Strongest Negative")
            if len(corr_pairs[corr_pairs['Correlation'] < 0]) > 0:
                top_neg = corr_pairs[corr_pairs['Correlation'] < 0].iloc[0]
                st.markdown(f"**{top_neg['Factor1']}** ↔ **{top_neg['Factor2']}**")
                st.markdown(f"<span style='color: #3498db; font-weight: bold;'>r = {top_neg['Correlation']:.3f}</span>", unsafe_allow_html=True)
            else:
                st.markdown("No strong negative correlations found")
            
            st.markdown("---")
            
            
        
        else:
            st.info("Not enough numeric data for analysis.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # --- 5. SHAP-BASED FEATURE IMPORTANCE SECTION ---
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#2c3e50; margin-bottom:20px;'> Feature Importance</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d; margin-bottom:30px;'>Understanding What Drives Happiness and Stress Predictions</p>", unsafe_allow_html=True)
    
    # Load feature importance data and SHAP images
    try:
        feature_importance_df = load_feature_importance()
        shap_images = load_shap_images()
        
        if feature_importance_df is not None:
            # 5.1 & 5.2: Top 5 Features Bar Charts (SHAP-based)
            st.markdown("### 📊 Top Contributing Factors")
            
            fi_cols = st.columns(2)
            # Top 5 Features for Happiness
            with fi_cols[0]:
                
                st.markdown("<p class='chart-title'>😊 Top 5 Happiness Drivers</p>", unsafe_allow_html=True)
                
                if 'Happiness_Importance' in feature_importance_df.columns:
                    happiness_fi = feature_importance_df.nlargest(5, 'Happiness_Importance')
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(happiness_fi['Feature'], happiness_fi['Happiness_Importance'], color='#2ecc71')
                    ax.set_title('Features Contributing to Happiness', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Happiness feature importance data not available.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Top 5 Features for Stress
            with fi_cols[1]:
                
                st.markdown("<p class='chart-title'>😰 Top 5 Stress Factors</p>", unsafe_allow_html=True)
                
                if 'Stress_Importance' in feature_importance_df.columns:
                    stress_fi = feature_importance_df.nlargest(5, 'Stress_Importance')
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(stress_fi['Feature'], stress_fi['Stress_Importance'], color='#e74c3c')
                    ax.set_xlabel('Feature Importance', fontsize=10)
                    ax.set_title('Features Contributing to Stress', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Stress feature importance data not available.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # 5.3-5.6: SHAP Visualizations
        if shap_images:
            st.markdown("### 🔍 SHAP Analysis Visualizations")
            
            # Create 2x2 grid for SHAP images
            shap_row1 = st.columns(2)
            shap_row2 = st.columns(2)
            
            # 5.3: SHAP Summary Plot - Happiness
            with shap_row1[0]:
                if "Happiness Summary" in shap_images:
                    
                    st.markdown("<p class='chart-title'>😊 SHAP Summary - Happiness</p>", unsafe_allow_html=True)
                    st.image(shap_images["Happiness Summary"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("SHAP Happiness Summary plot not available.")
            
            # 5.4: SHAP Summary Plot - Stress  
            with shap_row1[1]:
                if "Stress Summary" in shap_images:
                    
                    st.markdown("<p class='chart-title'>😰 SHAP Summary - Stress</p>", unsafe_allow_html=True)
                    st.image(shap_images["Stress Summary"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("SHAP Stress Summary plot not available.")
            
            # 5.5: SHAP Dot Plot - Happiness
            with shap_row2[0]:
                if "Happiness Dot" in shap_images:
                    
                    st.markdown("<p class='chart-title'>😊 SHAP Dot Plot - Happiness</p>", unsafe_allow_html=True)
                    st.image(shap_images["Happiness Dot"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("SHAP Happiness Dot plot not available.")
            
            # 5.6: SHAP Dot Plot - Stress
            with shap_row2[1]:
                if "Stress Dot" in shap_images:
                    
                    st.markdown("<p class='chart-title'>😰 SHAP Dot Plot - Stress</p>", unsafe_allow_html=True)
                    st.image(shap_images["Stress Dot"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("SHAP Stress Dot plot not available.")
    
    except Exception as e:
        st.error("Error loading SHAP visualizations or feature importance data.")
      # --- 6. PERSONALIZED INSIGHTS SECTION ---
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#2c3e50; margin-bottom:20px;'>💡 Personalized Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d; margin-bottom:30px;'>Dynamic insights based on your current filter selection</p>", unsafe_allow_html=True)
    
    # Generate comprehensive insights that work with ANY amount of data
    insights = []
    
    if not filtered_df.empty:
        try:
            # INSIGHT 1: Data Overview & Filter Impact
            total_entries = len(df)
            filtered_entries = len(filtered_df)
            filter_percentage = (filtered_entries / total_entries) * 100;
            
            insights.append({
                'icon': '📊',
                'title': 'Filter Overview',
                'text': f"Viewing {filtered_entries:,} entries ({filter_percentage:.1f}% of total dataset)"
            })
            
            # INSIGHT 2: Happiness Analysis (always available)
            avg_happiness = filtered_df['Happiness Score'].mean()
            overall_happiness = df['Happiness Score'].mean()
            happiness_diff = avg_happiness - overall_happiness
            
            if happiness_diff > 0.2:
                insights.append({
                    'icon': '😊📈',
                    'title': 'Happiness Boost',
                    'text': f"Your selection shows {happiness_diff:.1f} points higher happiness than average ({avg_happiness:.1f}/10)"
                })
            elif happiness_diff < -0.2:
                insights.append({
                    'icon': '😔📉',
                    'title': 'Happiness Alert',
                    'text': f"Your selection shows {abs(happiness_diff):.1f} points lower happiness than average ({avg_happiness:.1f}/10)"
                })
            else:
                insights.append({
                    'icon': '😐📊',
                    'title': 'Happiness Balance',
                    'text': f"Your selection shows average happiness levels ({avg_happiness:.1f}/10)"
                })
            
            # INSIGHT 3: Stress Analysis (always available)
            try:
                # Handle both numeric and categorical stress
                if filtered_df['Stress Level'].dtype == 'object':
                    stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
                    avg_stress = filtered_df['Stress Level'].map(stress_mapping).mean()
                    overall_stress = df['Stress Level'].map(stress_mapping).mean()
                    stress_scale = "/3"
                else:
                    avg_stress = filtered_df['Stress Level'].mean()
                    overall_stress = df['Stress Level'].mean()
                    stress_scale = "/10"
                
                stress_diff = avg_stress - overall_stress
                
                if stress_diff > 0.2:
                    insights.append({
                        'icon': '😰⚠️',
                        'title': 'Stress Alert',
                        'text': f"Your selection shows higher stress levels ({avg_stress:.1f}{stress_scale} vs {overall_stress:.1f}{stress_scale} average)"
                    })
                elif stress_diff < -0.2:
                    insights.append({
                        'icon': '😌✨',
                        'title': 'Lower Stress',
                        'text': f"Your selection shows lower stress levels ({avg_stress:.1f}{stress_scale} vs {overall_stress:.1f}{stress_scale} average)"
                    })
                else:
                    insights.append({
                        'icon': '😐📊',
                        'title': 'Average Stress',
                        'text': f"Your selection shows typical stress levels ({avg_stress:.1f}{stress_scale})"
                    })
            except:
                pass
            
            # INSIGHT 4: Sleep Pattern Analysis
            avg_sleep = filtered_df['Sleep Hours'].mean()
            overall_sleep = df['Sleep Hours'].mean()
            
            if avg_sleep < 6:
                insights.append({
                    'icon': '😴⚠️',
                    'title': 'Sleep Concern',
                    'text': f"Average sleep in selection: {avg_sleep:.1f}h - Consider aiming for 7-8 hours"
                })
            elif avg_sleep > 8.5:
                insights.append({
                    'icon': '😴💤',
                    'title': 'High Sleep',
                    'text': f"Average sleep in selection: {avg_sleep:.1f}h - Above typical range"
                })
            else:
                insights.append({
                    'icon': '😴✅',
                    'title': 'Good Sleep',
                    'text': f"Average sleep in selection: {avg_sleep:.1f}h - Within healthy range"
                })
            
            # INSIGHT 5: Exercise Pattern Analysis
            if len(filtered_df) > 0:
                exercise_counts = filtered_df['Exercise Level'].value_counts()
                top_exercise = exercise_counts.index[0]
                exercise_percentage = (exercise_counts.iloc[0] / len(filtered_df)) * 100
                
                if top_exercise == 'High':
                    insights.append({
                        'icon': '💪🔥',
                        'title': 'Active Lifestyle',
                        'text': f"{exercise_percentage:.0f}% of your selection exercises at high intensity"
                    })
                elif top_exercise == 'Low':
                    insights.append({
                        'icon': '🚶‍♂️📈',
                        'title': 'Exercise Opportunity',
                        'text': f"{exercise_percentage:.0f}% of your selection has low exercise - room for improvement"
                    })
                else:
                    insights.append({
                        'icon': '🏃‍♀️📊',
                        'title': 'Moderate Activity',
                        'text': f"{exercise_percentage:.0f}% of your selection exercises at moderate levels"
                    })
            
            # INSIGHT 6: Work-Life Balance Analysis
            avg_work = filtered_df['Work Hours per Week'].mean()
            
            if avg_work > 50:
                insights.append({
                    'icon': '💼⚠️',
                    'title': 'Work Intensity',
                    'text': f"Average work hours: {avg_work:.1f}h/week - High workload may impact wellbeing"
                })
            elif avg_work < 30:
                insights.append({
                    'icon': '💼😊',
                    'title': 'Work Balance',
                    'text': f"Average work hours: {avg_work:.1f}h/week - Good work-life balance"
                })
            else:
                insights.append({
                    'icon': '💼📊',
                    'title': 'Standard Workload',
                    'text': f"Average work hours: {avg_work:.1f}h/week - Typical full-time schedule"
                })
            
            # INSIGHT 7: Screen Time Analysis
            avg_screen = filtered_df['Screen Time per Day (Hours)'].mean()
            
            if avg_screen > 8:
                insights.append({
                    'icon': '📱⚠️',
                    'title': 'High Screen Time',
                    'text': f"Average screen time: {avg_screen:.1f}h/day - Consider digital wellness breaks"
                })
            elif avg_screen < 4:
                insights.append({
                    'icon': '📱✅',
                    'title': 'Moderate Screen Use',
                    'text': f"Average screen time: {avg_screen:.1f}h/day - Good digital balance"
                })
            else:
                insights.append({
                    'icon': '📱📊',
                    'title': 'Typical Screen Time',
                    'text': f"Average screen time: {avg_screen:.1f}h/day - Within normal range"
                })
            
            # INSIGHT 8: Social Interaction Analysis
            avg_social = filtered_df['Social Interaction Score'].mean()
            
            if avg_social < 4:
                insights.append({
                    'icon': '👥📉',
                    'title': 'Social Opportunity',
                    'text': f"Social interaction score: {avg_social:.1f}/10 - Consider increasing social connections"
                })
            elif avg_social > 7:
                insights.append({
                    'icon': '👥🌟',
                    'title': 'Strong Social Life',
                    'text': f"Social interaction score: {avg_social:.1f}/10 - Excellent social connections"
                })
            else:
                insights.append({
                    'icon': '👥📊',
                    'title': 'Moderate Social Life',
                    'text': f"Social interaction score: {avg_social:.1f}/10 - Balanced social interactions"
                })
            
        except Exception as e:
            # Fallback insight that always works
            insights = [{
                'icon': '📊',
                'title': 'Data Analysis',
                'text': f"Analyzing {len(filtered_df)} entries from your filter selection"
            }]
    
    else:
        # Empty dataset insight
        insights = [{
            'icon': '🔍',
            'title': 'No Data Found',
            'text': "No entries match your current filter criteria. Try adjusting your filters."
        }]
    
    # Display insights in a grid (always show at least 2, up to 4)
    num_insights_to_show = min(4, len(insights))
    if num_insights_to_show >= 2:
        insight_cols = st.columns(2)
        for i in range(num_insights_to_show):
            with insight_cols[i % 2]:
                insight = insights[i]
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{insight['icon']}</div>
                    <div style="font-size: 1rem; font-weight: bold; margin-bottom: 0.25rem;">{insight['title']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">{insight['text']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Single insight display
        insight = insights[0]
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{insight['icon']}</div>
            <div style="font-size: 1rem; font-weight: bold; margin-bottom: 0.25rem;">{insight['title']}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{insight['text']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- 7. SIMULATOR ACCESS SECTION ---
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#2c3e50; margin-bottom:20px;'>🚀 Explore Predictions with Simulator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7f8c8d; margin-bottom:30px;'>Use our AI-powered simulator to predict your personal wellness outcomes</p>", unsafe_allow_html=True)
      # Large Simulator Button
    simulator_container = st.container()
    with simulator_container:
        # Create a clickable button that sets session state
        if st.button("🎯 Go to Simulator & Predict Wellness", key="nav_to_simulator", use_container_width=True):
            # Set session state to switch to simulator
            st.session_state['active_tab'] = 1
            st.rerun()        # Style the button as a large card with beautiful header-matching CSS
        st.markdown("""
        <style>
        /* Style the simulator navigation button as a large card */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 3rem 4rem !important;
            border-radius: 25px !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4) !important;
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5) !important;
            backdrop-filter: blur(10px) !important;
            border: 3px solid rgba(255, 255, 255, 0.3) !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            width: 100% !important;
            margin: 2rem 0 !important;
            min-height: 120px !important;
            position: relative !important;
            overflow: hidden !important;
            transform: perspective(1000px) rotateX(0deg) !important;
        }
        
        /* Add subtle gradient overlay animation on hover */
        div[data-testid="stButton"] > button[kind="primary"]::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
            transition: left 0.6s ease !important;
        }
        
        /* Add floating particles effect */
        div[data-testid="stButton"] > button[kind="primary"]::after {
            content: '✨ 🚀 ⚡ 🎯' !important;
            position: absolute !important;
            top: 10px !important;
            right: 20px !important;
            font-size: 0.8rem !important;
            opacity: 0.7 !important;
            animation: float 2s ease-in-out infinite !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:hover::before {
            left: 100% !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-8px) scale(1.05) perspective(1000px) rotateX(5deg) !important;
            box-shadow: 0 25px 80px rgba(102, 126, 234, 0.8) !important;
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
            border: 3px solid rgba(255, 255, 255, 0.5) !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Enhanced pulsing glow animation */
        @keyframes pulseGlow {
            0% { 
                box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
            }
            50% { 
                box-shadow: 0 20px 70px rgba(102, 126, 234, 0.9);
            }
            100% { 
                box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
            }
        }
        
        @keyframes float {
            0%, 100% { 
                transform: translateY(0px) rotate(0deg);
                opacity: 0.7;
            }
            50% { 
                transform: translateY(-10px) rotate(180deg);
                opacity: 1;
            }
        }
        
        div[data-testid="stButton"] > button[kind="primary"] {
            animation: pulseGlow 4s ease-in-out infinite !important;
        }
        </style>
        """, unsafe_allow_html=True)
      # Utility Buttons
    # CSV Download Button
    csv_col1, csv_col2, csv_col3 = st.columns([1, 1, 1])
    with csv_col2:
        if st.download_button(
            label="📊 Download Dataset CSV",
            data=df.to_csv(index=False),
            file_name="Mental_Health_Lifestyle_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        ):
            st.success("Dataset downloaded successfully!")
    
    # Fixed Go-to-Top Button with improved JavaScript
    st.markdown("""
    <style>
        .go-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            transition: all 0.3s ease;
            display: none;
        }
        
        .go-to-top:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
        }
        
        .go-to-top.show {
            display: block !important;
        }
    </style>
    
    <button class="go-to-top" id="goToTopBtn" onclick="scrollToTop()" title="Go to Top">
        🔝
    </button>
    
    <script>
        // Improved Go-to-Top functionality
        function scrollToTop() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
        
        // Show/hide go-to-top button based on scroll position
        function toggleGoToTopButton() {
            const goToTopButton = document.getElementById('goToTopBtn');
            if (goToTopButton) {
                if (window.pageYOffset > 300) {
                    goToTopButton.classList.add('show');
                } else {
                    goToTopButton.classList.remove('show');
                }
            }
        }
        
        // Add scroll event listener
        window.addEventListener('scroll', toggleGoToTopButton);
        
        // Initial check when page loads
        document.addEventListener('DOMContentLoaded', function() {
            toggleGoToTopButton();
        });
        
        // Check again after Streamlit updates (for dynamic content)
        const observer = new MutationObserver(function(mutations) {
            toggleGoToTopButton();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
