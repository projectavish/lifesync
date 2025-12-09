"""
LifeSync Dashboard - Clean Simulator App
A streamlined version that eliminates duplicate visualizations and provides clear, non-redundant forecast displays.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import os
import shap
from datetime import datetime
import csv
import warnings
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import base64

# Configure matplotlib to handle font warnings
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph.*missing from font.*')

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'inputs' not in st.session_state:
    st.session_state.inputs = None
if 'processed_inputs' not in st.session_state:
    st.session_state.processed_inputs = None
if 'happiness_pred' not in st.session_state:
    st.session_state.happiness_pred = None
if 'stress_pred' not in st.session_state:
    st.session_state.stress_pred = None
if 'burnout_risk' not in st.session_state:
    st.session_state.burnout_risk = None
if 'happiness_forecast' not in st.session_state:
    st.session_state.happiness_forecast = None
if 'stress_forecast' not in st.session_state:
    st.session_state.stress_forecast = None
if 'burnout_forecast' not in st.session_state:
    st.session_state.burnout_forecast = None
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Enhanced CSS styling with Bootstrap integration
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

<style>
    /* Main container styling */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .simulator-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-card.happiness {
        border-color: #28a745;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }
    
    .prediction-card.stress {
        border-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    }
    
    .prediction-card.burnout {
        border-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }
    
    /* Form styling */
    .input-form {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Path configuration
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Mental_Health_Lifestyle_Dataset.csv")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Load dataset for reference values
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Load models
@st.cache_resource
def load_models():
    """Load the trained models."""
    try:
        happiness_model = joblib.load(os.path.join(MODELS_PATH, "lifesync_happiness_model.pkl"))
        stress_model = joblib.load(os.path.join(MODELS_PATH, "lifesync_stress_model.pkl"))
        return happiness_model, stress_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None

def preprocess_inputs(inputs):
    """
    Process user inputs to prepare for model prediction
    """
    # Load the data and model to get expected features
    happiness_model = joblib.load(os.path.join(MODELS_PATH, "lifesync_happiness_model.pkl"))
    feature_names = happiness_model.feature_names_in_
    
    # Create a DataFrame from inputs
    input_df = pd.DataFrame([inputs])
    
    # Process Exercise Level: Map to numeric
    exercise_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
    input_df['Exercise Level'] = input_df['Exercise Level'].map(exercise_mapping)
    
    # Initialize all possible one-hot encoded columns with zeros
    categorical_prefixes = ['Gender_', 'Diet_', 'MH_', 'Country_']
    for feature in feature_names:
        for prefix in categorical_prefixes:
            if feature.startswith(prefix):
                input_df[feature] = 0
    
    # Set the appropriate one-hot encoded columns to 1
    if f'Gender_{inputs["Gender"]}' in feature_names:
        input_df[f'Gender_{inputs["Gender"]}'] = 1
    
    if f'Diet_{inputs["Diet Type"]}' in feature_names:
        input_df[f'Diet_{inputs["Diet Type"]}'] = 1
    
    if f'MH_{inputs["Mental Health Condition"]}' in feature_names:
        input_df[f'MH_{inputs["Mental Health Condition"]}'] = 1
    
    # Use most common country from training data - default to USA
    if f'Country_{inputs["Country"]}' in feature_names:
        input_df[f'Country_{inputs["Country"]}'] = 1
    else:
        # Default to USA if country not in training data
        input_df['Country_USA'] = 1
    
    # Remove original categorical columns not needed for prediction
    input_df.drop(['Gender', 'Diet Type', 'Mental Health Condition', 'Country'], axis=1, inplace=True)
    
    # Ensure all features are in the correct order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    return input_df

def generate_forecast(happiness, stress, burnout):
    """Generate wellness forecasts with realistic progression."""
    # Base rates for realistic progression
    happiness_decline_rate = 0.12
    stress_increase_rate = 0.20
    burnout_increase_rate = 2.0
      # Ensure inputs are valid numbers
    try:
        happiness = float(happiness) if happiness is not None else 5.0
        stress = float(stress) if stress is not None else 5.0
        burnout = float(burnout) if burnout is not None else 50.0
    except (ValueError, TypeError):
        happiness = 5.0
        stress = 5.0
        burnout = 50.0
    
    # Happiness forecast
    happiness_forecast = {
        "Current": round(happiness, 1),
        "3 Days": round(max(0, happiness - (happiness_decline_rate * 0.4)), 1),
        "1 Week": round(max(0, happiness - happiness_decline_rate), 1),
        "1 Month": round(max(0, happiness - (happiness_decline_rate * 2.5)), 1),
        "3 Months": round(max(0, happiness - (happiness_decline_rate * 5)), 1),
    }
    
    # Stress forecast
    stress_forecast = {
        "Current": round(stress, 1),
        "3 Days": round(min(10, stress + (stress_increase_rate * 0.5)), 1),
        "1 Week": round(min(10, stress + stress_increase_rate), 1),
        "1 Month": round(min(10, stress + (stress_increase_rate * 2)), 1),
        "3 Months": round(min(10, stress + (stress_increase_rate * 3.5)), 1),
    }
    
    # Burnout forecast
    burnout_forecast = {
        "Current": round(burnout, 1),
        "3 Days": round(min(100, burnout + burnout_increase_rate), 1),
        "1 Week": round(min(100, burnout + (burnout_increase_rate * 1.8)), 1),
        "1 Month": round(min(100, burnout + (burnout_increase_rate * 4.5)), 1),
        "3 Months": round(min(100, burnout + (burnout_increase_rate * 8)), 1),
    }
    
    return happiness_forecast, stress_forecast, burnout_forecast

def get_recommendation_insights(inputs, happiness, stress, burnout):
    """Generate personalized recommendations based on predictions."""
    recommendations = []
    
    # Sleep recommendations
    if inputs['Sleep Hours'] < 7:
        recommendations.append({
            'category': 'Sleep Optimization',
            'icon': 'fas fa-bed',
            'color': 'primary',
            'priority': 'high' if inputs['Sleep Hours'] < 6 else 'medium',
            'title': 'Improve Sleep Quality',
            'message': f"Your sleep duration ({inputs['Sleep Hours']:.1f}h) is below optimal.",
            'actions': [
                "Aim for 7-9 hours of sleep per night",
                "Create a consistent bedtime routine",
                "Avoid screens 1 hour before bed",
                "Keep your bedroom cool and dark"
            ],
            'impact': 'High impact on happiness and stress reduction'
        })
    elif inputs['Sleep Hours'] > 9:
        recommendations.append({
            'category': 'Sleep Balance',
            'icon': 'fas fa-clock',
            'color': 'info',
            'priority': 'low',
            'title': 'Optimize Sleep Duration',
            'message': f"You're getting {inputs['Sleep Hours']:.1f}h of sleep, which may be excessive.",
            'actions': [
                "Try gradually reducing sleep to 7-8 hours",
                "Ensure sleep quality over quantity",
                "Check for underlying health issues"
            ],
            'impact': 'Moderate impact on energy levels'
        })
    
    # Work-life balance
    if inputs['Work Hours per Week'] > 50:
        recommendations.append({
            'category': 'Work-Life Balance',
            'icon': 'fas fa-briefcase',
            'color': 'warning',
            'priority': 'high' if inputs['Work Hours per Week'] > 60 else 'medium',
            'title': 'Manage Work Hours',
            'message': f"High work hours ({inputs['Work Hours per Week']:.0f}h/week) may increase burnout risk.",
            'actions': [
                "Set clear work boundaries",
                "Take regular breaks every 90 minutes",
                "Practice saying 'no' to non-essential tasks",
                "Consider delegating responsibilities"
            ],
            'impact': 'Critical for preventing burnout'
        })
    
    # Screen time
    if inputs['Screen Time per Day (Hours)'] > 6:
        recommendations.append({
            'category': 'Digital Wellness',
            'icon': 'fas fa-mobile-alt',
            'color': 'info',
            'priority': 'medium',
            'title': 'Reduce Screen Time',
            'message': f"Screen time ({inputs['Screen Time per Day (Hours)']:.1f}h/day) is above recommended levels.",
            'actions': [
                "Follow the 20-20-20 rule (every 20 min, look 20 feet away for 20 sec)",
                "Use app timers to limit social media",
                "Implement screen-free zones in your home",
                "Try digital detox periods"
            ],
            'impact': 'Reduces eye strain and improves focus'
        })
    
    # Social interaction
    if inputs['Social Interaction Score'] < 5:
        recommendations.append({
            'category': 'Social Connection',
            'icon': 'fas fa-users',
            'color': 'success',
            'priority': 'medium',
            'title': 'Enhance Social Connections',
            'message': f"Social interaction score ({inputs['Social Interaction Score']:.0f}/10) could be improved.",
            'actions': [
                "Schedule regular meetups with friends",
                "Join clubs or groups with similar interests",
                "Practice active listening in conversations",
                "Consider volunteering in your community"
            ],
            'impact': 'Significant boost to mental health and happiness'
        })
    
    # Exercise
    if inputs['Exercise Level'] == 'Low':
        recommendations.append({
            'category': 'Physical Activity',
            'icon': 'fas fa-dumbbell',
            'color': 'danger',
            'priority': 'high',
            'title': 'Increase Physical Activity',
            'message': "Regular exercise can significantly improve mood and reduce stress levels.",
            'actions': [
                "Start with 15-20 minutes of walking daily",
                "Try bodyweight exercises at home",
                "Find an activity you enjoy (dancing, swimming, cycling)",
                "Gradually increase intensity and duration"
            ],
            'impact': 'Powerful mood booster and stress reliever'
        })
    elif inputs['Exercise Level'] == 'Moderate':
        recommendations.append({
            'category': 'Fitness Enhancement',
            'icon': 'fas fa-running',
            'color': 'success',
            'priority': 'low',
            'title': 'Optimize Your Fitness Routine',
            'message': "You're doing well! Consider enhancing your routine.",
            'actions': [
                "Add strength training 2-3 times per week",
                "Try high-intensity interval training (HIIT)",
                "Include flexibility and balance exercises",
                "Set new fitness goals to stay motivated"
            ],
            'impact': 'Further improvements in energy and mood'
        })
    
    # Diet recommendations
    if inputs['Diet Type'] == 'Junk Food':
        recommendations.append({
            'category': 'Nutrition',
            'icon': 'fas fa-apple-alt',
            'color': 'warning',
            'priority': 'high',
            'title': 'Improve Nutrition',
            'message': "Diet significantly impacts mood and energy levels.",
            'actions': [
                "Gradually replace processed foods with whole foods",
                "Include more fruits and vegetables",
                "Stay hydrated with 8 glasses of water daily",
                "Plan meals in advance to avoid impulsive choices"
            ],
            'impact': 'Major improvement in energy and mental clarity'
        })
    
    # Mental health support
    if inputs['Mental Health Condition'] != 'None':
        recommendations.append({
            'category': 'Mental Health Support',
            'icon': 'fas fa-heart',
            'color': 'info',
            'priority': 'high',
            'title': 'Professional Support',
            'message': f"Managing {inputs['Mental Health Condition'].lower()} requires ongoing care.",
            'actions': [
                "Continue regular therapy or counseling sessions",
                "Practice mindfulness and meditation",
                "Build a strong support network",
                "Consider stress-reduction techniques like yoga"
            ],
            'impact': 'Essential for long-term mental wellness'
        })
    
    # Stress management (based on predictions)
    if stress > 6:
        recommendations.append({
            'category': 'Stress Management',
            'icon': 'fas fa-leaf',
            'color': 'success',
            'priority': 'high',
            'title': 'Reduce Stress Levels',
            'message': "Your stress levels are elevated and need attention.",
            'actions': [
                "Practice deep breathing exercises",
                "Try progressive muscle relaxation",
                "Engage in hobbies you enjoy",
                "Consider meditation or yoga classes"
            ],
            'impact': 'Immediate stress relief and better coping'
        })
    
    # Happiness boosters (based on predictions)
    if happiness < 6:
        recommendations.append({
            'category': 'Happiness Boost',
            'icon': 'fas fa-smile',
            'color': 'warning',
            'priority': 'medium',
            'title': 'Enhance Well-being',
            'message': "Let's work on boosting your happiness levels.",
            'actions': [
                "Practice gratitude journaling",
                "Engage in activities that bring you joy",
                "Spend time in nature",
                "Connect with positive, supportive people"
            ],
            'impact': 'Gradual improvement in overall life satisfaction'
        })
    
    # If no specific recommendations, provide general wellness tips
    if not recommendations:
        recommendations.append({
            'category': 'Wellness Maintenance',
            'icon': 'fas fa-star',
            'color': 'success',
            'priority': 'low',
            'title': 'Maintain Your Great Habits',
            'message': "Your lifestyle appears well-balanced! Keep up the excellent work.",
            'actions': [
                "Continue your current healthy routines",
                "Set new wellness goals to stay motivated",
                "Share your healthy habits with others",
                "Regular check-ins with your wellness progress"
            ],
            'impact': 'Sustained long-term health and happiness'
        })
    
    return recommendations

# Save prediction results to CSV
def save_prediction_to_csv(inputs, happiness_pred, stress_pred, burnout_risk, csv_path=None):
    """
    Save user inputs and prediction results to a CSV file with timestamp
    
    Parameters:
    -----------
    inputs : dict
        Dictionary containing user input values
    happiness_pred : float
        Happiness prediction value
    stress_pred : float
        Stress level prediction value
    burnout_risk : float
        Burnout risk percentage
    csv_path : str, optional
        Path to the CSV file. If None, defaults to 'prediction_history.csv'
        in the outputs directory.
    """
    if csv_path is None:
        # Use the outputs directory to store prediction history
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        
        # Ensure the outputs directory exists
        os.makedirs(outputs_dir, exist_ok=True)
        
        csv_path = os.path.join(outputs_dir, "prediction_history.csv")
    
    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(csv_path)
    
    try:
        # Create a list with the column values in the correct order
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Name", "Age", "Gender", "Sleep Hours", "Work Hours per Week", 
                    "Screen Time per Day (Hours)", "Social Interaction Score", 
                    "Exercise Level", "Diet Type", "Mental Health Condition", 
                    "Happiness Score", "Stress Level", "Burnout Risk"
                ])
            
            # Write the data row
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inputs.get("Name", ""), 
                inputs["Age"], 
                inputs["Gender"], 
                inputs["Sleep Hours"], 
                inputs["Work Hours per Week"],
                inputs["Screen Time per Day (Hours)"], 
                inputs["Social Interaction Score"],
                inputs["Exercise Level"], 
                inputs["Diet Type"], 
                inputs["Mental Health Condition"],
                happiness_pred, 
                stress_pred, 
                burnout_risk
            ])
        
        return True, csv_path
    except Exception as e:
        st.warning(f"Could not save prediction history: {e}")
        return False, str(e)

# Save predictions to a local CSV file in the predictions folder
def save_predictions_to_csv(inputs, happiness_pred, stress_pred, burnout_risk):
    """
    Save the prediction results to a CSV file in the predictions folder with a timestamp
    """
    try:
        # Prepare the directory
        os.makedirs("predictions", exist_ok=True)
        
        # File path
        file_path = os.path.join("predictions", "prediction_results.csv")
        
        # Check if file exists to determine if we need to write the header
        file_exists = os.path.isfile(file_path)
        
        # Write the data
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Name", "Age", "Gender", "Sleep Hours", "Work Hours per Week", 
                    "Screen Time per Day (Hours)", "Social Interaction Score", 
                    "Exercise Level", "Diet Type", "Mental Health Condition", 
                    "Happiness Score", "Stress Level", "Burnout Risk"
                ])
            
            # Write the data row
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inputs.get("Name", ""),
                inputs["Age"], 
                inputs["Gender"], 
                inputs["Sleep Hours"], 
                inputs["Work Hours per Week"],
                inputs["Screen Time per Day (Hours)"], 
                inputs["Social Interaction Score"],
                inputs["Exercise Level"], 
                inputs["Diet Type"], 
                inputs["Mental Health Condition"],
                happiness_pred, 
                stress_pred, 
                burnout_risk
            ])
        
        return True, file_path
    except Exception as e:
        st.warning(f"Could not save prediction history: {e}")
        return False, str(e)

# Generate PDF report with wellness predictions and recommendations
def generate_pdf_report(inputs, happiness_pred, stress_pred, burnout_risk, happiness_forecast, stress_forecast, burnout_forecast):
    """
    Create a professional PDF report with the user's wellness predictions, forecasts, and recommendations
    
    Returns the PDF as bytes that can be used for download
    """
    # Create a temporary buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           leftMargin=72, rightMargin=72, 
                           topMargin=72, bottomMargin=72)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    # Create a custom style for recommendations
    recommendation_style = ParagraphStyle('RecommendationStyle', 
                                        parent=normal_style,
                                        spaceBefore=12, 
                                        leftIndent=20)
    
    # Create custom header style
    header_style = ParagraphStyle('HeaderStyle',
                                parent=styles['Heading1'],
                                fontSize=16,
                                textColor=colors.darkblue)
    
    # Create custom subheader style
    subheader_style = ParagraphStyle('SubHeaderStyle',
                                   parent=styles['Heading2'],
                                   fontSize=14,
                                   textColor=colors.darkblue)
    
    # List to hold PDF content
    content = []
    
    # Add title and date
    content.append(Paragraph(f"LifeSync Wellness Report", title_style))
    content.append(Spacer(1, 0.25 * inch))
    
    # Add name and date
    content.append(Paragraph(f"Prepared for: {inputs.get('Name', 'Anonymous')}", header_style))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", normal_style))
    content.append(Spacer(1, 0.25 * inch))
    
    # Add executive summary
    content.append(Paragraph("Executive Summary:", subheader_style))
    
    # Determine overall wellness status
    overall_wellness = (happiness_pred + (10-stress_pred) + (100-burnout_risk)/10) / 3
    wellness_status = "excellent" if overall_wellness > 7.5 else "good" if overall_wellness > 6 else "moderate" if overall_wellness > 4.5 else "concerning"
    
    # Primary areas of focus
    focus_areas = []
    if happiness_pred < 6:
        focus_areas.append("happiness improvement")
    if stress_pred > 5:
        focus_areas.append("stress management")
    if burnout_risk > 40:
        focus_areas.append("burnout prevention")
    
    focus_text = ", ".join(focus_areas) if focus_areas else "maintenance of your current wellness"
    
    # Executive summary text
    summary_text = f"""Based on the information provided, your overall wellness score is {overall_wellness:.1f}/10, 
    which indicates a {wellness_status} wellness level. Your happiness score is {happiness_pred}/10, 
    stress level is {stress_pred:.1f}/10, and burnout risk is {burnout_risk}%. 
    """
    
    if focus_areas:
        summary_text += f"This report focuses primarily on {focus_text}."
    else:
        summary_text += "You're maintaining good wellness habits - this report offers strategies to sustain your progress."
    
    content.append(Paragraph(summary_text, normal_style))
    content.append(Spacer(1, 0.25 * inch))
    
    # Add introduction
    content.append(Paragraph("Personal Information:", subheader_style))
    
    # Add personal information
    personal_info = [
        ["Age", str(inputs["Age"])],
        ["Gender", inputs["Gender"]],
        ["Country", inputs["Country"]],
        ["Sleep Hours", f"{inputs['Sleep Hours']} hours/night"],
        ["Work Hours", f"{inputs['Work Hours per Week']} hours/week"],
        ["Screen Time", f"{inputs['Screen Time per Day (Hours)']} hours/day"],
        ["Social Interaction", f"{inputs['Social Interaction Score']}/10"],
        ["Exercise Level", inputs["Exercise Level"]],
        ["Diet Type", inputs["Diet Type"]],
        ["Mental Health Condition", inputs["Mental Health Condition"]]
    ]
    
    # Create a table for personal info
    t = Table(personal_info, colWidths=[2*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(t)
    
    content.append(Spacer(1, 0.25 * inch))
      # Add wellness scores
    content.append(Paragraph("Wellness Assessment Results:", subheader_style))
    
    # Create progress gauges for each metric
    happiness_gauge = create_progress_gauge(happiness_pred, 10, 150, 15)
    stress_gauge = create_progress_gauge(stress_pred, 10, 150, 15)
    burnout_gauge = create_progress_gauge(burnout_risk, 100, 150, 15)
    
    # Create a table for wellness scores with visual gauges
    wellness_scores = [
        ["Metric", "Score", "Visual", "Interpretation"],
        ["Happiness Score", f"{happiness_pred}/10", happiness_gauge, get_interpretation(happiness_pred, "happiness")],
        ["Stress Level", f"{stress_pred:.2f}/10", stress_gauge, get_interpretation(stress_pred, "stress")],
        ["Burnout Risk", f"{burnout_risk}%", burnout_gauge, get_interpretation(burnout_risk, "burnout")]
    ]
      # Create a table for wellness scores with wider columns to prevent text overflow
    t = Table(wellness_scores, colWidths=[1.2*inch, 0.8*inch, 1.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(t)
    
    content.append(Spacer(1, 0.25 * inch))
    
    # Add forecast data
    content.append(Paragraph("Wellness Forecast (If current patterns continue):", subheader_style))
    
    # Create forecast data
    forecast_data = [["Time Period", "Happiness", "Stress", "Burnout Risk"]]
    
    # Get time periods (keys from the forecast dictionaries)
    periods = list(happiness_forecast.keys())
    
    # Add each time period's data
    for period in periods:
        forecast_data.append([
            period,
            f"{happiness_forecast[period]}/10",
            f"{stress_forecast[period]:.2f}/10",
            f"{burnout_forecast[period]}%"
        ])
      # Create a table for forecast data
    t = Table(forecast_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.6*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(t)
    
    # Add visualization of forecasts
    try:
        # Create a figure for the graphs
        plt.figure(figsize=(8, 4))
        
        # Plot data from forecasts
        x = range(len(periods))
        plt.plot(x, [happiness_forecast[p] for p in periods], 'g-o', label='Happiness')
        plt.plot(x, [stress_forecast[p] for p in periods], 'r-s', label='Stress')
        plt.plot(x, [burnout_forecast[p]/10 for p in periods], 'y-^', label='Burnout Risk (÷10)')
        
        # Add labels and formatting
        plt.xlabel('Time Period')
        plt.ylabel('Score')
        plt.title('Wellness Trends Forecast')
        plt.xticks(x, periods, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure to a bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        
        # Create an Image object with the plot
        content.append(Spacer(1, 0.25 * inch))
        content.append(Paragraph("Wellness Forecast Visualization:", subheader_style))
        img = Image(img_buffer, width=6*inch, height=3*inch)
        content.append(img)
        
        plt.close()
    except Exception as e:
        # If plotting fails, just add a note
        content.append(Paragraph(f"Note: Visualization could not be generated.", normal_style))
    
    content.append(PageBreak())
    
    # Add recommendations section
    content.append(Paragraph("Personalized Recommendations:", subheader_style))
    content.append(Spacer(1, 0.1 * inch))
    
    # Generate recommendations
    recommendations = get_recommendation_insights(inputs, happiness_pred, stress_pred, burnout_risk)
    
    # Sort recommendations by priority
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations_sorted = sorted(recommendations, key=lambda x: priority_order.get(x['priority'], 3))
    
    # Add each recommendation
    for i, rec in enumerate(recommendations_sorted):
        # Add recommendation title
        content.append(Paragraph(f"{i+1}. {rec['title']} ({rec['priority'].upper()} Priority)", subheader_style))
        
        # Add message
        content.append(Paragraph(f"<i>{rec['message']}</i>", normal_style))
        content.append(Spacer(1, 0.1 * inch))
        
        # Add actions
        content.append(Paragraph("Suggested Actions:", normal_style))
        for action in rec['actions']:
            content.append(Paragraph(f"• {action}", recommendation_style))
        
        content.append(Paragraph(f"<i>Impact: {rec['impact']}</i>", normal_style))
        content.append(Spacer(1, 0.2 * inch))
      # Add lifestyle impact analysis
    content.append(PageBreak())
    content.append(Paragraph("Lifestyle Impact Analysis:", subheader_style))
    content.append(Spacer(1, 0.1 * inch))
    
    # Create a lifestyle impact analysis
    impact_text = """Below is an analysis of how specific lifestyle factors may be affecting your wellness:"""
    content.append(Paragraph(impact_text, normal_style))
    content.append(Spacer(1, 0.15 * inch))
    
    # Prepare impact factors
    impact_factors = []
    
    # Sleep analysis
    sleep_status = "optimal" if 7 <= inputs["Sleep Hours"] <= 9 else "excessive" if inputs["Sleep Hours"] > 9 else "insufficient"
    sleep_impact = "positive" if 7 <= inputs["Sleep Hours"] <= 9 else "neutral" if inputs["Sleep Hours"] > 9 else "negative"
    sleep_text = f"Your sleep duration ({inputs['Sleep Hours']} hours/night) is {sleep_status}, which has a {sleep_impact} impact on your wellness."
    impact_factors.append(("Sleep Pattern", sleep_text))
    
    # Work hours analysis
    work_status = "balanced" if inputs["Work Hours per Week"] <= 45 else "heavy" if inputs["Work Hours per Week"] <= 55 else "excessive"
    work_impact = "positive" if inputs["Work Hours per Week"] <= 45 else "moderate" if inputs["Work Hours per Week"] <= 55 else "negative"
    work_text = f"Your work schedule ({inputs['Work Hours per Week']} hours/week) is {work_status}, with a {work_impact} impact on work-life balance."
    impact_factors.append(("Work Schedule", work_text))
    
    # Screen time analysis
    screen_status = "healthy" if inputs["Screen Time per Day (Hours)"] < 4 else "moderate" if inputs["Screen Time per Day (Hours)"] < 7 else "high"
    screen_impact = "positive" if inputs["Screen Time per Day (Hours)"] < 4 else "neutral" if inputs["Screen Time per Day (Hours)"] < 7 else "negative"
    screen_text = f"Your screen time ({inputs['Screen Time per Day (Hours)']} hours/day) is {screen_status}, with a {screen_impact} impact on eye health and sleep quality."
    impact_factors.append(("Screen Time", screen_text))
    
    # Exercise impact
    exercise_impact = "significant positive" if inputs["Exercise Level"] == "High" else "moderate positive" if inputs["Exercise Level"] == "Moderate" else "minimal"
    exercise_text = f"Your {inputs['Exercise Level'].lower()} exercise level has a {exercise_impact} impact on both physical and mental health."
    impact_factors.append(("Physical Activity", exercise_text))
    
    # Social interaction impact
    social_status = "strong" if inputs["Social Interaction Score"] >= 8 else "moderate" if inputs["Social Interaction Score"] >= 5 else "limited"
    social_impact = "very positive" if inputs["Social Interaction Score"] >= 8 else "positive" if inputs["Social Interaction Score"] >= 5 else "potentially negative"
    social_text = f"Your {social_status} social connections (score: {inputs['Social Interaction Score']}/10) have a {social_impact} impact on happiness and stress resilience."
    impact_factors.append(("Social Connection", social_text))
    
    # Create a table for the impact analysis
    impact_data = []
    for factor, text in impact_factors:
        impact_data.append([factor, text])
        
    t = Table(impact_data, colWidths=[1.5*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('WORDWRAP', (1, 0), (1, -1), True),
    ]))
    content.append(t)
    
    content.append(Spacer(1, 0.25 * inch))
    content.append(Paragraph("Key Insights:", normal_style))
    
    # Generate overall key insights
    insights = []
    
    # Add personalized insights based on the data
    if inputs["Sleep Hours"] < 7:
        insights.append("Improving your sleep duration could significantly boost your happiness scores and reduce stress.")
    
    if inputs["Work Hours per Week"] > 50 and inputs["Screen Time per Day (Hours)"] > 6:
        insights.append("The combination of high work hours and screen time is a significant contributor to your burnout risk.")
    
    if inputs["Social Interaction Score"] < 5 and happiness_pred < 6:
        insights.append("Increasing your social connections could help improve your happiness levels.")
    
    if inputs["Exercise Level"] == "Low":
        insights.append("Adding regular exercise to your routine would likely improve all aspects of your wellness metrics.")
    
    # Add default insight if none were generated
    if not insights:
        insights.append("Your lifestyle factors are generally well-balanced. Focus on maintaining these healthy habits.")
    
    # Add each insight as a bullet point
    for insight in insights:
        content.append(Paragraph(f"• {insight}", normal_style))
    
    # Add disclaimer
    content.append(Spacer(1, 0.5 * inch))
    content.append(Paragraph("Disclaimer:", subheader_style))
    disclaimer_text = """This report provides insights for personal reflection based on the information you provided. 
    It is not a substitute for professional medical advice. The predictions are based on statistical models 
    and population data, which may not reflect individual variations. For serious mental health concerns, 
    please consult a healthcare professional."""
    
    content.append(Paragraph(disclaimer_text, normal_style))
      # Create footer with page numbers
    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"LifeSync Wellness Report - Page {page_num}"
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(7.5*inch, 0.5*inch, text)
        canvas.drawString(0.5*inch, 0.5*inch, f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}")
    
    # Build PDF with page numbers
    doc.build(content, onFirstPage=add_page_number, onLaterPages=add_page_number)
    
    # Get PDF content
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# Helper function for PDF report - get interpretation of scores
def get_interpretation(score, metric_type):
    if metric_type == "happiness":
        if score >= 8:
            return "Excellent - Very high happiness levels"
        elif score >= 6:
            return "Good - Above average happiness"
        elif score >= 4:
            return "Moderate - Average happiness levels"
        elif score >= 2:
            return "Low - May need attention"
        else:
            return "Very Low - Requires immediate attention"
    elif metric_type == "stress":
        if score <= 3:
            return "Low - Well-managed stress levels"
        elif score <= 6:
            return "Moderate - Average stress levels"
        else:
            return "High - Elevated stress requires attention"
    elif metric_type == "burnout":
        if score <= 30:
            return "Low Risk - Good work-life balance"
        elif score <= 60:
            return "Moderate Risk - Monitor and make adjustments"
        else:
            return "High Risk - Immediate attention recommended"

# Helper function for PDF report - create visual progress gauge
def create_progress_gauge(value, max_value, width, height):
    """
    Create a visual progress gauge for PDF report metrics
    
    Parameters:
    -----------
    value : float
        The current value to display (e.g., happiness score)
    max_value : float
        The maximum possible value (e.g., 10 for happiness, 100 for burnout)
    width : int
        Width of the gauge in pixels
    height : int
        Height of the gauge in pixels
    
    Returns:
    --------
    Image object with the gauge visualization
    """
    # Create a BytesIO buffer for the image
    img_buffer = io.BytesIO()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width/72, height/72), dpi=72)
    
    # Determine color based on value and metric type
    if max_value == 10:  # Happiness or Stress
        if value >= 7:
            color = '#28a745'  # Green for good happiness
        elif value >= 4:
            color = '#ffc107'  # Yellow for moderate
        else:
            color = '#dc3545'  # Red for low happiness
        
        # Adjust color if this is likely stress (inverse scale)
        if value > 5:
            color = '#dc3545'  # Red for high stress
        
    else:  # Burnout (0-100 scale)
        if value <= 30:
            color = '#28a745'  # Green for low burnout risk
        elif value <= 60:
            color = '#ffc107'  # Yellow for moderate risk
        else:
            color = '#dc3545'  # Red for high risk
    
    # Create progress bar
    progress_pct = min(value / max_value, 1.0)
    bar = ax.barh(0, progress_pct, color=color, height=0.5)
    
    # Add background for unfilled portion
    ax.barh(0, 1.0, color='#e9ecef', height=0.5, alpha=0.5, zorder=0)
    
    # Add value text
    if max_value == 100:
        value_text = f"{value:.0f}%"
    else:
        value_text = f"{value:.1f}/{max_value}"
    
    ax.text(progress_pct / 2, 0, value_text, 
            ha='center', va='center', color='white', 
            fontweight='bold', fontsize=8)
    
    # Remove axes and spines
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save figure to buffer
    plt.savefig(img_buffer, format='png', dpi=72, bbox_inches='tight', pad_inches=0)
    img_buffer.seek(0)
    plt.close(fig)
    
    # Return as an Image object for PDF
    return Image(img_buffer, width=width, height=height)

def main():
    # Header section
    st.markdown("""
    <div class="simulator-header">
        <h1>🔮 LifeSync Wellness Simulator</h1>
        <p>Predict your wellness trends and get personalized insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    happiness_model, stress_model = load_models()
    
    if happiness_model is None or stress_model is None:
        st.error("⚠️ Unable to load prediction models. Please check that model files exist in the outputs directory.")
        return
     
    
    with st.container():
        st.markdown("""
        <div class="card" style="border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #dee2e6;">
            <div class="card-header text-center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px 15px 0 0;">
                <h5 class="mb-0"><i class="fas fa-user-cog"></i> Personal Information & Lifestyle Factors</h5>
            </div>
            
        """, unsafe_allow_html=True)
        
        # Create input form with enhanced columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="mb-3">
                <h6 class="text-primary"><i class="fas fa-user"></i> Personal Info</h6>
            </div>
            """, unsafe_allow_html=True)
            name = st.text_input("Full Name", value="", help="Enter your full name for personalized report")
            age = st.slider("Age", 18, 80, 30, help="Your current age in years")
            gender = st.selectbox("Gender", ["Female", "Male", "Other"], help="Select your gender identity")
            country = st.selectbox("Country", ["USA", "Canada", "Australia", "Japan", "India", "Germany", "Brazil"], 
                                 help="Your country of residence")
        
        with col2:
            st.markdown("""
            <div class="mb-3">
                <h6 class="text-success"><i class="fas fa-heart"></i> Lifestyle Factors</h6>
            </div>
            """, unsafe_allow_html=True)
            exercise_level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"], 
                                        help="How often do you exercise? Low: rarely, Moderate: 2-3x/week, High: daily")
            diet_type = st.selectbox("Diet Type", ["Balanced", "Vegetarian", "Vegan", "Keto", "Junk Food"],
                                   help="What best describes your eating habits?")
            mental_health = st.selectbox("Mental Health Condition", ["None", "Anxiety", "Depression", "PTSD", "Bipolar"],
                                       help="Any mental health conditions you're managing (optional)")
        
        with col3:
            st.markdown("""
            <div class="mb-3">
                <h6 class="text-warning"><i class="fas fa-clock"></i> Daily Habits</h6>
            </div>
            """, unsafe_allow_html=True)
            sleep_hours = st.slider("Sleep Hours per Night", 3.0, 12.0, 7.5, 0.5, 
                                   help="Average hours of sleep you get per night")
            work_hours = st.slider("Work Hours per Week", 0, 80, 40,
                                 help="Total hours you work in a typical week")
            screen_time = st.slider("Screen Time per Day (Hours)", 0.0, 16.0, 4.0, 0.5,
                                  help="Daily screen time including phones, computers, TV")
            social_interaction = st.slider("Social Interaction Score (1-10)", 1, 10, 6,
                                         help="Rate your social activity level: 1=very isolated, 10=very social")
        
        # Enhanced prediction button
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            predict_button = st.button(
                "🔮 Generate Wellness Predictions", 
                use_container_width=True,
                help="Click to analyze your wellness trends and get personalized recommendations"
            )
    
    # Process predictions when button is clicked or if predictions have already been made
    if predict_button or st.session_state.predictions_made:# Prepare inputs
        inputs = {
            'Name': name,
            'Age': age,
            'Gender': gender,
            'Country': country,
            'Exercise Level': exercise_level,
            'Diet Type': diet_type,
            'Mental Health Condition': mental_health,
            'Sleep Hours': sleep_hours,
            'Work Hours per Week': work_hours,
            'Screen Time per Day (Hours)': screen_time,
            'Social Interaction Score': social_interaction
        }
        
        # Store in session state
        st.session_state.inputs = inputs
        st.session_state.predictions_made = True
          # Process inputs for model
        processed_inputs = preprocess_inputs(inputs)
        st.session_state.processed_inputs = processed_inputs
          # Make predictions
        happiness_pred = happiness_model.predict(processed_inputs)[0]
        happiness_pred = round(max(0, min(10, happiness_pred)), 1)
        
        stress_raw = stress_model.predict(processed_inputs)[0]
        stress_pred = round(max(0, min(10, ((stress_raw - 1) / 2) * 10)), 2)
        
        # Calculate burnout risk
        burnout_risk = (
            0.4 * inputs['Work Hours per Week'] +
            0.25 * inputs['Screen Time per Day (Hours)'] +
            0.2 * (10 - inputs['Sleep Hours']) +
            0.15 * (10 - inputs['Social Interaction Score'])
        )
        burnout_risk = round(max(0, min(100, burnout_risk)), 1)
        
        # Store predictions in session state for PDF generation
        st.session_state.happiness_pred = happiness_pred
        st.session_state.stress_pred = stress_pred
        st.session_state.burnout_risk = burnout_risk
          # Display immediate predictions with enhanced Bootstrap cards
        st.markdown("## 🎯 Your Wellness Predictions")
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            happiness_emoji = "😊" if happiness_pred >= 7 else "😐" if happiness_pred >= 4 else "😞"
            happiness_progress = int((happiness_pred / 10) * 100)
            st.markdown(f"""
            <div class="card h-100" style="background: linear-gradient(135deg, #d4edda, #c3e6cb); border: 2px solid #28a745; border-radius: 15px; box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);">
                <div class="card-body text-center p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <i class="fas fa-smile" style="font-size: 1.2em; color: #28a745;"></i>
                        <span class="badge bg-success">Happiness</span>
                    </div>
                    <h1 style="color: #28a745; margin: 15px 0; font-weight: bold;">
                        {happiness_pred}/10
                    </h1>
                    <div class="progress mb-2" style="height: 8px;">
                        <div class="progress-bar bg-success" style="width: {happiness_progress}%"></div>
                    </div>
                    <p style="color: #155724; margin: 0; font-size: 0.9em;">
                        {"Excellent mood outlook!" if happiness_pred >= 7 else 
                         "Moderate happiness levels" if happiness_pred >= 4 else 
                         "Consider lifestyle improvements"}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            stress_emoji = "😌" if stress_pred < 3 else "😟" if stress_pred < 7 else "😰"
            stress_progress = int((stress_pred / 10) * 100)
            st.markdown(f"""
            <div class="card h-100" style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 2px solid #dc3545; border-radius: 15px; box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);">
                <div class="card-body text-center p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <i class="fas fa-brain" style="font-size: 1.2em; color: #dc3545;"></i>
                        <span class="badge bg-danger">Stress</span>
                    </div>
                    <h1 style="color: #dc3545; margin: 15px 0; font-weight: bold;">
                        {stress_pred:.2f}/10
                    </h1>
                    <div class="progress mb-2" style="height: 8px;">
                        <div class="progress-bar bg-danger" style="width: {stress_progress}%"></div>
                    </div>
                    <p style="color: #721c24; margin: 0; font-size: 0.9em;">
                        {"Low stress - well managed!" if stress_pred < 3 else 
                         "Moderate stress levels" if stress_pred < 7 else 
                         "High stress - needs attention"}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col3:
            burnout_emoji = "✅" if burnout_risk < 40 else "⚠️" if burnout_risk < 60 else "🚨"
            burnout_status = "Low Risk" if burnout_risk < 40 else "Moderate Risk" if burnout_risk < 60 else "High Risk"
            burnout_color = "#28a745" if burnout_risk < 40 else "#ffc107" if burnout_risk < 60 else "#dc3545"
            burnout_bg = "linear-gradient(135deg, #fff3cd, #ffeaa7)" if burnout_risk < 60 else "linear-gradient(135deg, #f8d7da, #f5c6cb)"
            if burnout_risk < 40:
                burnout_bg = "linear-gradient(135deg, #d4edda, #c3e6cb)"
            
            st.markdown(f"""
            <div class="card h-100" style="background: {burnout_bg}; border: 2px solid {burnout_color}; border-radius: 15px; box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);">
                <div class="card-body text-center p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <i class="fas fa-fire" style="font-size: 1.2em; color: {burnout_color};"></i>
                        <span class="badge" style="background-color: {burnout_color};">Burnout Risk</span>
                    </div>
                    <h1 style="color: {burnout_color}; margin: 15px 0; font-weight: bold;">
                        {burnout_risk}%
                    </h1>
                    <div class="progress mb-2" style="height: 8px;">
                        <div class="progress-bar" style="width: {burnout_risk}%; background-color: {burnout_color};"></div>
                    </div>
                    <p style="color: #856404; margin: 0; font-size: 0.9em;">
                        {burnout_status}
                    </p>
                </div>
            </div>            """, unsafe_allow_html=True)        # Enhanced forecast visualization
        try:
            happiness_forecast, stress_forecast, burnout_forecast = generate_forecast(happiness_pred, stress_pred, burnout_risk)
            
            # Store forecasts in session state for PDF generation
            st.session_state.happiness_forecast = happiness_forecast
            st.session_state.stress_forecast = stress_forecast
            st.session_state.burnout_forecast = burnout_forecast
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            # Use session state values if they exist, otherwise initialize with defaults
            happiness_forecast = st.session_state.happiness_forecast if st.session_state.happiness_forecast else {"Current": 5.0, "3 Days": 4.8, "1 Week": 4.5, "1 Month": 4.0, "3 Months": 3.5}
            stress_forecast = st.session_state.stress_forecast if st.session_state.stress_forecast else {"Current": 5.0, "3 Days": 5.5, "1 Week": 6.0, "1 Month": 7.0, "3 Months": 7.5}
            burnout_forecast = st.session_state.burnout_forecast if st.session_state.burnout_forecast else {"Current": 50.0, "3 Days": 52.0, "1 Week": 55.0, "1 Month": 60.0, "3 Months": 65.0}
        
        st.markdown("---")
        st.markdown("""
        <div class="text-center mb-4">
            <h2><i class="fas fa-chart-line text-primary"></i> Wellness Trend Forecast</h2>
            <p class="text-muted"><em>Projected changes if current lifestyle patterns continue</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Wrap chart in Bootstrap card
        st.markdown("""
        <div class="card" style="border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div class="card-header text-center" style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-bottom: 2px solid #dee2e6;">
                <h5 class="mb-0"><i class="fas fa-analytics"></i> Comprehensive Wellness Forecast</h5>
            </div>
            <div class="card-body p-3">
        """, unsafe_allow_html=True)
        
        # Single comprehensive chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comprehensive Wellness Forecast', fontsize=16, fontweight='bold', y=0.98)
        
        periods = list(happiness_forecast.keys())
        x_pos = range(len(periods))
        
        # Happiness trend
        happiness_values = list(happiness_forecast.values())
        ax1.plot(x_pos, happiness_values, marker='o', linewidth=3, color='#28a745', markersize=8)
        ax1.fill_between(x_pos, happiness_values, alpha=0.3, color='#28a745')
        ax1.set_title('Happiness Score Trend', fontweight='bold')
        ax1.set_ylabel('Score (0-10)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(periods, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 10)
        
        # Stress trend
        stress_values = list(stress_forecast.values())
        ax2.plot(x_pos, stress_values, marker='s', linewidth=3, color='#dc3545', markersize=8)
        ax2.fill_between(x_pos, stress_values, alpha=0.3, color='#dc3545')
        ax2.set_title('Stress Level Trend', fontweight='bold')
        ax2.set_ylabel('Level (0-10)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(periods, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        
        # Burnout risk trend
        burnout_values = list(burnout_forecast.values())
        colors = ['#28a745' if v < 40 else '#ffc107' if v < 60 else '#dc3545' for v in burnout_values]
        ax3.bar(x_pos, burnout_values, color=colors, alpha=0.7)
        ax3.set_title('Burnout Risk Progression', fontweight='bold')
        ax3.set_ylabel('Risk Percentage')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(periods, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Wellness index (combined metric)
        wellness_values = [(h + (10-s) + (100-b)/10) / 3 for h, s, b in zip(happiness_values, stress_values, burnout_values)]
        ax4.plot(x_pos, wellness_values, marker='^', linewidth=3, color='#6f42c1', markersize=8)
        ax4.fill_between(x_pos, wellness_values, alpha=0.3, color='#6f42c1')
        ax4.set_title('Overall Wellness Index', fontweight='bold')
        ax4.set_ylabel('Index (0-10)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(periods, rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Close the chart card
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced forecast summary table
        st.markdown("### 📊 Detailed Forecast Data")
        forecast_df = pd.DataFrame({
            'Time Period': periods,
            'Happiness': [f"{v}/10" for v in happiness_values],
            'Stress': [f"{v:.2f}/10" for v in stress_values],
            'Burnout Risk': [f"{v}%" for v in burnout_values],
            'Wellness Index': [f"{v:.1f}/10" for v in wellness_values]
        })
        
        # Style the dataframe
        st.markdown("""
        <div class="card">
            <div class="card-body p-2">
        """, unsafe_allow_html=True)
        
        st.dataframe(
            forecast_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Time Period": st.column_config.TextColumn("📅 Time Period", width="medium"),
                "Happiness": st.column_config.TextColumn("😊 Happiness", width="small"),
                "Stress": st.column_config.TextColumn("🧠 Stress", width="small"),
                "Burnout Risk": st.column_config.TextColumn("🔥 Burnout Risk", width="small"),
                "Wellness Index": st.column_config.TextColumn("⭐ Wellness Index", width="small")
            }
        )
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
          # Personalized recommendations
        st.markdown("---")
        st.markdown("## 💡 Personalized Recommendations")
        
        recommendations = get_recommendation_insights(inputs, happiness_pred, stress_pred, burnout_risk)
        if recommendations:
            # Sort recommendations by priority
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            recommendations_sorted = sorted(recommendations, key=lambda x: priority_order.get(x['priority'], 3))

            # Emoji mapping for categories
            emoji_map = {
                'Physical Activity': '🏃',
                'Fitness Enhancement': '💪',
                'Sleep Optimization': '🛏️',
                'Sleep Balance': '⏰',
                'Work-Life Balance': '💼',
                'Digital Wellness': '📱',
                'Social Connection': '🤝',
                'Nutrition': '🍎',
                'Mental Health Support': '🧠',
                'Stress Management': '🌿',
                'Happiness Boost': '😊',
                'Wellness Maintenance': '⭐',
            }
            # Color mapping for priorities
            priority_colors = {
                'high': '#c53030',
                'medium': '#c05621',
                'low': '#2f855a'
            }
            for i, rec in enumerate(recommendations_sorted):
                if i % 2 == 0:
                    cols = st.columns(2)
                with cols[i % 2]:
                    emoji = emoji_map.get(rec['category'], '💡')
                    color = priority_colors.get(rec['priority'], '#2f855a')
                    st.markdown(f"""
**{emoji} {rec['title']}**  
<span style='color:{color}; font-weight:bold;'>{rec['priority'].upper()}</span>

<p style='font-size:0.95em; color:#555;'>{rec['message']}</p>

**Action Steps:**
""", unsafe_allow_html=True)
                    for action in rec['actions'][:3]:
                        st.markdown(f"- {action}")
                    st.markdown(f"""
<small style='color:#888;'><i>💡 {rec['impact']}</i></small>
---
""", unsafe_allow_html=True)
        else:
            st.success("Your lifestyle appears well-balanced! Keep up the great work.")
        
        # Enhanced download section
        st.markdown("---")
          # Create downloadable CSV in the same format as our prediction history CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a DataFrame that matches the prediction history format
        download_df = pd.DataFrame({
            "Timestamp": [timestamp],
            "Age": [inputs["Age"]],
            "Gender": [inputs["Gender"]],
            "Sleep Hours": [inputs["Sleep Hours"]],
            "Work Hours per Week": [inputs["Work Hours per Week"]],
            "Screen Time per Day (Hours)": [inputs["Screen Time per Day (Hours)"]],
            "Social Interaction Score": [inputs["Social Interaction Score"]],
            "Exercise Level": [inputs["Exercise Level"]],
            "Diet Type": [inputs["Diet Type"]],
            "Mental Health Condition": [inputs["Mental Health Condition"]],
            "Happiness Score": [happiness_pred],
            "Stress Level": [stress_pred],
            "Burnout Risk": [burnout_risk]
        })
        
        # Add forecast data in separate rows
        forecast_data = []
        for i, period in enumerate(periods):
            if period != "Current":  # Skip current since it's already included
                forecast_data.append({
                    "Timestamp": f"{timestamp} (Forecast: {period})",
                    "Age": inputs["Age"],
                    "Gender": inputs["Gender"],
                    "Sleep Hours": inputs["Sleep Hours"],
                    "Work Hours per Week": inputs["Work Hours per Week"],
                    "Screen Time per Day (Hours)": inputs["Screen Time per Day (Hours)"],
                    "Social Interaction Score": inputs["Social Interaction Score"],
                    "Exercise Level": inputs["Exercise Level"],
                    "Diet Type": inputs["Diet Type"],
                    "Mental Health Condition": inputs["Mental Health Condition"],
                    "Happiness Score": happiness_values[i],
                    "Stress Level": stress_values[i],
                    "Burnout Risk": burnout_values[i]
                })
          # Combine current and forecast data
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            download_df = pd.concat([download_df, forecast_df], ignore_index=True)
            
        # Add download button - PDF is only generated when this button is clicked
        col_download = st.columns([1, 2, 1])
        with col_download[1]:            # Create a button to generate PDF
            generate_pdf_button = st.button("📑 Generate PDF Report", 
                                          use_container_width=True,
                                          help="Click to prepare your personalized wellness report")
            
            # Only generate PDF when the button is clicked
            if generate_pdf_button:
                # Make sure all required variables exist and are not None
                if (happiness_forecast and stress_forecast and burnout_forecast and 
                    happiness_pred is not None and stress_pred is not None and burnout_risk is not None):
                    # Show a spinner while generating the PDF
                    with st.spinner("Creating your personalized wellness report..."):
                        try:
                            # Generate PDF content on demand
                            pdf_content = generate_pdf_report(
                                inputs,
                                happiness_pred,
                                stress_pred,
                                burnout_risk,
                                happiness_forecast,
                                stress_forecast,
                                burnout_forecast
                            )
                            # Store the generated PDF in session state
                            st.session_state.pdf_content = pdf_content
                            
                            # Show success message
                            st.success("Your PDF report is ready! Click below to download.")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.error("Missing data needed for PDF generation. Please try generating predictions again.")
            
            # If PDF has been generated successfully, show the download button
            if 'pdf_content' in st.session_state and st.session_state.pdf_content is not None:
                st.download_button(
                    label="📥 Download Your Wellness Report (PDF)",
                    data=st.session_state.pdf_content,
                    file_name=f"LifeSync_Wellness_Report_{inputs.get('Name', '').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Download your personalized wellness report with predictions and recommendations"
                )
        
        st.markdown("""
            </div>
        </div>        """, unsafe_allow_html=True)        # Save prediction history (after download section)
          # Save to both the outputs directory and the predictions folder
        
        # Save to the main outputs directory
        success, path = save_prediction_to_csv(inputs, happiness_pred, stress_pred, burnout_risk)
        
        # Also save to the predictions folder for easy access
        success2, path2 = save_predictions_to_csv(inputs, happiness_pred, stress_pred, burnout_risk)

    # Enhanced information section
    st.markdown("---")
    with st.expander("ℹ️ About This Simulator", expanded=False):
        st.markdown("""
        <div class="card">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h5 class="text-primary"><i class="fas fa-cogs"></i> How It Works</h5>
                        <ul class="list-unstyled">
                            <li class="mb-2">
                                <i class="fas fa-smile text-success"></i>
                                <strong>Happiness Predictions:</strong> Random Forest model trained on lifestyle factors
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-brain text-warning"></i>
                                <strong>Stress Predictions:</strong> XGBoost model with feature importance analysis
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-fire text-danger"></i>
                                <strong>Burnout Risk:</strong> Formula-based calculation considering work hours, screen time, sleep, and social factors
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-chart-line text-info"></i>
                                <strong>Forecasts:</strong> Realistic projections assuming current lifestyle patterns continue
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5 class="text-warning"><i class="fas fa-exclamation-triangle"></i> Important Notes</h5>
                        <ul class="list-unstyled">
                            <li class="mb-2">
                                <i class="fas fa-lightbulb text-primary"></i>
                                This tool provides insights for personal reflection, not medical advice
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-users text-secondary"></i>
                                Predictions are based on population data and may not reflect individual variations
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-user-md text-success"></i>
                                Consider consulting healthcare professionals for serious mental health concerns
                            </li>
                            <li class="mb-2">
                                <i class="fas fa-shield-alt text-info"></i>
                                All data processing happens locally - no personal information is stored externally
                            </li>
                        </ul>
                    </div>
                
                
                                
                
        
        """, unsafe_allow_html=True)
        st.markdown("""
                <div class="alert alert-info mt-3" style="border-left: 4px solid #17a2b8;">
                    <strong><i class="fas fa-info-circle"></i> Pro Tip:</strong> 
                    Use this tool regularly to track how lifestyle changes affect your predicted wellness trends. 
                    Small improvements in sleep, exercise, and work-life balance can lead to significant positive changes!
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
