# LifeSync - Mental Health & Lifestyle Analytics

LifeSync is a comprehensive analytics platform that explores the relationship between lifestyle factors and mental wellness. The project uses machine learning to predict happiness scores and stress levels based on various lifestyle inputs.

## Project Overview

LifeSync consists of two main modules:

1. **LifeView Dashboard**: For exploring lifestyle patterns and mental wellness trends through interactive visualizations.
2. **SyncPredict Simulator**: For real-time predictions and personal forecasting based on user inputs.

## Features

### Dashboard Module
- Interactive filters for demographic and lifestyle factors
- Comprehensive visualizations including correlation heatmaps, boxplots, scatterplots, and distributions
- Model performance metrics and feature importance analysis
- SHAP (SHapley Additive exPlanations) visualizations for model interpretability

### Simulator Module
- Interactive form for entering personal lifestyle factors
- Real-time predictions for:
  - Happiness Score
  - Stress Level
  - Burnout Risk
- Wellness forecasts for 3, 7, 30, and 90 days
- Personalized recommendations based on prediction results

## Models

LifeSync uses multiple machine learning models to predict wellness outcomes:

1. **Happiness Score Prediction**: Regression model that predicts happiness on a 0-10 scale.
2. **Stress Level Prediction**: Regression model that predicts stress levels on a 0-10 scale.
3. **Burnout Risk Calculation**: Custom formula that considers work hours, screen time, and sleep deficit.

## Project Structure

```
/LifeSync/
├── data/
├── notebooks/
│   └── model_training.ipynb
├── outputs/
│   ├── lifesync_happiness_model.pkl
│   ├── lifesync_stress_model.pkl
│   ├── shap_summary_happiness.png
│   ├── shap_summary_stress.png
│   ├── model_comparison_metrics.csv
│   └── feature_importance.csv
├── dashboard/
│   ├── app_dashboard.py
│   └── app_simulator.py
├── scripts/
├── run_dashboard.py
├── Mental_Health_Lifestyle_Dataset.csv
└── README.md
```

## Getting Started

### Requirements

- Python 3.7+
- Required libraries:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - shap
  - joblib

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/lifesync.git
   cd lifesync
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   pip install reportlab
   ```

3. Run the dashboard:
   ```
   streamlit run run_dashboard.py
   ```

## Running the Models

To train the models from scratch:

1. Run the model training notebook:
   ```
   jupyter notebook notebooks/model_training.ipynb
   ```

2. The trained models will be saved to the `outputs/` directory.

## Using the Dashboard

1. **Filters**: Use the sidebar filters to explore different subsets of the data.
2. **Tabs**: Navigate between different views using the tabs at the top of the dashboard.
3. **Visualizations**: Interact with charts to get additional information.

## Using the Simulator

1. Enter your lifestyle factors in the form.
2. Click "Generate Predictions" to get your wellness predictions.
3. View your forecasts and personalized recommendations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
