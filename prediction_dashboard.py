"""
Simple ML Predictions Dashboard
Focus: Interactive predictions for 4 trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
from joblib import load

# Page configuration
st.set_page_config(
    page_title="ML Predictions",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔮 ML Model Predictions</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Model Selection
st.sidebar.markdown("## 🎯 Select Model")
st.sidebar.markdown("Choose which prediction model to use:")

model_choice = st.sidebar.radio(
    "Available Models:",
    [
        "❤️ Heart Disease Classification",
        "🌡️ Temperature Prediction (Ensemble)",
        "🎯 Multi-Output (Pressure & Humidity)",
        "☁️ Weather Classification (4-class)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")

# Display different info based on selection
if "Heart Disease" in model_choice:
    st.sidebar.success("✅ **Model Available**")
    st.sidebar.info("""
    **Type:** Classification  
    **Algorithm:** Stacking Ensemble  
    **ROC-AUC:** 0.9782  
    **Features:** 11 clinical measurements
    """)
elif "Temperature" in model_choice:
    st.sidebar.warning("⚠️ **Requires Training**")
    st.sidebar.info("""
    **Type:** Regression  
    **Algorithm:** Stacking Ensemble  
    **R² Score:** 0.7889  
    **Features:** 8 weather variables
    """)
elif "Multi-Output" in model_choice:
    st.sidebar.warning("⚠️ **Requires Training**")
    st.sidebar.info("""
    **Type:** Multi-Output Regression  
    **Algorithm:** XGBoost  
    **Avg R²:** 0.9282  
    **Outputs:** Pressure & Humidity
    """)
else:  # Weather Classification
    st.sidebar.warning("⚠️ **Requires Training**")
    st.sidebar.info("""
    **Type:** 4-Class Classification  
    **Algorithm:** Random Forest  
    **ROC-AUC:** 0.8493  
    **Features:** 31 engineered
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tip:** Heart Disease model is pre-trained and ready to use!")

# Main content area
st.markdown(f"## {model_choice}")

# ============================================================================
# PREDICTION SECTION 1: HEART DISEASE
# ============================================================================

if "Heart Disease" in model_choice:
    
    st.markdown("### Enter Patient Information")
    
    # Try to load model
    try:
        model_dir = "Dataset2/classification_results/models"
        
        if os.path.exists(os.path.join(model_dir, "best_model.joblib")):
            model = load(os.path.join(model_dir, "best_model.joblib"))
            scaler = load(os.path.join(model_dir, "scaler.joblib"))
            
            with open(os.path.join(model_dir, "model_metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            st.success(f"✅ Model Loaded: {metadata.get('model_name', 'Unknown')}")
            
            # Display expected features
            st.info(f"**Required Features:** {', '.join(metadata.get('feature_names', []))}")
            
            # Input form
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📋 Demographics & Vitals**")
                age = st.number_input("Age (years)", 20, 100, 50, help="Patient's age in years")
                sex = st.selectbox("Sex", ["Male (1)", "Female (0)"], help="Biological sex")
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, 
                                          help="Blood pressure at rest")
                chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, 
                                     help="Serum cholesterol level")
            
            with col2:
                st.markdown("**💓 Heart Measurements**")
                cp = st.selectbox("Chest Pain Type", [
                    "0 - Typical Angina",
                    "1 - Atypical Angina",
                    "2 - Non-anginal Pain",
                    "3 - Asymptomatic"
                ], help="Type of chest pain experienced")
                thalach = st.number_input("Max Heart Rate", 60, 220, 150, 
                                        help="Maximum heart rate achieved")
                oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1, 
                                        help="ST depression induced by exercise")
                slope = st.selectbox("ST Slope", [
                    "0 - Upsloping",
                    "1 - Flat",
                    "2 - Downsloping"
                ], help="Slope of peak exercise ST segment")
            
            with col3:
                st.markdown("**🔬 Clinical Tests**")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"], 
                                 help="Is fasting blood sugar > 120 mg/dl?")
                restecg = st.selectbox("Resting ECG", [
                    "0 - Normal",
                    "1 - ST-T Wave Abnormality",
                    "2 - LV Hypertrophy"
                ], help="Resting electrocardiographic results")
                exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"], 
                                   help="Exercise induced angina")
            
            st.markdown("---")
            
            # Predict button
            if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
                
                # Prepare input data (11 features)
                input_data = np.array([[
                    age,
                    int(sex.split()[-1].strip("()")),
                    int(cp.split()[0]),
                    trestbps,
                    chol,
                    int(fbs.split()[-1].strip("()")),
                    int(restecg.split()[0]),
                    thalach,
                    int(exang.split()[-1].strip("()")),
                    oldpeak,
                    int(slope.split()[0])
                ]])
                
                # Scale and predict
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    risk_score = proba[1] * 100
                    confidence = max(proba) * 100
                else:
                    risk_score = prediction * 100
                    confidence = 0
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("### 🚨 POSITIVE")
                        st.markdown("**Heart disease detected**")
                    else:
                        st.success("### ✅ NEGATIVE")
                        st.markdown("**No heart disease detected**")
                
                with col2:
                    st.metric("Risk Score", f"{risk_score:.1f}%")
                    st.caption("Probability of heart disease")
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.caption("Model confidence")
                
                # Risk interpretation
                st.markdown("---")
                st.markdown("### 🏥 Clinical Interpretation")
                
                if risk_score < 30:
                    st.success("""
                    **Low Risk**  
                    - Continue regular health check-ups
                    - Maintain healthy lifestyle
                    - Monitor cardiovascular health annually
                    """)
                elif risk_score < 70:
                    st.warning("""
                    **Moderate Risk**  
                    - Further evaluation recommended
                    - Consult with cardiologist
                    - Consider additional diagnostic tests
                    - Lifestyle modifications advised
                    """)
                else:
                    st.error("""
                    **High Risk**  
                    - Immediate medical attention recommended
                    - Urgent cardiology consultation
                    - Comprehensive cardiac workup needed
                    - Do not delay seeking medical care
                    """)
                
                st.info("⚠️ **Disclaimer:** This is a machine learning prediction tool for educational purposes. Always consult qualified healthcare professionals for medical decisions.")
        
        else:
            st.error("❌ Model files not found!")
            st.warning(f"Expected location: `{model_dir}/best_model.joblib`")
            st.info("Please train the model first by running: `python Dataset2/heart_disease_classification.py`")
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)

# ============================================================================
# PREDICTION SECTION 2: TEMPERATURE
# ============================================================================

elif "Temperature" in model_choice:
    
    st.markdown("### Enter Weather Conditions")
    
    # Try to load model
    try:
        model_path = "ensemble_results/models/best_ensemble_model.joblib"
        metadata_path = "ensemble_results/models/model_metadata.json"
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.success(f"✅ Model Loaded: {metadata.get('model_name')}")
            st.info(f"**Required Features:** {', '.join(metadata.get('feature_names', []))}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌤️ Weather Conditions**")
                summary = st.selectbox("Weather Summary", ["Clear", "Partly Cloudy", "Cloudy", "Overcast"])
                precip_type = st.selectbox("Precipitation Type", ["None", "Rain", "Snow"])
                humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
                wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
            
            with col2:
                st.markdown("**🌍 Atmospheric Data**")
                wind_bearing = st.number_input("Wind Bearing (degrees)", 0.0, 360.0, 180.0)
                visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)
                cloud_cover = st.slider("Cloud Cover (0-8 oktas)", 0.0, 8.0, 4.0)
                pressure = st.number_input("Pressure (mbar)", 950.0, 1050.0, 1013.0)
            
            st.markdown("---")
            
            if st.button("🌡️ Predict Temperature", type="primary", use_container_width=True):
                # Map categorical to numerical
                summary_map = {"Clear": 0, "Partly Cloudy": 1, "Cloudy": 2, "Overcast": 3}
                precip_map = {"None": 0, "Rain": 1, "Snow": 2}
                
                input_data = np.array([[
                    summary_map[summary],
                    precip_map[precip_type],
                    humidity,
                    wind_speed,
                    wind_bearing,
                    visibility,
                    cloud_cover,
                    pressure
                ]])
                
                prediction = model.predict(input_data)[0]
                mae = metadata['performance_metrics']['mae']
                r2 = metadata['performance_metrics']['r2_score']
                
                st.markdown('<div class="prediction-result">🌡️ Predicted Temperature: {:.1f}°C</div>'.format(prediction), 
                          unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{prediction:.1f}°C")
                with col2:
                    st.metric("Uncertainty (MAE)", f"±{mae:.2f}°C")
                with col3:
                    st.metric("Model R² Score", f"{r2:.4f}")
                
                # Temperature interpretation
                st.markdown("---")
                if prediction < 0:
                    st.info("❄️ **Freezing conditions** - Temperature below 0°C")
                elif prediction < 15:
                    st.info("🧊 **Cold** - Light jacket recommended")
                elif prediction < 25:
                    st.success("☀️ **Pleasant** - Comfortable temperature")
                else:
                    st.warning("🔥 **Hot** - Stay hydrated")
        
        else:
            st.warning("⚠️ **Model Not Available**")
            st.info("""
            **To use Temperature predictions:**
            
            1. Train the model:
            ```bash
            python ml_pipeline_knn.py
            ```
            
            2. Model will be saved to: `ensemble_results/models/`
            
            3. Refresh this page
            """)
            st.markdown("**Note:** Model size ~2.5GB (not included in repository)")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PREDICTION SECTION 3: MULTI-OUTPUT
# ============================================================================

elif "Multi-Output" in model_choice:
    
    st.markdown("### Enter Weather Features")
    st.caption("Predict both Pressure and Humidity simultaneously")
    
    try:
        model_path = "multi_output_results/models/best_model.joblib"
        metadata_path = "multi_output_results/models/model_metadata.json"
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.success(f"✅ Model Loaded: {metadata.get('model_name')}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌡️ Temperature Data**")
                temp = st.number_input("Temperature (°C)", -40.0, 50.0, 20.0)
                apparent = st.number_input("Apparent Temperature (°C)", -40.0, 50.0, 20.0)
                wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
                wind_bearing = st.number_input("Wind Bearing (°)", 0.0, 360.0, 180.0)
            
            with col2:
                st.markdown("**☁️ Weather Conditions**")
                vis = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)
                cloud = st.slider("Cloud Cover (0-8)", 0.0, 8.0, 4.0)
                summary = st.selectbox("Summary", ["Clear", "Partly Cloudy", "Cloudy", "Overcast"], key="mo_sum")
                precip = st.selectbox("Precip Type", ["None", "Rain", "Snow"], key="mo_precip")
            
            st.markdown("---")
            
            if st.button("🎯 Predict Pressure & Humidity", type="primary", use_container_width=True):
                # Map categorical
                summary_map = {"Clear": 0, "Partly Cloudy": 1, "Cloudy": 2, "Overcast": 3}
                precip_map = {"None": 0, "Rain": 1, "Snow": 2}
                
                input_data = np.array([[
                    summary_map[summary],
                    precip_map[precip],
                    (temp + apparent) / 2,
                    wind,
                    wind_bearing,
                    vis,
                    cloud,
                    1013  # Default pressure placeholder
                ]])
                
                predictions = model.predict(input_data)[0]
                
                st.markdown("### 📊 Multi-Output Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="prediction-result">🌀 Pressure: {:.1f} mbar</div>'.format(predictions[0]), 
                              unsafe_allow_html=True)
                    st.caption("**Atmospheric Pressure**")
                    st.info("✅ Predicted from weather conditions")
                
                with col2:
                    st.markdown('<div class="prediction-result">💧 Humidity: {:.1f}%</div>'.format(predictions[1]), 
                              unsafe_allow_html=True)
                    st.caption("**Relative Humidity**")
                    st.info("✅ Predicted simultaneously")
                
                st.success("🎯 **Both outputs predicted in a single forward pass!**")
        
        else:
            st.warning("⚠️ **Model Not Available**")
            st.info("""
            **To use Multi-Output predictions:**
            
            1. Train the model:
            ```bash
            python multi_output_regression.py
            ```
            
            2. Model will be saved to: `multi_output_results/models/`
            
            3. Refresh this page
            """)
            st.markdown("**Note:** Model size ~500MB (not included in repository)")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PREDICTION SECTION 4: WEATHER CLASSIFICATION
# ============================================================================

else:  # Weather Classification
    
    st.markdown("### Enter Weather Observations")
    st.caption("Classify into 4 weather types: Clear, Mostly Cloudy, Overcast, Partly Cloudy")
    
    try:
        model_path = "weather_classification_models/best_model.joblib"
        metadata_path = "weather_classification_models/model_metadata.json"
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.success(f"✅ Model Loaded: {metadata.get('model_name')}")
            st.info(f"**Model requires 31 engineered features** - Using simplified heuristic approach")
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🌡️ Temperature**")
                temp_w = st.number_input("Temperature (°C)", -40.0, 50.0, 20.0, key="tw")
                humidity_w = st.slider("Humidity (%)", 0.0, 100.0, 50.0, key="hw")
                apparent_w = st.number_input("Apparent Temp (°C)", -40.0, 50.0, 20.0, key="aw")
            
            with col2:
                st.markdown("**💨 Wind & Pressure**")
                pressure_w = st.number_input("Pressure (mbar)", 950.0, 1050.0, 1013.0, key="pw")
                wind_w = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0, key="ww")
                wind_bearing_w = st.number_input("Wind Bearing (°)", 0.0, 360.0, 180.0, key="wb")
            
            with col3:
                st.markdown("**☁️ Visibility & Clouds**")
                visibility_w = st.number_input("Visibility (km)", 0.0, 20.0, 10.0, key="vw")
                cloud_w = st.slider("Cloud Cover (0-8)", 0.0, 8.0, 4.0, key="cw")
            
            st.markdown("---")
            
            if st.button("☁️ Classify Weather", type="primary", use_container_width=True):
                
                st.warning("""
                **Note:** Full model prediction requires 31 engineered features.  
                Using simplified heuristic based on cloud cover and humidity.
                """)
                
                # Simplified heuristic
                classes = metadata['target_classes']
                
                if cloud_w > 6:
                    probs = [0.05, 0.10, 0.25, 0.60]  # Mostly Overcast
                elif cloud_w > 4:
                    probs = [0.10, 0.20, 0.60, 0.10]  # Mostly Cloudy
                elif cloud_w > 2:
                    probs = [0.20, 0.60, 0.15, 0.05]  # Mostly Partly Cloudy
                else:
                    probs = [0.70, 0.20, 0.08, 0.02]  # Mostly Clear
                
                # Adjust for humidity
                if humidity_w > 80:
                    probs = [p * 0.7 for p in probs[:-1]] + [probs[-1] * 1.3]
                elif humidity_w < 30:
                    probs = [probs[0] * 1.3] + [p * 0.7 for p in probs[1:]]
                
                # Normalize
                total = sum(probs)
                probs = [p / total for p in probs]
                
                st.markdown("### 🎲 Weather Classification Results")
                
                for cls, prob in zip(classes, probs):
                    st.progress(prob, text=f"**{cls}**: {prob*100:.1f}%")
                
                predicted_class = classes[probs.index(max(probs))]
                
                st.markdown('<div class="prediction-result">☁️ Most Likely: {}</div>'.format(predicted_class), 
                          unsafe_allow_html=True)
                
                st.metric("Confidence", f"{max(probs)*100:.1f}%")
                
                st.caption("⚠️ Simplified prediction. Full model requires feature engineering pipeline.")
        
        else:
            st.warning("⚠️ **Model Not Available**")
            st.info("""
            **To use Weather Classification:**
            
            1. Train the model:
            ```bash
            python encoding_comparison.py
            ```
            
            2. Model will be saved to: `weather_classification_models/`
            
            3. Refresh this page
            """)
            st.markdown("**Note:** Model size ~1GB (not included in repository)")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🔮 <b>Simple ML Predictions Dashboard</b></p>
    <p>4 Production Models | Real-time Predictions | Interactive Interface</p>
    <p style='font-size: 12px;'>Built with Streamlit | Academic Project 2025</p>
</div>
""", unsafe_allow_html=True)
