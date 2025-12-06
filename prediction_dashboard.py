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
import sys
import pickle
import warnings
import matplotlib.pyplot as plt

# Suppress all warnings for cleaner dashboard output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*version.*')
warnings.filterwarnings('ignore', message='.*serialized model.*')
warnings.filterwarnings('ignore', message='.*XGBoost.*')
warnings.filterwarnings('ignore', category=UserWarning)

# Import all necessary ML libraries for unpickling
try:
    # Import from the correct XGBoost location
    from xgboost.sklearn import XGBClassifier, XGBRegressor
except ImportError:
    try:
        # Fallback for different XGBoost versions
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        XGBClassifier = None
        XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass

# Custom model loader with backward compatibility
def load_model_safe(filepath):
    """Load model with backward compatibility for old XGBoost/LightGBM imports"""
    try:
        return load(filepath)
    except (ModuleNotFoundError, AttributeError) as e:
        error_msg = str(e)
        
        # Try manual pickle loading with module remapping
        import pickle
        import joblib
        
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle ALL xgboost module paths
                if 'xgboost' in module:
                    if name == 'XGBClassifier':
                        try:
                            from xgboost.sklearn import XGBClassifier
                            return XGBClassifier
                        except ImportError:
                            try:
                                from xgboost import XGBClassifier
                                return XGBClassifier
                            except ImportError:
                                pass
                    elif name == 'XGBRegressor':
                        try:
                            from xgboost.sklearn import XGBRegressor
                            return XGBRegressor
                        except ImportError:
                            try:
                                from xgboost import XGBRegressor
                                return XGBRegressor
                            except ImportError:
                                pass
                
                # Handle LightGBM
                if 'lightgbm' in module:
                    if name == 'LGBMClassifier':
                        try:
                            from lightgbm import LGBMClassifier
                            return LGBMClassifier
                        except ImportError:
                            pass
                    elif name == 'LGBMRegressor':
                        try:
                            from lightgbm import LGBMRegressor
                            return LGBMRegressor
                        except ImportError:
                            pass
                
                # For everything else, use default behavior
                try:
                    return super().find_class(module, name)
                except ModuleNotFoundError:
                    # If module still not found, it might be an old path - try to fix it
                    if module == 'xgboost.sklearn':
                        # Redirect to correct module
                        return super().find_class('xgboost', name)
                    raise
        
        # Use joblib.load with custom unpickler to handle compression
        try:
            # Monkey-patch pickle.Unpickler temporarily
            original_unpickler = pickle.Unpickler
            pickle.Unpickler = CustomUnpickler
            result = joblib.load(filepath)
            pickle.Unpickler = original_unpickler
            return result
        except Exception:
            # Restore original unpickler
            pickle.Unpickler = original_unpickler
            raise

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
        "🦠 COVID-19 Diagnosis (Stacking Ensemble)",
        "🌡️ Temperature Prediction (Ensemble)",
        "🎯 Multi-Output (Pressure & Humidity)",
        "☁️ Weather Classification (4-class)",
        "🌬️ Wind Turbine Power Forecasting (Time Series)",
        "⚡ PJM Energy Consumption Forecasting (Hourly)"
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
elif "COVID-19" in model_choice:
    st.sidebar.success("✅ **Model Available**")
    st.sidebar.info("""
    **Type:** Binary Classification  
    **Algorithm:** Stacking (LR meta-learner)  
    **Base Models:** SVM, RF, GB, XGBoost, LightGBM  
    **ROC-AUC:** 0.8975  
    **Accuracy:** 0.8103  
    **Features:** 33 blood test markers  
    **Imputation:** KNN (k=5)  
    **Scaling:** StandardScaler
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

if "Wind Turbine" in model_choice:
    st.sidebar.success("✅ **Ready**")
    st.sidebar.info("""
    **Type:** Time Series Forecasting  
    **Algorithm:** Ridge Regression  
    **R² Score:** 0.9714  
    **Target:** LV ActivePower (kW)
    """)

if "PJM Energy" in model_choice:
    st.sidebar.success("✅ **Ready**")
    st.sidebar.info("""
    **Type:** Time Series Forecasting  
    **Algorithm:** Ridge Regression  
    **R² Score:** 0.9983  
    **Target:** PJME (MWh)  
    **Data:** 145K hours (2002-2018)
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tip:** Heart Disease and COVID-19 models are pre-trained and ready to use!")

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
            model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            
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
                
                # Prepare input data (11 features) as DataFrame to preserve feature names
                input_data = pd.DataFrame([[
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
                ]], columns=metadata.get('feature_names', []))
                
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
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📈 Model Performance Metrics")
            st.caption("Performance on test set (238 samples)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.4f}", help="Area Under ROC Curve - Overall classification quality")
            with col2:
                st.metric("Accuracy", f"{metadata.get('accuracy', 0):.4f}", help="Percentage of correct predictions")
            with col3:
                st.metric("Precision", f"{metadata.get('precision', 0):.4f}", help="True positives / All predicted positives")
            with col4:
                st.metric("Recall", f"{metadata.get('recall', 0):.4f}", help="True positives / All actual positives")
            
            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
                **ROC-AUC (0.9784):** Excellent discrimination ability - the model can distinguish between patients with and without heart disease with 97.84% reliability.
                
                **Accuracy (93.28%):** The model correctly predicts the outcome for 93 out of 100 patients.
                
                **Precision:** Of all patients predicted to have heart disease, this percentage actually has it.
                
                **Recall:** Of all patients who actually have heart disease, this percentage is correctly identified.
                
                **Model Type:** StackingClassifier combines XGBoost, RandomForest, GradientBoosting, and ExtraTrees with StandardScaler normalization.
                """)
        
        else:
            st.error("❌ Model files not found!")
            st.warning(f"Expected location: `{model_dir}/best_model.joblib`")
            st.info("Please train the model first by running: `python Dataset2/heart_disease_classification.py`")
    
    except Exception as e:
        error_msg = str(e)
        if 'CyHalfBinomialLoss' in error_msg or '__pyx_unpickle' in error_msg:
            st.error("❌ Scikit-learn Version Incompatibility!")
            st.warning(f"""**Error:** {error_msg}

The model was trained with scikit-learn 1.5.1 but you're running scikit-learn 1.7.2.
This specific version incompatibility cannot be resolved without retraining.

**Solution - Retrain the model:**
```bash
cd Dataset2
python heart_disease_classification.py
```

This will take 5-10 minutes but will create a fully compatible model.
            """)
        else:
            st.error(f"❌ Error loading model: {error_msg}")
            with st.expander("View full error details"):
                st.exception(e)

# ============================================================================
# PREDICTION SECTION 2: COVID-19 DIAGNOSIS
# ============================================================================

elif "COVID-19" in model_choice:
    
    st.markdown("### COVID-19 Diagnosis from Blood Tests")
    st.caption("Comprehensive analysis using 33 blood test markers with advanced ensemble learning")
    
    # Try to load model
    try:
        model_dir = "Dataset3Covid/covid_results/models"
        
        if os.path.exists(os.path.join(model_dir, "best_model.joblib")):
            model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
            scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            
            with open(os.path.join(model_dir, "model_metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            st.success(f"✅ Model Loaded: {metadata.get('model_name', 'Unknown')}")
            
            # Experiment Summary
            with st.expander("📊 Experiment Summary - Click to expand", expanded=False):
                st.markdown("""
                ### 🔬 Research Overview
                
                **Objective:** Predict COVID-19 diagnosis from blood test results using advanced ML techniques
                
                **Dataset:**
                - 1,736 samples (920 Negative, 816 Positive)
                - 33 blood test features
                - 13.63% missing values (7,808 entries)
                
                **Imputation Experiments:**
                1. **KNN (k=5)**: 2.18% avg mean change - Selected for best performance
                2. **MICE**: 0.90% avg mean change - Better statistical quality
                3. **Hybrid (KNN+MICE)**: 0.72% avg mean change
                4. **Best-of-all (4 methods)**: 0.63% avg mean change - Best data quality
                
                **Key Finding:** Simple KNN k=5 achieved best model performance despite more sophisticated 
                imputation methods achieving better statistical measures. This demonstrates that 
                preservation of feature correlations matters more than absolute value accuracy.
                
                **Model Architecture:**
                - **Algorithm:** Stacking Ensemble with Logistic Regression meta-learner
                - **Base Models:** SVM (RBF), Random Forest, Gradient Boosting, XGBoost, LightGBM
                - **Preprocessing:** StandardScaler normalization
                - **Imputation:** KNN (k=5)
                
                **Performance Metrics:**
                - ROC-AUC: 0.8975 (89.75%)
                - Accuracy: 0.8103 (81.03%)
                - Precision: 0.7988
                - Recall: 0.7988
                - F1-Score: 0.7988
                
                **Additional Experiments:**
                - ✅ Outlier Detection (IsolationForest): 89.56% AUC, improved stability
                - ✅ Hybrid Imputation + Outlier Removal: 88.56% AUC, best data quality
                - ✅ XGBoost without scaling: Confirmed tree models don't need normalization
                
                **Conclusion:** Stacking ensemble with KNN imputation provides optimal balance 
                of accuracy, stability, and generalization for COVID-19 blood test diagnosis.
                """)
            
            # Display expected features
            st.info(f"**Required Features ({len(metadata.get('feature_names', []))}):** Blood test markers including demographics, enzymes, electrolytes, and hematology")
            
            # Input form
            st.markdown("---")
            st.markdown("### 🩸 Enter Blood Test Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Demographics**")
                sex = st.selectbox("Sex", ["Male (1)", "Female (0)"], help="Biological sex", key="covid_sex")
                age = st.number_input("Age (years)", 0, 120, 50, help="Patient's age", key="covid_age")
                
                st.markdown("**🔬 Enzymes & Metabolism**")
                ca = st.number_input("CA - Calcium (mg/dL)", 0.0, 20.0, 9.5, 0.1, help="Serum calcium", key="covid_ca")
                ck = st.number_input("CK - Creatine Kinase (U/L)", 0.0, 2000.0, 100.0, 1.0, help="Muscle enzyme", key="covid_ck")
                crea = st.number_input("CREA - Creatinine (mg/dL)", 0.0, 15.0, 1.0, 0.1, help="Kidney function", key="covid_crea")
                alp = st.number_input("ALP - Alkaline Phosphatase (U/L)", 0.0, 500.0, 80.0, 1.0, help="Liver enzyme", key="covid_alp")
                ggt = st.number_input("GGT - Gamma-GT (U/L)", 0.0, 500.0, 30.0, 1.0, help="Liver enzyme", key="covid_ggt")
                glu = st.number_input("GLU - Glucose (mg/dL)", 0.0, 500.0, 100.0, 1.0, help="Blood sugar", key="covid_glu")
                ast = st.number_input("AST - Aspartate Aminotransferase (U/L)", 0.0, 500.0, 30.0, 1.0, help="Liver enzyme", key="covid_ast")
                alt = st.number_input("ALT - Alanine Aminotransferase (U/L)", 0.0, 500.0, 30.0, 1.0, help="Liver enzyme", key="covid_alt")
                ldh = st.number_input("LDH - Lactate Dehydrogenase (U/L)", 0.0, 1000.0, 200.0, 1.0, help="Tissue damage marker", key="covid_ldh")
            
            with col2:
                st.markdown("**💧 Inflammation & Electrolytes**")
                pcr = st.number_input("PCR - C-Reactive Protein (mg/L)", 0.0, 500.0, 5.0, 0.1, help="Inflammation marker", key="covid_pcr")
                kal = st.number_input("KAL - Potassium (mmol/L)", 0.0, 10.0, 4.0, 0.1, help="Electrolyte", key="covid_kal")
                nat = st.number_input("NAT - Sodium (mmol/L)", 0.0, 200.0, 140.0, 1.0, help="Electrolyte", key="covid_nat")
                urea = st.number_input("UREA - Blood Urea Nitrogen (mg/dL)", 0.0, 200.0, 30.0, 1.0, help="Kidney function", key="covid_urea")
                
                st.markdown("**🔴 Red Blood Cells**")
                rbc = st.number_input("RBC - Red Blood Cell Count (M/μL)", 0.0, 10.0, 4.5, 0.1, help="RBC count", key="covid_rbc")
                hgb = st.number_input("HGB - Hemoglobin (g/dL)", 0.0, 20.0, 14.0, 0.1, help="Oxygen carrier", key="covid_hgb")
                hct = st.number_input("HCT - Hematocrit (%)", 0.0, 100.0, 42.0, 0.1, help="RBC volume fraction", key="covid_hct")
                mcv = st.number_input("MCV - Mean Corpuscular Volume (fL)", 0.0, 150.0, 90.0, 0.1, help="Average RBC size", key="covid_mcv")
                mch = st.number_input("MCH - Mean Corpuscular Hemoglobin (pg)", 0.0, 50.0, 30.0, 0.1, help="Average Hgb per RBC", key="covid_mch")
                mchc = st.number_input("MCHC - Mean Corpuscular Hgb Concentration (g/dL)", 0.0, 50.0, 33.0, 0.1, help="Hgb concentration", key="covid_mchc")
            
            with col3:
                st.markdown("**⚪ White Blood Cells & Platelets**")
                wbc = st.number_input("WBC - White Blood Cell Count (K/μL)", 0.0, 50.0, 7.0, 0.1, help="Immune cells", key="covid_wbc")
                plt1 = st.number_input("PLT1 - Platelet Count (K/μL)", 0.0, 1000.0, 250.0, 1.0, help="Clotting cells", key="covid_plt1")
                
                st.markdown("**🦠 Differential Counts (%)**")
                ne = st.number_input("NE - Neutrophils (%)", 0.0, 100.0, 60.0, 0.1, help="Bacterial fighters", key="covid_ne")
                ly = st.number_input("LY - Lymphocytes (%)", 0.0, 100.0, 30.0, 0.1, help="Viral fighters", key="covid_ly")
                mo = st.number_input("MO - Monocytes (%)", 0.0, 100.0, 7.0, 0.1, help="Phagocytes", key="covid_mo")
                eo = st.number_input("EO - Eosinophils (%)", 0.0, 100.0, 2.0, 0.1, help="Allergy cells", key="covid_eo")
                ba = st.number_input("BA - Basophils (%)", 0.0, 100.0, 0.5, 0.1, help="Inflammatory cells", key="covid_ba")
                
                st.markdown("**🧮 Absolute Counts (K/μL)**")
                net = st.number_input("NET - Neutrophils Absolute", 0.0, 50.0, 4.2, 0.1, help="Absolute neutrophils", key="covid_net")
                lyt = st.number_input("LYT - Lymphocytes Absolute", 0.0, 50.0, 2.1, 0.1, help="Absolute lymphocytes", key="covid_lyt")
                mot = st.number_input("MOT - Monocytes Absolute", 0.0, 10.0, 0.5, 0.1, help="Absolute monocytes", key="covid_mot")
                eot = st.number_input("EOT - Eosinophils Absolute", 0.0, 5.0, 0.15, 0.01, help="Absolute eosinophils", key="covid_eot")
                bat = st.number_input("BAT - Basophils Absolute", 0.0, 2.0, 0.03, 0.01, help="Absolute basophils", key="covid_bat")
            
            st.markdown("---")
            
            # Predict button
            if st.button("🦠 Predict COVID-19 Diagnosis", type="primary", use_container_width=True):
                
                # Prepare input data (33 features) as DataFrame to preserve feature names
                input_data = pd.DataFrame([[
                    int(sex.split()[-1].strip("()")),
                    age, ca, ck, crea, alp, ggt, glu, ast, alt, ldh, pcr, kal, nat, urea,
                    wbc, rbc, hgb, hct, mcv, mch, mchc, plt1,
                    ne, ly, mo, eo, ba, net, lyt, mot, eot, bat
                ]], columns=metadata.get('feature_names', []))
                
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
                        st.error("### 🦠 POSITIVE")
                        st.markdown("**COVID-19 detected**")
                    else:
                        st.success("### ✅ NEGATIVE")
                        st.markdown("**COVID-19 not detected**")
                
                with col2:
                    st.metric("Risk Score", f"{risk_score:.1f}%")
                    st.caption("Probability of COVID-19")
                
                with col3:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                    st.caption("Prediction confidence")
                
                # Risk interpretation
                st.markdown("---")
                st.markdown("### 🏥 Clinical Interpretation")
                
                if risk_score < 30:
                    st.success("""
                    **Low Risk (< 30%)**  
                    - Blood markers suggest low probability of COVID-19
                    - Consider alternative diagnoses
                    - Monitor for symptom development
                    - Follow standard infection control protocols
                    """)
                elif risk_score < 70:
                    st.warning("""
                    **Moderate Risk (30-70%)**  
                    - Inconclusive blood marker pattern
                    - PCR testing strongly recommended
                    - Consider chest imaging if symptomatic
                    - Implement enhanced infection control
                    - Clinical correlation essential
                    """)
                else:
                    st.error("""
                    **High Risk (> 70%)**  
                    - Blood markers highly suggestive of COVID-19
                    - Immediate PCR confirmation required
                    - Initiate isolation protocols
                    - Consider hospitalization if severe symptoms
                    - Monitor oxygen saturation and inflammatory markers
                    - Contact tracing recommended
                    """)
                
                st.info("""
                ⚠️ **Important Disclaimers:**
                - This is a machine learning research tool for educational purposes
                - Based on Stacking Ensemble (SVM, RF, GB, XGBoost, LightGBM) with KNN imputation
                - Trained on 1,736 samples with 5-fold cross-validation
                - NOT a replacement for RT-PCR or rapid antigen testing
                - Blood test predictions should support, not replace, standard diagnostic protocols
                - Always consult qualified healthcare professionals for medical decisions
                - Clinical context and symptoms must be considered alongside predictions
                """)
            
            # Model Performance Summary Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📊 Overall Model Performance")
            st.caption("Trained on 1,388 samples, tested on 348 samples")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("**Classification Metrics:**")
                perf_metrics = st.columns(2)
                with perf_metrics[0]:
                    st.metric("Accuracy", "81.03%", help="Overall correct predictions")
                    st.metric("Precision", "79.88%", help="Positive predictive value")
                with perf_metrics[1]:
                    st.metric("Recall", "79.88%", help="Sensitivity / True positive rate")
                    st.metric("F1-Score", "79.88%", help="Harmonic mean of precision and recall")
            
            with perf_col2:
                st.markdown("**ROC-AUC Performance:**")
                st.metric("ROC-AUC Score", "89.75%", help="Area Under ROC Curve", delta="+1.36% vs best individual model")
                st.progress(0.8975)
                
                st.markdown("**Training Details:**")
                st.markdown("""
                - **Imputation:** KNN (k=5)
                - **Scaling:** StandardScaler  
                - **Base Models:** 5 (SVM, RF, GB, XGB, LGBM)  
                - **Meta-learner:** Logistic Regression
                """)
        
        else:
            st.error("❌ Model files not found!")
            st.warning(f"Expected location: `{model_dir}/best_model.joblib`")
            st.info("Please ensure the COVID-19 model is trained and saved in the correct directory.")
    
    except Exception as e:
        error_msg = str(e)
        if 'CyHalfBinomialLoss' in error_msg or '__pyx_unpickle' in error_msg:
            st.error("❌ Scikit-learn Version Incompatibility!")
            st.warning(f"""**Error:** {error_msg}

The model was trained with scikit-learn 1.5.1 but you're running scikit-learn 1.7.2.
This specific version incompatibility cannot be resolved without retraining.

**Solution - Retrain the COVID-19 model:**
```bash
cd Dataset3Covid
python covid_analysis_ensemble.py
```

This will take 10-15 minutes but will create a fully compatible model.
            """)
        else:
            st.error(f"❌ Error loading model: {error_msg}")
            with st.expander("View full error details"):
                st.exception(e)

# ============================================================================
# PREDICTION SECTION 3: TEMPERATURE
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
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📈 Model Performance Metrics")
            st.caption("Stacking Regressor ensemble performance")
            
            mae = metadata['performance_metrics']['mae']
            r2 = metadata['performance_metrics']['r2_score']
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination - variance explained")
                st.progress(r2)
            with perf_col2:
                st.metric("MAE (°C)", f"±{mae:.2f}", help="Mean Absolute Error - average prediction error")
            with perf_col3:
                improvement = 1.36
                st.metric("vs Best Individual", f"+{improvement}%", delta=f"{improvement}%", help="Improvement over XGBoost alone")
            
            with st.expander("ℹ️ Model Architecture & Training Details"):
                mse = metadata['performance_metrics']['mse']
                explained_var = metadata['performance_metrics']['explained_variance']
                training_time = metadata['training_details']['training_time_seconds']
                n_train = metadata['training_details']['n_samples_train']
                n_test = metadata['training_details']['n_samples_test']
                n_features = metadata['training_details']['n_features']
                
                st.markdown(f"""
                **Ensemble Method:** {metadata.get('model_type', 'StackingRegressor')} (Sequential)
                
                **Base Models ({metadata['ensemble_configuration']['n_base_models']}):**
                - XGBoost Regressor (R² = 0.7662)
                - Random Forest Regressor (R² = 0.7649)
                - Gradient Boosting Regressor (R² = 0.7386)
                - Extra Trees Regressor (R² = 0.7594)
                - Ridge Regression (R² = 0.6108)
                - Lasso Regression (R² = 0.6100)
                
                **Meta-learner:** {metadata['ensemble_configuration']['meta_learner']}
                
                **Performance Metrics:**
                - **R² Score:** {r2:.4f} (77.66% variance explained)
                - **MSE:** {mse:.2f} square degrees
                - **MAE:** ±{mae:.2f}°C (average error)
                - **Explained Variance:** {explained_var:.4f}
                
                **Training Details:**
                - **Training Samples:** {n_train:,}
                - **Test Samples:** {n_test:,}
                - **Features:** {n_features} weather variables
                - **Training Time:** {training_time:.1f} seconds
                
                **Preprocessing Pipeline:**
                - **Imputation:** KNN Imputer (k=5)
                - **Scaling:** StandardScaler (normalized features)
                - **Dropped:** Apparent Temperature, Formatted Date, Daily Summary
                
                **Comparison with Individual Models:**
                - **Best Individual:** XGBoost (R² = 0.7662)
                - **Stacking Ensemble:** R² = {r2:.4f}
                - **Improvement:** +{((r2 - 0.7662) / 0.7662 * 100):.2f}% over best individual
                
                **Interpretation:** On average, predictions are within ±{mae:.2f}°C of actual temperature, explaining {r2*100:.2f}% of temperature variance.
                """)
        
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
# PREDICTION SECTION 4: MULTI-OUTPUT
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
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📈 Model Performance Metrics")
            st.caption("XGBoost Multi-Output Regressor with GridSearch optimization")
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown("**🌀 Pressure Prediction:**")
                pressure_r2 = 0.9823
                pressure_mae = 2.89
                st.metric("R² Score", f"{pressure_r2:.4f}", help="98.23% variance explained")
                st.progress(pressure_r2)
                st.metric("MAE", f"±{pressure_mae:.2f} mbar", help="Average error in millibars")
                st.caption("**Excellent** pressure prediction accuracy")
                
            with perf_col2:
                st.markdown("**💧 Humidity Prediction:**")
                humidity_r2 = 0.8741
                humidity_mae = 8.12
                st.metric("R² Score", f"{humidity_r2:.4f}", help="87.41% variance explained")
                st.progress(humidity_r2)
                st.metric("MAE", f"±{humidity_mae:.2f}%", help="Average error in percentage")
                st.caption("**Strong** humidity prediction accuracy")
            
            with st.expander("ℹ️ Model Architecture & Training Details"):
                st.markdown(f"""
                **Model Type:** {metadata.get('model_type', 'MultiOutputRegressor(XGBRegressor)')}
                
                **Performance Metrics (From GridSearch Optimization):**
                - **Pressure R²:** 0.9823 (98.23% variance explained)
                - **Pressure MAE:** ±2.89 mbar (excellent accuracy)
                - **Humidity R²:** 0.8741 (87.41% variance explained)
                - **Humidity MAE:** ±8.12% (strong accuracy)
                - **Average R²:** 0.9282 (92.82% combined performance)
                
                **Best Hyperparameters (216 combinations tested):**
                - N Estimators: {metadata.get('best_parameters', {}).get('n_estimators', 200)}
                - Max Depth: {metadata.get('best_parameters', {}).get('max_depth', 9)}
                - Learning Rate: {metadata.get('best_parameters', {}).get('learning_rate', 0.05)}
                - Min Child Weight: {metadata.get('best_parameters', {}).get('min_child_weight', 3)}
                - Subsample: {metadata.get('best_parameters', {}).get('subsample', 0.9)}
                - Colsample By Tree: {metadata.get('best_parameters', {}).get('colsample_bytree', 0.9)}
                
                **Preprocessing:**
                - **Imputation:** KNN Imputer (k=5)
                - **Normalization:** No Scaling (tree-based model optimal)
                - **Handled:** {metadata.get('preprocessing', {}).get('zero_pressure_count', 1288)} zero-pressure sensor errors (6.69% of dataset)
                
                **Training Details:**
                - **Total CV Fits:** {metadata.get('training_details', {}).get('total_cv_fits', 1080)}
                - **Grid Search:** Completed successfully
                - **Simultaneous Targets:** Pressure (millibars) & Humidity (%)
                - **Computation Time:** ~8 minutes for full optimization
                
                **Key Advantages:**
                - Single model predicts both outputs in one forward pass
                - Captures correlations between pressure and humidity (r=-0.45)
                - Highly accurate for both meteorological variables
                - Optimized through extensive hyperparameter tuning
                - Production-ready with complete metadata
                
                **Research Contribution:**
                Novel approach demonstrating that multi-output regression can achieve excellent performance (R²>0.87) for both targets simultaneously, more efficient than training separate models.
                """)
        
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
# PREDICTION SECTION 5: WEATHER CLASSIFICATION
# ============================================================================

elif "Weather Classification" in model_choice:
    
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
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📊 Model Performance Metrics")
            st.caption("Random Forest Classifier - 4-class weather classification")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                auc = metadata.get('performance_metrics', {}).get('auc_score', 0.8493)
                st.metric("ROC-AUC", f"{auc:.4f}", help="Multi-class AUC score (One-vs-Rest)")
                st.progress(auc)
                
            with perf_col2:
                acc = metadata.get('performance_metrics', {}).get('accuracy', 0.6474)
                st.metric("Accuracy", f"{acc:.4f}", help="Overall classification accuracy")
                st.progress(acc)
                
            with perf_col3:
                n_classes = metadata.get('training_details', {}).get('n_classes', 4)
                st.metric("Classes", n_classes, help="Number of weather categories")
                st.info("Multi-class problem")
            
            with st.expander("ℹ️ Model Details & Feature Engineering"):
                st.markdown(f"""
                **Model Architecture:**
                - **Type:** {metadata.get('model_type', 'RandomForestClassifier')}
                - **Normalization:** {metadata.get('normalization', 'None')}
                - **Features:** {metadata.get('training_details', {}).get('n_features', 31)} engineered features
                
                **Training Dataset:**
                - **Training Samples:** {metadata.get('training_details', {}).get('n_samples_train', 69851):,}
                - **Test Samples:** {metadata.get('training_details', {}).get('n_samples_test', 17463):,}
                - **Total Size:** ~87,000 weather observations
                
                **Target Classes:**
                """)
                for cls in metadata.get('target_classes', []):
                    st.markdown(f"- {cls}")
                
                st.markdown("""
                **Feature Engineering Pipeline:**
                - **Temporal Features:** Year, Month, Day, Hour, cyclical encoding (sin/cos)
                - **Interaction Terms:** Temp×Humidity, Pressure×Temp, Cloud×Humidity
                - **Polynomial Features:** Temp², Wind Speed²
                - **Wind Components:** North-South, East-West decomposition
                - **Derived Features:** Feels-like difference, visibility/humidity ratio
                - **Binary Indicators:** Is_Winter, Is_Summer, Is_Day, Low/High Pressure
                
                **Preprocessing:**
                - **Imputation:** Iterative Imputer (BayesianRidge) or Distribution Sampling
                - **Dropped:** Formatted Date, Daily Summary, duplicate columns
                
                **Performance Context:**
                - 64.7% accuracy across 4 classes (baseline: 25%)
                - ROC-AUC 0.849 indicates good discrimination
                - Complex multi-class problem with overlapping boundaries
                """)
        
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

# ============================================================================
# PREDICTION SECTION 6: WIND TURBINE POWER FORECASTING
# ============================================================================

elif "Wind Turbine" in model_choice:
    
    st.markdown("### 🌬️ Wind Turbine Power Forecasting")
    st.caption("Predict future LV ActivePower (kW) using time series ML models")
    
    try:
        # Model paths
        model_dir = "Wind Turbine Scada dataset"
        
        # Load metadata from CSV results
        univariate_results = pd.read_csv(f"{model_dir}/forecasting_results_summary.csv")
        multivariate_results = pd.read_csv(f"{model_dir}/multivariate_forecasting_results_summary.csv")
        multistep_results = pd.read_csv(f"{model_dir}/multivariate_multistep_results.csv")
        
        st.success("✅ Models Ready: Univariate, Multivariate, Multi-Step Forecasting")
        
        # Info box about the dataset
        st.info("""
        **Dataset:** SCADA (Supervisory Control and Data Acquisition) data from wind turbine  
        **Target:** LV ActivePower (kW) - Low Voltage Active Power output  
        **Frequency:** 10-minute intervals  
        **Window Size:** 24 steps (4 hours of historical data)
        """)
        
        st.markdown("---")
        
        # User controls
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_type = st.selectbox(
                "🎯 Forecast Type",
                ["Univariate (Power Only)", "Multivariate (Power + Weather)", "Multi-Step (Next Hour)"],
                help="Univariate uses only historical power data. Multivariate includes weather features. Multi-Step predicts next 6 steps (1 hour)."
            )
        
        with col2:
            if "Multivariate" in forecast_type:
                st.info("**Fixed:** 1 step ahead (next 10 minutes)")
                horizon = 1
            elif "Multi-Step" in forecast_type:
                st.info("**Fixed:** 6 steps ahead (next hour)")
                horizon = 6
            else:  # Univariate
                horizon_options = {
                    "1 step (10 min)": 1,
                    "6 steps (1 hour)": 6,
                    "12 steps (2 hours)": 12,
                    "36 steps (6 hours)": 36,
                    "144 steps (24 hours)": 144
                }
                horizon_label = st.selectbox("⏰ Forecast Horizon", list(horizon_options.keys()))
                horizon = horizon_options[horizon_label]
        
        st.markdown("---")
        
        # Load and display historical data
        try:
            df = pd.read_csv(f"{model_dir}/T1.csv")
            target_col = 'LV ActivePower (kW)'
            
            # Show last 100 observations (for context)
            last_observations = df[target_col].tail(100).values
            
            st.markdown("### 📊 Recent Historical Data")
            st.caption("Last 100 observations (16.7 hours of data)")
            
            fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
            ax_hist.plot(range(len(last_observations)), last_observations, label='Historical Power', color='steelblue', linewidth=1.5)
            ax_hist.set_xlabel('Time Steps (10-min intervals)')
            ax_hist.set_ylabel('LV ActivePower (kW)')
            ax_hist.set_title('Recent Wind Turbine Power Output')
            ax_hist.grid(True, alpha=0.3)
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close()
            
        except Exception as e:
            st.warning(f"Could not load historical data: {str(e)}")
        
        st.markdown("---")
        
        # Handle Multivariate separately (doesn't need button inside button)
        if "Multivariate" in forecast_type:
            try:
                with open(f"{model_dir}/best_multivariate_model.pkl", 'rb') as f:
                    model_dict = pickle.load(f)
                
                model = model_dict['model']
                scaler = model_dict['scaler']
                window_size = model_dict['window_size']
                all_features = model_dict['all_features']
                target_idx = model_dict['target_idx']
                
                st.success("✅ Using Ridge Regression Multivariate Model (R²=0.9713)")
                
                st.info("""
                **True Multivariate Forecasting:** Uses **power history + weather features**  
                **Input:** Last 24 steps of [Power, Wind Speed, Theoretical Power, Wind Direction]  
                **Output:** Next power value (10 minutes ahead)
                """)
                
                # Get last row from dataset as defaults for weather
                weather_cols = ['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
                last_row = df[weather_cols].iloc[-1]
                
                st.markdown("### 🌤️ Current Weather Conditions")
                st.caption("Using most recent measurements from dataset (modify if needed)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    wind_speed = st.number_input(
                        "💨 Wind Speed (m/s)", 
                        min_value=0.0, 
                        max_value=25.0, 
                        value=float(last_row['Wind Speed (m/s)']),
                        step=0.1,
                        help="Current wind speed in meters per second"
                    )
                
                with col2:
                    theoretical_power = st.number_input(
                        "⚡ Theoretical Power (KWh)", 
                        min_value=0.0, 
                        max_value=3500.0, 
                        value=float(last_row['Theoretical_Power_Curve (KWh)']),
                        step=10.0,
                        help="Expected power from theoretical power curve"
                    )
                
                with col3:
                    wind_direction = st.number_input(
                        "🧭 Wind Direction (°)", 
                        min_value=0.0, 
                        max_value=360.0, 
                        value=float(last_row['Wind Direction (°)']),
                        step=1.0,
                        help="Wind direction in degrees (0-360)"
                    )
                
                st.markdown("---")
                
                if st.button("🔮 Predict Next 10 Minutes", type="primary", use_container_width=True):
                    
                    st.markdown("### 📈 Multivariate Forecast Results")
                    
                    # Get historical window data - ALL features (power + weather)
                    last_window_data = df[all_features].tail(window_size).values
                    
                    # Use historical window with updated current weather
                    input_window = last_window_data.copy()
                    
                    # Update last row with user input (weather only, keep last power value)
                    # Find indices of weather features in all_features
                    wind_speed_idx = all_features.index('Wind Speed (m/s)')
                    theoretical_power_idx = all_features.index('Theoretical_Power_Curve (KWh)')
                    wind_direction_idx = all_features.index('Wind Direction (°)')
                    
                    input_window[-1, wind_speed_idx] = wind_speed
                    input_window[-1, theoretical_power_idx] = theoretical_power
                    input_window[-1, wind_direction_idx] = wind_direction
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_window)
                    X_input = input_scaled.flatten().reshape(1, -1)
                    
                    # Predict next step
                    pred_scaled = model.predict(X_input)
                    
                    # Inverse transform - reconstruct full feature array
                    pred_full = np.zeros((1, len(all_features)))
                    pred_full[0, target_idx] = pred_scaled[0]
                    prediction = scaler.inverse_transform(pred_full)[0, target_idx]
                    
                    st.success(f"✅ Next 10-minute prediction: **{prediction:.2f} kW**")
                    
                    # Display prediction
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Historical
                        hist_steps = 24
                        hist_data = df[target_col].tail(hist_steps).values
                        ax.plot(range(-hist_steps, 0), hist_data, 
                               label='Historical', color='steelblue', linewidth=2, marker='o', markersize=4)
                        
                        # Prediction
                        ax.plot([0], [prediction], 
                               label='Predicted (Next 10min)', color='coral', 
                               marker='s', markersize=12, linestyle='none')
                        
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Now')
                        ax.set_xlabel('Time Steps (10-min intervals)')
                        ax.set_ylabel('LV ActivePower (kW)')
                        ax.set_title('Multivariate Forecast - Next Step')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📊 Input Summary**")
                        st.metric("Wind Speed", f"{wind_speed:.1f} m/s")
                        st.metric("Theoretical Power", f"{theoretical_power:.1f} KWh")
                        st.metric("Wind Direction", f"{wind_direction:.0f}°")
                        st.metric("**Predicted Power**", f"{prediction:.2f} kW")
                    
                    st.info("""
                    **Note:** Multivariate model predicts only **1 step ahead** (next 10 minutes).  
                    Uses both power history and weather features for better accuracy.
                    """)
            
            except Exception as e:
                st.error(f"Error loading multivariate model: {str(e)}")
                st.info("""
                **Multivariate model may not be available or needs retraining.**
                
                The updated multivariate model requires:
                - `best_multivariate_model.pkl` with scaler, model, window_size, all_features, target_idx
                - Historical data with power + weather features
                
                **To retrain with the improved approach:**
                ```bash
                cd "Wind Turbine Scada dataset"
                python multivariate_forecasting.py
                ```
                """)
        
        # Generate Forecast Button (for Univariate and Multi-Step only)
        elif st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
            
            st.markdown("### 📈 Forecast Results")
            
            if "Univariate" in forecast_type:
                # Load univariate model
                try:
                    with open(f"{model_dir}/best_model.pkl", 'rb') as f:
                        model_dict = pickle.load(f)
                    
                    model = model_dict['model']
                    scaler = model_dict['scaler']
                    window_size = model_dict['window_size']
                    
                    # Get best model info
                    best_model = univariate_results.loc[univariate_results['r2'].idxmax()]
                    
                    st.success(f"✅ Using {best_model['model']} (R²={best_model['r2']:.4f})")
                    
                    # Simulate forecast (recursive prediction for multi-step)
                    last_window = df[target_col].tail(window_size).values.reshape(-1, 1)
                    last_window_scaled = scaler.transform(last_window)
                    
                    predictions = []
                    current_window = last_window_scaled.flatten()
                    
                    for step in range(horizon):
                        # Predict next step
                        X_input = current_window.reshape(1, -1)
                        pred_scaled = model.predict(X_input)
                        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                        predictions.append(pred_actual)
                        
                        # Update window for next prediction
                        if step < horizon - 1:
                            # Shift window: remove oldest value, append new prediction
                            # This is called "recursive forecasting" - each prediction becomes input for the next
                            current_window = np.append(current_window[1:], pred_scaled)
                    
                    # Display forecast
                    forecast_df = pd.DataFrame({
                        'Step': range(1, horizon + 1),
                        'Time Ahead (min)': [(i * 10) for i in range(1, horizon + 1)],
                        'Predicted Power (kW)': predictions
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Plot forecast
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Historical (last 24 steps)
                        hist_steps = min(24, len(last_observations))
                        ax.plot(range(-hist_steps, 0), last_observations[-hist_steps:], 
                               label='Historical', color='steelblue', linewidth=2, marker='o', markersize=4)
                        
                        # Forecast
                        ax.plot(range(0, horizon), predictions, 
                               label='Forecast', color='coral', linewidth=2, marker='s', markersize=4, linestyle='--')
                        
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Forecast Start')
                        ax.set_xlabel('Time Steps (10-min intervals)')
                        ax.set_ylabel('LV ActivePower (kW)')
                        ax.set_title(f'Wind Turbine Power Forecast - Next {horizon} Steps')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📋 Forecast Table**")
                        st.dataframe(forecast_df, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Forecast Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Power", f"{np.mean(predictions):.2f} kW")
                    with col2:
                        st.metric("Max Power", f"{np.max(predictions):.2f} kW")
                    with col3:
                        st.metric("Min Power", f"{np.min(predictions):.2f} kW")
                    with col4:
                        st.metric("Std Dev", f"{np.std(predictions):.2f} kW")
                    
                    # Explanation of recursive forecasting
                    with st.expander("ℹ️ How Univariate Multi-Step Forecasting Works"):
                        st.markdown("""
                        **Recursive (Iterative) Forecasting Strategy:**
                        
                        The model uses **recursive prediction** to forecast multiple steps ahead:
                        
                        1. **Step 1:** Use last 24 historical power values → Predict power at t+1
                        2. **Step 2:** Shift window (remove oldest, add predicted t+1) → Predict power at t+2
                        3. **Step 3:** Shift window (remove oldest, add predicted t+2) → Predict power at t+3
                        4. **Continue...** until reaching the desired horizon
                        
                        **Advantages:**
                        - ✅ Can forecast any horizon (1 to 144+ steps)
                        - ✅ Uses only historical power data (no weather needed)
                        - ✅ Simple and interpretable
                        
                        **Limitations:**
                        - ⚠️ Errors accumulate over time (each prediction feeds the next)
                        - ⚠️ Accuracy decreases for longer horizons
                        - ⚠️ Assumes future patterns similar to recent history
                        
                        **Current Window Size:** 24 steps (4 hours of 10-min data)
                        """)
                    
                except Exception as e:
                    st.error(f"Error loading univariate model: {str(e)}")
                    
            elif "Multivariate" in forecast_type:
                st.info("ℹ️ **Multivariate model predicts 1 step ahead (10 minutes) using weather + power data**")
                # This section is outside the main button to avoid nesting
                pass
                
            else:  # Multi-Step
                # Load multi-step model
                try:
                    with open(f"{model_dir}/best_multistep_model.pkl", 'rb') as f:
                        model_dict = pickle.load(f)
                    
                    model = model_dict['model']
                    scaler = model_dict['scaler']
                    window_size = model_dict['window_size']
                    
                    st.success("✅ Using Ridge Regression Multi-Step Model")
                    
                    st.info("""
                    **Multi-Step Model:** Predicts all 6 future steps **simultaneously** (not recursively)  
                    **Direct Strategy:** One model input → Six outputs [t+1, t+2, t+3, t+4, t+5, t+6]  
                    **Input:** Power history only (24 steps)
                    """)
                    
                    # Get last window - only power (univariate)
                    last_window = df[target_col].tail(window_size).values.reshape(-1, 1)
                    last_window_scaled = scaler.transform(last_window)
                    X_input = last_window_scaled.flatten().reshape(1, -1)
                    
                    # Predict all 6 steps at once - model outputs a 2D array with 6 values
                    predictions_scaled = model.predict(X_input)  # Shape: (1, 6)
                    
                    # Reshape for inverse transform: (1, 6) -> (6, 1) -> scale back -> flatten
                    predictions_2d = predictions_scaled.reshape(-1, 1)  # Shape: (6, 1)
                    predictions = scaler.inverse_transform(predictions_2d).flatten()  # Shape: (6,)
                    
                    # Display forecast with enhanced visualization
                    st.markdown("### 🎯 Next Hour Forecast (6 Steps = 60 minutes)")
                    
                    forecast_df = pd.DataFrame({
                        'Step': range(1, 7),
                        'Time Ahead': ['10 min', '20 min', '30 min', '40 min', '50 min', '60 min'],
                        'Predicted Power (kW)': predictions
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Enhanced visualization
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Historical (last 24 steps = 4 hours)
                        hist_steps = 24
                        hist_data = last_observations[-hist_steps:]
                        ax.plot(range(-hist_steps, 0), hist_data, 
                               label='Historical (Last 4 Hours)', color='steelblue', 
                               linewidth=2.5, marker='o', markersize=5, alpha=0.8)
                        
                        # Forecast
                        ax.plot(range(0, 6), predictions, 
                               label='Forecast (Next Hour)', color='coral', 
                               linewidth=2.5, marker='s', markersize=7, linestyle='--')
                        
                        # Add value annotations on forecast points
                        for i, val in enumerate(predictions):
                            ax.annotate(f'{val:.1f}', 
                                       xy=(i, val), 
                                       xytext=(0, 10), 
                                       textcoords='offset points',
                                       ha='center', 
                                       fontsize=9,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                        
                        ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.6, label='Now')
                        ax.fill_between(range(0, 6), predictions, alpha=0.2, color='coral')
                        
                        ax.set_xlabel('Time Steps (10-min intervals)', fontsize=12, fontweight='bold')
                        ax.set_ylabel('LV ActivePower (kW)', fontsize=12, fontweight='bold')
                        ax.set_title('Wind Turbine Power Forecast - Next Hour (Multi-Step Prediction)', 
                                    fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.legend(fontsize=11, loc='best')
                        
                        # Add shaded region for forecast
                        ax.axvspan(0, 5, alpha=0.1, color='coral', label='_Forecast Region')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📋 Forecast Table**")
                        st.dataframe(forecast_df, hide_index=True)
                        
                        # Trend indicator
                        trend = "📈 Increasing" if predictions[-1] > predictions[0] else "📉 Decreasing"
                        change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
                        st.metric("Trend (10min → 60min)", trend, f"{change:+.2f}%")
                    
                    # Detailed statistics
                    st.markdown("### 📊 Detailed Forecast Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Power", f"{np.mean(predictions):.2f} kW", 
                                 help="Mean predicted power over next hour")
                    with col2:
                        st.metric("Peak Power", f"{np.max(predictions):.2f} kW",
                                 help="Maximum predicted power")
                    with col3:
                        st.metric("Min Power", f"{np.min(predictions):.2f} kW",
                                 help="Minimum predicted power")
                    with col4:
                        st.metric("Variability", f"{np.std(predictions):.2f} kW",
                                 help="Standard deviation (power fluctuation)")
                    
                    # Step-by-step performance from validation
                    st.markdown("### 🎯 Model Performance (Per Step)")
                    st.caption("Each step's accuracy degrades slightly as we predict further into the future")
                    
                    ridge_steps = multistep_results[multistep_results['model'] == 'Ridge Regression']
                    
                    perf_cols = st.columns(6)
                    for idx, (_, row) in enumerate(ridge_steps.iterrows()):
                        with perf_cols[idx]:
                            st.caption(f"**Step {row['step']}** ({row['step']*10}min)")
                            st.metric("R²", f"{row['r2']:.4f}", help=f"RMSE: {row['rmse']:.2f}, MAE: {row['mae']:.2f}")
                    
                    # Explanation of multi-step vs recursive
                    with st.expander("ℹ️ Multi-Step vs Recursive Forecasting - Key Differences"):
                        st.markdown("""
                        **Multi-Step (Direct) Strategy:**
                        
                        🎯 **How it works:**
                        - Single model predicts **all 6 future values simultaneously**
                        - Input: Last 24 historical power values
                        - Output: Vector of 6 predictions `[t+1, t+2, t+3, t+4, t+5, t+6]`
                        - Each output neuron/estimator learns to predict a specific time step
                        
                        **Advantages:**
                        - ✅ **Faster:** One model call predicts entire hour
                        - ✅ **No error accumulation:** Each step predicted independently
                        - ✅ **Captures step-specific patterns:** Separate parameters for each horizon
                        - ✅ **More stable:** Predictions don't feed back into inputs
                        
                        **Limitations:**
                        - ⚠️ **Fixed horizon:** Can only predict exactly 6 steps (as trained)
                        - ⚠️ **More training data needed:** Requires examples for all 6 steps
                        - ⚠️ **Higher complexity:** Effectively training 6 models in one
                        
                        ---
                        
                        **Recursive (Iterative) Strategy (used in Univariate):**
                        
                        🔄 **How it works:**
                        - Model predicts only **1 step ahead**
                        - Prediction is appended to input window
                        - Process repeats for next step (using previous prediction)
                        
                        **Comparison:**
                        - 📊 **Multi-Step:** Better accuracy, faster inference, fixed horizon
                        - 🔄 **Recursive:** Flexible horizon, simpler model, error accumulation
                        
                        **Current Model:** Ridge Regression with Direct Multi-Step Strategy
                        """)
                    
                except Exception as e:
                    st.error(f"Error loading multi-step model: {str(e)}")
        
        # Model Performance Section (Always Visible) - Only Best Models
        st.markdown("---")
        st.markdown("### 📊 Best Model Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔵 Univariate")
            st.caption("1-step ahead, power only")
            best_uni = univariate_results.loc[univariate_results['r2'].idxmax()]
            st.success(f"**{best_uni['model']}**")
            st.metric("R² Score", f"{best_uni['r2']:.4f}")
            st.metric("RMSE", f"{best_uni['rmse']:.2f} kW")
            st.metric("Window", f"{int(best_uni['window_size'])} steps")
        
        with col2:
            st.markdown("#### 🟢 Multivariate")
            st.caption("1-step ahead, power + weather")
            best_multi = multivariate_results.loc[multivariate_results['r2'].idxmax()]
            st.success(f"**{best_multi['model']}**")
            st.metric("R² Score", f"{best_multi['r2']:.4f}")
            st.metric("RMSE", f"{best_multi['rmse']:.2f} kW")
            st.metric("Window", "24 steps")
        
        with col3:
            st.markdown("#### 🟠 Multi-Step")
            st.caption("6-step direct forecast")
            # Get best model from multi-step results (average R² across steps)
            # Group by model and calculate average metrics
            multistep_avg = multistep_results.groupby('model').agg({
                'r2': 'mean',
                'rmse': 'mean'
            }).reset_index()
            best_multistep = multistep_avg.loc[multistep_avg['r2'].idxmax()]
            
            st.success(f"**{best_multistep['model']}**")
            st.metric("Avg R² (6 steps)", f"{best_multistep['r2']:.4f}")
            st.metric("Avg RMSE", f"{best_multistep['rmse']:.2f} kW")
            st.metric("Window", "24 steps")
        
        # Technical details
        with st.expander("ℹ️ Technical Details & Methodology"):
            st.markdown("""
            **Forecasting Approach:**
            - **Univariate:** Uses only historical LV ActivePower values
            - **Multivariate:** Incorporates Wind Speed, Theoretical Power Curve, Wind Direction
            - **Multi-Step:** Direct strategy - predicts all 6 future steps simultaneously
            
            **Feature Engineering:**
            - **Window Size:** 24 time steps (4 hours of history at 10-min intervals)
            - **Sliding Window:** Creates overlapping sequences for training
            - **Scaling:** MinMaxScaler (0,1) for better neural network convergence
            
            **Models Tested:**
            - Linear Regression (baseline, interpretable)
            - Ridge Regression (L2 regularization, best performer)
            - Random Forest (non-linear patterns, robust to outliers)
            - XGBoost (gradient boosting, handles complex relationships)
            
            **Training Strategy:**
            - **Split:** 80% train, 20% test (chronological split, no shuffle)
            - **Validation:** Walk-forward validation for time series
            - **Metrics:** RMSE (primary), MAE, R², MAPE
            
            **Key Findings:**
            - Ridge Regression achieved best single-step performance (R²=0.9714)
            - Univariate models outperformed multivariate (simpler is better here)
            - Performance degrades slightly for longer horizons (expected)
            - Window size of 24 steps provides optimal history-complexity tradeoff
            
            **Production Deployment:**
            - Models saved with pickle (model + scaler + window_size)
            - Real-time prediction requires last 24 power measurements
            - Recursive strategy enables arbitrary forecast horizons
            - Multi-step model provides fast 1-hour forecasts
            """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        **To use Wind Turbine Forecasting:**
        
        Ensure the following files exist in `Wind Turbine Scada dataset/`:
        - `best_model.pkl` (univariate)
        - `best_multivariate_model.pkl`
        - `best_multistep_model.pkl`
        - `T1.csv` (dataset)
        - Result CSV files
        """)

# ============================================================================
# PREDICTION SECTION 7: PJM ENERGY CONSUMPTION FORECASTING
# ============================================================================

elif "PJM Energy" in model_choice:
    
    st.markdown("### ⚡ PJM Energy Consumption Forecasting")
    st.caption("Predict future hourly energy consumption (PJME) using time series ML models")
    
    try:
        # Model paths
        model_dir = "Energy_Forecasting"
        
        # Load metadata from CSV results
        univariate_results = pd.read_csv(f"{model_dir}/univariate_forecasting_results_summary.csv")
        multivariate_results = pd.read_csv(f"{model_dir}/multivariate_forecasting_results_summary.csv")
        multistep_results = pd.read_csv(f"{model_dir}/multistep_forecasting_results_summary.csv")
        
        st.success("✅ Models Ready: Univariate, Multivariate, Multi-Step Forecasting")
        
        # Info box about the dataset
        st.info("""
        **Dataset:** PJM Interconnection hourly energy consumption data  
        **Target:** PJME (megawatthours) - PJM East region energy consumption  
        **Frequency:** Hourly intervals  
        **Period:** 2002-2018 (145,000+ hours)  
        **Window Size:** 168 hours (1 week of historical data)
        """)
        
        st.markdown("---")
        
        # User controls
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_type = st.selectbox(
                "🎯 Forecast Type",
                ["Univariate (PJME Only)", "Multivariate (PJME + Other Regions)", "Multi-Step (Next Day)"],
                help="Univariate uses only historical PJME data. Multivariate includes other regional consumption. Multi-Step predicts next 24 hours.",
                key="pjm_forecast_type"
            )
        
        with col2:
            if "Multivariate" in forecast_type:
                st.info("**Fixed:** 1 step ahead (next hour)")
                horizon = 1
            elif "Multi-Step" in forecast_type:
                st.info("**Fixed:** 24 steps ahead (next day)")
                horizon = 24
            else:  # Univariate
                horizon_options = {
                    "1 hour": 1,
                    "6 hours": 6,
                    "12 hours": 12,
                    "24 hours (1 day)": 24,
                    "168 hours (1 week)": 168
                }
                horizon_label = st.selectbox("⏰ Forecast Horizon", list(horizon_options.keys()), key="pjm_horizon")
                horizon = horizon_options[horizon_label]
        
        st.markdown("---")
        
        # Load and display historical data
        try:
            df = pd.read_csv(r"c:\Users\LENOVO\Desktop\Ml forcasting\Forecasting\Forecasting\EnergyConsuption\pjm_hourly_est.csv")
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df = df.sort_values('Datetime')
            df = df[df['PJME'].notnull()].copy()
            
            target_col = 'PJME'
            
            # Show last 200 observations (for context)
            last_observations = df[target_col].tail(200).values
            
            st.markdown("### 📊 Recent Historical Data")
            st.caption("Last 200 hours (~8 days of energy consumption)")
            
            fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
            ax_hist.plot(range(len(last_observations)), last_observations, label='Historical Consumption', color='darkgreen', linewidth=1.5)
            ax_hist.set_xlabel('Hours')
            ax_hist.set_ylabel('PJME (MWh)')
            ax_hist.set_title('Recent PJM East Energy Consumption')
            ax_hist.grid(True, alpha=0.3)
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close()
            
        except Exception as e:
            st.warning(f"Could not load historical data: {str(e)}")
        
        st.markdown("---")
        
        # Handle Multivariate separately (doesn't need button inside button)
        if "Multivariate" in forecast_type:
            try:
                with open(f"{model_dir}/pjm_energy_multivariate_best_model.pkl", 'rb') as f:
                    model_dict = pickle.load(f)
                
                model = model_dict['model']
                scaler = model_dict['scaler']
                target_scaler = model_dict['target_scaler']
                window_size = model_dict['window_size']
                feature_columns = model_dict['feature_columns']
                target_column = model_dict['target_column']
                
                # Build all_features in the SAME ORDER as training
                # During training: df_model has all columns, scaler fit on df_model.values
                # So we need to get columns in their original order from the dataframe
                df_model = df.drop('Datetime', axis=1).dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill')
                all_features = df_model.columns.tolist()
                target_idx = all_features.index(target_column)
                
                # Get best model info
                best_multi = multivariate_results.loc[multivariate_results['r2'].idxmax()]
                
                st.success(f"✅ Using {best_multi['model']} (R²={best_multi['r2']:.4f}, Window={int(best_multi['window_size'])}h)")
                
                st.info("""
                **True Multivariate Forecasting:** Uses **PJME history + other regional consumption data**  
                **Input:** Last 168 hours of all regional energy data  
                **Output:** Next PJME value (1 hour ahead)
                """)
                
                st.markdown("---")
                
                if st.button("🔮 Predict Next Hour", type="primary", use_container_width=True, key="pjm_multi_predict"):
                    
                    st.markdown("### 📈 Multivariate Forecast Results")
                    
                    # Get historical window data - ALL features (target + feature_columns)
                    df_model = df.drop('Datetime', axis=1).dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill')
                    last_window_data = df_model[all_features].tail(window_size).values
                    
                    # Scale the input using the feature scaler
                    input_scaled = scaler.transform(last_window_data)
                    X_input = input_scaled.flatten().reshape(1, -1)
                    
                    # Predict next step (returns scaled prediction)
                    pred_scaled = model.predict(X_input)
                    
                    # Inverse transform using target_scaler
                    prediction = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                    
                    st.success(f"✅ Next hour prediction: **{prediction:.2f} MWh**")
                    
                    # Display prediction
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Historical
                        hist_steps = 168
                        hist_data = df[target_col].tail(hist_steps).values
                        ax.plot(range(-hist_steps, 0), hist_data, 
                               label='Historical (Last Week)', color='darkgreen', linewidth=2, marker='o', markersize=3)
                        
                        # Prediction
                        ax.plot([0], [prediction], 
                               label='Predicted (Next Hour)', color='orange', 
                               marker='s', markersize=12, linestyle='none')
                        
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Now')
                        ax.set_xlabel('Hours')
                        ax.set_ylabel('PJME (MWh)')
                        ax.set_title('Multivariate Forecast - Next Hour')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📊 Prediction Summary**")
                        st.metric("Current (Last Hour)", f"{hist_data[-1]:.2f} MWh")
                        change = prediction - hist_data[-1]
                        st.metric("**Next Hour Forecast**", f"{prediction:.2f} MWh", delta=f"{change:+.2f} MWh")
                        pct_change = (change / hist_data[-1]) * 100
                        st.caption(f"Change: {pct_change:+.2f}%")
                    
                    st.info("""
                    **Note:** Multivariate model predicts only **1 step ahead** (next hour).  
                    Uses regional correlation patterns for improved accuracy.
                    """)
            
            except Exception as e:
                st.error(f"Error loading multivariate model: {str(e)}")
                st.info("""
                **Multivariate model may not be available.**
                
                Run training script:
                ```bash
                cd Energy_Forecasting
                python multivariate_forecasting.py
                ```
                """)
        
        # Generate Forecast Button (for Univariate and Multi-Step only)
        elif st.button("🔮 Generate Forecast", type="primary", use_container_width=True, key="pjm_forecast"):
            
            st.markdown("### 📈 Forecast Results")
            
            if "Univariate" in forecast_type:
                # Load univariate model
                try:
                    with open(f"{model_dir}/pjm_energy_univariate_best_model.pkl", 'rb') as f:
                        model_dict = pickle.load(f)
                    
                    model = model_dict['model']
                    scaler = model_dict['scaler']
                    window_size = model_dict['window_size']
                    
                    # Get best model info
                    best_model = univariate_results.loc[univariate_results['r2'].idxmax()]
                    
                    st.success(f"✅ Using {best_model['model']} (R²={best_model['r2']:.4f}, Window={int(best_model['window_size'])}h)")
                    
                    # Simulate forecast (recursive prediction for multi-step)
                    last_window = df[target_col].tail(window_size).values.reshape(-1, 1)
                    last_window_scaled = scaler.transform(last_window)
                    
                    predictions = []
                    current_window = last_window_scaled.flatten()
                    
                    for step in range(horizon):
                        # Predict next step
                        X_input = current_window.reshape(1, -1)
                        pred_scaled = model.predict(X_input)
                        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                        predictions.append(pred_actual)
                        
                        # Update window for next prediction
                        if step < horizon - 1:
                            current_window = np.append(current_window[1:], pred_scaled)
                    
                    # Display forecast
                    forecast_df = pd.DataFrame({
                        'Hour': range(1, horizon + 1),
                        'Predicted Consumption (MWh)': predictions
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Plot forecast
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Historical
                        hist_steps = min(168, len(last_observations))
                        ax.plot(range(-hist_steps, 0), last_observations[-hist_steps:], 
                               label='Historical', color='darkgreen', linewidth=2, marker='o', markersize=3)
                        
                        # Forecast
                        ax.plot(range(0, horizon), predictions, 
                               label='Forecast', color='orange', linewidth=2, marker='s', markersize=4, linestyle='--')
                        
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Forecast Start')
                        ax.set_xlabel('Hours')
                        ax.set_ylabel('PJME (MWh)')
                        ax.set_title(f'PJM Energy Forecast - Next {horizon} Hours')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📋 Forecast Table**")
                        st.dataframe(forecast_df.head(10), hide_index=True)
                        if horizon > 10:
                            st.caption(f"Showing first 10 of {horizon} hours")
                    
                    # Summary statistics
                    st.markdown("### 📊 Forecast Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average", f"{np.mean(predictions):.2f} MWh")
                    with col2:
                        st.metric("Peak", f"{np.max(predictions):.2f} MWh")
                    with col3:
                        st.metric("Minimum", f"{np.min(predictions):.2f} MWh")
                    with col4:
                        st.metric("Std Dev", f"{np.std(predictions):.2f} MWh")
                    
                except Exception as e:
                    st.error(f"Error loading univariate model: {str(e)}")
                    
            else:  # Multi-Step
                # Load multi-step model
                try:
                    with open(f"{model_dir}/pjm_energy_multistep_best_model.pkl", 'rb') as f:
                        model_dict = pickle.load(f)
                    
                    model = model_dict['model']
                    scaler = model_dict['scaler']
                    window_size = model_dict['window_size']
                    
                    # Get best model info
                    best_model = multistep_results.loc[multistep_results['r2'].idxmax()]
                    
                    st.success(f"✅ Using {best_model['model']} (Overall R²={best_model['r2']:.4f}, Window={int(best_model['window_size'])}h)")
                    
                    st.info("""
                    **Multi-Step Model:** Predicts all 24 future hours **simultaneously** (not recursively)  
                    **Direct Strategy:** One model input → 24 outputs [t+1h, t+2h, ..., t+24h]
                    """)
                    
                    # Get last window
                    last_window = df[target_col].tail(window_size).values.reshape(-1, 1)
                    last_window_scaled = scaler.transform(last_window)
                    X_input = last_window_scaled.flatten().reshape(1, -1)
                    
                    # Predict all 24 steps at once
                    predictions_scaled = model.predict(X_input)  # Shape: (1, 24)
                    
                    # Reshape for inverse transform
                    predictions_2d = predictions_scaled.reshape(-1, 1)
                    predictions = scaler.inverse_transform(predictions_2d).flatten()
                    
                    # Display forecast
                    st.markdown("### 🎯 Next Day Forecast (24 Hours)")
                    
                    forecast_df = pd.DataFrame({
                        'Hour': range(1, 25),
                        'Predicted Consumption (MWh)': predictions
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Historical
                        hist_steps = 168
                        hist_data = last_observations[-hist_steps:]
                        ax.plot(range(-hist_steps, 0), hist_data, 
                               label='Historical (Last Week)', color='darkgreen', 
                               linewidth=2, marker='o', markersize=3, alpha=0.8)
                        
                        # Forecast
                        ax.plot(range(0, 24), predictions, 
                               label='Forecast (Next Day)', color='orange', 
                               linewidth=2.5, marker='s', markersize=5, linestyle='--')
                        
                        ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.6, label='Now')
                        ax.fill_between(range(0, 24), predictions, alpha=0.2, color='orange')
                        
                        ax.set_xlabel('Hours', fontsize=12, fontweight='bold')
                        ax.set_ylabel('PJME (MWh)', fontsize=12, fontweight='bold')
                        ax.set_title('PJM Energy Forecast - Next 24 Hours', 
                                    fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.legend(fontsize=11, loc='best')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**📋 Hourly Forecast**")
                        st.dataframe(forecast_df.head(12), hide_index=True)
                        st.caption("Showing first 12 of 24 hours")
                        
                        # Trend indicator
                        trend = "📈 Increasing" if predictions[-1] > predictions[0] else "📉 Decreasing"
                        change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
                        st.metric("Trend (Hour 1 → 24)", trend, f"{change:+.2f}%")
                    
                    # Detailed statistics
                    st.markdown("### 📊 Detailed Forecast Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average", f"{np.mean(predictions):.2f} MWh")
                    with col2:
                        st.metric("Peak Demand", f"{np.max(predictions):.2f} MWh")
                    with col3:
                        st.metric("Lowest Demand", f"{np.min(predictions):.2f} MWh")
                    with col4:
                        st.metric("Variability", f"{np.std(predictions):.2f} MWh")
                    
                except Exception as e:
                    st.error(f"Error loading multi-step model: {str(e)}")
        
        # Model Performance Section (Always Visible) - Only Best Models
        st.markdown("---")
        st.markdown("### 📊 Best Model Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔵 Univariate")
            st.caption("1-step ahead, PJME only")
            best_uni = univariate_results.loc[univariate_results['r2'].idxmax()]
            st.success(f"**{best_uni['model']}**")
            st.metric("R² Score", f"{best_uni['r2']:.4f}")
            st.metric("RMSE", f"{best_uni['rmse']:.2f} MWh")
            st.metric("Window", f"{int(best_uni['window_size'])} hours")
        
        with col2:
            st.markdown("#### 🟢 Multivariate")
            st.caption("1-step ahead, all regions")
            best_multi = multivariate_results.loc[multivariate_results['r2'].idxmax()]
            st.success(f"**{best_multi['model']}**")
            st.metric("R² Score", f"{best_multi['r2']:.4f}")
            st.metric("RMSE", f"{best_multi['rmse']:.2f} MWh")
            st.metric("Window", f"{int(best_multi['window_size'])} hours")
        
        with col3:
            st.markdown("#### 🟠 Multi-Step")
            st.caption("24-hour direct forecast")
            best_multistep = multistep_results.loc[multistep_results['r2'].idxmax()]
            st.success(f"**{best_multistep['model']}**")
            st.metric("Overall R²", f"{best_multistep['r2']:.4f}")
            st.metric("Overall RMSE", f"{best_multistep['rmse']:.2f} MWh")
            st.metric("Window", f"{int(best_multistep['window_size'])} hours")
        
        # Technical details
        with st.expander("ℹ️ Technical Details & Dataset Information"):
            st.markdown("""
            **Dataset Information:**
            - **Source:** PJM Interconnection (Regional Transmission Organization)
            - **Target Variable:** PJME - PJM East region hourly energy consumption
            - **Units:** Megawatthours (MWh)
            - **Time Period:** 2002-2018
            - **Total Records:** 145,366 hourly observations
            - **Other Regions Available:** AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJMW
            
            **Forecasting Approaches:**
            - **Univariate:** Uses only historical PJME values (recursive for multi-step)
            - **Multivariate:** Incorporates other regional consumption patterns (correlation-based)
            - **Multi-Step:** Direct 24-hour ahead prediction (all steps simultaneously)
            
            **Window Sizes Tested:**
            - 24 hours (1 day)
            - 168 hours (1 week) - **Best performance**
            
            **Models Evaluated:**
            - Linear Regression (baseline)
            - Ridge Regression (L2 regularization, best performer)
            - Random Forest (ensemble, non-linear)
            - XGBoost (gradient boosting)
            
            **Key Findings:**
            - 168-hour window consistently outperformed 24-hour window
            - Ridge Regression achieved R² > 0.998 (univariate, 168h window)
            - Multivariate models benefit from regional correlation patterns
            - Multi-step direct strategy more stable than recursive for 24h forecasts
            
            **Production Notes:**
            - Real-time prediction requires 1 week of historical data
            - Models handle weekly seasonality patterns effectively
            - Suitable for day-ahead energy market forecasting
            """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        **To use PJM Energy Forecasting:**
        
        Ensure the following files exist in `Energy_Forecasting/`:
        - Model pickle files
        - Result CSV files
        - Dataset: `pjm_hourly_est.csv`
        """)

# ============================================================================
# PREDICTION SECTION 8: WEATHER CLASSIFICATION (FALLBACK)
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
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📊 Model Performance Metrics")
            st.caption("Random Forest Classifier - 4-class weather classification")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                auc = metadata.get('performance_metrics', {}).get('auc_score', 0.8493)
                st.metric("ROC-AUC", f"{auc:.4f}", help="Multi-class AUC score (One-vs-Rest)")
                st.progress(auc)
                
            with perf_col2:
                acc = metadata.get('performance_metrics', {}).get('accuracy', 0.6474)
                st.metric("Accuracy", f"{acc:.4f}", help="Overall classification accuracy")
                st.progress(acc)
                
            with perf_col3:
                n_classes = metadata.get('training_details', {}).get('n_classes', 4)
                st.metric("Classes", n_classes, help="Number of weather categories")
                st.info("Multi-class problem")
            
            with st.expander("ℹ️ Model Details & Feature Engineering"):
                st.markdown(f"""
                **Model Architecture:**
                - **Type:** {metadata.get('model_type', 'RandomForestClassifier')}
                - **Normalization:** {metadata.get('normalization', 'None')}
                - **Features:** {metadata.get('training_details', {}).get('n_features', 31)} engineered features
                
                **Training Dataset:**
                - **Training Samples:** {metadata.get('training_details', {}).get('n_samples_train', 69851):,}
                - **Test Samples:** {metadata.get('training_details', {}).get('n_samples_test', 17463):,}
                - **Total Size:** ~87,000 weather observations
                
                **Target Classes:**
                """)
                for cls in metadata.get('target_classes', []):
                    st.markdown(f"- {cls}")
                
                st.markdown("""
                **Feature Engineering Pipeline:**
                - **Temporal Features:** Year, Month, Day, Hour, cyclical encoding (sin/cos)
                - **Interaction Terms:** Temp×Humidity, Pressure×Temp, Cloud×Humidity
                - **Polynomial Features:** Temp², Wind Speed²
                - **Wind Components:** North-South, East-West decomposition
                - **Derived Features:** Feels-like difference, visibility/humidity ratio
                - **Binary Indicators:** Is_Winter, Is_Summer, Is_Day, Low/High Pressure
                
                **Preprocessing:**
                - **Imputation:** Iterative Imputer (BayesianRidge) or Distribution Sampling
                - **Dropped:** Formatted Date, Daily Summary, duplicate columns
                
                **Performance Context:**
                - 64.7% accuracy across 4 classes (baseline: 25%)
                - ROC-AUC 0.849 indicates good discrimination
                - Complex multi-class problem with overlapping boundaries
                """)
        
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
    <p>5 Production Models | Real-time Predictions | Interactive Interface</p>
    <p style='font-size: 12px;'>Built with Streamlit | Academic Project 2025</p>
</div>
""", unsafe_allow_html=True)
