"""
Simple ML Predictions Dashboard
Focus: Interactive predictions for 10 trained models
Enhanced with: Caching, Demo Data, Prediction History, Export, Batch Upload
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
from datetime import datetime
import io

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

# ============================================================================
# PYTORCH DNN MODEL SUPPORT
# ============================================================================
# Import PyTorch model utilities for DNN predictions
try:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    from pytorch_model_utils import FlexMLP, ResidualMLP, ResidualBlock, PyTorchMLPWrapper
    from novel_architectures import ScratchNovelNet, ScratchSqueezeExciteNet, ScratchMultiScaleNet
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

@st.cache_resource
def load_dnn_heart_model():
    """Load PyTorch DNN Heart Disease model (cached)"""
    try:
        return joblib.load("pytorch_results/models/heart_classification_pytorch_model.pkl")
    except:
        return None

@st.cache_resource
def load_dnn_temperature_model():
    """Load PyTorch DNN Temperature model (cached)"""
    try:
        return joblib.load("pytorch_results/models/temperature_regression_pytorch_model.pkl")
    except:
        return None

@st.cache_resource
def load_dnn_wind_model():
    """Load PyTorch DNN Wind Turbine model (cached)"""
    try:
        return joblib.load("pytorch_results/models/wind_forecasting_pytorch_model.pkl")
    except:
        return None

@st.cache_resource
def load_novel_heart_model():
    try:
        return joblib.load("pytorch_results/novel_models/heart_classification_novel_best.pkl")
    except:
        return None

@st.cache_resource
def load_novel_temperature_model():
    try:
        return joblib.load("pytorch_results/novel_models/temperature_regression_novel_best.pkl")
    except:
        return None

@st.cache_resource
def load_novel_wind_model():
    try:
        return joblib.load("pytorch_results/novel_models/wind_forecasting_novel_best.pkl")
    except:
        return None

@st.cache_resource
def load_novel_wine_model():
    try:
        return joblib.load("pytorch_results/novel_models/anomaly_wine_novel_best.pkl")
    except:
        return None


@st.cache_data
def load_pytorch_best_models_overview():
    """Load the compact best-model summary generated from the saved CSV artifacts."""
    path = "scratch_architectures/best_models_overview.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def show_pytorch_architecture_summary(task_key, section_title):
    """Show a compact architecture + parameter summary for the given task."""
    overview = load_pytorch_best_models_overview()
    if overview is None:
        return

    row = overview[overview["task"] == task_key]
    if row.empty:
        return

    row = row.iloc[0]
    summary_df = pd.DataFrame([
        {
            "Best DNN": row.get("best_dnn_model", "n/a"),
            "DNN Metric": f"{row.get('metric', 'n/a')} = {float(row.get('best_dnn_value', np.nan)):.4f}" if pd.notna(row.get("best_dnn_value", np.nan)) else "n/a",
            "ML Baseline": f"{row.get('ml_model', 'n/a')} = {float(row.get('ml_value', np.nan)):.4f}" if pd.notna(row.get("ml_value", np.nan)) else "n/a",
            "Winner": row.get("winner", "n/a"),
            "Scratch Best": row.get("scratch_best_config", "n/a"),
            "Scratch Value": f"{float(row.get('scratch_best_value', np.nan)):.4f}" if pd.notna(row.get("scratch_best_value", np.nan)) else "n/a",
        }
    ])

    with st.expander(section_title, expanded=False):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        notes = row.get("preprocessing_notes", "n/a")
        if pd.notna(notes):
            st.caption(f"Preprocessing notes: {notes}")

def show_novel_architecture_summary(pkg, section_title="Novel Architecture details"):
    if not pkg:
        return
    st.markdown(f"**Architecture:** `{pkg['architecture']}` (Class: `{pkg['model_meta']['arch_class']}`)")
    st.markdown(f"**Test Metric:** `{pkg['test_metric']:.4f}`")
    st.markdown(f"**Validation Metric:** `{pkg['val_metric']:.4f}`")
    st.markdown(f"**Training Time:** `{pkg['time_s']:.1f}s` ({pkg['epochs']} epochs)")
    with st.expander(section_title, expanded=False):
        st.json(pkg['model_meta'])

def predict_with_novel_model(pkg, X_np):
    """
    Given the loaded pkg (dict from joblib) and a numpy array X_np of shape (n_samples, n_features),
    instantiate the model, load weights, and return predictions.
    """
    if not pkg or not PYTORCH_AVAILABLE:
        return None
    meta = pkg['model_meta']
    cls_name = meta['arch_class']
    
    # Retrieve the class
    import novel_architectures
    if not hasattr(novel_architectures, cls_name):
        return None
    ArchClass = getattr(novel_architectures, cls_name)
    
    # Initialize
    model = ArchClass(
        input_dim=meta['input_dim'],
        output_dim=meta['output_dim'],
        task_type=meta['task_type'],
        **meta['arch_kwargs']
    )
    
    # Load weights
    model.load_state_dict(pkg['model_state_dict'])
    model.eval()
    
    # Predict
    X_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_t)
        
    if meta['task_type'] == 'classification':
        # out is probabilities after sigmoid, shape (n, 1)
        raw = out.numpy()
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw.ravel()  # flatten to (n,)
        # Build 2-column probability array [P(class=0), P(class=1)]
        if raw.ndim == 1:
            proba = np.column_stack([1 - raw, raw])  # shape (n, 2)
        else:
            proba = raw
        preds = (proba[:, 1] > 0.5).astype(int)
        return preds, proba
    elif meta['task_type'] == 'multiclass':
        # out is logits, apply softmax
        proba = torch.softmax(out, dim=1).numpy()
        preds = proba.argmax(axis=1)
        return preds, proba
    else: # regression
        return out.numpy().ravel(), None


def dnn_wind_predict(wrapper, raw_window):
    """
    Custom prediction for wind DNN model.
    The scaler was fit on 1D power values, but the model takes 24-feature window.
    raw_window: array of shape (24,) with raw power values.
    Returns: predicted power value (float).
    """
    scaled_vals = wrapper.scaler.transform(raw_window.reshape(-1, 1)).flatten()
    X_t = torch.tensor(scaled_vals.reshape(1, -1), dtype=torch.float32)
    wrapper.model.eval()
    with torch.no_grad():
        pred_scaled = wrapper.model(X_t).squeeze().numpy()
    return float(wrapper.scaler.inverse_transform([[float(pred_scaled)]])[0, 0])

# ============================================================================
# CACHED MODEL LOADING FUNCTIONS (5-10x faster predictions)
# ============================================================================
@st.cache_resource
def load_heart_model():
    """Load Heart Disease classification model (cached)"""
    try:
        return load_model_safe("best_heart_model.joblib")
    except:
        return None

@st.cache_resource
def load_covid_model():
    """Load COVID-19 model (cached)"""
    try:
        return load_model_safe("covid_stacking_classifier.joblib")
    except:
        return None

@st.cache_resource
def load_temperature_model():
    """Load Temperature prediction model (cached)"""
    try:
        with open("temperature_model_package.pkl", 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_multioutput_model():
    """Load Multi-output model (cached)"""
    try:
        with open("multioutput_model_package.pkl", 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_anomaly_employee_model():
    """Load Employee anomaly detection model (cached)"""
    try:
        with open("Anomaly detection/EmpolyeeClassification/best_anomaly_model.pkl", 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_anomaly_heart_model():
    """Load Heart anomaly detection model (cached)"""
    try:
        with open("Anomaly detection/HeartDesease/best_anomaly_model.pkl", 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_anomaly_wine_model():
    """Load Wine anomaly detection model (cached)"""
    try:
        with open("Anomaly detection/WineType/separate_class_evaluation_results/best_anomaly_model.pkl", 'rb') as f:
            return pickle.load(f)
    except:
        return None

# ============================================================================
# DEMO DATA FOR QUICK TESTING
# ============================================================================
DEMO_DATA = {
    'heart_classification': {
        'age': 55, 'sex': 1, 'chest_pain': 2, 'resting_bp': 140, 'cholesterol': 250,
        'fasting_bs': 0, 'resting_ecg': 1, 'max_hr': 150, 'exercise_angina': 0,
        'oldpeak': 1.5, 'st_slope': 1
    },
    'covid': {
        'sex': 1, 'age': 55, 'ca': 9.2, 'ck': 120.0, 'crea': 1.1, 'alp': 85.0, 'ggt': 35.0,
        'glu': 110.0, 'ast': 35.0, 'alt': 40.0, 'ldh': 220.0, 'pcr': 15.0, 'kal': 4.2,
        'nat': 138.0, 'urea': 35.0, 'wbc': 6.5, 'rbc': 4.3, 'hgb': 13.5, 'hct': 40.0,
        'mcv': 88.0, 'mch': 29.0, 'mchc': 32.5, 'plt1': 230.0, 'ne': 65.0, 'ly': 25.0,
        'mo': 6.0, 'eo': 2.5, 'ba': 0.5, 'net': 4.5, 'lyt': 1.8, 'mot': 0.4, 'eot': 0.18, 'bat': 0.04
    },
    'temperature': {
        'humidity': 65.0, 'wind_speed': 12.0, 'wind_bearing': 180.0, 'visibility': 10.0,
        'pressure': 1015.0, 'month': 6, 'hour': 14
    },
    'anomaly_employee': {
        'age': 35, 'gender': 'Male', 'years_at_company': 5, 'job_role': 'Engineering',
        'monthly_income': 5000, 'work_life_balance': 3, 'job_satisfaction': 3,
        'performance_rating': 3, 'num_promotions': 1, 'overtime': 'Yes',
        'distance_from_home': 10, 'education_level': 'Bachelor', 'marital_status': 'Married',
        'num_dependents': 2, 'job_level': 2, 'company_size': 'Large',
        'company_tenure': 5, 'remote_work': 'No', 'leadership_opps': 'Yes',
        'innovation_opps': 'Yes', 'company_reputation': 4, 'employee_recognition': 3
    },
    'anomaly_heart': {
        'age': 60, 'sex': 1, 'chest_pain': 3, 'resting_bp': 145, 'cholesterol': 280,
        'fasting_bs': 1, 'resting_ecg': 1, 'max_hr': 130, 'exercise_angina': 1,
        'oldpeak': 2.5, 'st_slope': 2
    },
    'anomaly_wine': {
        'fixed_acidity': 7.4, 'volatile_acidity': 0.7, 'citric_acid': 0.0,
        'residual_sugar': 1.9, 'chlorides': 0.076, 'free_so2': 11.0,
        'total_so2': 34.0, 'density': 0.9978, 'ph': 3.51, 'sulphates': 0.56,
        'alcohol': 9.4, 'quality': 5
    }
}

# Page configuration
st.set_page_config(
    page_title="ML Predictions",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
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
    .success-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .danger-result {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .demo-button {
        background-color: #ff9800;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .history-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION (Prediction History)
# ============================================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = {}

def add_to_history(model_name, inputs, prediction, probability=None):
    """Add a prediction to the session history"""
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'inputs': inputs,
        'prediction': prediction,
        'probability': probability
    }
    st.session_state.prediction_history.insert(0, record)  # Add to beginning
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[:50]

def export_history_to_csv():
    """Convert prediction history to CSV for download"""
    if not st.session_state.prediction_history:
        return None
    
    records = []
    for item in st.session_state.prediction_history:
        record = {
            'Timestamp': item['timestamp'],
            'Model': item['model'],
            'Prediction': item['prediction'],
            'Probability': item.get('probability', 'N/A')
        }
        # Flatten inputs
        if isinstance(item['inputs'], dict):
            for k, v in item['inputs'].items():
                record[f'Input_{k}'] = v
        records.append(record)
    
    return pd.DataFrame(records)

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
        "⚡ PJM Energy Consumption Forecasting (Hourly)",
        "🔍 Anomaly Detection: Employee Attrition",
        "🫀 Anomaly Detection: Heart Disease",
        "🍷 Anomaly Detection: Wine Type",
        "🧠 ML vs DNN Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")

# Display different info based on selection
if "Heart Disease Classification" in model_choice:
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
    st.sidebar.success("✅ **Model Available**")
    st.sidebar.info("""
    **Type:** Regression  
    **Algorithm:** Stacking Ensemble  
    **R² Score:** 0.7889  
    **Features:** 8 weather variables
    """)
elif "Multi-Output" in model_choice:
    st.sidebar.warning("✅ **Model Available**")
    st.sidebar.info("""
    **Type:** Multi-Output Regression  
    **Algorithm:** XGBoost  
    **Avg R²:** 0.9282  
    **Outputs:** Pressure & Humidity
    """)
elif "ML vs DNN Comparison" in model_choice:
    st.sidebar.success("✅ **Comparison Available**")
    st.sidebar.info("""
    **Type:** Benchmark Comparison  
    **Scope:** Same 10 dashboard tasks  
    **Models:** Existing ML vs DNN (MLP)  
    **Source:** `dnn_results/ml_vs_dnn_comparison.csv`
    """)
else:  # Weather Classification
    st.sidebar.warning("✅ **Model Available**")
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

if "Employee Attrition" in model_choice:
    st.sidebar.success("✅ **Results Available**")
    st.sidebar.info("""
    **Type:** Anomaly Detection  
    **Best Model:** Elliptic Envelope  
    **Accuracy:** 0.4896  
    **F1-Score:** 0.3382  
    **Problem:** Employee turnover detection
    """)

if "Anomaly Detection: Heart" in model_choice:
    st.sidebar.success("✅ **Results Available**")
    st.sidebar.info("""
    **Type:** Anomaly Detection  
    **Best Model:** Elliptic Envelope  
    **Balanced Acc:** 0.664  
    **TPR:** 0.4167  
    **Problem:** Heart disease as anomaly
    """)

if "Wine Type" in model_choice:
    st.sidebar.success("✅ **Results Available**")
    st.sidebar.info("""
    **Type:** Anomaly Detection  
    **Best Model:** Elliptic Envelope  
    **F1-Score:** 0.9188  
    **TPR:** 0.9312  
    **Problem:** Red wine as anomaly
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Tip:** Heart Disease and COVID-19 models are pre-trained and ready to use!")

# ============================================================================
# SIDEBAR: PREDICTION HISTORY & EXPORT
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 Prediction History")

if st.session_state.prediction_history:
    st.sidebar.success(f"**{len(st.session_state.prediction_history)}** predictions recorded")
    
    # Export button
    history_df = export_history_to_csv()
    if history_df is not None:
        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="📥 Export History (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()
    
    # Show recent predictions
    with st.sidebar.expander("📋 Recent Predictions"):
        for i, record in enumerate(st.session_state.prediction_history[:5]):
            st.markdown(f"""
            **{record['timestamp']}**  
            Model: {record['model']}  
            Result: {record['prediction']}
            """)
            if i < 4:
                st.markdown("---")
else:
    st.sidebar.info("No predictions yet. Make a prediction to start tracking!")

# Main content area
st.markdown(f"## {model_choice}")

# ============================================================================
# COMPARISON SECTION: ML VS DNN
# ============================================================================
if "ML vs DNN Comparison" in model_choice:
    comparison_path = "dnn_results/ml_vs_dnn_comparison.csv"
    dnn_details_path = "dnn_results/dnn_task_results.csv"

    st.markdown("### 🧠 ML vs DNN Benchmark (Same Dashboard Tasks)")

    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tasks Compared", int(len(comparison_df)))
        with col2:
            st.metric("DNN Wins", int((comparison_df['winner'] == 'DNN').sum()))
        with col3:
            st.metric("ML Wins", int((comparison_df['winner'] == 'ML').sum()))

        display_df = comparison_df.copy()
        for col in ['ml_value', 'dnn_value', 'delta_dnn_minus_ml']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True)

        winner_counts = comparison_df['winner'].value_counts().rename_axis('winner').reset_index(name='count')
        st.bar_chart(winner_counts.set_index('winner'))

        with st.expander("📋 DNN Task Details"):
            if os.path.exists(dnn_details_path):
                dnn_df = pd.read_csv(dnn_details_path)
                st.dataframe(dnn_df, use_container_width=True)
            else:
                st.info("DNN details file not found. Run `dnn_dashboard_benchmark.py` first.")

        csv_buffer = io.StringIO()
        comparison_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download ML vs DNN Comparison",
            data=csv_buffer.getvalue(),
            file_name="ml_vs_dnn_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ Comparison file not found")
        st.info("Run `dnn_dashboard_benchmark.py` to generate `dnn_results/ml_vs_dnn_comparison.csv`.")

# ============================================================================
# PREDICTION SECTION 1: HEART DISEASE
# ============================================================================

if "Heart Disease Classification" in model_choice:
    
    st.markdown("### Enter Patient Information")
    
    # Try to load model (using cached function)
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
            
            # Demo and Batch Upload buttons
            col_demo, col_batch = st.columns(2)
            with col_demo:
                if st.button("🎮 Load Demo Data", key="demo_heart_class", help="Fill form with sample patient data"):
                    st.session_state.use_demo_data['heart_class'] = True
                    st.rerun()
            with col_batch:
                batch_file = st.file_uploader("📤 Batch Upload (CSV)", type=['csv'], key="batch_heart_class", 
                                             help="Upload CSV with multiple patients")
            
            # Handle batch upload
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.markdown("### 📊 Batch Prediction Results")
                    
                    with st.spinner("Processing batch predictions..."):
                        # Scale and predict
                        batch_scaled = scaler.transform(batch_df)
                        batch_predictions = model.predict(batch_scaled)
                        batch_proba = model.predict_proba(batch_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        # Add results to dataframe
                        batch_df['Prediction'] = ['Heart Disease' if p == 1 else 'No Disease' for p in batch_predictions]
                        if batch_proba is not None:
                            batch_df['Probability'] = batch_proba
                        
                        # Display results
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Patients", len(batch_df))
                        with col2:
                            st.metric("Heart Disease", sum(batch_predictions))
                        with col3:
                            st.metric("No Disease", len(batch_predictions) - sum(batch_predictions))
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error processing batch file: {str(e)}")
            
            # Get demo values if requested
            demo = DEMO_DATA['heart_classification'] if st.session_state.use_demo_data.get('heart_class') else {}
            
            # Input form
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📋 Demographics & Vitals**")
                age = st.number_input("Age (years)", 20, 100, demo.get('age', 50), help="Patient's age in years")
                sex = st.selectbox("Sex", ["Male (1)", "Female (0)"], 
                                  index=0 if demo.get('sex', 1) == 1 else 1, help="Biological sex")
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, demo.get('resting_bp', 120), 
                                          help="Blood pressure at rest")
                chol = st.number_input("Cholesterol (mg/dl)", 100, 600, demo.get('cholesterol', 200), 
                                     help="Serum cholesterol level")
            
            with col2:
                st.markdown("**💓 Heart Measurements**")
                cp = st.selectbox("Chest Pain Type", [
                    "0 - Typical Angina",
                    "1 - Atypical Angina",
                    "2 - Non-anginal Pain",
                    "3 - Asymptomatic"
                ], index=demo.get('chest_pain', 0), help="Type of chest pain experienced")
                thalach = st.number_input("Max Heart Rate", 60, 220, demo.get('max_hr', 150), 
                                        help="Maximum heart rate achieved")
                oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, demo.get('oldpeak', 1.0), 0.1, 
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
                
                # Add to prediction history
                result_text = "Positive (Heart Disease)" if prediction == 1 else "Negative (Healthy)"
                add_to_history(
                    model_name="Heart Disease Classification",
                    inputs=f"Age: {age}, Sex: {sex}, Chest Pain: {cp}, BP: {trestbps}",
                    prediction=result_text,
                    probability=confidence
                )
                
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
            
            # ============================================================
            # DNN PREDICTION SECTION (Heart Disease)
            # ============================================================
            st.markdown("---")
            st.markdown("### 🧠 Deep Neural Network Prediction")
            
            dnn_heart = load_dnn_heart_model()
            if dnn_heart is not None and PYTORCH_AVAILABLE:
                st.success(f"✅ DNN Model Loaded: **{dnn_heart.config_desc}** (Config: {dnn_heart.config_name})")
                show_pytorch_architecture_summary("heart_classification", "Best architecture summary")
                
                if st.button("🧠 Predict with DNN (PyTorch)", type="secondary", use_container_width=True, key="dnn_heart_predict"):
                    # Prepare input for DNN model
                    dnn_input = pd.DataFrame([[
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
                    ]], columns=dnn_heart.feature_columns)
                    
                    dnn_pred = dnn_heart.predict(dnn_input)
                    dnn_proba = dnn_heart.predict_proba(dnn_input)
                    # Handle scalar vs array returns
                    dnn_pred_val = int(dnn_pred) if np.ndim(dnn_pred) == 0 else int(dnn_pred[0])
                    dnn_risk = float(dnn_proba[0, 1]) * 100 if dnn_proba.ndim == 2 else float(dnn_proba[1]) * 100
                    dnn_confidence = max(float(dnn_proba[0, 0]), float(dnn_proba[0, 1])) * 100 if dnn_proba.ndim == 2 else max(float(dnn_proba[0]), float(dnn_proba[1])) * 100
                    
                    dnn_result_text = "Positive (Heart Disease)" if dnn_pred_val == 1 else "Negative (Healthy)"
                    add_to_history(
                        model_name="Heart Disease DNN (PyTorch)",
                        inputs=f"Age: {age}, Sex: {sex}, BP: {trestbps}",
                        prediction=dnn_result_text,
                        probability=dnn_confidence
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if dnn_pred_val == 1:
                            st.error("### 🚨 DNN: POSITIVE")
                            st.markdown("**Heart disease detected**")
                        else:
                            st.success("### ✅ DNN: NEGATIVE")
                            st.markdown("**No heart disease detected**")
                    with col2:
                        st.metric("DNN Risk Score", f"{dnn_risk:.1f}%")
                    with col3:
                        st.metric("DNN Confidence", f"{dnn_confidence:.1f}%")
            else:
                st.warning("⚠️ PyTorch DNN model not available. Run `pytorch_advanced_experiments.py` to train it.")
            
            # ============================================================
            # NOVEL SCRATCH ARCHITECTURE PREDICTION (Heart Disease)
            # ============================================================
            st.markdown("---")
            st.markdown("### 🛠️ Novel Scratch Architecture Prediction")
            
            novel_heart = load_novel_heart_model()
            if novel_heart is not None:
                show_novel_architecture_summary(novel_heart, "Best Novel Architecture Summary")
                
                if st.button("🛠️ Predict with Novel Arch", type="secondary", use_container_width=True, key="novel_heart_predict"):
                    novel_input = np.array([[
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
                    # Standard scaler was used for ML and saved, use it for novel model too since the training pipeline used StandardScaler
                    # wait, let's just use the exact same scaler that the original ML loaded.
                    novel_input_scaled = scaler.transform(pd.DataFrame(novel_input, columns=metadata.get('feature_names', [])))
                    
                    preds, proba = predict_with_novel_model(novel_heart, novel_input_scaled)
                    
                    if preds is not None:
                        novel_pred_val = int(preds[0]) if np.ndim(preds) > 0 else int(preds)
                        novel_risk = float(proba[0, 1]) * 100 if proba is not None and proba.ndim == 2 else float(proba[0]) * 100 if proba is not None else (100 if novel_pred_val==1 else 0)
                        novel_confidence = max(float(proba[0, 0]), float(proba[0, 1])) * 100 if proba is not None and proba.ndim == 2 else max(float(proba[0]), 1-float(proba[0])) * 100 if proba is not None else 100
                        
                        novel_result_text = "Positive (Heart Disease)" if novel_pred_val == 1 else "Negative (Healthy)"
                        add_to_history(
                            model_name=f"Heart Novel ({novel_heart['architecture']})",
                            inputs=f"Age: {age}, Sex: {sex}, BP: {trestbps}",
                            prediction=novel_result_text,
                            probability=novel_confidence
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if novel_pred_val == 1:
                                st.error("### 🚨 NOVEL: POSITIVE")
                                st.markdown("**Heart disease detected**")
                            else:
                                st.success("### ✅ NOVEL: NEGATIVE")
                                st.markdown("**No heart disease detected**")
                        with col2:
                            st.metric("Novel Risk Score", f"{novel_risk:.1f}%")
                        with col3:
                            st.metric("Novel Confidence", f"{novel_confidence:.1f}%")
            else:
                st.warning("⚠️ Novel architecture model not found. Run `train_all_novel_architectures.py` first.")
            
            # ============================================================
            # ML vs DNN COMPARISON (Heart Disease)
            # ============================================================
            st.markdown("---")
            st.markdown("### 📊 ML vs DNN Performance Comparison")
            st.caption("Heart Disease Classification — Same test set")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.markdown("#### 🏆 ML (Stacking)")
                st.metric("ROC-AUC", "0.9784")
                st.metric("Accuracy", f"{metadata.get('accuracy', 0.9328):.4f}")
                st.caption("XGBoost + RF + GB + ET")
            with comp_col2:
                st.markdown("#### 🧠 DNN (ResidualMLP)")
                st.metric("ROC-AUC", "0.9517", delta="-0.0267")
                st.metric("Accuracy", "0.9034", delta=f"{0.9034 - metadata.get('accuracy', 0.9328):+.4f}")
                st.caption("512-dim × 4 blocks + AdamW")
            with comp_col3:
                st.markdown("#### 📈 Verdict")
                st.info("**ML wins** by +2.67% ROC-AUC")
                st.caption("Stacking ensemble benefits from diverse base learners on small tabular data (952 rows)")
            
            with st.expander("📋 All DNN Configurations Tested"):
                try:
                    heart_std = pd.read_csv("pytorch_results/heart_pytorch_results.csv")
                    heart_adv = pd.read_csv("pytorch_results/heart_advanced_results.csv")
                    all_heart = pd.concat([
                        heart_std[['config', 'description', 'test_auc', 'test_acc', 'test_f1']],
                        heart_adv[['config', 'description', 'test_auc', 'test_acc', 'test_f1']]
                    ], ignore_index=True).sort_values('test_auc', ascending=False)
                    all_heart.columns = ['Config', 'Description', 'ROC-AUC', 'Accuracy', 'F1-Score']
                    st.dataframe(all_heart, use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Run PyTorch experiments to generate comparison data.")
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📈 ML Model Performance Metrics")
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
            
            # Demo and Batch Upload section
            col_demo, col_batch = st.columns([1, 2])
            with col_demo:
                if st.button("🎮 Load Demo Data", key="demo_covid", help="Fill form with sample blood test data"):
                    st.session_state['use_demo_data'] = st.session_state.get('use_demo_data', {})
                    st.session_state['use_demo_data']['covid'] = True
                    st.rerun()
            
            with col_batch:
                batch_file = st.file_uploader("📤 Upload CSV for batch predictions", type=['csv'], key="batch_covid")
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.info(f"📊 Loaded {len(batch_df)} records")
                    
                    if st.button("🔮 Run Batch Predictions", key="run_batch_covid"):
                        batch_scaled = scaler.transform(batch_df)
                        batch_preds = model.predict(batch_scaled)
                        batch_proba = model.predict_proba(batch_scaled)[:, 1] if hasattr(model, 'predict_proba') else batch_preds
                        
                        results_df = batch_df.copy()
                        results_df['Prediction'] = ['Positive' if p == 1 else 'Negative' for p in batch_preds]
                        results_df['Probability'] = batch_proba * 100
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        st.download_button("📥 Download Results", csv_buffer.getvalue(), "covid_batch_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error processing batch file: {e}")
            
            # Get demo data if available
            demo = {}
            if st.session_state.get('use_demo_data', {}).get('covid', False):
                demo = DEMO_DATA.get('covid', {})
                st.session_state['use_demo_data']['covid'] = False
            
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
                
                # Add to prediction history
                result_text = "Positive (COVID-19 Detected)" if prediction == 1 else "Negative (No COVID-19)"
                add_to_history(
                    model_name="COVID-19 Diagnosis",
                    inputs=f"Age: {age}, WBC: {wbc}, CRP: {pcr}, Lymph: {ly}%",
                    prediction=result_text,
                    probability=confidence
                )
                
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
            
            # Demo button
            col_demo, col_space = st.columns([1, 3])
            with col_demo:
                if st.button("🎮 Load Demo Data", key="demo_temp", help="Fill with sample weather data"):
                    st.session_state['use_demo_data'] = st.session_state.get('use_demo_data', {})
                    st.session_state['use_demo_data']['temp'] = True
                    st.rerun()
            
            demo = DEMO_DATA.get('temperature', {}) if st.session_state.get('use_demo_data', {}).get('temp', False) else {}
            if demo:
                st.session_state['use_demo_data']['temp'] = False
            
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
                
                # Add to prediction history
                add_to_history(
                    model_name="Temperature Prediction",
                    inputs=f"Humidity: {humidity}%, Wind: {wind_speed}km/h, Pressure: {pressure}mbar",
                    prediction=f"{prediction:.1f}°C",
                    probability=r2 * 100
                )
                
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
            
            # ============================================================
            # DNN PREDICTION SECTION (Temperature)
            # ============================================================
            st.markdown("---")
            st.markdown("### 🧠 Deep Neural Network Prediction")
            
            dnn_temp = load_dnn_temperature_model()
            if dnn_temp is not None and PYTORCH_AVAILABLE:
                st.success(f"✅ DNN Model Loaded: **{dnn_temp.config_desc}** (Config: {dnn_temp.config_name})")
                show_pytorch_architecture_summary("temperature_regression", "Best architecture summary")
                
                if st.button("🧠 Predict with DNN (PyTorch)", type="secondary", use_container_width=True, key="dnn_temp_predict"):
                    # DNN model expects string categories for Summary and Precip Type
                    # Map dashboard options to the DNN label encoder classes
                    summary_dnn_map = {"Clear": "Clear", "Partly Cloudy": "Partly Cloudy", 
                                       "Cloudy": "Mostly Cloudy", "Overcast": "Overcast"}
                    precip_dnn_map = {"None": np.nan, "Rain": "rain", "Snow": "snow"}
                    
                    # IMPORTANT: Dataset uses Humidity in [0,1], not [0,100]
                    # Dataset uses Loud Cover always 0 (not 0-8 oktas)
                    dnn_input = pd.DataFrame([[
                        summary_dnn_map[summary],
                        precip_dnn_map[precip_type],
                        humidity / 100.0,             # Convert 0-100% to 0-1
                        wind_speed,
                        wind_bearing,
                        visibility,
                        0.0,                          # Loud Cover is always 0 in dataset
                        pressure
                    ]], columns=dnn_temp.feature_columns)
                    
                    dnn_pred = dnn_temp.predict(dnn_input)
                    if hasattr(dnn_pred, '__len__'):
                        dnn_pred = float(dnn_pred[0]) if len(dnn_pred.shape) > 0 else float(dnn_pred)
                    else:
                        dnn_pred = float(dnn_pred)
                    
                    add_to_history(
                        model_name="Temperature DNN (PyTorch)",
                        inputs=f"Humidity: {humidity}%, Wind: {wind_speed}km/h",
                        prediction=f"{dnn_pred:.1f}°C",
                        probability=None
                    )
                    
                    st.markdown('<div class="prediction-result">🧠 DNN Predicted Temperature: {:.1f}°C</div>'.format(dnn_pred), 
                              unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("DNN Temperature", f"{dnn_pred:.1f}°C")
                    with col2:
                        st.metric("DNN R² Score", "0.7520")
            else:
                st.warning("⚠️ PyTorch DNN model not available. Run `pytorch_advanced_experiments.py` to train it.")

            # ============================================================
            # NOVEL SCRATCH ARCHITECTURE PREDICTION (Temperature)
            # ============================================================
            st.markdown("---")
            st.markdown("### 🛠️ Novel Scratch Architecture Prediction")
            
            novel_temp = load_novel_temperature_model()
            if novel_temp is not None:
                show_novel_architecture_summary(novel_temp, "Best Novel Architecture Summary")
                
                if st.button("🛠️ Predict with Novel Arch", type="secondary", use_container_width=True, key="novel_temp_predict"):
                    try:
                        # The novel model was trained on 7 features:
                        # Precip Type (encoded), Apparent Temperature (C), Humidity, Wind Speed (km/h),
                        # Wind Bearing (degrees), Visibility (km), Pressure (millibars)
                        # We reconstruct the StandardScaler from the same data subset used in training
                        from sklearn.preprocessing import LabelEncoder
                        from sklearn.impute import SimpleImputer
                        
                        @st.cache_resource
                        def get_novel_temp_scaler():
                            import pandas as pd
                            from sklearn.preprocessing import StandardScaler, LabelEncoder
                            from sklearn.impute import SimpleImputer
                            df_sc = pd.read_csv("Dataset1.csv")
                            drop_cols = ["Temperature (C)", "Formatted Date", "Summary", "Daily Summary", "Loud Cover"]
                            X_sc = df_sc.drop(columns=drop_cols, errors="ignore")
                            for c in X_sc.select_dtypes(['object']).columns:
                                X_sc[c] = LabelEncoder().fit_transform(X_sc[c].astype(str))
                            X_sc = SimpleImputer(strategy='mean').fit_transform(X_sc)[:15000]
                            sc = StandardScaler()
                            sc.fit(X_sc)
                            return sc
                        
                        novel_sc = get_novel_temp_scaler()
                        
                        # Map dashboard inputs to the 7 training features
                        # The LabelEncoder for Precip Type encodes: nan→0, rain→1, snow→2 (approx)
                        precip_map_novel = {"None": 0, "Rain": 1, "Snow": 2}
                        # Apparent Temperature ≈ Temperature + wind chill offset; use humidity-adjusted estimate
                        apparent_temp_estimate = (humidity / 100.0) * 15.0 + (1 - wind_speed / 100.0) * 5.0
                        
                        raw_input = np.array([[
                            precip_map_novel.get(precip_type, 0),   # Precip Type
                            apparent_temp_estimate,                   # Apparent Temperature (C)
                            humidity / 100.0,                         # Humidity [0,1]
                            wind_speed,                               # Wind Speed (km/h)
                            wind_bearing,                             # Wind Bearing (degrees)
                            visibility,                               # Visibility (km)
                            pressure                                  # Pressure (millibars)
                        ]], dtype=np.float32)
                        
                        novel_input_scaled = novel_sc.transform(raw_input)
                        preds, _ = predict_with_novel_model(novel_temp, novel_input_scaled)
                        
                        if preds is not None:
                            novel_pred = float(preds[0]) if np.ndim(preds) > 0 else float(preds)
                            
                            add_to_history(
                                model_name=f"Temperature Novel ({novel_temp['architecture']})",
                                inputs=f"Humidity: {humidity}%, Wind: {wind_speed}km/h",
                                prediction=f"{novel_pred:.1f}°C",
                                probability=None
                            )
                            
                            st.markdown('<div class="prediction-result">🛠️ Novel Predicted Temperature: {:.1f}°C</div>'.format(novel_pred), 
                                      unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Novel Temperature", f"{novel_pred:.1f}°C")
                            with col2:
                                st.metric("Novel Test R²", f"{novel_temp['test_metric']:.4f}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Novel architecture model not found. Run `train_all_novel_architectures.py` first.")
            
            # ============================================================
            # ML vs DNN COMPARISON (Temperature)
            # ============================================================
            st.markdown("---")
            st.markdown("### 📊 ML vs DNN Performance Comparison")
            st.caption("Temperature Regression — Same test set")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.markdown("#### 🏆 ML (Stacking)")
                st.metric("R² Score", "0.7766")
                st.metric("MAE", f"±{metadata['performance_metrics']['mae']:.2f}°C")
                st.caption("XGBoost + RF + GB + ET + Ridge + Lasso")
            with comp_col2:
                st.markdown("#### 🧠 DNN (ResidualMLP)")
                st.metric("R² Score", "0.7520", delta="-0.0246")
                st.metric("MAE", "±3.70°C")
                st.caption("512-dim × 4 blocks + AdamW")
            with comp_col3:
                st.markdown("#### 📈 Verdict")
                st.info("**ML wins** by +2.46% R²")
                st.caption("Stacking ensemble captures complex interactions that MLP cannot replicate on 62K weather samples")
            
            with st.expander("📋 All DNN Configurations Tested"):
                try:
                    temp_std = pd.read_csv("pytorch_results/temperature_pytorch_results.csv")
                    temp_adv = pd.read_csv("pytorch_results/temperature_advanced_results.csv")
                    all_temp = pd.concat([
                        temp_std[['config', 'description', 'test_r2', 'test_mae', 'test_rmse']],
                        temp_adv[['config', 'description', 'test_r2', 'test_mae', 'test_rmse']]
                    ], ignore_index=True).sort_values('test_r2', ascending=False)
                    all_temp.columns = ['Config', 'Description', 'R²', 'MAE (°C)', 'RMSE (°C)']
                    st.dataframe(all_temp, use_container_width=True, hide_index=True)
                except Exception:
                    st.info("Run PyTorch experiments to generate comparison data.")
            
            # Model Performance Section (Always Visible)
            st.markdown("---")
            st.markdown("### 📈 ML Model Performance Metrics")
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
                # Map categorical to match training encoding
                summary_map = {"Clear": 0, "Partly Cloudy": 3, "Cloudy": 1, "Overcast": 2}  # LabelEncoder order
                precip_map = {"None": 0, "Rain": 1, "Snow": 2}
                
                # Input features: Summary, Precip Type, Temperature, Apparent Temp, Wind Speed, Wind Bearing, Visibility, Cloud Cover
                # Note: Pressure and Humidity are OUTPUTS, not inputs!
                input_data = np.array([[
                    summary_map[summary],
                    precip_map[precip],
                    temp,
                    apparent,
                    wind,
                    wind_bearing,
                    vis,
                    cloud
                ]])
                
                predictions = model.predict(input_data)[0]
                
                # Add to prediction history
                add_to_history(
                    model_name="Multi-Output Prediction",
                    inputs=f"Temp: {temp}°C, Wind: {wind}km/h, Cloud: {cloud}/8",
                    prediction=f"Pressure: {predictions[0]:.1f}mbar, Humidity: {predictions[1]:.1f}%",
                    probability=None
                )
                
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
            st.info(f"**Model requires 31 engineered features** ")
            
            # Demo button
            col_demo, col_space = st.columns([1, 3])
            with col_demo:
                if st.button("🎮 Load Demo Data", key="demo_weather_class", help="Fill with sample data"):
                    st.session_state['use_demo_data'] = st.session_state.get('use_demo_data', {})
                    st.session_state['use_demo_data']['weather_class'] = True
                    st.rerun()
            
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
                precip_type_w = st.selectbox("Precipitation Type", ["None", "Rain", "Snow"], key="precip_w")
            
            # Additional temporal inputs
            st.markdown("**📅 Date & Time** (for temporal features)")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                month_w = st.slider("Month", 1, 12, 6, key="month_w")
                day_w = st.slider("Day", 1, 31, 15, key="day_w")
            with col_date2:
                hour_w = st.slider("Hour (0-23)", 0, 23, 12, key="hour_w")
                year_w = st.number_input("Year", 2006, 2025, 2016, key="year_w")
            
            st.markdown("---")
            
            if st.button("☁️ Classify Weather", type="primary", use_container_width=True):
                
                # ============ FEATURE ENGINEERING PIPELINE ============
                # Generate all 31 features required by the model
                
                # Cyclical encoding for Month and Hour
                month_sin = np.sin(2 * np.pi * month_w / 12)
                month_cos = np.cos(2 * np.pi * month_w / 12)
                hour_sin = np.sin(2 * np.pi * hour_w / 24)
                hour_cos = np.cos(2 * np.pi * hour_w / 24)
                
                # Precipitation type encoding
                precip_map = {"None": 0, "Rain": 1, "Snow": 2}
                precip_encoded = precip_map[precip_type_w]
                
                # Interaction terms
                temp_humidity_interaction = temp_w * humidity_w / 100
                feels_like_diff = temp_w - apparent_w
                pressure_temp_interaction = pressure_w * temp_w / 1000
                cloud_humidity_interaction = cloud_w * humidity_w / 100
                
                # Polynomial features
                temp_squared = temp_w ** 2
                wind_speed_squared = wind_w ** 2
                
                # Wind decomposition (bearing to N-S and E-W components)
                wind_bearing_rad = np.radians(wind_bearing_w)
                wind_n_s = wind_w * np.cos(wind_bearing_rad)  # North-South component
                wind_e_w = wind_w * np.sin(wind_bearing_rad)  # East-West component
                
                # Binary indicators
                low_pressure = 1 if pressure_w < 1000 else 0
                high_pressure = 1 if pressure_w > 1020 else 0
                is_winter = 1 if month_w in [12, 1, 2] else 0
                is_summer = 1 if month_w in [6, 7, 8] else 0
                is_day = 1 if 6 <= hour_w <= 18 else 0
                
                # Visibility/Humidity ratio (avoid division by zero)
                visibility_humidity_ratio = visibility_w / max(humidity_w, 1)
                
                # Build feature array in exact order expected by model
                feature_values = [
                    temp_w,                        # Temperature (C)
                    apparent_w,                    # Apparent Temperature (C)
                    humidity_w,                    # Humidity
                    wind_w,                        # Wind Speed (km/h)
                    wind_bearing_w,                # Wind Bearing (degrees)
                    visibility_w,                  # Visibility (km)
                    cloud_w,                       # Loud Cover (cloud cover)
                    pressure_w,                    # Pressure (millibars)
                    year_w,                        # Year
                    month_w,                       # Month
                    day_w,                         # Day
                    hour_w,                        # Hour
                    month_sin,                     # Month_sin
                    month_cos,                     # Month_cos
                    hour_sin,                      # Hour_sin
                    hour_cos,                      # Hour_cos
                    precip_encoded,                # Precip_Type_encoded
                    temp_humidity_interaction,     # Temp_Humidity_Interaction
                    feels_like_diff,               # Feels_Like_Diff
                    temp_squared,                  # Temp_Squared
                    wind_speed_squared,            # Wind_Speed_Squared
                    wind_n_s,                      # Wind_N_S
                    wind_e_w,                      # Wind_E_W
                    pressure_temp_interaction,     # Pressure_Temp_Interaction
                    low_pressure,                  # Low_Pressure
                    high_pressure,                 # High_Pressure
                    visibility_humidity_ratio,     # Visibility_Humidity_Ratio
                    cloud_humidity_interaction,    # Cloud_Humidity_Interaction
                    is_winter,                     # Is_Winter
                    is_summer,                     # Is_Summer
                    is_day                         # Is_Day
                ]
                
                # Create DataFrame with feature names
                input_df = pd.DataFrame([feature_values], columns=metadata['feature_names'])
                
                # Predict using actual model
                prediction_idx = model.predict(input_df)[0]
                
                # Get probabilities and class names
                classes = metadata['target_classes']
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(input_df)[0]
                else:
                    probs = [1.0 if i == prediction_idx else 0.0 for i in range(len(classes))]
                
                # Convert numeric prediction to class name
                predicted_class = classes[int(prediction_idx)] if isinstance(prediction_idx, (int, np.integer)) else prediction_idx
                confidence = max(probs) * 100
                
                # Add to prediction history
                add_to_history(
                    model_name="Weather Classification",
                    inputs=f"Temp: {temp_w}°C, Humidity: {humidity_w}%, Cloud: {cloud_w}/8",
                    prediction=predicted_class,
                    probability=confidence
                )
                
                st.success("✅ **Full Feature Engineering Pipeline Applied!** (31 features)")
                
                st.markdown("### 🎲 Weather Classification Results")
                
                for cls, prob in zip(classes, probs):
                    st.progress(prob, text=f"**{cls}**: {prob*100:.1f}%")
                
                st.markdown('<div class="prediction-result">☁️ Predicted: {}</div>'.format(predicted_class), 
                          unsafe_allow_html=True)
                
                st.metric("Model Confidence", f"{confidence:.1f}%")
                
                # Show engineered features
                with st.expander("🔧 View Engineered Features (31 total)"):
                    feat_df = input_df.T.rename(columns={0: 'Value'})
                    st.dataframe(feat_df, use_container_width=True)
            
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
                    
                    # Add to prediction history
                    add_to_history(
                        model_name="Wind Turbine Forecasting",
                        inputs=f"Wind: {wind_speed}m/s, Dir: {wind_direction}°",
                        prediction=f"{prediction:.2f} kW",
                        probability=None
                    )
                    
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
        
        # ============================================================
        # DNN PREDICTION & COMPARISON (Wind Turbine)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🧠 Deep Neural Network Forecasting")
        
        dnn_wind = load_dnn_wind_model()
        if dnn_wind is not None and PYTORCH_AVAILABLE:
            st.success(f"✅ DNN Model Loaded: **{dnn_wind.config_desc}** (Config: {dnn_wind.config_name})")
            show_pytorch_architecture_summary("wind_forecasting", "Best architecture summary")
            
            if st.button("🧠 DNN Forecast (PyTorch)", type="secondary", use_container_width=True, key="dnn_wind_predict"):
                try:
                    # Get last 24 raw power values from dataset
                    dnn_window = df[target_col].tail(24).values.astype(np.float64)
                    
                    # Recursive DNN forecast
                    dnn_predictions = []
                    current_window = dnn_window.copy()
                    
                    for step in range(horizon):
                        pred = dnn_wind_predict(dnn_wind, current_window)
                        dnn_predictions.append(pred)
                        if step < horizon - 1:
                            current_window = np.append(current_window[1:], pred)
                    
                    # Display DNN forecast
                    st.markdown("### 📈 DNN Forecast Results")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        hist_steps = min(24, len(last_observations))
                        ax.plot(range(-hist_steps, 0), last_observations[-hist_steps:],
                               label='Historical', color='steelblue', linewidth=2, marker='o', markersize=4)
                        ax.plot(range(0, len(dnn_predictions)), dnn_predictions,
                               label='DNN Forecast', color='#9b59b6', linewidth=2, marker='s', markersize=5, linestyle='--')
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Forecast Start')
                        ax.set_xlabel('Time Steps (10-min intervals)')
                        ax.set_ylabel('LV ActivePower (kW)')
                        ax.set_title(f'DNN Wind Turbine Forecast — Next {horizon} Steps')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        dnn_forecast_df = pd.DataFrame({
                            'Step': range(1, len(dnn_predictions) + 1),
                            'Time (min)': [(i * 10) for i in range(1, len(dnn_predictions) + 1)],
                            'Power (kW)': [f"{p:.2f}" for p in dnn_predictions]
                        })
                        st.dataframe(dnn_forecast_df, hide_index=True)
                    
                    add_to_history(
                        model_name="Wind Turbine DNN (PyTorch)",
                        inputs=f"Window: last 24 steps, Horizon: {horizon}",
                        prediction=f"Avg: {np.mean(dnn_predictions):.2f} kW",
                        probability=None
                    )
                except Exception as e:
                    st.error(f"Error in DNN forecast: {str(e)}")
        else:
            st.warning("⚠️ PyTorch DNN model not available. Run `pytorch_advanced_experiments.py` to train it.")

        # ============================================================
        # NOVEL SCRATCH ARCHITECTURE PREDICTION (Wind Turbine)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🛠️ Novel Scratch Architecture Prediction")
        
        novel_wind = load_novel_wind_model()
        if novel_wind is not None:
            show_novel_architecture_summary(novel_wind, "Best Novel Architecture Summary")
            
            if st.button("🛠️ Predict with Novel Arch", type="secondary", use_container_width=True, key="novel_wind_predict"):
                try:
                    # Get last 24 raw power values from dataset
                    from sklearn.preprocessing import MinMaxScaler
                    sc = MinMaxScaler()
                    # Fit on first 20k to match training script
                    sc.fit(df[target_col].values[:20000].reshape(-1, 1))
                    
                    dnn_window = df[target_col].tail(24).values.astype(np.float64)
                    
                    novel_predictions = []
                    current_window = dnn_window.copy()
                    
                    for step in range(horizon):
                        scaled_vals = sc.transform(current_window.reshape(-1, 1)).flatten()
                        preds, _ = predict_with_novel_model(novel_wind, scaled_vals.reshape(1, -1))
                        if preds is not None:
                            pred_scaled = float(preds[0])
                            pred = float(sc.inverse_transform([[pred_scaled]])[0, 0])
                        else:
                            pred = 0.0
                        novel_predictions.append(pred)
                        if step < horizon - 1:
                            current_window = np.append(current_window[1:], pred)
                            
                    st.markdown("### 📈 Novel Forecast Results")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        hist_steps = min(24, len(last_observations))
                        ax.plot(range(-hist_steps, 0), last_observations[-hist_steps:],
                               label='Historical', color='steelblue', linewidth=2, marker='o', markersize=4)
                        ax.plot(range(0, len(novel_predictions)), novel_predictions,
                               label='Novel Forecast', color='#e67e22', linewidth=2, marker='s', markersize=5, linestyle='--')
                        ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Forecast Start')
                        ax.set_xlabel('Time Steps (10-min intervals)')
                        ax.set_ylabel('LV ActivePower (kW)')
                        ax.set_title(f'Novel Architecture Forecast — Next {horizon} Steps')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        novel_forecast_df = pd.DataFrame({
                            'Step': range(1, len(novel_predictions) + 1),
                            'Time (min)': [(i * 10) for i in range(1, len(novel_predictions) + 1)],
                            'Power (kW)': [f"{p:.2f}" for p in novel_predictions]
                        })
                        st.dataframe(novel_forecast_df, hide_index=True)
                except Exception as e:
                    st.error(f"Error in Novel forecast: {str(e)}")
        else:
            st.warning("⚠️ Novel architecture model not found. Run `train_all_novel_architectures.py` first.")
        
        # ML vs DNN Comparison
        st.markdown("---")
        st.markdown("### 📊 ML vs DNN Performance Comparison")
        st.caption("Wind Turbine Power Forecasting — Univariate, same chronological test set")
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            st.markdown("#### 🏆 ML (Ridge)")
            st.metric("R² Score", "0.9714")
            st.metric("RMSE", f"{best_uni['rmse']:.2f} kW")
            st.caption("Ridge Regression + MinMaxScaler")
        with comp_col2:
            st.markdown("#### 🧠 DNN (FlexMLP)")
            st.metric("R² Score", "0.9710", delta="-0.0004")
            st.metric("RMSE", "228.47 kW")
            st.caption("Adam + CosineAnnealingLR")
        with comp_col3:
            st.markdown("#### 📈 Verdict")
            st.success("**Near parity!** Gap only 0.04%")
            st.caption("DNN essentially matches ML Ridge on this univariate time series task")
        
        with st.expander("📋 All DNN Configurations Tested"):
            try:
                wind_results = pd.read_csv("pytorch_results/wind_pytorch_results.csv")
                wind_display = wind_results[['config', 'description', 'test_r2', 'test_mae', 'test_rmse']].sort_values('test_r2', ascending=False)
                wind_display.columns = ['Config', 'Description', 'R²', 'MAE (kW)', 'RMSE (kW)']
                st.dataframe(wind_display, use_container_width=True, hide_index=True)
            except Exception:
                st.info("Run PyTorch experiments to generate comparison data.")
        
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
            df = pd.read_csv("Energy_Forecasting/pjm_hourly_est.csv")
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
                    
                    # Add to prediction history
                    add_to_history(
                        model_name="PJM Energy Forecasting",
                        inputs=f"Historical window: {window_size} hours",
                        prediction=f"{prediction:.2f} MWh",
                        probability=None
                    )
                    
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
# ANOMALY DETECTION SECTION 1: EMPLOYEE ATTRITION
# ============================================================================

elif "Employee Attrition" in model_choice:
    
    st.markdown("### 🔍 Employee Attrition Anomaly Detection")
    st.caption("Detecting employee turnover (attrition) as anomalies using unsupervised learning")
    
    try:
        # Load results and model
        results_path = "Anomaly detection/EmpolyeeClassification/anomaly_detection_results.csv"
        model_path = "Anomaly detection/EmpolyeeClassification/best_anomaly_model.pkl"
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, index_col=0)
            
            # Find best model (by F1-score)
            best_model_name = results_df['f1_score'].idxmax()
            best_model_metrics = results_df.loc[best_model_name]
            
            st.success(f"✅ **Best Model: {best_model_name}**")
            
            # Check if model exists for prediction
            model_available = os.path.exists(model_path)
            
            if model_available:
                # Load the model
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                st.markdown("### 🎯 Enter Employee Information for Prediction")
                
                # Create input form with 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=18, max_value=65, value=35)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
                    job_role = st.selectbox("Job Role", ["Sales", "Engineering", "Marketing", "HR", "Finance", "Operations", "IT", "Research"])
                    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000, value=5000)
                    work_life_balance = st.slider("Work-Life Balance", 1, 5, 3)
                    job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
                    performance_rating = st.slider("Performance Rating", 1, 5, 3)
                
                with col2:
                    num_promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
                    overtime = st.selectbox("Overtime", ["Yes", "No"])
                    distance_from_home = st.number_input("Distance from Home (km)", min_value=0, max_value=100, value=10)
                    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
                    job_level = st.slider("Job Level", 1, 5, 2)
                
                with col3:
                    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
                    company_tenure = st.number_input("Company Tenure (years)", min_value=0, max_value=40, value=5)
                    remote_work = st.selectbox("Remote Work", ["Yes", "No"])
                    leadership_opps = st.selectbox("Leadership Opportunities", ["Yes", "No"])
                    innovation_opps = st.selectbox("Innovation Opportunities", ["Yes", "No"])
                    company_reputation = st.slider("Company Reputation", 1, 5, 3)
                    employee_recognition = st.slider("Employee Recognition", 1, 5, 3)
                
                if st.button("🔍 Detect Anomaly", type="primary", use_container_width=True):
                    # Prepare input data
                    input_data = {
                        'Age': age,
                        'Gender': gender,
                        'Years at Company': years_at_company,
                        'Job Role': job_role,
                        'Monthly Income': monthly_income,
                        'Work-Life Balance': work_life_balance,
                        'Job Satisfaction': job_satisfaction,
                        'Performance Rating': performance_rating,
                        'Number of Promotions': num_promotions,
                        'Overtime': overtime,
                        'Distance from Home': distance_from_home,
                        'Education Level': education_level,
                        'Marital Status': marital_status,
                        'Number of Dependents': num_dependents,
                        'Job Level': job_level,
                        'Company Size': company_size,
                        'Company Tenure': company_tenure,
                        'Remote Work': remote_work,
                        'Leadership Opportunities': leadership_opps,
                        'Innovation Opportunities': innovation_opps,
                        'Company Reputation': company_reputation,
                        'Employee Recognition': employee_recognition
                    }
                    
                    # Create dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode categorical variables using stored encoders
                    label_encoders = model_package['label_encoders']
                    for col in input_df.select_dtypes(include=['object']).columns:
                        if col in label_encoders:
                            le = label_encoders[col]
                            try:
                                input_df[col] = le.transform(input_df[col])
                            except ValueError:
                                # Handle unseen labels by using the most frequent class
                                input_df[col] = 0
                    
                    # Scale features
                    scaler = model_package['scaler']
                    input_scaled = scaler.transform(input_df)
                    
                    # Predict using Elliptic Envelope model
                    model = model_package['model']
                    prediction_raw = model.predict(input_scaled)
                    prediction = 1 if prediction_raw[0] == -1 else 0  # -1 = anomaly
                    
                    # Add to prediction history
                    result_text = "Anomaly (Attrition Risk)" if prediction == 1 else "Normal (Low Risk)"
                    add_to_history(
                        model_name="Anomaly: Employee Attrition",
                        inputs=f"Age: {age}, Role: {job_role}, Income: ${monthly_income}",
                        prediction=result_text,
                        probability=None
                    )
                    
                    st.markdown("---")
                    st.markdown("### 🎲 Anomaly Detection Result")
                    
                    if prediction == 1:
                        st.error("⚠️ **ANOMALY DETECTED - Potential Attrition Risk**")
                        st.markdown("""
                        This employee profile shows characteristics similar to employees who have left the company.
                        
                        **Recommended Actions:**
                        - Schedule a 1-on-1 meeting to understand their concerns
                        - Review compensation and career growth opportunities
                        - Consider work-life balance improvements
                        - Evaluate leadership and innovation opportunities
                        """)
                    else:
                        st.success("✅ **NORMAL - Low Attrition Risk**")
                        st.markdown("""
                        This employee profile appears stable with characteristics similar to employees who stay.
                        
                        **Recommendations:**
                        - Continue regular engagement
                        - Maintain career development conversations
                        - Monitor for any changes in behavior or satisfaction
                        """)
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        st.dataframe(pd.DataFrame([input_data]).T.rename(columns={0: 'Value'}))
            
            st.markdown("---")
            
            # Best Model Performance
            st.markdown("### 🏆 Best Model Performance (Elliptic Envelope)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{best_model_metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{best_model_metrics['precision']:.4f}")
            with col3:
                st.metric("Recall (TPR)", f"{best_model_metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{best_model_metrics['f1_score']:.4f}")
            
            # Technical details
            with st.expander("ℹ️ Technical Details & Methodology"):
                st.markdown("""
                **Problem Definition:**
                - **Normal Class:** Employees who stayed (majority)
                - **Anomaly Class:** Employees who left (attrition)
                - **Goal:** Detect attrition-prone employees using unsupervised methods
                
                **Best Model: Elliptic Envelope**
                - Fits a robust covariance estimate to the data
                - Assumes data follows Gaussian distribution
                - Points outside the ellipsoid are marked as anomalies
                
                **Preprocessing:**
                - Label encoding for categorical variables
                - StandardScaler for feature normalization
                - Contamination rate: ~30%
                
                **Key Insights:**
                - Elliptic Envelope achieved best accuracy (0.4896)
                - Challenging dataset due to overlapping class distributions
                - Uses robust covariance for anomaly detection
                """)
        else:
            st.warning("⚠️ Results file not found")
            st.info("Run `anomaly_detection_analysis.py` in the EmpolyeeClassification folder")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# ANOMALY DETECTION SECTION 2: HEART DISEASE
# ============================================================================

elif "Anomaly Detection: Heart" in model_choice:
    
    st.markdown("### 🫀 Heart Disease Anomaly Detection")
    st.caption("Detecting heart disease patients as anomalies using unsupervised learning")
    
    try:
        # Load results and model
        results_path = "Anomaly detection/HeartDesease/heart_evaluation_comparison.csv"
        model_path = "Anomaly detection/HeartDesease/best_anomaly_model.pkl"
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            results_df.set_index('Algorithm', inplace=True)
            
            # Find best model (by Balanced Accuracy)
            best_model_name = results_df['Balanced_Acc'].idxmax()
            best_model_metrics = results_df.loc[best_model_name]
            
            st.success(f"✅ **Best Model: {best_model_name}**")
            
            # Check if model exists for prediction
            model_available = os.path.exists(model_path)
            
            if model_available:
                # Load the model
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                st.markdown("### 🎯 Enter Patient Information for Prediction")
                
                # Create input form with 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=20, max_value=100, value=55)
                    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                    chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                                             format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                                                                   3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
                    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130)
                
                with col2:
                    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
                    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                                             format_func=lambda x: "No" if x == 0 else "Yes")
                    resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                                              format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 
                                                                    2: "Left Ventricular Hypertrophy"}[x])
                    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
                
                with col3:
                    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1],
                                                  format_func=lambda x: "No" if x == 0 else "Yes")
                    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                    st_slope = st.selectbox("ST Slope", [1, 2, 3],
                                           format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
                
                if st.button("🔍 Detect Anomaly", type="primary", use_container_width=True):
                    # Prepare input data
                    input_data = pd.DataFrame([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                                               resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
                                             columns=['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                                                     'fasting blood sugar', 'resting ecg', 'max heart rate',
                                                     'exercise angina', 'oldpeak', 'ST slope'])
                    
                    # Scale features
                    scaler = model_package['scaler']
                    input_scaled = scaler.transform(input_data)
                    
                    # Predict using Elliptic Envelope
                    model = model_package['model']
                    prediction = model.predict(input_scaled)
                    prediction = 1 if prediction[0] == -1 else 0  # -1 = anomaly
                    
                    # Add to prediction history
                    result_text = "Anomaly (Potential Heart Disease)" if prediction == 1 else "Normal (Low Risk)"
                    add_to_history(
                        model_name="Anomaly: Heart Disease",
                        inputs=f"Age: {age}, BP: {resting_bp}, Chol: {cholesterol}",
                        prediction=result_text,
                        probability=None
                    )
                    
                    st.markdown("---")
                    st.markdown("### 🎲 Anomaly Detection Result")
                    
                    if prediction == 1:
                        st.error("⚠️ **ANOMALY DETECTED - Potential Heart Disease**")
                        st.markdown("""
                        This patient profile shows characteristics similar to heart disease patients.
                        
                        **Recommended Actions:**
                        - Schedule comprehensive cardiac evaluation
                        - Consider stress test and echocardiogram
                        - Review lifestyle factors (diet, exercise, smoking)
                        - Monitor blood pressure and cholesterol levels
                        
                        **Note:** This is a screening tool. Always consult with a cardiologist for diagnosis.
                        """)
                    else:
                        st.success("✅ **NORMAL - Low Heart Disease Risk**")
                        st.markdown("""
                        This patient profile appears similar to patients without heart disease.
                        
                        **Recommendations:**
                        - Continue regular health checkups
                        - Maintain healthy lifestyle habits
                        - Monitor cardiovascular risk factors
                        
                        **Note:** This is a screening tool and doesn't guarantee absence of heart disease.
                        """)
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        display_df = input_data.T.copy()
                        display_df.columns = ['Value']
                        st.dataframe(display_df)
            
            st.markdown("---")
            
            # Best Model Performance
            st.markdown("### 🏆 Best Model Performance (Elliptic Envelope)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{best_model_metrics['Sep_Accuracy']:.4f}")
            with col2:
                st.metric("F1-Score", f"{best_model_metrics['Sep_F1']:.4f}")
            with col3:
                st.metric("TPR (Recall)", f"{best_model_metrics['TPR']:.4f}")
            with col4:
                st.metric("TNR (Specificity)", f"{best_model_metrics['TNR']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Balanced Accuracy", f"{best_model_metrics['Balanced_Acc']:.4f}")
            with col2:
                st.metric("TP / FN / TN / FP", f"{int(best_model_metrics['TP'])} / {int(best_model_metrics['FN'])} / {int(best_model_metrics['TN'])} / {int(best_model_metrics['FP'])}")
            
            # Technical details
            with st.expander("ℹ️ Technical Details & Methodology"):
                st.markdown("""
                **Problem Definition:**
                - **Normal Class:** Patients without heart disease (majority)
                - **Anomaly Class:** Patients with heart disease (80 samples)
                - **Goal:** Detect heart disease as anomaly in clinical data
                
                **Best Model: Elliptic Envelope**
                - Fits a robust covariance estimate to the data
                - Assumes data follows Gaussian distribution
                - Points outside the ellipsoid are marked as anomalies
                - Contamination rate: ~12.5%
                
                **Key Insights:**
                - Balanced Accuracy: 0.664 (best among all models)
                - High TNR (0.9112) - very few false alarms
                - TPR (0.4167) - catches ~42% of disease cases
                - Good for screening with low false positive rate
                """)
        else:
            st.warning("⚠️ Results file not found")
            st.info("Run `heart_anomaly_detection.py` in the HeartDesease folder")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# ANOMALY DETECTION SECTION 3: WINE TYPE
# ============================================================================

elif "Wine Type" in model_choice:
    
    st.markdown("### 🍷 Wine Type Anomaly Detection")
    st.caption("Detecting red wine as anomalies in a dataset dominated by white wine")
    
    try:
        # Load results and model
        results_path = "Anomaly detection/WineType/separate_class_evaluation_results/wine_separate_evaluation_results.csv"
        model_path = "Anomaly detection/WineType/separate_class_evaluation_results/best_anomaly_model.pkl"
        metadata_path = "Anomaly detection/WineType/separate_class_evaluation_results/model_metadata.pkl"
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path, index_col=0)
            
            # Find best model (by F1-score)
            best_model_name = results_df['f1_score'].idxmax()
            best_model_metrics = results_df.loc[best_model_name]
            
            st.success(f"✅ **Best Model: {best_model_name}**")
            
            # Check if model exists for prediction
            model_available = os.path.exists(model_path)
            
            if model_available:
                st.markdown("### 🎯 Enter Wine Chemical Properties for Prediction")
                
                # Create input form with 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fixed_acidity = st.number_input("Fixed Acidity (g/L)", min_value=3.0, max_value=16.0, value=7.0, step=0.1)
                    volatile_acidity = st.number_input("Volatile Acidity (g/L)", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
                    citric_acid = st.number_input("Citric Acid (g/L)", min_value=0.0, max_value=1.5, value=0.3, step=0.01)
                    residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.0, max_value=70.0, value=5.0, step=0.5)
                
                with col2:
                    chlorides = st.number_input("Chlorides (g/L)", min_value=0.0, max_value=0.7, value=0.05, step=0.01)
                    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=0.0, max_value=300.0, value=30.0, step=1.0)
                    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=0.0, max_value=500.0, value=120.0, step=5.0)
                    density = st.number_input("Density (g/cm³)", min_value=0.98, max_value=1.05, value=0.995, step=0.001, format="%.4f")
                
                with col3:
                    ph = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.2, step=0.01)
                    sulphates = st.number_input("Sulphates (g/L)", min_value=0.2, max_value=2.0, value=0.5, step=0.01)
                    alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
                    quality = st.number_input("Quality Score (1-10)", min_value=1, max_value=10, value=6, step=1)
                
                if st.button("🔍 Detect Wine Type Anomaly", type="primary", use_container_width=True):
                    # Load the model package
                    with open(model_path, 'rb') as f:
                        model_package = pickle.load(f)
                    
                    # Get feature names from model package
                    feature_names = model_package.get('feature_names', [
                        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol', 'quality'
                    ])
                    
                    input_values = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                   chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                   ph, sulphates, alcohol, quality]
                    
                    input_data = pd.DataFrame([input_values], columns=feature_names)
                    
                    # Get model components
                    scaler = model_package.get('scaler')
                    model = model_package.get('model')
                    
                    try:
                        # Scale the input
                        input_scaled = scaler.transform(input_data)
                        
                        # Predict using Elliptic Envelope
                        prediction_raw = model.predict(input_scaled)
                        prediction = 1 if prediction_raw[0] == -1 else 0
                        
                        # Add to prediction history
                        result_text = "Anomaly (Red Wine)" if prediction == 1 else "Normal (White Wine)"
                        add_to_history(
                            model_name="Anomaly: Wine Type",
                            inputs=f"Alcohol: {alcohol}%, pH: {ph}, Acidity: {fixed_acidity}",
                            prediction=result_text,
                            probability=None
                        )
                        
                        st.markdown("---")
                        st.markdown("### 🎲 Wine Type Detection Result")
                        
                        if prediction == 1:
                            st.error("🍷 **ANOMALY DETECTED - Likely RED WINE**")
                            st.markdown("""
                            This wine sample has chemical properties characteristic of **red wine**.
                            
                            **Red Wine Characteristics Detected:**
                            - Higher volatile acidity
                            - Lower sulfur dioxide levels
                            - Different acid profile
                            
                            **In this anomaly detection context:**
                            - Red wine is the minority class (~25%)
                            - Model detected this as an outlier from white wine distribution
                            """)
                        else:
                            st.success("🥂 **NORMAL - Likely WHITE WINE**")
                            st.markdown("""
                            This wine sample has chemical properties characteristic of **white wine**.
                            
                            **White Wine Characteristics:**
                            - Lower volatile acidity
                            - Higher sulfur dioxide levels
                            - Typical white wine acid profile
                            
                            **In this anomaly detection context:**
                            - White wine is the majority class (~75%)
                            - Sample fits within normal distribution
                            """)
                        
                        # Show input summary
                        with st.expander("📋 Input Summary"):
                            st.dataframe(input_data.T.rename(columns={0: 'Value'}))
                    
                    except Exception as pred_error:
                        st.error(f"Prediction error: {str(pred_error)}")
            
            # ============================================================
            # NOVEL SCRATCH ARCHITECTURE PREDICTION (Wine Anomaly)
            # ============================================================
            st.markdown("---")
            st.markdown("### 🛠️ Novel Scratch Architecture Prediction")
            
            novel_wine = load_novel_wine_model()
            if novel_wine is not None:
                show_novel_architecture_summary(novel_wine, "Best Novel Architecture Summary")
                
                if st.button("🛠️ Predict with Novel Arch", type="secondary", use_container_width=True, key="novel_wine_predict"):
                    try:
                        with open(model_path, 'rb') as f:
                            ml_package = pickle.load(f)
                        scaler_wine = ml_package.get('scaler')
                        
                        feature_names = ml_package.get('feature_names', [
                            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'quality'
                        ])
                        input_values = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                       ph, sulphates, alcohol, quality]
                        
                        input_data_novel = pd.DataFrame([input_values], columns=feature_names)
                        input_scaled_novel = scaler_wine.transform(input_data_novel)
                        
                        preds, proba = predict_with_novel_model(novel_wine, input_scaled_novel)
                        
                        if preds is not None:
                            novel_pred_val = int(preds[0]) if np.ndim(preds) > 0 else int(preds)
                            novel_risk = float(proba[0, 1]) * 100 if proba is not None and proba.ndim == 2 else float(proba[0]) * 100 if proba is not None else (100 if novel_pred_val==1 else 0)
                            
                            novel_result_text = "Anomaly (Red Wine)" if novel_pred_val == 1 else "Normal (White Wine)"
                            add_to_history(
                                model_name=f"Wine Novel ({novel_wine['architecture']})",
                                inputs=f"Alcohol: {alcohol}%, pH: {ph}, Acidity: {fixed_acidity}",
                                prediction=novel_result_text,
                                probability=novel_risk
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if novel_pred_val == 1:
                                    st.error("### 🚨 NOVEL: ANOMALY (RED WINE)")
                                    st.markdown("**Detected as anomaly**")
                                else:
                                    st.success("### ✅ NOVEL: NORMAL (WHITE WINE)")
                                    st.markdown("**Detected as normal**")
                            with col2:
                                st.metric("Novel Risk Score", f"{novel_risk:.1f}%")
                    except Exception as e:
                        st.error(f"Error in Novel prediction: {str(e)}")
            else:
                st.warning("⚠️ Novel architecture model not found. Run `train_all_novel_architectures.py` first.")
            
            st.markdown("---")
            
            # Best Model Performance
            st.markdown("### 🏆 Best Model Performance (Elliptic Envelope)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{best_model_metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{best_model_metrics['precision']:.4f}")
            with col3:
                st.metric("Recall (TPR)", f"{best_model_metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{best_model_metrics['f1_score']:.4f}")
            
            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("TPR", f"{best_model_metrics['tpr']:.4f}", 
                         help="True Positive Rate - Red wines correctly identified")
            with col2:
                st.metric("TNR (Specificity)", f"{best_model_metrics['tnr']:.4f}",
                         help="True Negative Rate - White wines correctly identified")
            with col3:
                st.metric("FPR", f"{best_model_metrics['fpr']:.4f}",
                         help="False Positive Rate - White wines wrongly flagged as red")
            with col4:
                st.metric("FNR", f"{best_model_metrics['fnr']:.4f}",
                         help="False Negative Rate - Red wines missed")
            
            # Confusion matrix counts
            st.markdown("#### Confusion Matrix")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives", f"{int(best_model_metrics['true_positives'])}")
            with col2:
                st.metric("False Negatives", f"{int(best_model_metrics['false_negatives'])}")
            with col3:
                st.metric("True Negatives", f"{int(best_model_metrics['true_negatives'])}")
            with col4:
                st.metric("False Positives", f"{int(best_model_metrics['false_positives'])}")
            
            st.markdown("---")
            
            # All models comparison
            st.markdown("### 📈 All Models Comparison")
            
            # Create comparison dataframe
            comparison_df = results_df[['accuracy', 'precision', 'recall', 'specificity', 'f1_score']].copy()
            comparison_df.columns = ['Accuracy', 'Precision', 'Recall (TPR)', 'Specificity (TNR)', 'F1-Score']
            
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            # Rates table
            st.markdown("#### Detection Rates")
            rates_df = results_df[['tpr', 'tnr', 'fpr', 'fnr']].copy()
            rates_df.columns = ['TPR', 'TNR', 'FPR', 'FNR']
            st.dataframe(
                rates_df.style.format("{:.4f}"),
                use_container_width=True
            )
            
            # Visualization
            st.markdown("### 📊 Performance Visualization")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # TPR vs TNR comparison
            x = np.arange(len(results_df.index))
            width = 0.35
            
            axes[0].bar(x - width/2, results_df['tpr'], width, label='TPR', color='red', alpha=0.7, edgecolor='black')
            axes[0].bar(x + width/2, results_df['tnr'], width, label='TNR', color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Algorithm')
            axes[0].set_ylabel('Rate')
            axes[0].set_title('TPR vs TNR by Algorithm')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(results_df.index, rotation=45, ha='right')
            axes[0].legend()
            axes[0].set_ylim([0, 1.1])
            axes[0].grid(axis='y', alpha=0.3)
            
            # F1-Score comparison
            colors = ['#3498db' if n != best_model_name else '#e74c3c' for n in results_df.index]
            axes[1].bar(results_df.index, results_df['f1_score'], color=colors, edgecolor='black')
            axes[1].set_xlabel('Algorithm')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_title('F1-Score Comparison (Best in Red)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_ylim([0, 1.1])
            for i, (name, v) in enumerate(zip(results_df.index, results_df['f1_score'])):
                axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
            axes[1].grid(axis='y', alpha=0.3)
            
            # Best model breakdown
            metrics = ['Accuracy', 'Precision', 'TPR', 'TNR', 'F1-Score']
            values = [best_model_metrics['accuracy'], best_model_metrics['precision'], 
                     best_model_metrics['tpr'], best_model_metrics['tnr'], best_model_metrics['f1_score']]
            bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
            
            axes[2].bar(metrics, values, color=bar_colors, edgecolor='black')
            axes[2].set_title(f'Best: {best_model_name}')
            axes[2].set_ylabel('Score')
            axes[2].set_ylim([0, 1.1])
            for i, v in enumerate(values):
                axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
            axes[2].tick_params(axis='x', rotation=30)
            axes[2].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Technical details
            with st.expander("ℹ️ Technical Details & Methodology"):
                st.markdown("""
                **Problem Definition:**
                - **Normal Class:** White wine (majority - ~75%)
                - **Anomaly Class:** Red wine (minority - ~25%)
                - **Goal:** Detect red wine samples as anomalies based on chemical properties
                
                **Dataset Features:**
                - 11 chemical properties: fixed acidity, volatile acidity, citric acid, residual sugar, 
                  chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
                - Wine quality merged dataset (red + white wines)
                
                **Evaluation Methodology:**
                - **Separate Class Evaluation:** 
                  - TPR calculated on red wine samples only
                  - TNR calculated on white wine samples only
                - This gives clearer insight than mixed test set evaluation
                
                **Results Analysis:**
                - **Elliptic Envelope** achieved exceptional performance:
                  - 93.12% of red wines correctly identified (TPR)
                  - 96.87% of white wines correctly identified (TNR)
                  - Only 3.13% false positive rate
                  - F1-Score of 0.9188 (excellent)
                
                **Why Elliptic Envelope Works Well:**
                - Wine chemical properties follow roughly Gaussian distribution
                - Red and white wines have distinct chemical signatures
                - Robust covariance estimation handles minor outliers
                
                **Algorithm Comparison:**
                - DBSCAN fails completely (100% FPR) - density-based approach unsuitable
                - Isolation Forest moderate (43.5% TPR, 83.2% TNR)
                - Local Outlier Factor struggles (23.75% TPR)
                """)
        else:
            st.warning("⚠️ Results file not found")
            st.info("Run `wine_separate_class_evaluation.py` in the WineType folder")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

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
            st.info(f"**Using full feature engineering pipeline** - 31 engineered features")
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🌡️ Temperature**")
                temp_w = st.number_input("Temperature (°C)", -40.0, 50.0, 20.0, key="tw_f")
                humidity_w = st.slider("Humidity (%)", 0.0, 100.0, 50.0, key="hw_f")
                apparent_w = st.number_input("Apparent Temp (°C)", -40.0, 50.0, 20.0, key="aw_f")
            
            with col2:
                st.markdown("**💨 Wind & Pressure**")
                pressure_w = st.number_input("Pressure (mbar)", 950.0, 1050.0, 1013.0, key="pw_f")
                wind_w = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0, key="ww_f")
                wind_bearing_w = st.number_input("Wind Bearing (°)", 0.0, 360.0, 180.0, key="wb_f")
            
            with col3:
                st.markdown("**☁️ Visibility & Clouds**")
                visibility_w = st.number_input("Visibility (km)", 0.0, 20.0, 10.0, key="vw_f")
                cloud_w = st.slider("Cloud Cover (0-8)", 0.0, 8.0, 4.0, key="cw_f")
                precip_type_w = st.selectbox("Precipitation Type", ["None", "Rain", "Snow"], key="precip_w_f")
            
            # Additional temporal inputs
            st.markdown("**📅 Date & Time** (for temporal features)")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                month_w = st.slider("Month", 1, 12, 6, key="month_w_f")
                day_w = st.slider("Day", 1, 31, 15, key="day_w_f")
            with col_date2:
                hour_w = st.slider("Hour (0-23)", 0, 23, 12, key="hour_w_f")
                year_w = st.number_input("Year", 2006, 2025, 2016, key="year_w_f")
            
            st.markdown("---")
            
            if st.button("☁️ Classify Weather", type="primary", use_container_width=True, key="classify_weather_f"):
                
                # ============ FEATURE ENGINEERING PIPELINE ============
                # Cyclical encoding
                month_sin = np.sin(2 * np.pi * month_w / 12)
                month_cos = np.cos(2 * np.pi * month_w / 12)
                hour_sin = np.sin(2 * np.pi * hour_w / 24)
                hour_cos = np.cos(2 * np.pi * hour_w / 24)
                
                # Precipitation type encoding
                precip_map = {"None": 0, "Rain": 1, "Snow": 2}
                precip_encoded = precip_map[precip_type_w]
                
                # Interaction terms
                temp_humidity_interaction = temp_w * humidity_w / 100
                feels_like_diff = temp_w - apparent_w
                pressure_temp_interaction = pressure_w * temp_w / 1000
                cloud_humidity_interaction = cloud_w * humidity_w / 100
                
                # Polynomial features
                temp_squared = temp_w ** 2
                wind_speed_squared = wind_w ** 2
                
                # Wind decomposition
                wind_bearing_rad = np.radians(wind_bearing_w)
                wind_n_s = wind_w * np.cos(wind_bearing_rad)
                wind_e_w = wind_w * np.sin(wind_bearing_rad)
                
                # Binary indicators
                low_pressure = 1 if pressure_w < 1000 else 0
                high_pressure = 1 if pressure_w > 1020 else 0
                is_winter = 1 if month_w in [12, 1, 2] else 0
                is_summer = 1 if month_w in [6, 7, 8] else 0
                is_day = 1 if 6 <= hour_w <= 18 else 0
                
                visibility_humidity_ratio = visibility_w / max(humidity_w, 1)
                
                # Build feature array
                feature_values = [
                    temp_w, apparent_w, humidity_w, wind_w, wind_bearing_w, visibility_w, cloud_w, pressure_w,
                    year_w, month_w, day_w, hour_w, month_sin, month_cos, hour_sin, hour_cos, precip_encoded,
                    temp_humidity_interaction, feels_like_diff, temp_squared, wind_speed_squared,
                    wind_n_s, wind_e_w, pressure_temp_interaction, low_pressure, high_pressure,
                    visibility_humidity_ratio, cloud_humidity_interaction, is_winter, is_summer, is_day
                ]
                
                input_df = pd.DataFrame([feature_values], columns=metadata['feature_names'])
                prediction_idx = model.predict(input_df)[0]
                
                classes = metadata['target_classes']
                probs = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else [1.0 if i == prediction_idx else 0.0 for i in range(len(classes))]
                
                # Convert numeric prediction to class name
                predicted_class = classes[int(prediction_idx)] if isinstance(prediction_idx, (int, np.integer)) else prediction_idx
                
                st.success("✅ **Full Feature Engineering Pipeline Applied!** (31 features)")
                
                st.markdown("### 🎲 Weather Classification Results")
                for cls, prob in zip(classes, probs):
                    st.progress(prob, text=f"**{cls}**: {prob*100:.1f}%")
                
                st.markdown('<div class="prediction-result">☁️ Predicted: {}</div>'.format(predicted_class), unsafe_allow_html=True)
                st.metric("Model Confidence", f"{max(probs)*100:.1f}%")
            
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
    <p>10 Models | Prediction & Forecasting & Anomaly Detection | Interactive Interface</p>
    <p style='font-size: 12px;'>Built with Streamlit | Academic Project 2025</p>
</div>
""", unsafe_allow_html=True)
