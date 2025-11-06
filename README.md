# 🎯 Comprehensive Machine Learning Experiments

A complete machine learning analysis spanning **4 major experimental domains** with production-ready models, interactive dashboard, and comprehensive documentation.

## 📊 Overview

This project demonstrates advanced ML techniques across regression and classification tasks using weather (96,453 samples) and heart disease (1,190 samples) datasets.

### 🏆 Key Achievements

- **4 Production Models** saved with complete metadata
- **18,400+ CV Fits** across comprehensive GridSearch experiments
- **Interactive Dashboard** with real-time predictions (Streamlit)
- **39-page LaTeX Report** with detailed methodology and visualizations

## 🎯 Experimental Domains

### 1️⃣ Temperature Regression (Ensemble)
- **Best Model:** Stacking Ensemble (6 base models + Ridge meta-learner)
- **Performance:** R² = 0.7889, MAE = 3.47°C
- **Features:** 8 weather variables
- **Innovation:** Improved from R²=0.7667 (single model) to 0.7889 (ensemble)

### 2️⃣ Heart Disease Classification
- **Best Model (AUC):** ExtraTrees - ROC-AUC = 0.9782
- **Best Model (Accuracy):** XGBoost - 93.70%
- **Features:** 11 clinical measurements
- **Innovation:** ROC-AUC prioritization for medical ML applications

### 3️⃣ Multi-Output Regression
- **Best Model:** XGBoost MultiOutputRegressor
- **Performance:** 
  - Pressure: R² = 0.9823
  - Humidity: R² = 0.8741
- **Innovation:** Simultaneous prediction of 2 targets in single forward pass

### 4️⃣ Weather Classification (4-class)
- **Best Model:** Random Forest
- **Performance:** ROC-AUC = 0.8493, Accuracy = 64.74%
- **Features:** 31 engineered features from 11 raw variables
- **Classes:** Clear, Mostly Cloudy, Overcast, Partly Cloudy

## 🚀 Features

### Advanced Techniques
- ✅ **Multi-Output Learning** - Single model predicting multiple targets
- ✅ **Ensemble Methods** - Voting & Stacking with meta-learners
- ✅ **Feature Engineering** - 31 engineered weather features (cyclical encoding, interactions, polynomials)
- ✅ **Comprehensive GridSearch** - 18,400+ parameter combinations tested
- ✅ **SVM Kernel Analysis** - 5 variants across 3 normalizations
- ✅ **Distribution-Preserving Imputation** - Maintaining statistical integrity

### Production Ready
- 📦 **Saved Models** - 4 models with joblib + metadata JSON
- 🎨 **Interactive Dashboard** - Streamlit app with prediction interfaces
- 📊 **35+ Visualizations** - Performance comparisons, feature importance, distributions
- 📄 **Complete Documentation** - 39-page LaTeX report + Markdown

## 🛠️ Technology Stack

- **ML Libraries:** scikit-learn, XGBoost, LightGBM
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Dashboard:** Streamlit
- **Documentation:** LaTeX, Markdown

## 📁 Project Structure

```
ML tp0/
├── dashboard_v2.py                    # Interactive Streamlit dashboard
├── ml_pipeline_knn.py                 # Temperature regression pipeline
├── ml_analysis.py                     # Heart disease classification
├── multi_output_regression.py         # Multi-output experiments
├── encoding_comparison.py             # Weather classification
├── Dataset1.csv                       # Weather data (96,453 samples)
├── Dataset2/                          # Heart disease data
│   └── classification_results/
│       └── models/                    # Saved heart disease models
├── ensemble_results/models/           # Temperature ensemble models
├── multi_output_results/models/       # Multi-output models
├── weather_classification_models/     # Weather classification models
└── Overleaf_Upload/
    └── ML_Experiments_COMPLETE.tex    # 39-page LaTeX report
```

## 🎮 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard_v2.py
```

The dashboard will open at `http://localhost:8502`

### 3. Explore Experiments

Navigate through 6 sections:
- 🌡️ Temperature Regression
- ❤️ Heart Disease Classification
- 🎯 Multi-Output Regression
- ☁️ Weather Classification
- 🤝 Ensemble Methods Comparison
- 🔮 Model Predictions (Interactive)

## 🔮 Model Predictions

The dashboard includes interactive prediction interfaces for all 4 production models:

1. **Heart Disease Risk Assessment** - 11 clinical features → Risk score + confidence
2. **Temperature Prediction** - 8 weather variables → Temperature ± uncertainty
3. **Multi-Output Prediction** - Simultaneous Pressure & Humidity prediction
4. **Weather Classification** - 4-class probability distribution

## 📊 Results Summary

| Task | Best Model | Primary Metric | Score | Saved |
|------|-----------|----------------|-------|-------|
| Temperature | Stacking Ensemble | R² | 0.7889 | ✅ |
| Heart Disease (AUC) | ExtraTrees | ROC-AUC | 0.9782 | ✅ |
| Heart Disease (Acc) | XGBoost | Accuracy | 93.70% | - |
| Pressure (Multi) | XGBoost Multi | R² | 0.9823 | ✅ |
| Humidity (Multi) | XGBoost Multi | R² | 0.8741 | ✅ |
| Weather Summary | Random Forest | ROC-AUC | 0.8493 | ✅ |

## 🧪 Experimental Highlights

### Heart Disease Classification
- **47 model configurations** tested (13 models × 3 normalizations + GridSearch + Ensembles)
- **ROC-AUC prioritization** for medical diagnosis (better than accuracy)
- **SVM kernel comparison:** Quantified normalization impact (+15%)
- **Production deployment:** Complete metadata for clinical integration

### Multi-Output Regression
- **1,080 CV fits** in XGBoost GridSearch alone
- **Simultaneous prediction** of correlated weather variables
- **Computational efficiency:** Single model vs separate models
- **Distribution preservation:** Handling 1,288 zero-pressure anomalies

### Weather Classification
- **31 engineered features:**
  - Cyclical temporal encoding (Month_sin, Hour_cos, etc.)
  - Interaction terms (Temp×Humidity, Pressure×Temp)
  - Polynomial features (Temp², WindSpeed²)
  - Domain indicators (Low_Pressure, Is_Winter, Is_Day)
- **4-class imbalance handling**
- **Random Forest superiority** without normalization

## 📈 Key Insights

1. **Ensemble Stacking** improved temperature R² by +2.2% over best single model
2. **ROC-AUC** is superior to accuracy for medical ML (98% vs 94%)
3. **Multi-output learning** enables efficient correlated prediction
4. **Feature engineering** is critical (31 features from 11 raw variables)
5. **Normalization matters** for distance-based models (+15% for SVM)

## 📚 Documentation

- **LaTeX Report:** `Overleaf_Upload/ML_Experiments_COMPLETE.tex` (39 pages)
- **Compiled PDF:** Complete with visualizations and methodology
- **Dashboard:** Interactive exploration of all experiments

## 🤝 Contributing

This is an academic project demonstrating comprehensive ML experimentation. Feel free to:
- Explore the code and methodology
- Adapt techniques for your own projects
- Provide feedback or suggestions

## 📄 License

Academic project - 2025/2026

## 👨‍💻 Author

Machine Learning Practical Work - Complete Experimental Suite

---

**⭐ Features:**
- 4 Production Models | 100+ Configurations | 18,400+ CV Fits
- Interactive Dashboard | 35+ Visualizations | 39-Page Report
- Multi-Output Learning | Advanced Feature Engineering | Ensemble Methods
