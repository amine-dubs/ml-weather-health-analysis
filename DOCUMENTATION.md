# ML TPs Documentation - Semester Overview

**Academic Year:** 2024/2025  
**Last Updated:** November 4, 2025

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset 1: Weather Temperature Prediction](#dataset-1-weather-temperature-prediction)
3. [Dataset 2: Heart Disease Classification](#dataset-2-heart-disease-classification)
4. [Dashboard Usage](#dashboard-usage)
5. [Key Findings & Results](#key-findings--results)

---

## 🎯 Project Overview

This repository contains comprehensive machine learning practical work (TPs) covering:
- **Regression Analysis**: Weather temperature prediction
- **Classification Analysis**: Heart disease detection
- **Interactive Dashboard**: Streamlit-based visualization and prediction interface

### Technologies Used
- **Python 3.x**
- **Scikit-learn**: Machine learning models and preprocessing
- **XGBoost, LightGBM**: Gradient boosting implementations
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **Streamlit**: Interactive dashboard
- **Joblib**: Model persistence

---

## 🌡️ Dataset 1: Weather Temperature Prediction

### Objective
Predict maximum daily temperature using meteorological features.

### Dataset Information
- **Task Type**: Regression
- **Target Variable**: Maximum temperature (°C)
- **Features**: Multiple weather-related variables

### Methods Applied

#### 1. Data Preprocessing
- Data cleaning and validation
- Feature engineering
- Train-test split
- Normalization techniques (StandardScaler, MinMaxScaler)

#### 2. Models Evaluated
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)

#### 3. Optimization Techniques
- **Hyperparameter Tuning**: GridSearchCV on top models
- **XGBoost Deep Dive**: Comprehensive parameter exploration
- **Advanced Methods**: Custom ensemble approaches
- **Feature Selection**: Identifying most important predictors

#### 4. Key Results
*[To be updated after running all experiments]*

### Files Structure
```
regression_results/
├── data_analysis/          # EDA visualizations
├── model_comparison/       # Model performance plots
├── tuning_results/         # GridSearch results
├── xgboost_optimization/   # XGBoost-specific analysis
├── advanced_results/       # Advanced techniques
└── ensemble_results/       # Ensemble methods
```

---

## ❤️ Dataset 2: Heart Disease Classification

### Objective
Binary classification to predict presence of heart disease based on clinical indicators.

### Dataset Information
- **Task Type**: Binary Classification
- **Target Variable**: Heart disease (0: No, 1: Yes)
- **Samples**: 1,190 patients
- **Features**: 11 clinical indicators
- **Class Distribution**: 
  - Positive cases: 629 (52.86%)
  - Negative cases: 561 (47.14%)

### Clinical Features
1. **age**: Patient age (years)
2. **sex**: Gender (0: Female, 1: Male)
3. **chest pain type**: Type of chest pain (0-3)
4. **resting bp s**: Resting blood pressure (mm Hg)
5. **cholesterol**: Serum cholesterol (mg/dl)
6. **fasting blood sugar**: Fasting blood sugar > 120 mg/dl (1: True, 0: False)
7. **resting ecg**: Resting ECG results (0-2)
8. **max heart rate**: Maximum heart rate achieved
9. **exercise angina**: Exercise-induced angina (1: Yes, 0: No)
10. **oldpeak**: ST depression induced by exercise
11. **ST slope**: Slope of peak exercise ST segment (0-2)

### Methods Applied

#### 1. PRIMARY METRIC: ROC-AUC Score
**Why ROC-AUC over Accuracy?**
- Medical diagnosis requires careful consideration of false negatives/positives
- ROC-AUC better handles class imbalance
- Provides probability-based predictions (useful for risk assessment)
- Standard metric in medical ML applications

#### 2. Models Evaluated (13 Base Models)

**Tree-Based Models:**
- XGBoost
- RandomForest
- GradientBoosting
- ExtraTrees
- LightGBM
- DecisionTree

**Linear Models:**
- LogisticRegression
- LinearSVC

**Distance-Based:**
- K-Nearest Neighbors (KNN)

**Support Vector Machines (5 Variants):**
- SVC_RBF (Radial Basis Function kernel)
- SVC_Poly (Polynomial kernel, degree=3)
- SVC_Sigmoid (Sigmoid kernel)
- SVC_Linear (Linear kernel)
- LinearSVC (Linear SVM, optimized for large datasets)

**Total Configurations:** 13 models × 3 normalizations (None, StandardScaler, MinMaxScaler) = **39 configurations**

#### 3. GridSearch Optimization

**Enhanced GridSearch on 4 Models:**

**a) XGBoost - Comprehensive Search**
```python
Parameters:
- n_estimators: [50, 100, 150, 200]
- max_depth: [3, 5, 7, 9]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- subsample: [0.7, 0.8, 0.9, 1.0]
- colsample_bytree: [0.7, 0.8, 0.9, 1.0]
- gamma: [0, 0.1, 0.2]
Total combinations: 3,072
Scoring: ROC-AUC
```

**b) KNN GridSearch (NEW)**
```python
Parameters:
- n_neighbors: [3, 5, 7, 9, 11, 15, 20]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan', 'minkowski']
- p: [1, 2]
Total combinations: ~84
Scoring: ROC-AUC
```

**c) RandomForest - Enhanced**
```python
Parameters:
- n_estimators: [50, 100, 150, 200]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
Scoring: ROC-AUC
```

**d) GradientBoosting - Enhanced**
```python
Parameters:
- n_estimators: [50, 100, 150, 200]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- max_depth: [3, 5, 7]
- subsample: [0.7, 0.8, 0.9, 1.0]
Scoring: ROC-AUC
```

#### 4. Ensemble Methods
- **Voting Classifier**: Hard and soft voting
- **Stacking Classifier**: Meta-learner approach
- Both with/without normalization

#### 5. Model Persistence for Production

**Best Model Saved:**
- `best_model.joblib`: Trained model (selected by highest ROC-AUC)
- `scaler.joblib`: Preprocessing scaler (if normalization used)
- `model_metadata.json`: Complete model information

**Metadata Includes:**
```json
{
  "model_name": "Best model name",
  "normalization": "StandardScaler/MinMaxScaler/None",
  "roc_auc": 0.xxxx,
  "accuracy": 0.xxxx,
  "feature_names": ["age", "sex", ...],
  "target_name": "target"
}
```

### Key Results

#### Performance Metrics (Preliminary - Running)
*Individual Models (Top 10 by ROC-AUC):*

| Rank | Model | Normalization | ROC-AUC | Accuracy | F1-Score |
|------|-------|---------------|---------|----------|----------|
| 1 | ExtraTrees | None/Standard/MinMax | 0.9782 | 0.9076 | 0.9106 |
| 2 | XGBoost | None/Standard/MinMax | 0.9717 | 0.9370 | 0.9398 |
| 3 | RandomForest | None/MinMax | 0.9708 | 0.9244 | 0.9286 |
| ... | ... | ... | ... | ... | ... |

*GridSearch Results:* (In Progress)
- XGBoost: Testing 3,072 combinations
- KNN: Testing ~84 combinations
- RandomForest: Enhanced search
- GradientBoosting: Enhanced search

#### Normalization Impact
- **Tree-based models**: Minimal impact (XGBoost, RandomForest, ExtraTrees)
- **SVC models**: Significant improvement with StandardScaler
  - SVC_RBF: 0.8112 → 0.9352 ROC-AUC with normalization
  - SVC_Poly: 0.8060 → 0.9274 ROC-AUC with normalization
- **KNN**: Large improvement with normalization
  - 0.7872 → 0.9179 ROC-AUC with StandardScaler

#### Key Insights
1. **ExtraTrees achieved highest ROC-AUC** (0.9782) - consistent across normalizations
2. **XGBoost best for accuracy** (0.9370) with high ROC-AUC (0.9717)
3. **SVM kernels comparison**: RBF > Polynomial > Sigmoid when normalized
4. **LinearSVC efficient alternative**: 0.9051 ROC-AUC with 0.06s training time
5. **Normalization essential** for distance-based and linear models

### Files Structure
```
Dataset2/
├── Dataset2.csv                                    # Raw data
├── heart_disease_classification.py                # Main analysis script
└── classification_results/
    ├── individual_models_comparison.csv           # All individual models
    ├── gridsearch_results.csv                     # GridSearch results
    ├── ensemble_methods_comparison.csv            # Ensemble comparisons
    ├── final_comparison_all_methods.csv           # Combined results
    ├── models/
    │   ├── best_model.joblib                      # Saved best model
    │   ├── scaler.joblib                          # Saved scaler
    │   └── model_metadata.json                    # Model metadata
    └── plots/
        ├── 01_top_models_comparison.png           # Top models visualization
        ├── 02_normalization_impact.png            # Normalization analysis
        ├── 03_ensemble_gridsearch_analysis.png    # Advanced methods
        └── 04_best_model_details.png              # Best model details
```

---

## 📊 Dashboard Usage

### Installation

```bash
# Install required packages
pip install streamlit pandas numpy matplotlib seaborn pillow scikit-learn xgboost lightgbm joblib
```

### Running the Dashboard

```bash
# Navigate to project directory
cd "ML tp0"

# Run Streamlit dashboard
streamlit run dashboard_v2.py
```

### Dashboard Features

#### Section 1: Weather Temperature Prediction
**Tabs:**
1. **EDA**: Exploratory data analysis and visualizations
2. **Models**: Individual model comparisons
3. **Tuning**: Hyperparameter tuning results
4. **XGBoost**: Deep dive into XGBoost optimization
5. **Advanced**: Advanced techniques and feature engineering
6. **Ensemble**: Ensemble method comparisons

#### Section 2: Heart Disease Classification
**Tabs:**
1. **📊 All Results**: Complete results table with all 47+ configurations
2. **🏆 Top Models**: Top 10 models ranked by ROC-AUC
3. **🔧 Normalization**: Impact analysis of different normalization techniques
4. **📈 Best Model**: Detailed analysis of best performing model
5. **🔮 Make Predictions**: Interactive prediction interface

### Making Predictions (Heart Disease)

The dashboard includes a **live prediction interface**:

1. Navigate to **Heart Disease Classification** section
2. Select **"🔮 Make Predictions"** tab
3. Enter patient clinical data in the form
4. Click **"🔍 Predict Heart Disease"**
5. View prediction with confidence scores

**Features:**
- Uses the best model (selected by ROC-AUC)
- Automatic preprocessing (applies saved scaler)
- Probability breakdown for both classes
- Medical disclaimer for ethical use

**Example Input Fields:**
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Cholesterol
- Fasting Blood Sugar, Resting ECG
- Max Heart Rate, Exercise Angina
- Oldpeak, ST Slope

---

## 🔬 Key Findings & Results

### Overall Methodology

**Systematic Approach:**
1. ✅ Comprehensive data preprocessing
2. ✅ Multiple model architectures tested
3. ✅ Extensive hyperparameter tuning
4. ✅ Proper train-test validation
5. ✅ Cross-validation for robustness
6. ✅ Model persistence for deployment

### Best Practices Demonstrated

#### For Regression (Weather)
- Feature engineering importance
- Model selection based on metrics (MAE, RMSE, R²)
- Ensemble methods for improved performance
- XGBoost tuning for optimal results

#### For Classification (Heart Disease)
- **ROC-AUC as primary metric** for medical applications
- Multiple SVM kernel comparisons
- Normalization impact analysis
- Model deployment preparation (saved artifacts)
- Ethical considerations (medical disclaimer)

### Technical Achievements

1. **Model Diversity**: 13+ different model architectures tested
2. **Systematic Tuning**: GridSearchCV with comprehensive parameter grids
3. **Production Ready**: Model persistence with metadata
4. **Interactive Dashboard**: User-friendly interface for exploration and prediction
5. **Documentation**: Complete analysis documentation

### Future Work & Improvements

**Potential Enhancements:**
1. Cross-dataset validation
2. Deep learning approaches (Neural Networks)
3. Feature importance analysis and selection
4. Model interpretability (SHAP, LIME)
5. Real-time prediction API
6. Model monitoring and retraining pipeline

---

## 📖 References

### Libraries & Tools
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Streamlit: https://streamlit.io/

### Datasets
- Weather Dataset: [Source to be added]
- Heart Disease Dataset: UCI Machine Learning Repository

### Methodologies
- ROC-AUC for medical classification: Standard practice in healthcare ML
- GridSearchCV: Exhaustive hyperparameter search
- Ensemble methods: Combining multiple models for better performance

---

## 👨‍💻 Repository Structure

```
ML tp0/
├── dashboard_v2.py                    # Main Streamlit dashboard
├── DOCUMENTATION.md                   # This file
├── Dataset1.csv                       # Weather data
├── Dataset2/
│   ├── Dataset2.csv                  # Heart disease data
│   ├── heart_disease_classification.py
│   └── classification_results/       # All results and saved models
├── regression_results/               # Weather prediction results
└── [other analysis scripts]
```

---

## ⚠️ Important Notes

### Medical Disclaimer
The heart disease classification model is developed for **educational purposes only**. 

**DO NOT use for:**
- Actual medical diagnosis
- Clinical decision-making
- Patient treatment planning

**Always consult qualified healthcare professionals** for medical advice and diagnosis.

### Data Privacy
- No real patient data used in this project
- Dataset is publicly available for research
- Follow appropriate data protection guidelines for any extensions

---

## 📝 Version History

- **v2.0** (Nov 4, 2025): 
  - Enhanced classification with ROC-AUC primary metric
  - Added 5 SVM kernel variants
  - Comprehensive GridSearch (3,072 combinations for XGBoost)
  - NEW: KNN GridSearch
  - Model persistence for production
  - Interactive prediction interface

- **v1.0** (Initial): 
  - Basic model comparisons
  - Dashboard structure
  - Initial documentation

---

**End of Documentation**

*Last Updated: November 4, 2025*
*Status: In Progress - GridSearch Running*
