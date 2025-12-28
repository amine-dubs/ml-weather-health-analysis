# Wine Anomaly Detection - Separate Class Evaluation Method

## Overview

This experiment uses a **specialized evaluation technique** for imbalanced anomaly detection. Instead of evaluating on the full test set, we **separately predict on positive and negative classes** to better understand algorithm performance on each class independently.

This method was recommended by your professor for handling imbalanced data scenarios.

---

## Evaluation Methodology - Key Difference

### Traditional Approach (Experiment 1)
```
1. Train on imbalanced data (3,428 normal + 1,119 anomaly)
2. Predict on combined test set (1,470 normal + 480 anomaly)
3. Calculate overall metrics
```

### Separate Class Evaluation (Experiment 2) ✨
```
1. Train on imbalanced data (3,428 normal + 1,119 anomaly)
2. Separate test set into TWO subsets:
   - X_test_negative: ONLY normal samples (1,470 white wines)
   - X_test_positive: ONLY anomaly samples (480 red wines)
3. Predict separately:
   - Predict on X_test_negative → measure Specificity (TNR)
   - Predict on X_test_positive → measure Sensitivity (TPR)
4. Calculate metrics for each class independently
```

---

## Why This Method is Better

### Advantages

1. **Clear Class-Specific Performance**
   - See exactly how well the model detects normal samples
   - See exactly how well the model detects anomalies
   - No confusion from mixed predictions

2. **Reveals True Trade-offs**
   - High TNR but low TPR? → Model is too conservative
   - Low TNR but high TPR? → Model is too aggressive
   - High both? → Model is excellent!

3. **Better for Imbalanced Data**
   - Traditional accuracy can be misleading (e.g., 75% by always predicting majority)
   - Separate evaluation shows true performance on minority class

4. **Clinically Meaningful Metrics**
   - TPR (True Positive Rate) = Sensitivity = Recall
   - TNR (True Negative Rate) = Specificity
   - These are standard in medical/scientific applications

---

## Dataset & Setup

### Same as Experiment 1
- **Training**: 4,547 samples (3,428 white + 1,119 red)
- **Test**: 1,950 samples (1,470 white + 480 red)
- **Contamination**: 0.246 (24.6%)

### Test Set Split for Evaluation
- **X_test_negative**: 1,470 white wines (normal class only)
- **X_test_positive**: 480 red wines (anomaly class only)

---

## Results Summary

### Performance Comparison - ALL Metrics

| Algorithm | Accuracy | Precision | Recall (TPR) | Specificity (TNR) | F1-Score |
|-----------|----------|-----------|--------------|-------------------|----------|
| **Elliptic Envelope** ⭐ | **0.9595** | **0.9067** | **0.9313** | **0.9687** | **0.9188** |
| Isolation Forest | 0.7344 | 0.4583 | 0.4354 | 0.8320 | 0.4466 |
| One-Class SVM | 0.6841 | 0.3583 | 0.3583 | 0.7905 | 0.3583 |
| Local Outlier Factor | 0.6369 | 0.2500 | 0.2375 | 0.7673 | 0.2436 |
| DBSCAN | 0.2462 | 0.2462 | 1.0000 | 0.0000 | 0.3951 |

### Detailed Performance Breakdown

| Algorithm | True Positives (TP) | False Negatives (FN) | True Negatives (TN) | False Positives (FP) |
|-----------|---------------------|----------------------|---------------------|----------------------|
| **Elliptic Envelope** | **447** | **33** | **1,424** | **46** |
| Isolation Forest | 209 | 271 | 1,223 | 247 |
| One-Class SVM | 172 | 308 | 1,162 | 308 |
| Local Outlier Factor | 114 | 366 | 1,128 | 342 |
| DBSCAN | 480 | 0 | 0 | 1,470 |

### Rate Comparison

| Algorithm | TPR (Sensitivity) | FNR (Miss Rate) | TNR (Specificity) | FPR (False Alarm) |
|-----------|-------------------|-----------------|-------------------|-------------------|
| **Elliptic Envelope** | **93.13%** | **6.88%** | **96.87%** | **3.13%** |
| Isolation Forest | 43.54% | 56.46% | 83.20% | 16.80% |
| One-Class SVM | 35.83% | 64.17% | 79.05% | 20.95% |
| Local Outlier Factor | 23.75% | 76.25% | 76.73% | 23.27% |
| DBSCAN | 100% | 0% | 0% | 100% |

---

## Detailed Algorithm Analysis

### 🏆 1. Elliptic Envelope - Near Perfect Performance

#### Prediction on X_test_negative (White Wines Only)
- **True Negatives**: 1,424 / 1,470 = **96.87% TNR** ✨
- **False Positives**: 46 / 1,470 = 3.13% FPR
- **Interpretation**: Correctly identifies 97% of white wines as normal

#### Prediction on X_test_positive (Red Wines Only)
- **True Positives**: 447 / 480 = **93.13% TPR** ✨
- **False Negatives**: 33 / 480 = 6.88% FNR
- **Interpretation**: Correctly detects 93% of red wines as anomalies

#### Overall Confusion Matrix
```
                    Predicted Normal  Predicted Anomaly
Actual Normal (White)     1,424            46          TNR = 96.87%
Actual Anomaly (Red)         33           447          TPR = 93.13%
```

#### Balanced Accuracy
- **Balanced Acc = (TPR + TNR) / 2 = (93.13% + 96.87%) / 2 = 95.00%**
- This accounts for class imbalance better than standard accuracy

#### Why It Excels
✅ Excellent on BOTH classes
✅ Only 79 total errors out of 1,950 samples
✅ Very low false positive rate (3.13%)
✅ Very low false negative rate (6.88%)
✅ Production-ready performance

---

### 2. Isolation Forest - Conservative Detector

#### Prediction on X_test_negative (White Wines)
- **TNR**: 1,223 / 1,470 = **83.20%**
- **FPR**: 247 / 1,470 = 16.80%
- Good at identifying normal samples, but flags 247 white wines as anomalies

#### Prediction on X_test_positive (Red Wines)
- **TPR**: 209 / 480 = **43.54%**
- **FNR**: 271 / 480 = 56.46%
- **Critical Issue**: Misses 56% of red wines!

#### Trade-off Analysis
- **Conservative**: Prefers to predict "normal" to avoid false alarms
- **Result**: Good specificity (83%) but poor sensitivity (44%)
- **Use Case**: When false positives are more costly than false negatives

---

### 3. DBSCAN - Complete Failure Mode

#### Prediction on X_test_negative (White Wines)
- **TNR**: 0 / 1,470 = **0%** ⚠️
- **FPR**: 1,470 / 1,470 = **100%**
- Flags ALL white wines as anomalies!

#### Prediction on X_test_positive (Red Wines)
- **TPR**: 480 / 480 = **100%** ✅
- **FNR**: 0 / 480 = 0%
- Catches all red wines, but...

#### Why This Happened
- eps=0.5 parameter is too strict for this dataset
- Treats almost everything as sparse outliers
- **Useless in practice**: Can't distinguish between classes
- **Lesson**: DBSCAN needs careful hyperparameter tuning

---

### 4. One-Class SVM - Balanced Poor Performance

#### Prediction on X_test_negative (White Wines)
- **TNR**: 1,162 / 1,470 = **79.05%**
- **FPR**: 308 / 1,470 = 20.95%

#### Prediction on X_test_positive (Red Wines)
- **TPR**: 172 / 480 = **35.83%**
- **FNR**: 308 / 480 = 64.17%

#### Symmetrical Errors
- Exactly 308 false positives = 308 false negatives
- Poor on both classes equally
- **Use Case**: Not recommended for this problem

---

### 5. Local Outlier Factor - Worst Recall

#### Prediction on X_test_negative (White Wines)
- **TNR**: 1,128 / 1,470 = **76.73%**
- **FPR**: 342 / 1,470 = 23.27%

#### Prediction on X_test_positive (Red Wines)
- **TPR**: 114 / 480 = **23.75%** ⚠️
- **FNR**: 366 / 480 = **76.25%**
- **Critical**: Misses 366 out of 480 red wines!

#### Why It's Worst
- Density-based local neighborhoods don't capture global separation
- Only detects 24% of anomalies
- High false negative rate = dangerous for critical applications

---

## Visualization Analysis

### 1. Separate Class Evaluation Plot
**File:** `wine_separate_class_evaluation.png`

Each subplot shows:
- **Blue circles**: True Negatives (TN) - correctly identified white wines
- **Orange X's**: False Positives (FP) - white wines wrongly flagged as red
- **Red triangles**: True Positives (TP) - correctly detected red wines
- **Light green squares**: False Negatives (FN) - red wines missed

**Visual Insights:**
- **Elliptic Envelope**: Mostly blue and red, minimal orange/green errors
- **DBSCAN**: All orange and red (no blue), showing it flags everything
- **LOF**: Lots of green squares, showing poor anomaly detection

---

### 2. Performance Metrics Comparison
**File:** `wine_separate_evaluation_metrics.png`

Four charts:

#### Chart 1: TPR vs TNR
- **Elliptic Envelope**: Both bars near 100% - excellent balance
- **DBSCAN**: TPR = 100%, TNR = 0% - useless trade-off
- **Others**: Higher TNR than TPR - conservative

#### Chart 2: FPR vs FNR
- **Elliptic Envelope**: Both bars near 0% - minimal errors
- **LOF**: High FNR bar - misses many anomalies
- **DBSCAN**: FPR = 100% - catastrophic false alarms

#### Chart 3: F1-Score
- **Elliptic Envelope**: 0.919 - dominant
- **Others**: < 0.5 - poor to moderate

#### Chart 4: Accuracy vs Balanced Accuracy
- Shows why balanced accuracy matters
- DBSCAN has low accuracy but 50% balanced accuracy (random guessing level)

---

## Comparison with Standard Evaluation

### Metrics That Are the Same
- **Accuracy**: 0.9595 (same in both experiments)
- **Precision**: 0.9067 (same in both experiments)
- **Recall**: 0.9313 (same in both experiments)
- **F1-Score**: 0.9188 (same in both experiments)

### NEW Metrics from Separate Evaluation
- **TNR (Specificity)**: 0.9687 - how well we detect normal samples
- **FPR**: 0.0313 - false alarm rate
- **FNR**: 0.0688 - miss rate
- **Balanced Accuracy**: 0.95 - accounts for imbalance

---

## Key Insights from Separate Evaluation

### 1. Understanding Trade-offs
| Algorithm | Strategy | TPR | TNR | Best For |
|-----------|----------|-----|-----|----------|
| Elliptic Envelope | **Balanced** | 93% | 97% | ✅ Production use |
| Isolation Forest | Conservative | 44% | 83% | When FP costly |
| DBSCAN | Aggressive | 100% | 0% | ❌ Not usable |
| One-Class SVM | Uncertain | 36% | 79% | ❌ Neither class well |
| LOF | Very Conservative | 24% | 77% | ❌ Misses too many |

### 2. Clinical/Scientific Interpretation
- **High TPR + High TNR** = Excellent diagnostic tool (Elliptic Envelope)
- **High TPR + Low TNR** = Over-diagnosis (DBSCAN)
- **Low TPR + High TNR** = Under-diagnosis (LOF, Isolation Forest)

### 3. Real-World Applications
In production:
- **Medical screening**: Need high TPR (catch all diseases)
- **Fraud detection**: Need balanced TPR/TNR
- **Quality control**: Depends on cost of false alarms vs defects

For this wine dataset, **Elliptic Envelope** is ideal because:
- Won't reject good wines (low FPR = 3%)
- Won't accept wrong wine type (low FNR = 7%)
- Reliable for both classes

---

## Advantages of This Evaluation Method

### ✅ Pros
1. **Transparent Performance**: See exactly what happens on each class
2. **Better for Imbalanced Data**: Accounts for class distribution
3. **Clinically Meaningful**: TPR/TNR used in medical/scientific fields
4. **Reveals Algorithm Behavior**: Conservative vs aggressive strategies
5. **Guides Threshold Tuning**: Understand trade-offs for adjustments

### 🤔 Considerations
1. **More Complex**: Need to track more metrics
2. **Requires Careful Interpretation**: TPR/TNR less familiar than accuracy
3. **Same Overall Results**: Final accuracy/F1 identical to standard method

---

## Best Performing Algorithm Summary

### 🥇 Elliptic Envelope - CHAMPION

**Separate Class Results:**
- ✅ **TPR**: 93.13% (447 / 480 red wines detected)
- ✅ **TNR**: 96.87% (1,424 / 1,470 white wines identified)
- ✅ **FPR**: 3.13% (only 46 false alarms)
- ✅ **FNR**: 6.88% (only 33 missed)
- ✅ **Balanced Accuracy**: 95.00%

**Why It Wins:**
- Near-perfect performance on BOTH classes
- Minimal errors in both directions
- Gaussian assumption matches wine chemistry
- Production-ready without further tuning

---

## Conclusions

### Main Takeaways

1. **Separate class evaluation reveals true performance**
   - Standard metrics can hide class-specific issues
   - TPR/TNR show the complete picture

2. **Elliptic Envelope is exceptional**
   - 93% TPR + 97% TNR = nearly perfect
   - Works equally well on both classes
   - Only 4% error rate overall

3. **Other algorithms have clear weaknesses**
   - DBSCAN: 100% TPR but 0% TNR (useless)
   - LOF: 24% TPR (misses 3 out of 4 anomalies)
   - Isolation Forest: 44% TPR (conservative, misses half)

4. **This evaluation method is valuable**
   - Especially important for imbalanced data
   - Required for medical/critical applications
   - Helps understand algorithm behavior

---

## Files Generated

1. **wine_separate_class_evaluation.png** - Visualizations with TP/FP/TN/FN color-coded
2. **wine_separate_evaluation_metrics.png** - TPR vs TNR, FPR vs FNR comparisons
3. **wine_separate_evaluation_results.csv** - Complete numerical results
4. **wine_separate_class_evaluation.py** - Implementation script

All files located in: `separate_class_evaluation_results/` folder

---

## Practical Recommendations

### For Your Professor's Assignment ✅

This separate class evaluation method demonstrates:
1. ✅ Understanding of imbalanced data challenges
2. ✅ Proper handling of minority class evaluation
3. ✅ Knowledge of TPR/TNR/FPR/FNR metrics
4. ✅ Ability to interpret trade-offs between classes
5. ✅ Critical thinking about algorithm behavior

### For Real Projects

Use this method when:
- Working with imbalanced datasets (< 30% minority class)
- Costs of FP vs FN are different
- Need to report sensitivity and specificity
- Medical, security, or quality control applications
- Want to understand algorithm behavior deeply

Don't need this method when:
- Classes are balanced (50/50 split)
- Only care about overall accuracy
- Quick prototyping phase
- Standard metrics (precision/recall) are sufficient
