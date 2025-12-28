# Employee Attrition Anomaly Detection Analysis

## Dataset Overview

### Class Distribution
The dataset has **2 balanced classes**:
- **Stayed** (Normal): 52.45% in training, 52.81% in test
- **Left** (Anomaly): 47.55% in training, 47.19% in test

This is a relatively balanced dataset, making it suitable for anomaly detection analysis.

### Data Characteristics
- **Training samples**: 59,598 employees
- **Test samples**: 14,900 employees
- **Features**: 22 attributes (after removing Employee ID)
- **Target**: Attrition (Stayed vs Left)

## Algorithms Tested

Five different anomaly detection algorithms were evaluated:

1. **Isolation Forest** - Tree-based ensemble method
2. **One-Class SVM** - Support Vector Machine for outlier detection
3. **Local Outlier Factor (LOF)** - Density-based local outlier detection
4. **Elliptic Envelope** - Gaussian distribution-based method
5. **DBSCAN** - Density-based clustering for outlier detection

## Performance Results

### Overall Performance Summary

| Algorithm | Accuracy | Precision | Recall | F1-Score | Anomalies Detected |
|-----------|----------|-----------|--------|----------|-------------------|
| **DBSCAN** | 0.4468 | 0.4288 | **0.5185** | **0.4694** | 8,502 |
| **Elliptic Envelope** | **0.4896** | 0.4357 | 0.2763 | 0.3382 | 4,459 |
| **Local Outlier Factor** | 0.4825 | 0.4277 | 0.2856 | 0.3425 | 4,695 |
| **One-Class SVM** | 0.4795 | 0.4182 | 0.2634 | 0.3232 | 4,428 |
| **Isolation Forest** | 0.4588 | 0.3839 | 0.2426 | 0.2973 | 4,444 |

### Best Performing Algorithms

- 🏆 **Best F1-Score**: DBSCAN (0.4694)
- 🏆 **Best Accuracy**: Elliptic Envelope (0.4896)
- 🏆 **Best Recall**: DBSCAN (0.5185)

## Key Findings

### 1. DBSCAN Performance
- **Strengths**: 
  - Highest recall (0.5185) - detects 51.85% of actual employee departures
  - Best F1-score (0.4694) - best balance between precision and recall
  - Most sensitive to detecting attrition cases
- **Trade-off**: 
  - Lower accuracy due to more false positives
  - Detected 8,502 anomalies (57% of test set)

### 2. Elliptic Envelope Performance
- **Strengths**:
  - Highest accuracy (0.4896)
  - Good precision (0.4357) - when it predicts "Left", it's right 43.57% of the time
- **Trade-off**:
  - Lower recall (0.2763) - misses many actual departures
  - More conservative in flagging anomalies

### 3. Algorithm Comparison Insights

- **Conservative algorithms** (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope):
  - Detect 26-30% of anomalies (contamination parameter set to 0.3)
  - Higher precision but lower recall
  - Miss many actual employee departures

- **Aggressive algorithm** (DBSCAN):
  - Detects 57% of test data as anomalies
  - Better at catching actual departures (51.85% recall)
  - More false positives but useful when cost of missing departures is high

### 4. Why Performance is Moderate

The relatively moderate performance across all algorithms (F1-scores 0.30-0.47) suggests:

1. **Complex Decision Boundary**: Employee attrition is influenced by many subtle factors
2. **Feature Overlap**: Employees who stay and leave may have similar characteristics
3. **Non-linear Patterns**: Simple distance or density-based methods struggle with complex relationships
4. **Balanced Classes**: With nearly 50-50 split, this is more of a classification problem than true anomaly detection

## Recommendations

### For Different Use Cases:

1. **If goal is to catch most potential departures** (High Recall Priority):
   - Use **DBSCAN**
   - Accept more false alarms
   - Good for proactive retention programs

2. **If goal is to accurately identify high-risk employees** (High Precision Priority):
   - Use **Elliptic Envelope** or **Local Outlier Factor**
   - More targeted interventions
   - Fewer false alarms

3. **For balanced approach**:
   - Use **DBSCAN** (best F1-score)
   - Provides reasonable trade-off between catching departures and accuracy

### Potential Improvements:

1. **Feature Engineering**: Create interaction features, tenure ratios, satisfaction trends
2. **Supervised Learning**: Given balanced classes, try classification algorithms (Random Forest, XGBoost, Neural Networks)
3. **Ensemble Methods**: Combine predictions from multiple anomaly detectors
4. **Adjust Contamination Parameter**: Fine-tune based on business requirements
5. **Cost-Sensitive Learning**: Weight the cost of false negatives vs false positives

## Visualizations Generated

1. **anomaly_detection_comparison.png**: Side-by-side comparison of all 5 algorithms showing detected anomalies in red
2. **actual_data_distribution.png**: Ground truth visualization showing actual attrition patterns
3. **performance_metrics_comparison.png**: Bar charts comparing accuracy, precision, recall, and F1-score
4. **anomaly_detection_results.csv**: Detailed numerical results

## Conclusion

For this employee attrition dataset, **DBSCAN** provides the best overall performance with an F1-score of 0.469 and recall of 0.519. However, the moderate performance across all algorithms suggests that this problem might benefit more from supervised classification approaches given the balanced nature of the classes. Anomaly detection algorithms work best when the anomalous class is rare (<10%), but in this case, departures represent nearly 47% of the data.
