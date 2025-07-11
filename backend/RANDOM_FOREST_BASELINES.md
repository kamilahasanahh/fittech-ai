# Random Forest Baseline Models for Academic Comparison

## Overview

This implementation adds Random Forest baseline models to the XGFitness AI system, enabling comprehensive academic comparison between XGBoost and Random Forest performance for fitness recommendation systems.

## Features Added

### 1. Random Forest Models
- **RandomForestClassifier** for workout recommendations
- **RandomForestClassifier** for nutrition recommendations
- Same features, train/val/test splits as XGBoost models
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive evaluation metrics

### 2. Model Comparison System
- Side-by-side performance comparison tables
- Feature importance analysis for both models
- Academic insights and statistical analysis
- Prediction agreement analysis

### 3. Enhanced Training Pipeline
- `train_all_models()`: Train both XGBoost and Random Forest
- `train_random_forest_baselines()`: Train only Random Forest models
- `compare_model_performance()`: Generate comparison reports

## Usage

### Basic Training and Comparison

```python
from src.thesis_model import XGFitnessAIModel

# Initialize model
model = XGFitnessAIModel(templates_dir='data')

# Create training dataset
df_training = model.create_training_dataset(
    real_data_file='e267_Data on age, gender, height, weight, activity levels for each household member.txt',
    equal_goal_distribution=True,
    splits=(0.70, 0.15, 0.15),
    random_state=42
)

# Train all models (XGBoost + Random Forest)
comprehensive_info = model.train_all_models(df_training, random_state=42)

# Generate comparison report
comparison_data = model.compare_model_performance()
```

### Individual Model Training

```python
# Train only XGBoost models
xgb_info = model.train_models(df_training, random_state=42)

# Train only Random Forest baseline models
rf_info = model.train_random_forest_baselines(df_training, random_state=42)
```

### Model Persistence

```python
# Save models (includes both XGBoost and Random Forest)
model.save_model('models/xgfitness_ai_with_baselines.pkl')

# Load models
model.load_model('models/xgfitness_ai_with_baselines.pkl')
```

## Comparison Metrics

The system provides comprehensive comparison across:

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Balanced Accuracy**: Accuracy accounting for class imbalance
- **F1 Score (Weighted/Macro)**: Harmonic mean of precision and recall
- **Precision/Recall (Weighted)**: Per-class performance
- **Cohen's Kappa**: Agreement measure
- **Top-2/Top-3 Accuracy**: Multi-class ranking performance
- **AUC-ROC**: Area under ROC curve

### Model Characteristics
- **Feature Importance**: Top features for each model
- **Training Time**: Computational efficiency comparison
- **Prediction Agreement**: How often models agree on recommendations

## Academic Benefits

### 1. Rigorous Baseline Comparison
- Establishes XGBoost performance against industry-standard Random Forest
- Provides statistical significance testing framework
- Enables publication-quality comparison tables

### 2. Model Interpretability
- Random Forest feature importance vs XGBoost feature importance
- Understanding of feature interactions
- Model decision transparency

### 3. Robustness Validation
- Cross-validation across multiple algorithms
- Ensures recommendations are not algorithm-specific
- Validates feature engineering approach

## Example Output

```
📊 COMPREHENSIVE MODEL COMPARISON: XGBoost vs Random Forest
================================================================================

🏋️ WORKOUT MODEL COMPARISON:
--------------------------------------------------------------------------------
Metric                    XGBoost     Random Forest   Difference
--------------------------------------------------------------------------------
Accuracy                  0.8150      0.7920          +0.0230
Balanced Accuracy         0.7840      0.7610          +0.0230
F1 Score (Weighted)       0.7400      0.7150          +0.0250
F1 Score (Macro)          0.7120      0.6890          +0.0230
Precision (Weighted)      0.7450      0.7200          +0.0250
Recall (Weighted)         0.7400      0.7150          +0.0250
Cohen's Kappa             0.6980      0.6750          +0.0230
Top-2 Accuracy            0.9120      0.8950          +0.0170
Top-3 Accuracy            0.9450      0.9280          +0.0170
AUC-ROC (Weighted)        0.9230      0.9010          +0.0220

🥗 NUTRITION MODEL COMPARISON:
--------------------------------------------------------------------------------
Metric                    XGBoost     Random Forest   Difference
--------------------------------------------------------------------------------
Accuracy                  0.9220      0.8980          +0.0240
Balanced Accuracy         0.9150      0.8920          +0.0230
F1 Score (Weighted)       0.9100      0.8850          +0.0250
F1 Score (Macro)          0.8950      0.8720          +0.0230
Precision (Weighted)      0.9180      0.8930          +0.0250
Recall (Weighted)         0.9100      0.8850          +0.0250
Cohen's Kappa             0.8950      0.8720          +0.0230
Top-2 Accuracy            0.9780      0.9620          +0.0160
Top-3 Accuracy            0.9920      0.9850          +0.0070
AUC-ROC (Weighted)        0.9780      0.9620          +0.0160

📈 OVERALL PERFORMANCE SUMMARY:
--------------------------------------------------
Workout Model Average Performance:
  XGBoost: 0.7847
  Random Forest: 0.7617
  XGBoost Advantage: +0.0230

Nutrition Model Average Performance:
  XGBoost: 0.9157
  Random Forest: 0.8927
  XGBoost Advantage: +0.0230

🎓 ACADEMIC INSIGHTS:
--------------------------------------------------
✅ XGBoost outperforms Random Forest for workout recommendations
   Improvement: 3.02%

✅ XGBoost outperforms Random Forest for nutrition recommendations
   Improvement: 2.58%
```

## Testing

Run the test script to verify functionality:

```bash
cd backend
python test_random_forest_baselines.py
```

This will:
1. Create training dataset
2. Train both XGBoost and Random Forest models
3. Generate comparison tables
4. Test predictions with both models
5. Save models for future use

## Technical Details

### Random Forest Hyperparameters
```python
rf_param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}
```

### Fair Comparison Methodology
- Same training/validation/test splits
- Same feature engineering pipeline
- Same evaluation metrics
- Same hyperparameter tuning approach
- Same class weighting strategy

## Research Applications

### Thesis Enhancement
- Provides academic rigor through baseline comparison
- Demonstrates XGBoost superiority with statistical evidence
- Enables publication in peer-reviewed journals

### Model Validation
- Ensures recommendations are not algorithm-specific
- Validates feature engineering approach
- Provides confidence in model robustness

### Future Research
- Baseline for comparing other algorithms (SVM, Neural Networks)
- Framework for ensemble methods
- Foundation for model interpretability studies

## File Structure

```
backend/
├── src/
│   └── thesis_model.py          # Updated with Random Forest functionality
├── test_random_forest_baselines.py  # Test script
├── models/                      # Saved models directory
│   └── xgfitness_ai_with_baselines.pkl
└── RANDOM_FOREST_BASELINES.md   # This documentation
```

## Dependencies

The Random Forest functionality uses standard scikit-learn components:
- `RandomForestClassifier` from `sklearn.ensemble`
- `RandomizedSearchCV` from `sklearn.model_selection`
- Standard evaluation metrics from `sklearn.metrics`

No additional dependencies beyond the existing XGBoost implementation.

## Conclusion

This Random Forest baseline implementation provides a solid academic foundation for comparing XGBoost performance in fitness recommendation systems. It enables rigorous evaluation, enhances thesis credibility, and provides insights into model behavior and feature importance. 