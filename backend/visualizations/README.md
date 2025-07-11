# XGFitness AI Visualization Suite

## Overview

This comprehensive visualization suite generates publication-ready visualizations for the XGFitness AI model analysis, providing academic-quality charts and graphs for thesis presentation and model comparison.

## Features

### 🎯 **Model Comparison Visualizations**
- **Performance comparison bar charts**: XGBoost vs Random Forest for both workout and nutrition models
- **Side-by-side comparison charts**: Accuracy, F1-score, Precision, Recall
- **Comprehensive metrics tables**: All performance indicators in tabular format

### 📊 **Confusion Matrices**
- **All 4 models**: XGBoost Workout (9 classes), XGBoost Nutrition (8 classes), Random Forest Workout (9 classes), Random Forest Nutrition (8 classes)
- **Seaborn heatmaps**: With class labels and percentages
- **Professional styling**: High-quality PNG output (300 DPI)

### 📈 **ROC/AUC Curves**
- **Multi-class ROC curves**: One-vs-Rest for each class
- **Macro and micro-averaged AUC scores**: Comprehensive performance analysis
- **Class-wise AUC comparison**: Between XGBoost and Random Forest models

### 🔍 **Feature Importance Analysis**
- **Top 10 features**: For each model (XGBoost vs Random Forest)
- **XGBoost feature importance**: By type (gain, weight, cover)
- **Feature importance correlation**: Between models
- **Feature interaction analysis**: Where available

### 📋 **Classification Reports**
- **Per-class precision, recall, F1-score tables**: Detailed performance breakdown
- **Support visualization**: Sample count per class
- **Class imbalance impact analysis**: Understanding data distribution effects
- **Misclassification pattern analysis**: Error analysis

### 📊 **Dataset Analysis**
- **Goal distribution pie chart**: Real vs final after augmentation
- **BMI category distribution**: Visual representation of data composition
- **Activity level distribution**: Understanding user activity patterns
- **Real vs Synthetic data composition**: Data source analysis
- **Train/Val/Test split visualization**: Data partitioning overview
- **Template assignment distribution**: Workout and nutrition template usage

### 🎛️ **Performance Metrics Dashboard**
- **All metrics comparison**: Accuracy, precision, recall, F1, AUC, Cohen's Kappa
- **Top-2 and Top-3 accuracy visualization**: Multi-class ranking performance
- **Balanced vs standard accuracy comparison**: Handling class imbalance
- **Model performance by class size**: High-sample vs low-sample classes

### 🌳 **XGBoost Diagnostics**
- **Learning curves**: Training vs validation loss over iterations
- **Feature importance plots**: Gain, weight, cover analysis
- **Hyperparameter tuning results**: Best params vs performance
- **Training convergence plots**: Model training progression
- **Validation score progression**: During hyperparameter search

## Directory Structure

```
visualizations/
├── comparisons/              # Performance comparison charts
├── confusion_matrices/       # Confusion matrices for all models
├── roc_curves/              # ROC curves and AUC analysis
├── feature_importance/      # Feature importance visualizations
├── data_analysis/           # Dataset distribution analysis
├── xgboost_analysis/        # XGBoost-specific diagnostics
├── diagnostics/             # Model diagnostic plots
└── README.md               # This documentation
```

## Usage

### Basic Usage

```python
from visualizations.model_visualization_suite import create_comprehensive_visualizations

# After training your models
model_results = {
    'comparison_data': metrics_dict,
    'feature_importance': importance_dict,
    'feature_names': feature_names,
    'class_labels': class_labels_dict,
    'n_classes': n_classes_dict,
    'y_true': y_true_dict,
    'y_pred': y_pred_dict,
    'y_score': y_score_dict,
    'xgboost_model': xgb_model,
    'evals_result': evals_result,
    'diagnostics': diagnostics_data
}

# Generate all visualizations
create_comprehensive_visualizations(model_results, data_df, 'visualizations/')
```

### Individual Visualization Functions

```python
from visualizations.model_visualization_suite import (
    plot_performance_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_data_distribution,
    plot_xgboost_analysis,
    plot_diagnostics
)

# Performance comparison
plot_performance_comparison(metrics_dict, ['XGBoost', 'Random Forest'])

# Confusion matrices
plot_confusion_matrices(y_true_dict, y_pred_dict, class_labels_dict)

# ROC curves
plot_roc_curves(y_true_dict, y_score_dict, n_classes_dict, class_labels_dict)

# Feature importance
plot_feature_importance(importance_dict, feature_names, model_names)

# Data analysis
plot_data_distribution(data_df)

# XGBoost analysis
plot_xgboost_analysis(xgb_model, evals_result)

# Diagnostics
plot_diagnostics(results_dict)
```

## Integration with Model Training

### Automatic Integration

The visualization suite is designed to work seamlessly with the XGFitness AI model training pipeline:

```python
from src.thesis_model import XGFitnessAIModel
from visualizations.model_visualization_suite import create_comprehensive_visualizations

# Initialize and train models
model = XGFitnessAIModel(templates_dir='data')
df_training = model.create_training_dataset(...)
comprehensive_info = model.train_all_models(df_training, random_state=42)

# Prepare visualization data
model_results = prepare_visualization_data(model, comprehensive_info, df_training)

# Generate all visualizations
create_comprehensive_visualizations(model_results, df_training, 'visualizations/')
```

### Manual Integration

For custom integration, prepare the data structure as follows:

```python
model_results = {
    'comparison_data': {
        'workout': {
            'accuracy': {'XGBoost': 0.85, 'Random Forest': 0.82},
            'f1_weighted': {'XGBoost': 0.84, 'Random Forest': 0.81},
            # ... other metrics
        },
        'nutrition': {
            'accuracy': {'XGBoost': 0.92, 'Random Forest': 0.89},
            'f1_weighted': {'XGBoost': 0.91, 'Random Forest': 0.88},
            # ... other metrics
        }
    },
    'feature_importance': {
        'XGBoost Workout': {...},
        'Random Forest Workout': {...},
        # ... other models
    },
    'feature_names': ['age', 'gender_encoded', 'height_cm', ...],
    'class_labels': {
        'xgboost_workout': ['Template 1', 'Template 2', ...],
        'rf_nutrition': ['Template 1', 'Template 2', ...],
        # ... other models
    },
    'n_classes': {
        'xgboost_workout': 9,
        'rf_nutrition': 8,
        # ... other models
    },
    'y_true': {
        'xgboost_workout': y_true_encoded,
        'rf_nutrition': y_true_encoded,
        # ... other models
    },
    'y_pred': {
        'xgboost_workout': y_pred_encoded,
        'rf_nutrition': y_pred_encoded,
        # ... other models
    },
    'y_score': {
        'xgboost_workout': y_score_proba,
        'rf_nutrition': y_score_proba,
        # ... other models
    },
    'xgboost_model': trained_xgb_model,
    'evals_result': training_evaluation_results,
    'diagnostics': {
        'prediction_confidence': confidence_scores,
        'cv_scores': cross_validation_scores,
        # ... other diagnostic data
    }
}
```

## Output Specifications

### File Format
- **Format**: High-quality PNG files
- **Resolution**: 300 DPI
- **Color scheme**: Professional, consistent styling
- **File naming**: `metric_name_model_type.png`

### Styling
- **Color scheme**: Consistent professional colors
- **Fonts**: Clear, readable typography
- **Layout**: Publication-ready formatting
- **Annotations**: Value labels and percentages where relevant

## Testing

Run the test script to verify the visualization suite:

```bash
cd backend
python test_visualization_suite.py
```

This will:
1. Create sample visualizations with dummy data
2. Train real models and generate comprehensive visualizations
3. Save all plots to the appropriate directories

## Academic Benefits

### Thesis Enhancement
- **Publication-ready figures**: High-quality visualizations for academic papers
- **Comprehensive analysis**: All aspects of model performance covered
- **Professional presentation**: Consistent styling and formatting
- **Statistical rigor**: Proper metrics and confidence intervals

### Model Validation
- **Performance comparison**: XGBoost vs Random Forest baseline
- **Feature analysis**: Understanding model decisions
- **Error analysis**: Identifying model weaknesses
- **Data quality assessment**: Understanding dataset characteristics

### Research Applications
- **Baseline establishment**: Comparing against industry standards
- **Model interpretability**: Understanding feature importance
- **Performance optimization**: Identifying improvement areas
- **Reproducibility**: Consistent visualization pipeline

## Dependencies

- **matplotlib**: Core plotting library
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots (optional)
- **scikit-learn**: Metrics and preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages
   ```bash
   pip install matplotlib seaborn plotly scikit-learn pandas numpy
   ```

2. **Path issues**: Ensure correct directory structure
   ```bash
   mkdir -p visualizations/{comparisons,confusion_matrices,roc_curves,feature_importance,data_analysis,xgboost_analysis,diagnostics}
   ```

3. **Data format**: Ensure data is in the correct format
   - Metrics should be nested dictionaries
   - Feature importance should be dictionaries with feature names as keys
   - Class labels should be lists of strings

4. **Memory issues**: For large datasets, consider sampling
   - Use subset of data for visualization
   - Reduce figure sizes for memory-intensive plots

### Error Handling

The visualization suite includes comprehensive error handling:
- Graceful degradation for missing data
- Informative error messages
- Fallback options for unavailable metrics
- Progress indicators for long-running operations

## Future Enhancements

- **Interactive plots**: Plotly integration for web-based viewing
- **Animation support**: Training progression animations
- **Export options**: PDF, SVG, and other formats
- **Custom themes**: User-defined color schemes and styles
- **Batch processing**: Multiple model comparison
- **Real-time updates**: Live visualization during training

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test script for usage examples
3. Ensure all dependencies are installed
4. Verify data format matches expected structure

## License

This visualization suite is part of the XGFitness AI project and follows the same licensing terms. 