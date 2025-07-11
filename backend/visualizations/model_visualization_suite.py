#!/usr/bin/env python3
"""
Comprehensive Visualization Suite for XGFitness AI Model Analysis
Generates publication-ready visualizations for thesis and academic comparison
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (confusion_matrix, roc_curve, auc, classification_report,
                           precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color scheme for consistent styling
COLORS = {
    'xgboost': '#1f77b4',
    'random_forest': '#ff7f0e',
    'xgboost_light': '#aec7e8',
    'random_forest_light': '#ffbb78',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd'
}

# Utility functions
def save_fig(fig, path, dpi=300, bbox_inches='tight'):
    """Save figure with consistent settings"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✅ Saved: {path}")

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%" if value < 1 else f"{value:.1f}"

def add_value_labels(ax, bars, values, fmt='.3f'):
    """Add value labels on top of bars"""
    for bar, value in zip(bars, values):
        # Handle both vertical and horizontal bars
        if hasattr(bar, 'get_height'):
            # Vertical bars
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:{fmt}}', ha='center', va='bottom', fontsize=8)
        else:
            # Horizontal bars
            width = bar.get_width()
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:{fmt}}', ha='left', va='center', fontsize=8)

# ============================================================================
# 1. PERFORMANCE COMPARISON VISUALIZATIONS
# ============================================================================

def plot_performance_comparison(metrics_dict, model_names, save_dir='visualizations/comparisons/'):
    """
    Create performance comparison bar charts for XGBoost vs Random Forest
    
    Args:
        metrics_dict: Dict with structure {'workout': {'accuracy': {'XGBoost': 0.85, 'Random Forest': 0.82}, ...}, 'nutrition': {...}}
        model_names: List of model names
        save_dir: Directory to save plots
    """
    print("📊 Creating performance comparison visualizations...")
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison: XGBoost vs Random Forest', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i//2, i%2]
        
        # Extract data for this metric
        workout_values = [metrics_dict['workout'][metric][model] for model in model_names]
        nutrition_values = [metrics_dict['nutrition'][metric][model] for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, workout_values, width, label='Workout Model', 
                      color=COLORS['xgboost'], alpha=0.8)
        bars2 = ax.bar(x + width/2, nutrition_values, width, label='Nutrition Model', 
                      color=COLORS['random_forest'], alpha=0.8)
        
        # Add value labels
        add_value_labels(ax, bars1, workout_values)
        add_value_labels(ax, bars2, nutrition_values)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits for better visualization
        all_values = workout_values + nutrition_values
        ax.set_ylim(0, max(all_values) * 1.15)
    
    plt.tight_layout()
    save_fig(fig, f'{save_dir}performance_comparison.png')
    
    # Create summary table
    create_performance_summary_table(metrics_dict, model_names, save_dir)

def create_performance_summary_table(metrics_dict, model_names, save_dir):
    """Create a summary table of all performance metrics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Model', 'Task', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Balanced Acc', 'Cohen\'s Kappa']
    
    for model in model_names:
        for task in ['workout', 'nutrition']:
            row = [model, task.title()]
            for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'balanced_accuracy', 'cohen_kappa']:
                value = metrics_dict[task].get(metric, {}).get(model, 0)
                row.append(f'{value:.4f}')
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Comprehensive Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    save_fig(fig, f'{save_dir}performance_summary_table.png')

# ============================================================================
# 2. CONFUSION MATRICES
# ============================================================================

def plot_confusion_matrices(y_true_dict, y_pred_dict, class_labels_dict, save_dir='visualizations/confusion_matrices/'):
    """
    Create confusion matrices for all models
    
    Args:
        y_true_dict: Dict with keys like 'xgboost_workout', 'rf_nutrition'
        y_pred_dict: Dict with keys like 'xgboost_workout', 'rf_nutrition'
        class_labels_dict: Dict with class labels for each model
    """
    print("📈 Creating confusion matrices...")
    
    for model_key in y_true_dict.keys():
        y_true = y_true_dict[model_key]
        y_pred = y_pred_dict[model_key]
        class_labels = class_labels_dict[model_key]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        # Calculate percentages for annotation
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Add percentage annotations
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                if cm[i, j] > 0:
                    ax.text(j + 0.5, i + 0.5, f'\n{cm_percent[i, j]:.1f}%',
                           ha='center', va='center', fontsize=8, color='red')
        
        # Calculate metrics
        accuracy = np.trace(cm) / np.sum(cm)
        total_samples = np.sum(cm)
        
        # Set title and labels
        model_name = model_key.replace('_', ' ').title()
        ax.set_title(f'Confusion Matrix: {model_name}\n'
                    f'Accuracy: {accuracy:.3f} | Total Samples: {total_samples}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        save_fig(fig, f'{save_dir}confusion_matrix_{model_key}.png')

# ============================================================================
# 3. ROC CURVES
# ============================================================================

def plot_roc_curves(y_true_dict, y_score_dict, n_classes_dict, class_labels_dict, save_dir='visualizations/roc_curves/'):
    """
    Create multi-class ROC curves for all models
    
    Args:
        y_true_dict: Dict with true labels
        y_score_dict: Dict with prediction probabilities
        n_classes_dict: Dict with number of classes for each model
        class_labels_dict: Dict with class labels
    """
    print("📊 Creating ROC curves...")
    
    for model_key in y_true_dict.keys():
        y_true = y_true_dict[model_key]
        y_score = y_score_dict[model_key]
        n_classes = n_classes_dict[model_key]
        class_labels = class_labels_dict[model_key]
        
        # Binarize the output for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot individual class ROC curves
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_labels[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        ax1.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        # Plot diagonal line
        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curves by Class: {model_key.replace("_", " ").title()}')
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot AUC comparison bar chart
        classes = list(range(n_classes)) + ['Micro-avg']
        auc_values = [roc_auc[i] for i in range(n_classes)] + [roc_auc['micro']]
        colors_bar = list(colors) + ['deeppink']
        
        bars = ax2.bar(classes, auc_values, color=colors_bar, alpha=0.8)
        add_value_labels(ax2, bars, auc_values, '.3f')
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('AUC Score')
        ax2.set_title(f'AUC Scores by Class: {model_key.replace("_", " ").title()}')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels([class_labels[i] if i < n_classes else 'Micro-avg' for i in classes], rotation=45)
        
        plt.tight_layout()
        save_fig(fig, f'{save_dir}roc_curves_{model_key}.png')

# ============================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def plot_feature_importance(importance_dict, feature_names, model_names, save_dir='visualizations/feature_importance/'):
    """
    Create feature importance visualizations
    
    Args:
        importance_dict: Dict with feature importance scores for each model
        feature_names: List of feature names
        model_names: List of model names
    """
    print("🔍 Creating feature importance visualizations...")
    
    # Create individual feature importance plots for each model
    for model_name in model_names:
        if model_name in importance_dict:
            importance_scores = importance_dict[model_name]
            
            # Sort features by importance
            feature_importance = list(zip(feature_names, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 15 features
            top_features = feature_importance[:15]
            features, scores = zip(*top_features)
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(features)), scores, color=COLORS['xgboost'] if 'xgboost' in model_name.lower() else COLORS['random_forest'])
            
            # Add value labels
            max_score = max(scores) if scores else 0
            for i, (bar, score) in enumerate(zip(bars, scores)):
                try:
                    ax.text(bar.get_width() + max_score * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{score:.4f}', ha='left', va='center', fontsize=9)
                except AttributeError:
                    # Fallback for different bar types
                    ax.text(score + max_score * 0.01, i,
                           f'{score:.4f}', ha='left', va='center', fontsize=9)
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance Score')
            ax.set_title(f'Top 15 Feature Importance: {model_name.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_fig(fig, f'{save_dir}feature_importance_{model_name.lower().replace(" ", "_")}.png')
    
    # Create comparison plot for top 10 features
    create_feature_importance_comparison(importance_dict, feature_names, model_names, save_dir)

def create_feature_importance_comparison(importance_dict, feature_names, model_names, save_dir):
    """Create comparison plot of feature importance between models"""
    # Get top 10 features from the best performing model
    best_model = max(importance_dict.keys(), key=lambda x: max(importance_dict[x]))
    top_features_idx = np.argsort(importance_dict[best_model])[-10:]
    top_features = [feature_names[i] for i in top_features_idx]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_features))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        if model_name in importance_dict:
            scores = [importance_dict[model_name][idx] for idx in top_features_idx]
            color = COLORS['xgboost'] if 'xgboost' in model_name.lower() else COLORS['random_forest']
            bars = ax.bar(x + i * width, scores, width, label=model_name, color=color, alpha=0.8)
            add_value_labels(ax, bars, scores, '.3f')
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Importance Score')
    ax.set_title('Feature Importance Comparison: Top 10 Features', fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, f'{save_dir}feature_importance_comparison.png')

# ============================================================================
# 5. DATA ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_data_distribution(data_df, save_dir='visualizations/data_analysis/'):
    """
    Create data distribution visualizations
    
    Args:
        data_df: DataFrame with training data
    """
    print("📊 Creating data distribution visualizations...")
    
    # 1. Goal distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    goal_counts = data_df['fitness_goal'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    wedges, texts, autotexts = ax.pie(goal_counts.values, labels=goal_counts.index, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
    ax.set_title('Fitness Goal Distribution', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, f'{save_dir}goal_distribution.png')
    
    # 2. BMI category distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bmi_counts = data_df['bmi_category'].value_counts()
    bars = ax.bar(bmi_counts.index, bmi_counts.values, color=COLORS['info'])
    add_value_labels(ax, bars, bmi_counts.values)
    ax.set_xlabel('BMI Category')
    ax.set_ylabel('Count')
    ax.set_title('BMI Category Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{save_dir}bmi_distribution.png')
    
    # 3. Activity level distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    activity_counts = data_df['activity_level'].value_counts()
    bars = ax.bar(activity_counts.index, activity_counts.values, color=COLORS['success'])
    add_value_labels(ax, bars, activity_counts.values)
    ax.set_xlabel('Activity Level')
    ax.set_ylabel('Count')
    ax.set_title('Activity Level Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{save_dir}activity_distribution.png')
    
    # 4. Data split distribution
    if 'split' in data_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        split_counts = data_df['split'].value_counts()
        colors_split = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax.bar(split_counts.index, split_counts.values, color=colors_split)
        add_value_labels(ax, bars, split_counts.values)
        ax.set_xlabel('Data Split')
        ax.set_ylabel('Count')
        ax.set_title('Train/Validation/Test Split Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, f'{save_dir}data_split_distribution.png')
    
    # 5. Age distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data_df['age'], bins=20, color=COLORS['xgboost'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Age Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f'{save_dir}age_distribution.png')
    
    # 6. Gender distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    gender_counts = data_df['gender'].value_counts()
    colors_gender = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                                     colors=colors_gender, startangle=90)
    ax.set_title('Gender Distribution', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, f'{save_dir}gender_distribution.png')

# ============================================================================
# 6. XGBOOST ANALYSIS
# ============================================================================

def plot_xgboost_analysis(xgb_model, evals_result=None, save_dir='visualizations/xgboost_analysis/'):
    """
    Create XGBoost-specific analysis plots
    
    Args:
        xgb_model: Trained XGBoost model
        evals_result: Evaluation results from training
    """
    print("🌳 Creating XGBoost analysis visualizations...")
    
    # 1. Feature importance by type (if available)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        importance_types = ['weight', 'gain', 'cover']
        for i, imp_type in enumerate(importance_types):
            try:
                importance = xgb_model.get_booster().get_score(importance_type=imp_type)
                if importance:
                    features = list(importance.keys())
                    scores = list(importance.values())
                    
                    # Sort by importance
                    sorted_idx = np.argsort(scores)[::-1]
                    features = [features[i] for i in sorted_idx[:10]]
                    scores = [scores[i] for i in sorted_idx[:10]]
                    
                    bars = axes[i].barh(range(len(features)), scores, color=COLORS['xgboost'])
                    axes[i].set_yticks(range(len(features)))
                    axes[i].set_yticklabels(features)
                    axes[i].set_xlabel(f'{imp_type.title()} Importance')
                    axes[i].set_title(f'Feature Importance ({imp_type.title()})')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels
                    for j, (bar, score) in enumerate(zip(bars, scores)):
                        axes[i].text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{score:.2f}', ha='left', va='center', fontsize=8)
                else:
                    axes[i].text(0.5, 0.5, f'No {imp_type} importance available', 
                               ha='center', va='center', transform=axes[i].transAxes)
            except:
                axes[i].text(0.5, 0.5, f'Error getting {imp_type} importance', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        save_fig(fig, f'{save_dir}xgboost_feature_importance_types.png')
    except Exception as e:
        print(f"Warning: Could not create XGBoost feature importance plots: {e}")
    
    # 2. Learning curves (if evaluation results available)
    if evals_result:
        create_learning_curves(evals_result, save_dir)

def create_learning_curves(evals_result, save_dir):
    """Create learning curves from evaluation results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training and validation loss
    for i, metric in enumerate(['train', 'eval']):
        if f'{metric}-mlogloss' in evals_result:
            epochs = range(len(evals_result[f'{metric}-mlogloss']))
            axes[0].plot(epochs, evals_result[f'{metric}-mlogloss'], 
                        label=f'{metric.title()} Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Learning Curves: Log Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy if available
    for i, metric in enumerate(['train', 'eval']):
        if f'{metric}-merror' in evals_result:
            epochs = range(len(evals_result[f'{metric}-merror']))
            axes[1].plot(epochs, evals_result[f'{metric}-merror'], 
                        label=f'{metric.title()} Error', linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error Rate')
    axes[1].set_title('Learning Curves: Error Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, f'{save_dir}xgboost_learning_curves.png')

# ============================================================================
# 7. DIAGNOSTICS
# ============================================================================

def plot_diagnostics(results_dict, save_dir='visualizations/diagnostics/'):
    """
    Create diagnostic plots
    
    Args:
        results_dict: Dictionary with diagnostic results
    """
    print("🔧 Creating diagnostic visualizations...")
    
    # 1. Prediction confidence distribution
    if 'prediction_confidence' in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        confidence_scores = results_dict['prediction_confidence']
        ax.hist(confidence_scores, bins=20, color=COLORS['xgboost'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Confidence Scores', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, f'{save_dir}prediction_confidence_distribution.png')
    
    # 2. Cross-validation scores
    if 'cv_scores' in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        cv_scores = results_dict['cv_scores']
        ax.boxplot(cv_scores, labels=['CV Scores'])
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Score Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, f'{save_dir}cross_validation_scores.png')

# ============================================================================
# 8. MAIN VISUALIZATION PIPELINE
# ============================================================================

def create_comprehensive_visualizations(model_results, data_df, save_base_dir='visualizations/'):
    """
    Main function to create all visualizations
    
    Args:
        model_results: Dictionary containing all model results and metrics
        data_df: Training dataset DataFrame
        save_base_dir: Base directory for saving visualizations
    """
    print("🎨 Starting comprehensive visualization pipeline...")
    
    # Extract data from model_results
    metrics_dict = model_results.get('comparison_data', {})
    y_true_dict = model_results.get('y_true', {})
    y_pred_dict = model_results.get('y_pred', {})
    y_score_dict = model_results.get('y_score', {})
    importance_dict = model_results.get('feature_importance', {})
    
    # Create all visualizations
    try:
        # 1. Performance comparisons
        if metrics_dict:
            plot_performance_comparison(metrics_dict, ['XGBoost', 'Random Forest'], 
                                      f'{save_base_dir}comparisons/')
        
        # 2. Confusion matrices
        if y_true_dict and y_pred_dict:
            class_labels_dict = model_results.get('class_labels', {})
            plot_confusion_matrices(y_true_dict, y_pred_dict, class_labels_dict, 
                                  f'{save_base_dir}confusion_matrices/')
        
        # 3. ROC curves
        if y_true_dict and y_score_dict:
            n_classes_dict = model_results.get('n_classes', {})
            class_labels_dict = model_results.get('class_labels', {})
            plot_roc_curves(y_true_dict, y_score_dict, n_classes_dict, class_labels_dict, 
                           f'{save_base_dir}roc_curves/')
        
        # 4. Feature importance
        if importance_dict:
            feature_names = model_results.get('feature_names', [])
            model_names = list(importance_dict.keys())
            plot_feature_importance(importance_dict, feature_names, model_names, 
                                  f'{save_base_dir}feature_importance/')
        
        # 5. Data analysis
        if data_df is not None:
            plot_data_distribution(data_df, f'{save_base_dir}data_analysis/')
        
        # 6. XGBoost analysis
        xgb_model = model_results.get('xgboost_model')
        evals_result = model_results.get('evals_result')
        if xgb_model:
            plot_xgboost_analysis(xgb_model, evals_result, f'{save_base_dir}xgboost_analysis/')
        
        # 7. Diagnostics
        diagnostics_data = model_results.get('diagnostics', {})
        if diagnostics_data:
            plot_diagnostics(diagnostics_data, f'{save_base_dir}diagnostics/')
        
        print("✅ All visualizations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in visualization pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Visualization suite ready for use!")
    print("Import and use create_comprehensive_visualizations() function.") 