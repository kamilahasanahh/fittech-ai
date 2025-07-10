#!/usr/bin/env python3
"""
Generate proper confusion matrices and detailed metrics for XGFitness AI model
Shows actual template IDs (W1-W9, N1-N8) and comprehensive performance metrics
"""
import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from thesis_model import XGFitnessAIModel

def create_proper_confusion_matrices():
    """Generate confusion matrices with actual template IDs and detailed metrics"""
    
    print("=" * 80)
    print("XGFITNESS AI - TEMPLATE-BASED CONFUSION MATRICES & METRICS")
    print("=" * 80)
    
    # Initialize model
    model = XGFitnessAIModel('../data')
    model.load_model('models/xgfitness_ai_model.pkl')
    
    # Create test dataset
    print("\nCreating test dataset...")
    df = model.create_training_dataset(
        real_data_file='../data/backups/e267_Data on age, gender, height, weight, activity levels for each household member.txt',
        total_samples=1000
    )
    
    # Prepare data for testing
    X, y_workout, y_nutrition, df_enhanced = model.prepare_training_data(df)
    
    # Use test set only
    test_mask = df_enhanced['split'] == 'test'
    X_test = X[test_mask]
    y_w_test = y_workout[test_mask]
    y_n_test = y_nutrition[test_mask]
    
    # Scale features
    X_test_scaled = model.scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions on test set...")
    y_w_pred = model.workout_model.predict(X_test_scaled)
    y_w_pred_proba = model.workout_model.predict_proba(X_test_scaled)
    
    y_n_pred = model.nutrition_model.predict(X_test_scaled)
    y_n_pred_proba = model.nutrition_model.predict_proba(X_test_scaled)
    
    # Convert template IDs directly - they should already be in correct format
    y_w_test_ids = y_w_test.values
    y_w_pred_ids = y_w_pred
    
    y_n_test_ids = y_n_test.values
    y_n_pred_ids = y_n_pred
    
    # Create template labels
    workout_labels = [f'W{i}' for i in sorted(np.unique(np.concatenate([y_w_test_ids, y_w_pred_ids])))]
    nutrition_labels = [f'N{i}' for i in sorted(np.unique(np.concatenate([y_n_test_ids, y_n_pred_ids])))]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Workout Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    workout_cm = confusion_matrix(y_w_test_ids, y_w_pred_ids)
    
    # Create proper labels for workout templates
    unique_workout_ids = sorted(np.unique(np.concatenate([y_w_test_ids, y_w_pred_ids])))
    workout_display_labels = [f'W{i}' for i in unique_workout_ids]
    
    sns.heatmap(workout_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=workout_display_labels, yticklabels=workout_display_labels,
                ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Workout Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Template')
    ax1.set_ylabel('True Template')
    
    # 2. Nutrition Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    nutrition_cm = confusion_matrix(y_n_test_ids, y_n_pred_ids)
    
    # Create proper labels for nutrition templates
    unique_nutrition_ids = sorted(np.unique(np.concatenate([y_n_test_ids, y_n_pred_ids])))
    nutrition_display_labels = [f'N{i}' for i in unique_nutrition_ids]
    
    sns.heatmap(nutrition_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=nutrition_display_labels, yticklabels=nutrition_display_labels,
                ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Nutrition Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Template')
    ax2.set_ylabel('True Template')
    
    # 3. Workout Template Performance Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    workout_report = classification_report(y_w_test_ids, y_w_pred_ids, output_dict=True)
    
    # Extract per-template metrics
    workout_template_metrics = []
    for template_id in unique_workout_ids:
        if str(template_id) in workout_report:
            metrics = workout_report[str(template_id)]
            workout_template_metrics.append({
                'Template': f'W{template_id}',
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    workout_df = pd.DataFrame(workout_template_metrics)
    
    # Create heatmap for workout metrics
    workout_metrics_matrix = workout_df[['Precision', 'Recall', 'F1-Score']].T
    sns.heatmap(workout_metrics_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=workout_df['Template'], yticklabels=['Precision', 'Recall', 'F1-Score'],
                ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Workout Template\nPerformance Metrics', fontsize=14, fontweight='bold')
    
    # 4. Nutrition Template Performance Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    nutrition_report = classification_report(y_n_test_ids, y_n_pred_ids, output_dict=True)
    
    # Extract per-template metrics
    nutrition_template_metrics = []
    for template_id in unique_nutrition_ids:
        if str(template_id) in nutrition_report:
            metrics = nutrition_report[str(template_id)]
            nutrition_template_metrics.append({
                'Template': f'N{template_id}',
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    nutrition_df = pd.DataFrame(nutrition_template_metrics)
    
    # Create heatmap for nutrition metrics
    nutrition_metrics_matrix = nutrition_df[['Precision', 'Recall', 'F1-Score']].T
    sns.heatmap(nutrition_metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=nutrition_df['Template'], yticklabels=['Precision', 'Recall', 'F1-Score'],
                ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Nutrition Template\nPerformance Metrics', fontsize=14, fontweight='bold')
    
    # 5. Overall Model Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Calculate overall metrics
    workout_accuracy = accuracy_score(y_w_test_ids, y_w_pred_ids)
    workout_f1 = f1_score(y_w_test_ids, y_w_pred_ids, average='weighted')
    workout_precision = precision_score(y_w_test_ids, y_w_pred_ids, average='weighted')
    workout_recall = recall_score(y_w_test_ids, y_w_pred_ids, average='weighted')
    workout_kappa = cohen_kappa_score(y_w_test_ids, y_w_pred_ids)
    
    nutrition_accuracy = accuracy_score(y_n_test_ids, y_n_pred_ids)
    nutrition_f1 = f1_score(y_n_test_ids, y_n_pred_ids, average='weighted')
    nutrition_precision = precision_score(y_n_test_ids, y_n_pred_ids, average='weighted')
    nutrition_recall = recall_score(y_n_test_ids, y_n_pred_ids, average='weighted')
    nutrition_kappa = cohen_kappa_score(y_n_test_ids, y_n_pred_ids)
    
    # Create comparison chart
    metrics_comparison = pd.DataFrame({
        'Workout Model': [workout_accuracy, workout_precision, workout_recall, workout_f1, workout_kappa],
        'Nutrition Model': [nutrition_accuracy, nutrition_precision, nutrition_recall, nutrition_f1, nutrition_kappa]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Cohen\'s Kappa'])
    
    metrics_comparison.plot(kind='bar', ax=ax5, color=['#3498db', '#2ecc71'])
    ax5.set_title('Overall Model Performance\nComparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Score')
    ax5.set_ylim(0, 1.0)
    ax5.legend(loc='lower right')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Template Distribution in Test Set
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Count template occurrences in test set
    workout_counts = pd.Series(y_w_test_ids).value_counts().sort_index()
    workout_counts.index = [f'W{i}' for i in workout_counts.index]
    
    workout_counts.plot(kind='bar', ax=ax6, color='skyblue', alpha=0.7)
    ax6.set_title('Workout Template\nDistribution in Test Set', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Count')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Nutrition Template Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    
    nutrition_counts = pd.Series(y_n_test_ids).value_counts().sort_index()
    nutrition_counts.index = [f'N{i}' for i in nutrition_counts.index]
    
    nutrition_counts.plot(kind='bar', ax=ax7, color='lightgreen', alpha=0.7)
    ax7.set_title('Nutrition Template\nDistribution in Test Set', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Count')
    ax7.tick_params(axis='x', rotation=45)
    
    # 8. Template Mapping Info
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Load template descriptions
    import json
    with open('../data/workout_templates.json', 'r') as f:
        workout_templates = json.load(f)['workout_templates']
    with open('../data/templates/nutrition_templates.json', 'r') as f:
        nutrition_templates = json.load(f)['nutrition_templates']
    
    # Create template mapping text
    mapping_text = "TEMPLATE MAPPINGS:\n\n"
    mapping_text += "WORKOUT TEMPLATES:\n"
    for wt in workout_templates[:6]:  # Show first 6 to fit
        mapping_text += f"W{wt['template_id']}: {wt['goal']} - {wt['activity_level']} ({wt['workout_type']})\n"
    
    mapping_text += "\nNUTRITION TEMPLATES:\n"
    for nt in nutrition_templates:
        mapping_text += f"N{nt['template_id']}: {nt['goal']} - {nt['bmi_category']} (×{nt['caloric_intake_multiplier']})\n"
    
    ax8.text(0.02, 0.95, mapping_text, transform=ax8.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('XGFitness AI - Template-Based Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    plt.savefig('template_confusion_matrices_detailed.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Detailed confusion matrices saved as 'template_confusion_matrices_detailed.png'")
    
    # Print detailed metrics to console
    print("\n" + "=" * 80)
    print("DETAILED TEMPLATE PERFORMANCE METRICS")
    print("=" * 80)
    
    print(f"\n📊 OVERALL MODEL PERFORMANCE:")
    print(f"Workout Model:")
    print(f"  Accuracy: {workout_accuracy:.4f}")
    print(f"  Precision: {workout_precision:.4f}")
    print(f"  Recall: {workout_recall:.4f}")
    print(f"  F1-Score: {workout_f1:.4f}")
    print(f"  Cohen's Kappa: {workout_kappa:.4f}")
    
    print(f"\nNutrition Model:")
    print(f"  Accuracy: {nutrition_accuracy:.4f}")
    print(f"  Precision: {nutrition_precision:.4f}")
    print(f"  Recall: {nutrition_recall:.4f}")
    print(f"  F1-Score: {nutrition_f1:.4f}")
    print(f"  Cohen's Kappa: {nutrition_kappa:.4f}")
    
    print(f"\n🏋️ WORKOUT TEMPLATE PERFORMANCE:")
    print(workout_df.to_string(index=False))
    
    print(f"\n🍎 NUTRITION TEMPLATE PERFORMANCE:")
    print(nutrition_df.to_string(index=False))
    
    # Calculate and print per-template accuracy
    print(f"\n📈 PER-TEMPLATE ACCURACY:")
    print("Workout Templates:")
    for i, template_id in enumerate(unique_workout_ids):
        template_accuracy = workout_cm[i, i] / workout_cm[i, :].sum() if workout_cm[i, :].sum() > 0 else 0
        print(f"  W{template_id}: {template_accuracy:.4f} ({workout_cm[i, i]}/{workout_cm[i, :].sum()})")
    
    print("Nutrition Templates:")
    for i, template_id in enumerate(unique_nutrition_ids):
        template_accuracy = nutrition_cm[i, i] / nutrition_cm[i, :].sum() if nutrition_cm[i, :].sum() > 0 else 0
        print(f"  N{template_id}: {template_accuracy:.4f} ({nutrition_cm[i, i]}/{nutrition_cm[i, :].sum()})")
    
    print(f"\n✅ Analysis complete! Test set size: {len(X_test)} samples")
    
    return {
        'workout_metrics': workout_df,
        'nutrition_metrics': nutrition_df,
        'workout_cm': workout_cm,
        'nutrition_cm': nutrition_cm,
        'overall_metrics': {
            'workout': {
                'accuracy': workout_accuracy,
                'precision': workout_precision,
                'recall': workout_recall,
                'f1': workout_f1,
                'kappa': workout_kappa
            },
            'nutrition': {
                'accuracy': nutrition_accuracy,
                'precision': nutrition_precision,
                'recall': nutrition_recall,
                'f1': nutrition_f1,
                'kappa': nutrition_kappa
            }
        }
    }

if __name__ == "__main__":
    results = create_proper_confusion_matrices()
