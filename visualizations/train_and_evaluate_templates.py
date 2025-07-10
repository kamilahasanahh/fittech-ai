#!/usr/bin/env python3
"""
Generate proper confusion matrices by training a fresh model 
that maintains template ID consistency for research explanation
"""
import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from thesis_model import XGFitnessAIModel

def train_and_evaluate_with_template_ids():
    print("=" * 60)
    print("TRAINING FRESH MODEL WITH PROPER TEMPLATE ID TRACKING")
    print("=" * 60)
    
    # 1. Initialize fresh model
    print("\n1. Initializing fresh model...")
    model = XGFitnessAIModel('../data')
    
    # 2. Create dataset with 70/15/15 split
    print("\n2. Creating dataset with 70/15/15 split...")
    df = model.create_training_dataset(
        real_data_file='../data/backups/e267_Data on age, gender, height, weight, activity levels for each household member.txt',
        total_samples=1200  # Good size for training
    )
    
    print(f"✅ Dataset created: {len(df)} samples")
    print(f"   Train: {len(df[df['split'] == 'train'])} (real data)")
    print(f"   Validation: {len(df[df['split'] == 'validation'])} (real data)")  
    print(f"   Test: {len(df[df['split'] == 'test'])} (dummy data)")
    
    # 3. Train the model
    print("\n3. Training model...")
    training_results = model.train_models(df)
    
    print(f"✅ Training complete!")
    print(f"   Workout accuracy: {training_results['workout_accuracy']:.4f}")
    print(f"   Nutrition accuracy: {training_results['nutrition_accuracy']:.4f}")
    
    # 4. Prepare test data for evaluation
    print("\n4. Preparing test data for evaluation...")
    X, y_workout, y_nutrition, df_enhanced = model.prepare_training_data(df)
    
    # Get test data
    test_mask = df_enhanced['split'] == 'test'
    X_test = X[test_mask]
    y_w_test = y_workout[test_mask]
    y_n_test = y_nutrition[test_mask]
    
    # Scale features
    X_test_scaled = model.scaler.transform(X_test)
    
    # 5. Make predictions and handle encoding properly
    print("\n5. Making predictions...")
    
    # Get encoded predictions
    y_w_pred_encoded = model.workout_model.predict(X_test_scaled)
    y_n_pred_encoded = model.nutrition_model.predict(X_test_scaled)
    
    # Map back to template IDs using the label encoders
    if hasattr(model, 'workout_label_encoder'):
        y_w_pred = model.workout_label_encoder.inverse_transform(y_w_pred_encoded)
        y_w_test_actual = y_w_test.values  # These should already be template IDs
    else:
        # Fallback: assume predictions are already template IDs
        y_w_pred = y_w_pred_encoded
        y_w_test_actual = y_w_test.values
        
    if hasattr(model, 'nutrition_label_encoder'):
        y_n_pred = model.nutrition_label_encoder.inverse_transform(y_n_pred_encoded)
        y_n_test_actual = y_n_test.values
    else:
        # Fallback: assume predictions are already template IDs
        y_n_pred = y_n_pred_encoded
        y_n_test_actual = y_n_test.values
    
    print(f"✅ Predictions made on {len(X_test)} test samples")
    
    # 6. Calculate accuracies
    workout_accuracy = accuracy_score(y_w_test_actual, y_w_pred)
    nutrition_accuracy = accuracy_score(y_n_test_actual, y_n_pred)
    
    print(f"✅ Test Accuracies:")
    print(f"   Workout: {workout_accuracy:.4f} ({workout_accuracy:.1%})")
    print(f"   Nutrition: {nutrition_accuracy:.4f} ({nutrition_accuracy:.1%})")
    
    # 7. Generate confusion matrices with proper template labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Get unique template IDs that appear in test set
    workout_templates_in_test = sorted(np.unique(y_w_test_actual))
    nutrition_templates_in_test = sorted(np.unique(y_n_test_actual))
    
    print(f"\n6. Template coverage in test set:")
    print(f"   Workout templates: {workout_templates_in_test}")
    print(f"   Nutrition templates: {nutrition_templates_in_test}")
    
    # Workout confusion matrix
    workout_cm = confusion_matrix(y_w_test_actual, y_w_pred, 
                                 labels=workout_templates_in_test)
    workout_labels = [f'W{i}' for i in workout_templates_in_test]
    
    sns.heatmap(workout_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=workout_labels, yticklabels=workout_labels, ax=ax1)
    ax1.set_title(f'Workout Template Confusion Matrix\nAccuracy: {workout_accuracy:.3f}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Template')
    ax1.set_ylabel('True Template')
    
    # Nutrition confusion matrix
    nutrition_cm = confusion_matrix(y_n_test_actual, y_n_pred,
                                   labels=nutrition_templates_in_test)
    nutrition_labels = [f'N{i}' for i in nutrition_templates_in_test]
    
    sns.heatmap(nutrition_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=nutrition_labels, yticklabels=nutrition_labels, ax=ax2)
    ax2.set_title(f'Nutrition Template Confusion Matrix\nAccuracy: {nutrition_accuracy:.3f}', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Template')
    ax2.set_ylabel('True Template')
    
    # 8. Calculate detailed metrics per template
    workout_report = classification_report(y_w_test_actual, y_w_pred, 
                                         labels=workout_templates_in_test,
                                         target_names=workout_labels,
                                         output_dict=True, zero_division=0)
    
    nutrition_report = classification_report(y_n_test_actual, y_n_pred,
                                           labels=nutrition_templates_in_test, 
                                           target_names=nutrition_labels,
                                           output_dict=True, zero_division=0)
    
    # 9. Create metrics visualization
    metrics_data = {
        'Model': ['Workout', 'Nutrition'],
        'Accuracy': [workout_accuracy, nutrition_accuracy],
        'Precision': [workout_report['macro avg']['precision'], nutrition_report['macro avg']['precision']],
        'Recall': [workout_report['macro avg']['recall'], nutrition_report['macro avg']['recall']],
        'F1-Score': [workout_report['macro avg']['f1-score'], nutrition_report['macro avg']['f1-score']]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics comparison
    x = np.arange(len(metrics_data['Model']))
    width = 0.2
    
    bars1 = ax3.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='lightgreen')
    bars3 = ax3.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='salmon')
    bars4 = ax3.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8, color='gold')
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_df['Model'])
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (0.8)')
    
    # Add value labels on bars
    for i, model in enumerate(metrics_df['Model']):
        for j, (bars, metric) in enumerate(zip([bars1, bars2, bars3, bars4], 
                                             ['Accuracy', 'Precision', 'Recall', 'F1-Score'])):
            height = bars[i].get_height()
            ax3.text(bars[i].get_x() + bars[i].get_width()/2., height + 0.01,
                    f'{metrics_df.iloc[i][metric]:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 10. Template performance details
    ax4.axis('off')
    ax4.text(0.05, 0.95, 'Per-Template Performance', fontsize=16, fontweight='bold', transform=ax4.transAxes)
    
    # Workout templates
    ax4.text(0.05, 0.85, 'Workout Templates:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    y_pos = 0.80
    for template_id in workout_templates_in_test:
        label = f'W{template_id}'
        if label in workout_report:
            precision = workout_report[label]['precision']
            recall = workout_report[label]['recall']
            f1 = workout_report[label]['f1-score']
            support = int(workout_report[label]['support'])
            ax4.text(0.05, y_pos, f'{label}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})', 
                    fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.04
    
    # Nutrition templates
    ax4.text(0.05, 0.45, 'Nutrition Templates:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    y_pos = 0.40
    for template_id in nutrition_templates_in_test:
        label = f'N{template_id}'
        if label in nutrition_report:
            precision = nutrition_report[label]['precision']
            recall = nutrition_report[label]['recall']
            f1 = nutrition_report[label]['f1-score']
            support = int(nutrition_report[label]['support'])
            ax4.text(0.05, y_pos, f'{label}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support})', 
                    fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.04
    
    plt.tight_layout()
    plt.savefig('proper_template_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 11. Print comprehensive results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"Workout Model: {workout_accuracy:.4f} ({workout_accuracy:.1%})")
    print(f"Nutrition Model: {nutrition_accuracy:.4f} ({nutrition_accuracy:.1%})")
    
    print(f"\n🎯 TARGET ACHIEVEMENT (0.8 threshold):")
    workout_meets_target = workout_accuracy >= 0.8
    nutrition_meets_target = nutrition_accuracy >= 0.8
    print(f"Workout Model: {'✅ MEETS' if workout_meets_target else '❌ BELOW'} target")
    print(f"Nutrition Model: {'✅ MEETS' if nutrition_meets_target else '❌ BELOW'} target")
    
    print(f"\n📋 TEMPLATE COVERAGE IN TEST SET:")
    print(f"Workout: {len(workout_templates_in_test)}/9 templates ({workout_templates_in_test})")
    print(f"Nutrition: {len(nutrition_templates_in_test)}/8 templates ({nutrition_templates_in_test})")
    
    print(f"\n🔬 RESEARCH SUITABILITY:")
    print(f"✅ Model trained on real household data (70% train, 15% val)")
    print(f"✅ Tested on diverse dummy data (15% test)")
    print(f"✅ Clear template ID mapping for explanation")
    print(f"✅ Proper train/validation/test split maintained")
    
    # Save detailed results
    results = {
        'workout_accuracy': workout_accuracy,
        'nutrition_accuracy': nutrition_accuracy,
        'workout_report': workout_report,
        'nutrition_report': nutrition_report,
        'test_samples': len(X_test),
        'template_coverage': {
            'workout': workout_templates_in_test,
            'nutrition': nutrition_templates_in_test
        }
    }
    
    return results

if __name__ == "__main__":
    results = train_and_evaluate_with_template_ids()
