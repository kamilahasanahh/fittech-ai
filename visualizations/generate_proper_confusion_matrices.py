#!/usr/bin/env python3
"""
Generate proper confusion matrices and metrics from existing trained model
Using actual template IDs (W1-W9, N1-N8) as requested
"""
import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
from thesis_model import XGFitnessAIModel

def generate_template_confusion_matrices():
    print("=" * 60)
    print("GENERATING CONFUSION MATRICES FOR TEMPLATE PREDICTIONS")
    print("=" * 60)
    
    # 1. Load existing trained model
    print("\n1. Loading existing trained model...")
    model = XGFitnessAIModel('../data')
    model.load_model('models/xgfitness_ai_model.pkl')
    print("✅ Model loaded successfully")
    
    # 2. Create test dataset (same as model was trained on)
    print("\n2. Creating test dataset with 70/15/15 split...")
    df = model.create_training_dataset(
        real_data_file='../data/backups/e267_Data on age, gender, height, weight, activity levels for each household member.txt',
        total_samples=1500  # Larger dataset for better evaluation
    )
    
    # 3. Get test data only
    test_df = df[df['split'] == 'test'].copy()
    print(f"✅ Test dataset: {len(test_df)} samples")
    print(f"   Real data: {len(test_df[test_df['data_source'] == 'real'])}")
    print(f"   Dummy data: {len(test_df[test_df['data_source'] == 'dummy'])}")
    
    # 4. Prepare test features
    X, y_workout, y_nutrition, df_enhanced = model.prepare_training_data(test_df)
    test_mask = df_enhanced['split'] == 'test'
    X_test = X[test_mask]
    y_w_test = y_workout[test_mask]
    y_n_test = y_nutrition[test_mask]
    
    # Scale features
    X_test_scaled = model.scaler.transform(X_test)
    
    # 5. Make predictions
    print("\n3. Making predictions on test set...")
    y_w_pred = model.workout_model.predict(X_test_scaled)
    y_n_pred = model.nutrition_model.predict(X_test_scaled)
    y_w_pred_proba = model.workout_model.predict_proba(X_test_scaled)
    y_n_pred_proba = model.nutrition_model.predict_proba(X_test_scaled)
    
    # 6. Convert encoded predictions back to template IDs
    # The model uses LabelEncoder, so we need to map back to template IDs
    # Workout templates: 1-9, Nutrition templates: 1-8
    
    # Get unique template IDs that were actually used
    workout_templates_used = sorted(y_w_test.unique())
    nutrition_templates_used = sorted(y_n_test.unique())
    
    print(f"✅ Workout templates in test set: {workout_templates_used}")
    print(f"✅ Nutrition templates in test set: {nutrition_templates_used}")
    
    # 7. Generate confusion matrices
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Workout confusion matrix
    workout_cm = confusion_matrix(y_w_test, y_w_pred)
    workout_labels = [f'W{i}' for i in workout_templates_used]
    
    sns.heatmap(workout_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=workout_labels, yticklabels=workout_labels, ax=ax1)
    ax1.set_title('Workout Template Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Template')
    ax1.set_ylabel('True Template')
    
    # Nutrition confusion matrix
    nutrition_cm = confusion_matrix(y_n_test, y_n_pred)
    nutrition_labels = [f'N{i}' for i in nutrition_templates_used]
    
    sns.heatmap(nutrition_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=nutrition_labels, yticklabels=nutrition_labels, ax=ax2)
    ax2.set_title('Nutrition Template Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Template')
    ax2.set_ylabel('True Template')
    
    # 8. Calculate detailed metrics
    print("\n4. Calculating detailed metrics...")
    
    # Workout metrics
    workout_accuracy = accuracy_score(y_w_test, y_w_pred)
    workout_precision, workout_recall, workout_f1, workout_support = precision_recall_fscore_support(
        y_w_test, y_w_pred, average=None, zero_division=0
    )
    workout_macro_precision = np.mean(workout_precision)
    workout_macro_recall = np.mean(workout_recall)
    workout_macro_f1 = np.mean(workout_f1)
    
    # Nutrition metrics
    nutrition_accuracy = accuracy_score(y_n_test, y_n_pred)
    nutrition_precision, nutrition_recall, nutrition_f1, nutrition_support = precision_recall_fscore_support(
        y_n_test, y_n_pred, average=None, zero_division=0
    )
    nutrition_macro_precision = np.mean(nutrition_precision)
    nutrition_macro_recall = np.mean(nutrition_recall)
    nutrition_macro_f1 = np.mean(nutrition_f1)
    
    # 9. Create metrics visualization
    metrics_data = {
        'Model': ['Workout', 'Nutrition'],
        'Accuracy': [workout_accuracy, nutrition_accuracy],
        'Precision': [workout_macro_precision, nutrition_macro_precision],
        'Recall': [workout_macro_recall, nutrition_macro_recall],
        'F1-Score': [workout_macro_f1, nutrition_macro_f1]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot metrics comparison
    x = np.arange(len(metrics_data['Model']))
    width = 0.2
    
    ax3.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax3.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax3.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax3.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_df['Model'])
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, model in enumerate(metrics_df['Model']):
        ax3.text(i-1.5*width, metrics_df.iloc[i]['Accuracy'] + 0.01, 
                f'{metrics_df.iloc[i]["Accuracy"]:.3f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i-0.5*width, metrics_df.iloc[i]['Precision'] + 0.01, 
                f'{metrics_df.iloc[i]["Precision"]:.3f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i+0.5*width, metrics_df.iloc[i]['Recall'] + 0.01, 
                f'{metrics_df.iloc[i]["Recall"]:.3f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i+1.5*width, metrics_df.iloc[i]['F1-Score'] + 0.01, 
                f'{metrics_df.iloc[i]["F1-Score"]:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 10. Template coverage analysis
    ax4.axis('off')
    ax4.text(0.1, 0.9, 'Template Coverage Analysis', fontsize=16, fontweight='bold', transform=ax4.transAxes)
    
    # Workout template coverage
    ax4.text(0.1, 0.8, 'Workout Templates:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    for i, (template_id, support) in enumerate(zip(workout_templates_used, workout_support)):
        precision = workout_precision[i]
        recall = workout_recall[i]
        f1 = workout_f1[i]
        ax4.text(0.1, 0.75-i*0.05, f'W{template_id}: {support} samples | P={precision:.3f} R={recall:.3f} F1={f1:.3f}', 
                fontsize=10, transform=ax4.transAxes)
    
    # Nutrition template coverage
    ax4.text(0.1, 0.45, 'Nutrition Templates:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    for i, (template_id, support) in enumerate(zip(nutrition_templates_used, nutrition_support)):
        precision = nutrition_precision[i]
        recall = nutrition_recall[i]
        f1 = nutrition_f1[i]
        ax4.text(0.1, 0.4-i*0.05, f'N{template_id}: {support} samples | P={precision:.3f} R={recall:.3f} F1={f1:.3f}', 
                fontsize=10, transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('template_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 11. Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"Workout Model Accuracy: {workout_accuracy:.4f} ({workout_accuracy:.1%})")
    print(f"Nutrition Model Accuracy: {nutrition_accuracy:.4f} ({nutrition_accuracy:.1%})")
    
    print(f"\n🎯 WORKOUT MODEL METRICS:")
    print(f"Macro Precision: {workout_macro_precision:.4f}")
    print(f"Macro Recall: {workout_macro_recall:.4f}")
    print(f"Macro F1-Score: {workout_macro_f1:.4f}")
    
    print(f"\n🥗 NUTRITION MODEL METRICS:")
    print(f"Macro Precision: {nutrition_macro_precision:.4f}")
    print(f"Macro Recall: {nutrition_macro_recall:.4f}")
    print(f"Macro F1-Score: {nutrition_macro_f1:.4f}")
    
    print(f"\n📋 TEMPLATE COVERAGE:")
    print(f"Workout Templates Used: {len(workout_templates_used)}/9 ({workout_templates_used})")
    print(f"Nutrition Templates Used: {len(nutrition_templates_used)}/8 ({nutrition_templates_used})")
    
    print(f"\n🎯 ASSESSMENT:")
    if workout_accuracy >= 0.8 and nutrition_accuracy >= 0.8:
        print("✅ Both models meet the 0.8 target accuracy!")
    elif workout_accuracy >= 0.8:
        print("✅ Workout model meets 0.8 target, nutrition model needs improvement")
    elif nutrition_accuracy >= 0.8:
        print("✅ Nutrition model meets 0.8 target, workout model needs improvement")
    else:
        print("⚠️ Both models below 0.8 target - consider model tuning")
    
    return {
        'workout_accuracy': workout_accuracy,
        'nutrition_accuracy': nutrition_accuracy,
        'workout_metrics': {
            'precision': workout_macro_precision,
            'recall': workout_macro_recall,
            'f1': workout_macro_f1
        },
        'nutrition_metrics': {
            'precision': nutrition_macro_precision,
            'recall': nutrition_macro_recall,
            'f1': nutrition_macro_f1
        }
    }

if __name__ == "__main__":
    results = generate_template_confusion_matrices()
