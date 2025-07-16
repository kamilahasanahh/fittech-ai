#!/usr/bin/env python3
"""
XGFitness AI Visualization Runner
Run this script after training to generate comprehensive visualizations
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc

# Paths
MODEL_PATH = os.path.join('models', 'xgfitness_ai_model.pkl')
DATA_PATH = 'training_data.csv'
OUTPUT_DIR = os.path.join('backend', 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
workout_model = model_dict['workout_model']
nutrition_model = model_dict['nutrition_model']
scaler = model_dict['scaler']
workout_id_reverse_map = model_dict['workout_id_reverse_map']
nutrition_id_reverse_map = model_dict['nutrition_id_reverse_map']

# Load data
full_df = pd.read_csv(DATA_PATH)

# Features used for prediction
feature_cols = [
    'age', 'gender_encoded', 'height_cm', 'weight_kg',
    'bmi', 'goal_encoded', 'activity_level_encoded',
    'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
    'BMR_per_weight', 'TDEE_BMR_ratio', 'activity_efficiency',
    'BMI_deviation', 'weight_height_ratio', 'metabolic_score',
    'high_metabolism', 'very_active', 'young_adult', 'optimal_BMI'
]

splits = ['train', 'validation', 'test']
label_types = [
    ('workout_template_id', workout_model, workout_id_reverse_map, 'Workout'),
    ('nutrition_template_id', nutrition_model, nutrition_id_reverse_map, 'Nutrition')
]

for split in splits:
    split_df = full_df[full_df['split'] == split]
    X = split_df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    for label_col, model, id_reverse_map, label_name in label_types:
        y_true = split_df[label_col].values
        # Predict
        y_pred_enc = model.predict(X_scaled)
        # Decode predictions
        y_pred = [id_reverse_map.get(i, np.nan) for i in y_pred_enc]
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_true) | set(y_pred)))
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f'{split.capitalize()} {label_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{split}_{label_name.lower()}_confusion.png'))
        plt.close()
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, f'{split}_{label_name.lower()}_classification_report.csv'))
        # AUROC (one-vs-rest, macro average)
        if hasattr(model, 'predict_proba'):
            y_true_bin = pd.get_dummies(y_true)
            y_score = model.predict_proba(X_scaled)
            # Ensure columns match
            aucs = []
            for i, class_label in enumerate(y_true_bin.columns):
                if i < y_score.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true_bin.iloc[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    plt.plot(fpr, tpr, label=f'Class {class_label} (AUC={roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'{split.capitalize()} {label_name} ROC Curve (OvR)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{split}_{label_name.lower()}_roc.png'))
            plt.close()
