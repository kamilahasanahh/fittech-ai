#!/usr/bin/env python3
"""
Standalone Advanced Visualization Suite for FitTech AI
Automatically loads the latest model and CSVs, generates comprehensive plots for XGBoost and Random Forest models.
"""
import os
import glob
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.utils.validation import check_is_fitted

# --- Utility functions ---
def find_latest_file(patterns, directory):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(directory, pat)))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def ensure_label_encoder_fitted(encoder, y, extra_classes=None):
    try:
        check_is_fitted(encoder)
    except Exception:
        classes = list(set(list(y)))
        if extra_classes:
            classes = list(set(classes + list(extra_classes)))
        encoder.fit(classes)
    return encoder

def ensure_scaler_fitted(scaler, X):
    try:
        check_is_fitted(scaler)
    except Exception:
        scaler.fit(X)
    return scaler

def add_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    return df[columns]

# --- Main script ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = BASE_DIR
    VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
    os.makedirs(VIZ_DIR, exist_ok=True)

    # 1. Find latest model
    model_file = find_latest_file([
        "research_model_comparison.pkl", "xgfitness_ai_model.pkl"
    ], MODEL_DIR)
    if not model_file:
        print("❌ No model pickle found in models/")
        exit(1)
    print(f"📥 Using model: {model_file}")

    # 2. Load model
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    class ModelWrapper:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    model = ModelWrapper(model_data)

    # 3. Load train/val/test CSVs directly
    train_df = pd.read_csv(os.path.join(DATA_DIR, "strict_balanced_train_250.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "real_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "real_test.csv"))

    # 4. Use test_df directly for test set visualizations
    feature_cols = [
        'age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'tdee',
        'activity_multiplier', 'Mod_act', 'Vig_act'
    ]
    test_df = test_df.copy()
    X_test = test_df[feature_cols]
    y_w_test = test_df['workout_template_id']
    y_n_test = test_df['nutrition_template_id']

    # 5. Setup output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(VIZ_DIR, f"advanced_run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 6. Fit encoders/scalers if needed
    setattr(model, 'workout_label_encoder', ensure_label_encoder_fitted(getattr(model, 'workout_label_encoder', LabelEncoder()), y_w_test))
    setattr(model, 'nutrition_label_encoder', ensure_label_encoder_fitted(getattr(model, 'nutrition_label_encoder', LabelEncoder()), y_n_test))
    setattr(model, 'scaler', ensure_scaler_fitted(getattr(model, 'scaler', None), X_test))
    if hasattr(model, 'rf_scaler') or getattr(model, 'rf_scaler', None) is not None:
        setattr(model, 'rf_scaler', ensure_scaler_fitted(getattr(model, 'rf_scaler', None), X_test))

    # 7. Encode labels
    y_w_test_enc = getattr(model, 'workout_label_encoder', LabelEncoder()).transform(y_w_test)
    y_n_test_enc = getattr(model, 'nutrition_label_encoder', LabelEncoder()).transform(y_n_test)

    # 8. Scale features
    X_test_xgb = getattr(model, 'scaler', None)
    if X_test_xgb is not None:
        X_test_xgb = X_test_xgb.transform(X_test)
    else:
        X_test_xgb = X_test
    X_test_rf = getattr(model, 'rf_scaler', None)
    if X_test_rf is not None:
        X_test_rf = X_test_rf.transform(X_test)
    else:
        X_test_rf = X_test_xgb

    # 9. Class labels
    workout_labels = [f"W{i+1}" for i in range(len(np.unique(y_w_test_enc)))]
    nutrition_labels = [f"N{i+1}" for i in range(len(np.unique(y_n_test_enc)))]

    # --- Plotting functions ---
    def plot_confusion(y_true, y_pred, labels, title, fname):
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        label_names = [f"W{l}" if 'workout' in title.lower() else f"N{l}" for l in unique_labels]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    def plot_roc(y_true, y_score, n_classes, labels, title, fname):
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{labels[i]} (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    def plot_per_class(y_true, y_pred, labels, title, fname):
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        label_names = [f"W{l}" if 'workout' in title.lower() else f"N{l}" for l in unique_labels]
        report = classification_report(y_true, y_pred, labels=unique_labels, target_names=label_names, output_dict=True, zero_division="warn")
        metrics = ['precision', 'recall', 'f1-score']
        data = {m: [] for m in metrics}
        for l in label_names:
            if isinstance(report, dict) and l in report and isinstance(report[l], dict):
                for m in metrics:
                    data[m].append(report[l][m])
            else:
                available_keys = list(report.keys()) if isinstance(report, dict) else str(type(report))
                print(f"Label {l} not found in report. Available keys: {available_keys}")
                for m in metrics:
                    data[m].append(0)
        x = np.arange(len(label_names))
        width = 0.2
        plt.figure(figsize=(12, 6))
        for i, m in enumerate(metrics):
            plt.bar(x + i*width, data[m], width, label=m)
        plt.xticks(x + width, label_names, rotation=45)
        plt.ylim(0, 1)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    def plot_feature_importance(importances, features, title, fname):
        idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[idx], tick_label=np.array(features)[idx])
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    # --- XGBoost Plots ---
    print("\n🎨 Generating XGBoost visualizations...")
    workout_model = getattr(model, 'workout_model', None)
    nutrition_model = getattr(model, 'nutrition_model', None)
    if workout_model is not None and nutrition_model is not None:
        xgb_w_pred = workout_model.predict(X_test_xgb)
        xgb_n_pred = nutrition_model.predict(X_test_xgb)
        xgb_w_proba = workout_model.predict_proba(X_test_xgb)
        xgb_n_proba = nutrition_model.predict_proba(X_test_xgb)
        plot_confusion(y_w_test_enc, xgb_w_pred, workout_labels, "XGBoost Workout Confusion Matrix", "xgb_workout_confusion.png")
        plot_confusion(y_n_test_enc, xgb_n_pred, nutrition_labels, "XGBoost Nutrition Confusion Matrix", "xgb_nutrition_confusion.png")
        plot_roc(y_w_test_enc, xgb_w_proba, len(workout_labels), workout_labels, "XGBoost Workout ROC", "xgb_workout_roc.png")
        plot_roc(y_n_test_enc, xgb_n_proba, len(nutrition_labels), nutrition_labels, "XGBoost Nutrition ROC", "xgb_nutrition_roc.png")
        plot_per_class(y_w_test_enc, xgb_w_pred, workout_labels, "XGBoost Workout Per-Class Metrics", "xgb_workout_perclass.png")
        plot_per_class(y_n_test_enc, xgb_n_pred, nutrition_labels, "XGBoost Nutrition Per-Class Metrics", "xgb_nutrition_perclass.png")
        if hasattr(workout_model, 'feature_importances_'):
            plot_feature_importance(workout_model.feature_importances_, feature_cols, "XGBoost Workout Feature Importance", "xgb_workout_featimp.png")
        if hasattr(nutrition_model, 'feature_importances_'):
            plot_feature_importance(nutrition_model.feature_importances_, feature_cols, "XGBoost Nutrition Feature Importance", "xgb_nutrition_featimp.png")

    # --- Random Forest Plots ---
    workout_rf_model = getattr(model, 'workout_rf_model', None)
    nutrition_rf_model = getattr(model, 'nutrition_rf_model', None)
    if workout_rf_model is not None and nutrition_rf_model is not None:
        print("\n🎨 Generating Random Forest visualizations...")
        rf_w_pred = workout_rf_model.predict(X_test_rf)
        rf_n_pred = nutrition_rf_model.predict(X_test_rf)
        rf_w_proba = workout_rf_model.predict_proba(X_test_rf)
        rf_n_proba = nutrition_rf_model.predict_proba(X_test_rf)
        plot_confusion(y_w_test_enc, rf_w_pred, workout_labels, "RF Workout Confusion Matrix", "rf_workout_confusion.png")
        plot_confusion(y_n_test_enc, rf_n_pred, nutrition_labels, "RF Nutrition Confusion Matrix", "rf_nutrition_confusion.png")
        plot_roc(y_w_test_enc, rf_w_proba, len(workout_labels), workout_labels, "RF Workout ROC", "rf_workout_roc.png")
        plot_roc(y_n_test_enc, rf_n_proba, len(nutrition_labels), nutrition_labels, "RF Nutrition ROC", "rf_nutrition_roc.png")
        plot_per_class(y_w_test_enc, rf_w_pred, workout_labels, "RF Workout Per-Class Metrics", "rf_workout_perclass.png")
        plot_per_class(y_n_test_enc, rf_n_pred, nutrition_labels, "RF Nutrition Per-Class Metrics", "rf_nutrition_perclass.png")
        if hasattr(workout_rf_model, 'feature_importances_'):
            plot_feature_importance(workout_rf_model.feature_importances_, feature_cols, "RF Workout Feature Importance", "rf_workout_featimp.png")
        if hasattr(nutrition_rf_model, 'feature_importances_'):
            plot_feature_importance(nutrition_rf_model.feature_importances_, feature_cols, "RF Nutrition Feature Importance", "rf_nutrition_featimp.png")

    print(f"\n✅ All advanced visualizations saved to: {out_dir}\n") 