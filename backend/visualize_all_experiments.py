import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

EXPERIMENTS = [
    ("REAL", "backend/outputs/real"),
    ("AUGMENTED", "backend/outputs/augmented"),
    ("FULLY_SYNTHETIC", "backend/outputs/fully_synthetic"),
]
SPLITS = ["train", "val", "test"]
LABEL_TYPES = [
    ("nutrition_template_id", "Nutrition"),
    ("workout_template_id", "Workout"),
]
OUTPUT_DIR = "backend/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load template mappings for axis labels
def load_template_mapping(json_path, id_key, desc_keys):
    df = pd.read_json(json_path)
    return {row[id_key]: ' | '.join(str(row[k]) for k in desc_keys) for _, row in df.iterrows()}

nutrition_map = load_template_mapping("backend/data/nutrition_templates.json", "template_id", ["goal", "bmi_category"])
workout_map = load_template_mapping("backend/data/workout_templates.json", "template_id", ["goal", "activity_level"])

for exp_name, exp_dir in EXPERIMENTS:
    # Load model for this experiment
    model_path = os.path.join(exp_dir, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    xgb_workout = model_data["workout_model"]
    xgb_nutrition = model_data["nutrition_model"]
    scaler = model_data["scaler"]
    workout_id_reverse_map = model_data["workout_id_reverse_map"]
    nutrition_id_reverse_map = model_data["nutrition_id_reverse_map"]
    # For each split
    for split in SPLITS:
        path = os.path.join(exp_dir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # Features for prediction
        feature_cols = [
            'age', 'gender_encoded', 'height_cm', 'weight_kg',
            'bmi', 'goal_encoded', 'activity_level_encoded',
            'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
            'BMR_per_weight', 'TDEE_BMR_ratio', 'activity_efficiency',
            'BMI_deviation', 'weight_height_ratio', 'metabolic_score',
            'high_metabolism', 'very_active', 'young_adult', 'optimal_BMI'
        ]
        X = df[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X)
        for label_col, label_name in LABEL_TYPES:
            if label_col not in df.columns:
                continue
            y_true = np.array(df[label_col].values)
            # Predict using model
            if label_name == "Workout":
                model = xgb_workout
                id_reverse_map = workout_id_reverse_map
                id_map = workout_map
            else:
                model = xgb_nutrition
                id_reverse_map = nutrition_id_reverse_map
                id_map = nutrition_map
            # Encode y_true to 0-based indices for comparison
            unique_ids = sorted(set(y_true) | set(id_reverse_map.values()))
            id_to_idx = {tid: idx for idx, tid in enumerate(unique_ids)}
            idx_to_id = {idx: tid for tid, idx in id_to_idx.items()}
            y_true_idx = np.array([id_to_idx[tid] for tid in y_true])
            # Predict and decode
            y_pred_idx = model.predict(X_scaled)
            y_pred = np.array([id_reverse_map.get(idx_to_id.get(idx, -1), -1) for idx in y_pred_idx])
            # Confusion Matrix
            labels = sorted(set(y_true) | set(y_pred))
            label_names = [id_map.get(l, str(l)) for l in labels]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
            disp.plot(cmap='Blues')
            plt.title(f'{exp_name} {split.capitalize()} {label_name} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{exp_name}_{split}_{label_name}_confusion.png'))
            plt.close()
            # Classification Report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division='warn')
            pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, f'{exp_name}_{split}_{label_name}_classification_report.csv'))
            # AUROC (one-vs-rest)
            classes = np.unique(y_true)
            if len(classes) > 1:
                y_true_bin = label_binarize(y_true, classes=classes)
                y_pred_bin = label_binarize(y_pred, classes=classes)
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i, c in enumerate(classes):
                    fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
                    roc_auc[c] = auc(fpr[c], tpr[c])
                plt.figure()
                for c in classes:
                    plt.plot(fpr[c], tpr[c], label=f'{id_map.get(c, c)} (AUC = {roc_auc[c]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{exp_name} {split.capitalize()} {label_name} AUROC')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'{exp_name}_{split}_{label_name}_auroc.png'))
                plt.close() 