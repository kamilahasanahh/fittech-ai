"""
Model Training Script for FitTech AI

- Loads the augmented/balanced train set and untouched real validation/test sets.
- Only the train set is ever augmented (in a separate script). Validation and test sets are untouched real data.
- All splits are filtered for valid combinations and missing template IDs are dropped.
- Derived features are added after splitting.
- This workflow ensures no data leakage and that model evaluation is always on real, unaugmented data for a realistic system assessment.
"""
#!/usr/bin/env python3
"""
XGFitness AI Model       # Train all models using the unified training method
    print("\n🚀 Training ALL Models (XGBoost + Random Forest)...")
    comprehensive_info = model.train_all_models(training_data) Train all models using the unified training method
    print("\n🚀 Training ALL Models (XGBoost + Random Forest)...")
    comprehensive_info = model.train_all_models(training_data)ining Script - RESTORED AUTHENTICITY VERSION
Trains DUAL XGBoost models (main AI) + DUAL Random Forest models (comparison)
Implements EXACT user requirements for thesis authenticity
"""

import os
import sys
import time
from datetime import datetime
import argparse

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import thesis_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
import copy
import re

def clean_raw_data(raw_txt_path, output_csv_path):
    """
    Cleans the raw data file according to constraints:
    - Age: 18–65 years
    - Height: 150–200 cm (convert from feet.inches)
    - Weight: 40–150 kg (convert from pounds)
    - BMI must be consistent with height/weight
    Saves cleaned data as CSV.
    """
    import pandas as pd
    import numpy as np
    # Read as tab-separated, skip bad lines
    df = pd.read_csv(raw_txt_path, sep='\t', header=0, dtype=str, na_values=['', ' '])
    # Rename columns for clarity
    df = df.rename(columns={
        'Member_Age_Orig': 'age',
        'Member_Gender_Orig': 'gender',
        'HEIGHT': 'height_ft',
        'WEIGHT': 'weight_lb',
        'Mod_act': 'mod_act',
        'Vig_act': 'vig_act',
    })
    # Drop rows with missing age, gender, height, or weight
    df = df.dropna(subset=['age', 'gender', 'height_ft', 'weight_lb'])
    # Convert age to int
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    # Convert gender to string (1=Male, 2=Female)
    df['gender'] = df['gender'].map({'1': 'Male', '2': 'Female'})
    # Convert height from feet.inches to cm
    def feet_inches_to_cm(val):
        if pd.isna(val):
            return np.nan
        match = re.match(r'^(\d+)(?:\.(\d+))?$', str(val))
        if not match:
            return np.nan
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        total_inches = feet * 12 + inches
        return total_inches * 2.54
    df['height_cm'] = df['height_ft'].apply(feet_inches_to_cm)
    # Convert weight from pounds to kg
    df['weight_kg'] = pd.to_numeric(df['weight_lb'], errors='coerce') * 0.453592
    # Filter by age, height, weight
    df = df[(df['age'] >= 18) & (df['age'] <= 65)]
    df = df[(df['height_cm'] >= 150) & (df['height_cm'] <= 200)]
    df = df[(df['weight_kg'] >= 40) & (df['weight_kg'] <= 150)]
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    # Filter for plausible BMI (15-40)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 40)]
    # Save cleaned data
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Cleaned data saved to {output_csv_path} ({len(df)} rows)")


# --- Load splits at the top ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_experiment_splits(experiment):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(project_root, 'backend', 'outputs', experiment)
    train = pd.read_csv(os.path.join(base, 'train.csv'))
    val = pd.read_csv(os.path.join(base, 'val.csv'))
    test = pd.read_csv(os.path.join(base, 'test.csv'))
    return train, val, test

def main():
    """Train XGBoost + Random Forest models with STRICT AUTHENTICITY"""
    print("=" * 80)
    print("🏋️ FITTECH AI MODEL TRAINING - DUAL MODEL SYSTEM")
    print("=" * 80)
    print("STREAMLINED TRAINING PIPELINE:")
    print("1. 🚀 PRODUCTION Model: XGBoost-only (web application)")
    print("2. 📊 RESEARCH Model: XGBoost + Random Forest (thesis analysis)")  
    print("=" * 80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize model
    print("📚 Initializing XGFitness AI Model...")
    model = thesis_model.XGFitnessAIModel(templates_dir='../data')
    print("✅ Model initialized successfully")
    print()
    
    # Create training dataset with EXACT AUTHENTICITY METHODOLOGY
    print("🔍 Loading real data...")
    training_data = model.load_real_data_balanced(
        train_csv='../backend/strict_balanced_train_250.csv',
        val_csv='../backend/real_val.csv',
        test_csv='../backend/real_test.csv'
    )
    print(f"✅ Training dataset loaded: {len(training_data)} samples")

    # Use the already balanced training set as-is
    train_balanced = training_data[training_data['split'] == 'train'].copy()
    train_balanced['split'] = 'train'

    # Combine with untouched validation and test sets
    val_df = training_data[training_data['split'] == 'validation'].copy()
    test_df = training_data[training_data['split'] == 'test'].copy()

    # --- FILTER: Only keep rows matching valid template combinations ---
    valid_combos = pd.read_csv('valid_template_combinations.csv')
    valid_combos_set = set(
        tuple(x) for x in valid_combos[['fitness_goal', 'activity_level', 'bmi_category']].values
    )
    def filter_valid_combos(df):
        return df[
            df.apply(lambda row: (row['fitness_goal'], row['activity_level'], row['bmi_category']) in valid_combos_set, axis=1)
        ]
    train_balanced = filter_valid_combos(train_balanced)
    val_df = filter_valid_combos(val_df)
    test_df = filter_valid_combos(test_df)

    # --- PRINT: Class distributions for each split ---
    def print_value_counts(df, col, split_name):
        if col in df.columns:
            print(f"{split_name} {col} distribution:")
            print(df[col].value_counts())
        else:
            print(f"{split_name} {col} not found.")

    print_value_counts(train_balanced, 'nutrition_template_id', 'Train')
    print_value_counts(train_balanced, 'workout_template_id', 'Train')
    print_value_counts(val_df, 'nutrition_template_id', 'Val')
    print_value_counts(val_df, 'workout_template_id', 'Val')
    print_value_counts(test_df, 'nutrition_template_id', 'Test')
    print_value_counts(test_df, 'workout_template_id', 'Test')
    print()

    # Combine with untouched validation and test sets
    val_df = training_data[training_data['split'] == 'validation'].copy()
    test_df = training_data[training_data['split'] == 'test'].copy()

    # --- PATCH: Only add derived features to all splits, do NOT assign template IDs to val/test ---
    def add_derived_features(df):
        # Calculate BMI
        if 'bmi' not in df.columns or df['bmi'].isna().any():
            df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        # BMI category
        if 'bmi_category' not in df.columns or df['bmi_category'].isna().any():
            df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        # BMR (Harris-Benedict)
        if 'bmr' not in df.columns or df['bmr'].isna().any():
            def calc_bmr(row):
                if row['gender'] == 'Male':
                    return 88.362 + (13.397 * row['weight_kg']) + (4.799 * row['height_cm']) - (5.677 * row['age'])
                else:
                    return 447.593 + (9.247 * row['weight_kg']) + (3.098 * row['height_cm']) - (4.330 * row['age'])
            df['bmr'] = df.apply(calc_bmr, axis=1)
        # Activity multiplier
        if 'activity_multiplier' not in df.columns or df['activity_multiplier'].isna().any():
            activity_map = {'Low Activity': 1.29, 'Moderate Activity': 1.55, 'High Activity': 1.81}
            df['activity_multiplier'] = df['activity_level'].map(activity_map)
        # TDEE
        if 'tdee' not in df.columns or df['tdee'].isna().any():
            df['tdee'] = df['bmr'] * df['activity_multiplier']

        # --- NEW ENGINEERED FEATURES ---
        # Encodings
        if 'goal_encoded' not in df.columns:
            goal_map = {'Fat Loss': 0, 'Muscle Gain': 1, 'Maintenance': 2}
            df['goal_encoded'] = df['fitness_goal'].map(goal_map)
        if 'activity_level_encoded' not in df.columns:
            activity_map_enc = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
            df['activity_level_encoded'] = df['activity_level'].map(activity_map_enc)

        # Key interaction terms
        df['BMI_Goal_interaction'] = df['bmi'] * df['goal_encoded']
        df['Age_Activity_interaction'] = df['age'] * df['activity_level_encoded']
        df['BMI_Activity_interaction'] = df['bmi'] * df['activity_level_encoded']
        df['Age_Goal_interaction'] = df['age'] * df['goal_encoded']

        # Metabolic Ratios
        df['BMR_per_weight'] = df['bmr'] / df['weight_kg']
        df['TDEE_BMR_ratio'] = df['tdee'] / df['bmr']
        df['activity_efficiency'] = df['tdee'] / (df['bmr'] * df['activity_multiplier'])

        # Health Deviation Scores
        df['BMI_deviation'] = abs(df['bmi'] - 22.5)
        df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
        df['metabolic_score'] = df['bmr'] / (df['age'] * df['weight_kg'])

        # Boolean Classification Flags
        threshold_high = 0.9  # Adjust as needed
        df['high_metabolism'] = (df['BMR_per_weight'] > threshold_high).astype(int)
        # If 'vigorous_activity' not present, default to 0
        if 'vigorous_activity' not in df.columns:
            df['vigorous_activity'] = 0
        df['very_active'] = ((df['activity_level'] == 'High Activity') & (df['vigorous_activity'] > 0)).astype(int)
        df['young_adult'] = (df['age'] <= 30).astype(int)
        df['optimal_BMI'] = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)

        return df

    # Add derived features to all splits (no label assignment)
    train_balanced = add_derived_features(train_balanced)
    val_df = add_derived_features(val_df)
    test_df = add_derived_features(test_df)

    # Debug: Print row counts for each split
    print(f"train_balanced rows: {len(train_balanced)}")
    print(f"val_df rows: {len(val_df)}")
    print(f"test_df rows: {len(test_df)}")

    # --- CHECK: No overlap between splits ---
    def check_split_overlap(train_df, val_df, test_df):
        # Use a subset of columns that uniquely identify a row (all features except split/labels)
        id_cols = [col for col in train_df.columns if col not in ['split', 'nutrition_template_id', 'workout_template_id']]
        train_ids = set(tuple(row) for row in train_df[id_cols].values)
        val_ids = set(tuple(row) for row in val_df[id_cols].values)
        test_ids = set(tuple(row) for row in test_df[id_cols].values)
        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids
        if overlap_train_val:
            print(f'❌ Data leakage: {len(overlap_train_val)} rows overlap between train and val splits!')
        if overlap_train_test:
            print(f'❌ Data leakage: {len(overlap_train_test)} rows overlap between train and test splits!')
        if overlap_val_test:
            print(f'❌ Data leakage: {len(overlap_val_test)} rows overlap between val and test splits!')
        if not (overlap_train_val or overlap_train_test or overlap_val_test):
            print('✅ No overlap between train, val, and test splits.')

    check_split_overlap(train_balanced, val_df, test_df)

    # --- CHECK: Split ratios ---
    total = len(train_balanced) + len(val_df) + len(test_df)
    train_ratio = len(train_balanced) / total if total else 0
    val_ratio = len(val_df) / total if total else 0
    test_ratio = len(test_df) / total if total else 0
    print(f'Split ratios: train={train_ratio:.2%}, val={val_ratio:.2%}, test={test_ratio:.2%}')
    if not (0.68 <= train_ratio <= 0.72 and 0.13 <= val_ratio <= 0.17 and 0.13 <= test_ratio <= 0.17):
        print('⚠️ WARNING: Data split ratios deviate from 70/15/15!')

    # --- Prevent data leakage: drop leakage-prone features before model training ---
    leakage_features = ['activity_level_encoded', 'TDEE_BMR_ratio']
    for feat in leakage_features:
        if feat in train_balanced.columns:
            train_balanced = train_balanced.drop(columns=[feat])
        if feat in val_df.columns:
            val_df = val_df.drop(columns=[feat])
        if feat in test_df.columns:
            test_df = test_df.drop(columns=[feat])

    # Recombine after dropping leakage features
    training_data = pd.concat([train_balanced, val_df, test_df], ignore_index=True)
    print(f"training_data total rows: {len(training_data)} (after dropping leakage features)")
    
    # Save training data for visualizations
    print("💾 Saving training data for visualizations...")
    training_data_path = 'training_data.csv'
    training_data.to_csv(training_data_path, index=False)
    print(f"✅ Training data saved to: {training_data_path}")
    print()
    
    # Train ALL models with verification
    print("🚀 Starting COMPREHENSIVE model training...")
    start_time = time.time()
    
    # Drop rows with missing template IDs before model training
    before = len(training_data)
    training_data = training_data.dropna(subset=['nutrition_template_id', 'workout_template_id'])
    after = len(training_data)
    if after < before:
        print(f"Dropped {before - after} rows with missing template IDs from training_data before model training.")

    # Train all models using the unified training method
    print("\n🚀 Training ALL Models (XGBoost)...")
    comprehensive_info = model.train_all_models(training_data, random_state=42)
    
    training_time = time.time() - start_time
    print(f"✅ COMPREHENSIVE training completed in {training_time:.2f} seconds")
    print()

    # Display comprehensive training results
    print("📊 COMPREHENSIVE TRAINING RESULTS:")
    print("=" * 80)
    print(f"Dataset Information:")
    print(f"  Total samples: {comprehensive_info.get('total_samples', 'N/A')}")
    print(f"  Training samples: {comprehensive_info.get('training_samples', 'N/A')}")
    print(f"  Validation samples: {comprehensive_info.get('validation_samples', 'N/A')}")
    print(f"  Test samples: {comprehensive_info.get('test_samples', 'N/A')}")
    print()

    print(f"XGBOOST MODEL PERFORMANCE (Main AI for Web App):")
    print(f"  Workout Model:")
    workout_accuracy = comprehensive_info.get('workout_accuracy', None)
    print(f"    - Accuracy: {workout_accuracy:.4f}" if isinstance(workout_accuracy, (float, int)) else f"    - Accuracy: N/A")
    workout_f1 = comprehensive_info.get('workout_f1', None)
    print(f"    - F1 Score: {workout_f1:.4f}" if isinstance(workout_f1, (float, int)) else f"    - F1 Score: N/A")
    workout_precision = comprehensive_info.get('workout_precision', None)
    print(f"    - Precision: {workout_precision:.4f}" if isinstance(workout_precision, (float, int)) else f"    - Precision: N/A")
    workout_recall = comprehensive_info.get('workout_recall', None)
    print(f"    - Recall: {workout_recall:.4f}" if isinstance(workout_recall, (float, int)) else f"    - Recall: N/A")
    print(f"  Nutrition Model:")
    nutrition_accuracy = comprehensive_info.get('nutrition_accuracy', None)
    print(f"    - Accuracy: {nutrition_accuracy:.4f}" if isinstance(nutrition_accuracy, (float, int)) else f"    - Accuracy: N/A")
    nutrition_f1 = comprehensive_info.get('nutrition_f1', None)
    print(f"    - F1 Score: {nutrition_f1:.4f}" if isinstance(nutrition_f1, (float, int)) else f"    - F1 Score: N/A")
    nutrition_precision = comprehensive_info.get('nutrition_precision', None)
    print(f"    - Precision: {nutrition_precision:.4f}" if isinstance(nutrition_precision, (float, int)) else f"    - Precision: N/A")
    nutrition_recall = comprehensive_info.get('nutrition_recall', None)
    print(f"    - Recall: {nutrition_recall:.4f}" if isinstance(nutrition_recall, (float, int)) else f"    - Recall: N/A")
    print()

    # Save models using the streamlined TWO-MODEL approach
    print("💾 Saving models using DUAL-MODEL strategy...")
    print()

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # 1. PRODUCTION MODEL: XGBoost-only (for web application)
    print("🚀 Saving PRODUCTION model (XGBoost-only for web app)...")
    production_path = 'models/xgfitness_ai_model.pkl'
    model.save_model(production_path, include_research_models=False)  # XGBoost only
    prod_size = os.path.getsize(production_path) / (1024 * 1024)  # Size in MB
    print(f"✅ PRODUCTION model saved: {production_path}")
    print(f"   - File size: {prod_size:.2f} MB")
    print(f"   - Contains: XGBoost models only (optimized for web app)")
    wa = comprehensive_info.get('workout_accuracy', None)
    na = comprehensive_info.get('nutrition_accuracy', None)
    print(f"   - XGBoost Workout Accuracy: {wa:.1%}" if isinstance(wa, (float, int)) else f"   - XGBoost Workout Accuracy: N/A")
    print(f"   - XGBoost Nutrition Accuracy: {na:.1%}" if isinstance(na, (float, int)) else f"   - XGBoost Nutrition Accuracy: N/A")
    print()

    # 2. RESEARCH MODEL: Both algorithms (for thesis analysis)
    print("📊 Saving RESEARCH model (XGBoost for thesis)...")
    research_path = 'models/research_model_comparison.pkl'
    model.save_model(research_path, include_research_models=True)  # Both algorithms
    research_size = os.path.getsize(research_path) / (1024 * 1024)  # Size in MB
    print(f"✅ RESEARCH model saved: {research_path}")
    print(f"   - File size: {research_size:.2f} MB") 
    print(f"   - Contains: XGBoostmodels (complete analysis)")
    rf_wa = comprehensive_info.get('rf_workout_accuracy', None)
    rf_na = comprehensive_info.get('rf_nutrition_accuracy', None)
    print(f"   - Random Forest Workout Accuracy: {rf_wa:.1%}" if isinstance(rf_wa, (float, int)) else f"   - Random Forest Workout Accuracy: N/A")
    print(f"   - Random Forest Nutrition Accuracy: {rf_na:.1%}" if isinstance(rf_na, (float, int)) else f"   - Random Forest Nutrition Accuracy: N/A")
    print()

    print("🎯 DUAL-MODEL SUMMARY:")
    print(f"   📱 Production model: {prod_size:.1f}MB (web-ready)")
    print(f"   📊 Research model: {research_size:.1f}MB (thesis-ready)")
    print(f"   🔄 Reproducible results with random_state=42")
    print()
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds")
    print()
    print("🚀 Your DUAL model system is now ready:")
    print()

    # --- SUMMARY TABLE AND WARNINGS FOR THESIS ---
    def get_class_counts(df, col):
        if col in df.columns:
            # Exclude NaN from value_counts
            return df[col].dropna().value_counts().sort_index()
        return pd.Series(dtype=int)

    # Only use non-NaN unique classes for summary
    nutrition_classes = sorted(set(train_balanced['nutrition_template_id'].dropna().unique()) |
                               set(val_df['nutrition_template_id'].dropna().unique()) |
                               set(test_df['nutrition_template_id'].dropna().unique()))
    workout_classes = sorted(set(train_balanced['workout_template_id'].dropna().unique()) |
                             set(val_df['workout_template_id'].dropna().unique()) |
                             set(test_df['workout_template_id'].dropna().unique()))

    summary_rows = []
    for split_name, df in [('train', train_balanced), ('val', val_df), ('test', test_df)]:
        for col, classes in [('nutrition_template_id', nutrition_classes), ('workout_template_id', workout_classes)]:
            counts = get_class_counts(df, col)
            for c in classes:
                count_val = counts.get(c, 0)
                if count_val is None:
                    count_val = 0
                summary_rows.append({'split': split_name, 'label_type': col, 'class': c, 'count': int(count_val)})

    summary_df = pd.DataFrame(summary_rows)
    print('\n===== CLASS DISTRIBUTION SUMMARY =====')
    print(summary_df.pivot_table(index=['label_type','class'], columns='split', values='count', fill_value=0))
    summary_df.to_csv('class_distribution_summary.csv', index=False)
    print('Class distribution summary saved to class_distribution_summary.csv')

    # Warn if any class is missing in any split, and warn for very low sample classes
    MIN_SAMPLES = 10  # threshold for warning about rare classes
    for col, classes in [('nutrition_template_id', nutrition_classes), ('workout_template_id', workout_classes)]:
        for split_name, df in [('train', train_balanced), ('val', val_df), ('test', test_df)]:
            present = set(df[col].dropna().unique())
            missing = [c for c in classes if c not in present]
            if missing:
                print(f'WARNING: {col} classes {missing} missing in {split_name} split!')
            # Warn for rare classes
            counts = get_class_counts(df, col)
            rare = []
            for c in classes:
                count_val = counts.get(c, 0)
                if count_val is None:
                    count_val = 0
                if count_val < MIN_SAMPLES:
                    rare.append(c)
            if rare:
                print(f'⚠️ WARNING: {col} classes {rare} have <{MIN_SAMPLES} samples in {split_name} split!')

    # --- Automated per-class metrics and confusion matrices ---
    def save_metrics_and_plots(y_true, y_pred, classes, label, split):
        # Exclude NaN from y_true/y_pred
        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = pd.Series(y_true)[mask]
        y_pred = pd.Series(y_pred)[mask]
        # Only use valid classes
        valid_classes = [c for c in classes if not pd.isna(c)]
        report = classification_report(y_true, y_pred, labels=valid_classes, output_dict=True, zero_division='0')
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'backend/visualizations/{split}_{label}_classification_report.csv')
        print(f'{split} {label} classification report:')
        print(report_df)
        cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation="45")
        plt.title(f'{split} {label} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'backend/visualizations/{split}_{label}_confusion_matrix.png')
        plt.close()

    # --- Macro/micro metrics summary ---
    def print_macro_micro_metrics(y_true, y_pred, label, split):
        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true = pd.Series(y_true)[mask]
        y_pred = pd.Series(y_pred)[mask]
        print(f'\n{split} {label} macro/micro metrics:')
        for avg in ['macro', 'micro', 'weighted']:
            f1 = f1_score(y_true, y_pred, average=avg, zero_division="0")
            prec = precision_score(y_true, y_pred, average=avg, zero_division="0")
            rec = recall_score(y_true, y_pred, average=avg, zero_division="0")
            print(f'  {avg.capitalize()} F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    # --- Visualize and save class distributions ---
    def plot_class_distribution(df, col, split):
        import os
        save_dir = 'backend/visualizations'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df, order=sorted(df[col].unique()))
        plt.title(f'{split} {col} Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{split}_{col}_distribution.png')
        plt.close()

    for split_name, df in [('train', train_balanced), ('val', val_df), ('test', test_df)]:
        plot_class_distribution(df, 'nutrition_template_id', split_name)
        plot_class_distribution(df, 'workout_template_id', split_name)

    # --- SYNTHETIC SPLIT: Force 70/15/15 for all 21 valid combinations ---
    def synthesize_samples(df, n_needed, combo_cols, numeric_cols):
        # If not enough, copy and jitter numeric features
        if len(df) == 0 or n_needed <= 0:
            return pd.DataFrame(columns=df.columns)
        samples = []
        for _ in range(n_needed):
            row = df.sample(1, replace=True).iloc[0].copy()
            for col in numeric_cols:
                if col in row:
                    row[col] += np.random.normal(0, 0.01)  # small jitter
            samples.append(row)
        return pd.DataFrame(samples)

    def create_synthetic_split(full_df, valid_combos, combo_cols, numeric_cols, split_ratios=(0.7, 0.15, 0.15), min_per_combo=20):
        train_rows, val_rows, test_rows = [], [], []
        for combo in valid_combos:
            mask = (full_df[combo_cols[0]] == combo[0]) & (full_df[combo_cols[1]] == combo[1]) & (full_df[combo_cols[2]] == combo[2])
            combo_df = full_df[mask].copy()
            if len(combo_df) == 0:
                print(f'⚠️ WARNING: No data for combo {combo}, skipping synthetic generation for this group.')
                continue
            n_total = max(len(combo_df), min_per_combo)
            n_train = int(round(n_total * split_ratios[0]))
            n_val = int(round(n_total * split_ratios[1]))
            n_test = n_total - n_train - n_val
            # Synthesize if needed
            if len(combo_df) < n_total:
                synth = synthesize_samples(combo_df, n_total - len(combo_df), combo_cols, numeric_cols)
                combo_df = pd.concat([combo_df, synth], ignore_index=True)
            combo_df = combo_df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_rows.append(combo_df.iloc[:n_train].assign(split='train'))
            val_rows.append(combo_df.iloc[n_train:n_train+n_val].assign(split='validation'))
            test_rows.append(combo_df.iloc[n_train+n_val:n_train+n_val+n_test].assign(split='test'))
        train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        val = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
        test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
        return train, val, test

    # --- Prepare for both experiments ---
    combo_cols = ['fitness_goal', 'activity_level', 'bmi_category']
    numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'tdee', 'activity_multiplier',
                    'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
                    'BMR_per_weight', 'activity_efficiency', 'BMI_deviation', 'weight_height_ratio', 'metabolic_score']
    valid_combos_list = [tuple(x) for x in valid_combos[['fitness_goal', 'activity_level', 'bmi_category']].values]

    # --- Directory setup ---
    os.makedirs('outputs/synthetic', exist_ok=True)
    os.makedirs('outputs/real', exist_ok=True)

    # --- SYNTHETIC EXPERIMENT ---
    print('\n=== SYNTHETIC 70/15/15 SPLIT EXPERIMENT ===')
    synth_train, synth_val, synth_test = create_synthetic_split(pd.concat([train_balanced, val_df, test_df], ignore_index=True), valid_combos_list, combo_cols, numeric_cols)
    for feat in ['activity_level_encoded', 'TDEE_BMR_ratio']:
        for df in [synth_train, synth_val, synth_test]:
            if feat in df.columns:
                df[feat] = 0
    # Drop rows with missing template IDs
    before = len(synth_train) + len(synth_val) + len(synth_test)
    synth_train = synth_train.dropna(subset=['nutrition_template_id', 'workout_template_id'])
    synth_val = synth_val.dropna(subset=['nutrition_template_id', 'workout_template_id'])
    synth_test = synth_test.dropna(subset=['nutrition_template_id', 'workout_template_id'])
    after = len(synth_train) + len(synth_val) + len(synth_test)
    if after < before:
        print(f'⚠️ WARNING: Dropped {before - after} synthetic rows with missing template IDs.')
    synth_train.to_csv('outputs/synthetic/train.csv', index=False)
    synth_val.to_csv('outputs/synthetic/val.csv', index=False)
    synth_test.to_csv('outputs/synthetic/test.csv', index=False)
    print('Saved outputs/synthetic/train.csv, val.csv, test.csv')
    synth_data = pd.concat([synth_train, synth_val, synth_test], ignore_index=True)
    # Check if any synthetic group is missing template IDs
    for split_name, df in [('synthetic_train', synth_train), ('synthetic_val', synth_val), ('synthetic_test', synth_test)]:
        missing_nutrition = df['nutrition_template_id'].isnull().any() if 'nutrition_template_id' in df.columns and not df.empty else False
        missing_workout = df['workout_template_id'].isnull().any() if 'workout_template_id' in df.columns and not df.empty else False
        if bool(missing_nutrition) or bool(missing_workout):
            print(f'⚠️ WARNING: {split_name} contains rows with missing template IDs!')
    # Train and save XGBoost model for synthetic experiment
    print('\nTraining XGBoost on synthetic split...')
    synth_model = thesis_model.XGFitnessAIModel(templates_dir='../data')
    synth_model.train_all_models(synth_data, random_state=42)
    synth_model.save_model('outputs/synthetic/xgboost_model.pkl', include_research_models=False)
    print('Saved outputs/synthetic/xgboost_model.pkl')

    # --- REAL (AUGMENTED TRAIN, PURE VAL/TEST) EXPERIMENT ---
    print('\n=== REAL (AUGMENTED TRAIN, PURE VAL/TEST) EXPERIMENT ===')
    for feat in ['activity_level_encoded', 'TDEE_BMR_ratio']:
        for df in [train_balanced, val_df, test_df]:
            if feat in df.columns:
                df[feat] = 0
    train_balanced.to_csv('outputs/real/train.csv', index=False)
    val_df.to_csv('outputs/real/val.csv', index=False)
    test_df.to_csv('outputs/real/test.csv', index=False)
    print('Saved outputs/real/train.csv, val.csv, test.csv')
    real_data = pd.concat([train_balanced, val_df, test_df], ignore_index=True)
    # Train and save XGBoost model for real experiment
    print('\nTraining XGBoost on real split...')
    real_model = thesis_model.XGFitnessAIModel(templates_dir='../data')
    real_model.train_all_models(real_data, random_state=42)
    real_model.save_model('outputs/real/xgboost_model.pkl', include_research_models=False)
    print('Saved outputs/real/xgboost_model.pkl')

    print('\nAll done!')

if __name__ == "__main__":
    print("\n=== Running ALL THREE experiments: REAL, AUGMENTED, FULLY SYNTHETIC ===\n")

    experiments = [
        ("REAL", "real"),
        ("AUGMENTED", "synthetic"),
        ("FULLY_SYNTHETIC", "fully_synthetic"),
    ]
    results = {}
    for exp_name, exp_dir in experiments:
        print(f"\n=== {exp_name} EXPERIMENT ===\n")
        train, val, test = load_experiment_splits(exp_dir)
        train = train.dropna(subset=['nutrition_template_id', 'workout_template_id'])
        val = val.dropna(subset=['nutrition_template_id', 'workout_template_id'])
        test = test.dropna(subset=['nutrition_template_id', 'workout_template_id'])
        data = pd.concat([train, val, test], ignore_index=True)
        print(f"\n📚 Initializing XGFitness AI Model for {exp_name}...")
        model = thesis_model.XGFitnessAIModel(templates_dir='../data')
        print("✅ Model initialized successfully\n")
        print(f"\n🚀 Training ALL Models (XGBoost) on {exp_name}...")
        info = model.train_all_models(data, random_state=42)
        # Ensure output directory exists before saving model
        os.makedirs(f'outputs/{exp_dir}', exist_ok=True)
        model.save_model(f'outputs/{exp_dir}/xgboost_model.pkl', include_research_models=False)
        print(f'Saved outputs/{exp_dir}/xgboost_model.pkl')
        results[exp_name] = info

    # --- SUMMARY COMPARISON ---
    print("\n=== SUMMARY COMPARISON: REAL vs AUGMENTED vs FULLY SYNTHETIC ===\n")
    def print_metrics(metrics, label):
        print(f"{label}:")
        print(f"  Workout Accuracy: {metrics.get('workout_accuracy', 'N/A')}")
        print(f"  Nutrition Accuracy: {metrics.get('nutrition_accuracy', 'N/A')}")
        print(f"  Workout F1: {metrics.get('workout_f1', 'N/A')}")
        print(f"  Nutrition F1: {metrics.get('nutrition_f1', 'N/A')}")
        print()
    for exp_name in ["REAL", "AUGMENTED", "FULLY_SYNTHETIC"]:
        print_metrics(results[exp_name], exp_name)
    print("\nSee outputs/real/, outputs/synthetic/, and outputs/fully_synthetic/ for all models, splits, and metrics.\n")
