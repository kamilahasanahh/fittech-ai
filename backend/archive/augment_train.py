"""
Augmentation Script for FitTech AI

- Only the train set is augmented (upsampled/downsampled) to ensure all valid combinations are well-represented.
- Validation and test sets are untouched real data and are never augmented.
- Template IDs are assigned after augmentation.
- The script enforces balanced representation for all valid (fitness_goal, activity_level, bmi_category) combinations in the train set only.
- This ensures no data leakage and that model evaluation is always on real, unaugmented data.
"""
import pandas as pd
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from src.fitness_data_augmenter import FitnessDataAugmenter
from src.templates import get_template_manager
import numpy as np

REAL_TRAIN_CSV = os.path.join(BASE_DIR, 'real_train.csv')
STRICT_BALANCED_TRAIN_250_CSV = os.path.join(BASE_DIR, 'strict_balanced_train_250.csv')

# Load valid combinations from file
def load_valid_combinations():
    combos_path = os.path.join(BASE_DIR, 'valid_template_combinations.csv')
    df = pd.read_csv(combos_path)
    return [tuple(x) for x in df[['fitness_goal', 'activity_level', 'bmi_category']].values]

valid_combinations = load_valid_combinations()

def print_summary(df, label):
    print(f'\nSummary for {label}:')
    group_cols = ['fitness_goal', 'activity_level', 'bmi_category']
    dist = df.groupby(group_cols).size().reset_index(name='count')
    print(dist)
    if 'data_source' in df.columns:
        print(df['data_source'].value_counts())
        print(f"Total: {len(df)}")
    else:
        print('No data_source column found in strict balanced data.')

# --- AUGMENTATION CONFIG ---
# Set UPSAMPLING_MODE to 'none', 'min', or 'strict'
#   'none'  = no upsampling, just use real data
#   'min'   = upsample each class to at least UPSAMPLE_MIN_PER_CLASS (if possible)
#   'strict'= upsample/downsample to exactly UPSAMPLE_TARGET_PER_CLASS (current behavior)
UPSAMPLING_MODE = 'none'  # 'none', 'min', or 'strict'
UPSAMPLE_MIN_PER_CLASS = 50
UPSAMPLE_TARGET_PER_CLASS = 20  # Drastically reduced to minimize overfitting

def enforce_targets(df, valid_combinations, mode, min_per_class, target_per_class, label):
    group_cols = ['fitness_goal', 'activity_level', 'bmi_category']
    filtered = []
    logs = []
    combos_in_data = set(tuple(x) for x in df[group_cols].drop_duplicates().values)
    for combo in valid_combinations:
        if combo not in combos_in_data:
            logs.append(f"[{label}] WARNING: No data for {combo}, skipping.")
            continue
        goal, activity, bmi_cat = combo
        group_df = df[(df['fitness_goal'] == goal) & (df['activity_level'] == activity) & (df['bmi_category'] == bmi_cat)]
        count = len(group_df)
        if mode == 'none':
            logs.append(f"[{label}] Using {combo}: {count} samples (no upsampling)")
            filtered.append(group_df)
        elif mode == 'min':
            if count < min_per_class:
                n_needed = min_per_class - count
                upsampled = group_df.sample(n_needed, replace=True, random_state=42)
                group_df = pd.concat([group_df, upsampled], ignore_index=True)
                logs.append(f"[{label}] Upsampled {combo}: {count} -> {min_per_class}")
            else:
                logs.append(f"[{label}] Using {combo}: {count} samples (no upsampling needed)")
            filtered.append(group_df)
        elif mode == 'strict':
            if count > target_per_class:
                group_df = group_df.sample(target_per_class, random_state=42)
                logs.append(f"[{label}] Downsampled {combo}: {count} -> {target_per_class}")
            elif count < target_per_class:
                n_needed = target_per_class - count
                upsampled = group_df.sample(n_needed, replace=True, random_state=42)
                group_df = pd.concat([group_df, upsampled], ignore_index=True)
                logs.append(f"[{label}] Upsampled {combo}: {count} -> {target_per_class}")
            filtered.append(group_df)
    df_out = pd.concat(filtered, ignore_index=True)
    summary = df_out.groupby(group_cols).size().reset_index().rename(columns={0: 'count'})
    print(f"\nSummary for {label} (after enforcement, mode={mode}):")
    print(summary)
    for log in logs:
        print(log)
    return df_out

def enforce_strict_targets(df, valid_combinations, target_per_comb, label):
    import numpy as np
    from collections import Counter
    group_cols = ['fitness_goal', 'activity_level', 'bmi_category']
    filtered = []
    logs = []
    combos_in_data = set(tuple(x) for x in df[group_cols].drop_duplicates().values)
    for combo in valid_combinations:
        if combo not in combos_in_data:
            logs.append(f"[{label}] WARNING: No data for {combo}, skipping.")
            continue
        goal, activity, bmi_cat = combo
        group_df = df[(df['fitness_goal'] == goal) & (df['activity_level'] == activity) & (df['bmi_category'] == bmi_cat)]
        count = len(group_df)
        if count > target_per_comb:
            group_df = group_df.sample(target_per_comb, random_state=42)
            logs.append(f"[{label}] Downsampled {combo}: {count} -> {target_per_comb}")
        elif count < target_per_comb:
            n_needed = target_per_comb - count
            upsampled = group_df.sample(n_needed, replace=True, random_state=42)
            group_df = pd.concat([group_df, upsampled], ignore_index=True)
            logs.append(f"[{label}] Upsampled {combo}: {count} -> {target_per_comb}")
        filtered.append(group_df)
    df_strict = pd.concat(filtered, ignore_index=True)
    summary = df_strict.groupby(group_cols).size().reset_index().rename(columns={0: 'count'})
    print(f"\nStrict summary for {label} (after enforcement):")
    print(summary)
    for log in logs:
        print(log)
    return df_strict

def compute_bmr(row):
    if row['gender'] == 'Male':
        return 88.362 + (13.397 * row['weight_kg']) + (4.799 * row['height_cm']) - (5.677 * row['age'])
    else:
        return 447.593 + (9.247 * row['weight_kg']) + (3.098 * row['height_cm']) - (4.330 * row['age'])

activity_map = {
    'Low Activity': 1.29,
    'Moderate Activity': 1.55,
    'High Activity': 1.81
}

# --- Feature Engineering Functions ---
def add_features(df):
    # BMI
    if 'bmi' not in df.columns or df['bmi'].isna().any():
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    # BMI category
    if 'bmi_category' not in df.columns or df['bmi_category'].isna().any():
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    # BMR
    df['bmr'] = df.apply(compute_bmr, axis=1)
    # Activity multiplier
    df['activity_multiplier'] = df['activity_level'].map(activity_map)
    # TDEE
    df['tdee'] = df['bmr'] * df['activity_multiplier']
    # Ratios
    df['tdee_bmr_ratio'] = df['tdee'] / df['bmr']
    # Height/weight bins
    df['height_bin'] = pd.cut(df['height_cm'], bins=[150, 165, 180, 200], labels=['150-165', '166-180', '181-200'], right=True, include_lowest=True)
    df['weight_bin'] = pd.cut(df['weight_kg'], bins=[40, 70, 100, 150], labels=['40-70', '71-100', '101-150'], right=True, include_lowest=True)
    # Ensure gender_encoded is present
    if 'gender' in df.columns and 'gender_encoded' not in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
    return df

# --- Outlier Removal & Label Error Checks ---
def clean_data(df):
    # Remove outliers for age, height, weight
    df = df[(df['age'] >= 18) & (df['age'] <= 65)]
    df = df[(df['height_cm'] >= 150) & (df['height_cm'] <= 200)]
    df = df[(df['weight_kg'] >= 40) & (df['weight_kg'] <= 150)]
    # Remove label errors: e.g., Fat Loss for Underweight, Muscle Gain for Obese
    df = df[~((df['bmi_category'] == 'Underweight') & (df['fitness_goal'] == 'Fat Loss'))]
    df = df[~((df['bmi_category'] == 'Obese') & (df['fitness_goal'] == 'Muscle Gain'))]
    return df

# --- Class Weight Calculation (for use in model training) ---
def compute_class_weights(df, label_col):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(df[label_col])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df[label_col])
    return dict(zip(classes, weights))

def main():
    if not os.path.exists(REAL_TRAIN_CSV):
        print(f'File not found: {REAL_TRAIN_CSV}')
        exit(1)

    print(f'Loading real train data from: {REAL_TRAIN_CSV}')
    df_train = pd.read_csv(REAL_TRAIN_CSV)
    df_train = add_features(df_train)
    df_train = clean_data(df_train)
    if not isinstance(df_train, pd.DataFrame):
        df_train = pd.DataFrame(df_train)
    augmenter = FitnessDataAugmenter(df_train)

    # Load template manager
    template_manager = get_template_manager(os.path.join(BASE_DIR, '../data'))

    def assign_template_ids(row):
        workout_id, nutrition_id = template_manager.get_template_assignments(
            row['fitness_goal'], row['activity_level'], row['bmi_category']
        )
        return pd.Series({'workout_template_id': workout_id, 'nutrition_template_id': nutrition_id})

    print(f"AUGMENTATION MODE: {UPSAMPLING_MODE}")
    # Run augmentation according to mode
    if UPSAMPLING_MODE == 'strict':
        df_aug = augmenter.augment_training_data_strict_combinations(valid_combinations, target_per_comb=UPSAMPLE_TARGET_PER_CLASS)
        df_aug = enforce_targets(df_aug, valid_combinations, 'strict', UPSAMPLE_MIN_PER_CLASS, UPSAMPLE_TARGET_PER_CLASS, 'strict_balanced_train')
    elif UPSAMPLING_MODE == 'min':
        df_aug = augmenter.augment_training_data_strict_combinations(valid_combinations, target_per_comb=UPSAMPLE_MIN_PER_CLASS)
        df_aug = enforce_targets(df_aug, valid_combinations, 'min', UPSAMPLE_MIN_PER_CLASS, UPSAMPLE_TARGET_PER_CLASS, 'min_balanced_train')
    else:  # 'none'
        df_aug = df_train[df_train.apply(lambda row: (row['fitness_goal'], row['activity_level'], row['bmi_category']) in valid_combinations, axis=1)].copy()
        df_aug = enforce_targets(df_aug, valid_combinations, 'none', UPSAMPLE_MIN_PER_CLASS, UPSAMPLE_TARGET_PER_CLASS, 'real_train')

    # Assign template IDs
    ids = df_aug.apply(assign_template_ids, axis=1)
    df_aug['workout_template_id'] = ids['workout_template_id']
    df_aug['nutrition_template_id'] = ids['nutrition_template_id']
    # Drop rows with missing template IDs
    before_drop = len(df_aug)
    df_aug = df_aug.dropna(subset=['workout_template_id', 'nutrition_template_id'])
    after_drop = len(df_aug)
    if after_drop < before_drop:
        print(f"Dropped {before_drop - after_drop} rows with missing template IDs after assignment.")
    # Add engineered features again (for synthetic rows)
    df_aug = add_features(df_aug)
    # Ensure template IDs are integers
    df_aug['workout_template_id'] = df_aug['workout_template_id'].astype(int)
    df_aug['nutrition_template_id'] = df_aug['nutrition_template_id'].astype(int)
    # Shuffle and drop duplicates to further reduce overfitting
    df_aug = df_aug.sample(frac=1, random_state=42).drop_duplicates().reset_index(drop=True)
    # Save
    df_aug.to_csv(STRICT_BALANCED_TRAIN_250_CSV, index=False)
    print(f'Saved train set: {STRICT_BALANCED_TRAIN_250_CSV} ({len(df_aug)})')
    print_summary(df_aug, 'final_train')

    # Save class distribution summary for thesis
    class_dist = df_aug.groupby(['fitness_goal','activity_level','bmi_category','workout_template_id','nutrition_template_id']).size().reset_index(drop=False)
    class_dist = class_dist.rename(columns={0: 'count'})
    class_dist.to_csv('strict_balanced_train_250_class_distribution.csv', index=False)
    print('Saved class distribution summary to strict_balanced_train_250_class_distribution.csv')

    # Print class weights for use in model training
    print('Class weights (nutrition_template_id):', compute_class_weights(df_aug, 'nutrition_template_id'))
    print('Class weights (workout_template_id):', compute_class_weights(df_aug, 'workout_template_id'))

if __name__ == '__main__':
    main() 