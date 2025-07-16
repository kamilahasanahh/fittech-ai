import pandas as pd
import numpy as np
import os

CLEANED_PATH = 'backend/cleaned_real_data.csv'
OUTPUT_SYNTH = 'backend/outputs/synthetic'
OUTPUT_REAL = 'backend/outputs/real'
OUTPUT_FULLY_SYNTH = 'backend/outputs/fully_synthetic'

os.makedirs(OUTPUT_SYNTH, exist_ok=True)
os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FULLY_SYNTH, exist_ok=True)

# --- Assign derived features and class labels ---
def add_features_and_labels(df):
    # BMI category
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    # Activity level (simple logic based on mod_act + vig_act)
    df['mod_act'] = pd.Series(pd.to_numeric(df['mod_act'], errors='coerce')).fillna(0)
    df['vig_act'] = pd.Series(pd.to_numeric(df['vig_act'], errors='coerce')).fillna(0)
    total_act = df['mod_act'] + 2 * df['vig_act']
    df['activity_level'] = pd.cut(total_act, bins=[-1, 10, 30, 1000], labels=['Low Activity', 'Moderate Activity', 'High Activity'])
    # Fitness goal (simple logic: BMI/age)
    def assign_goal(row):
        if row['bmi'] >= 27:
            return 'Fat Loss'
        elif row['bmi'] <= 21 and row['age'] <= 35:
            return 'Muscle Gain'
        else:
            return 'Maintenance'
    df['fitness_goal'] = df.apply(assign_goal, axis=1)
    # Nutrition/workout template IDs (dummy: hash of combo for now)
    df['nutrition_template_id'] = df.apply(lambda r: hash((r['fitness_goal'], r['activity_level'], r['bmi_category'])) % 100, axis=1)
    df['workout_template_id'] = df.apply(lambda r: hash((r['fitness_goal'], r['activity_level'], r['bmi_category'], 'workout')) % 100, axis=1)
    return df

def synthesize_samples(df, n_needed, numeric_cols):
    if len(df) == 0 or n_needed <= 0:
        return pd.DataFrame(columns=df.columns)
    samples = []
    for _ in range(n_needed):
        row = df.sample(1, replace=True).iloc[0].copy()
        for col in numeric_cols:
            if col in row:
                row[col] += np.random.normal(0, 0.01)
        samples.append(row)
    return pd.DataFrame(samples)

def create_synthetic_split(full_df, combo_cols, numeric_cols, split_ratios=(0.7, 0.15, 0.15), min_per_combo=30):
    train_rows, val_rows, test_rows = [], [], []
    combos = full_df[combo_cols].drop_duplicates().values
    for combo in combos:
        mask = (full_df[combo_cols[0]] == combo[0]) & (full_df[combo_cols[1]] == combo[1]) & (full_df[combo_cols[2]] == combo[2])
        combo_df = full_df[mask].copy()
        n_total = max(len(combo_df), min_per_combo)
        n_train = int(round(n_total * split_ratios[0]))
        n_val = int(round(n_total * split_ratios[1]))
        n_test = n_total - n_train - n_val
        # Synthesize if needed
        if len(combo_df) < n_total:
            synth = synthesize_samples(combo_df, n_total - len(combo_df), numeric_cols)
            combo_df = pd.concat([combo_df, synth], ignore_index=True)
        combo_df = combo_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_rows.append(combo_df.iloc[:n_train].assign(split='train'))
        val_rows.append(combo_df.iloc[n_train:n_train+n_val].assign(split='validation'))
        test_rows.append(combo_df.iloc[n_train+n_val:n_train+n_val+n_test].assign(split='test'))
    train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    val = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    return train, val, test

def create_real_split(df, split_ratios=(0.7, 0.15, 0.15)):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    n_test = n - n_train - n_val
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    train['split'] = 'train'
    val['split'] = 'validation'
    test['split'] = 'test'
    return train, val, test

def create_fully_synthetic_split(combo_cols, numeric_cols, n_per_combo=100, split_ratios=(0.7, 0.15, 0.15)):
    # Define all possible class combinations
    combos = []
    for goal_idx, goal in enumerate(['Fat Loss', 'Muscle Gain', 'Maintenance']):
        for act_idx, act in enumerate(['Low Activity', 'Moderate Activity', 'High Activity']):
            for bmi_idx, bmi_cat in enumerate(['Underweight', 'Normal', 'Overweight', 'Obese']):
                combos.append((goal, act, bmi_cat, goal_idx, act_idx, bmi_idx))
    # Assign unique, non-overlapping feature values for each class
    def perfect_features(goal, act, bmi_cat, goal_idx, act_idx, bmi_idx, class_idx):
        # Each class gets a unique block in feature space
        base = {
            'age': 20 + 20 * goal_idx + 5 * act_idx + bmi_idx,  # 20, 40, 60 for goal; 0,5,10 for act; 0-3 for bmi
            'height_cm': 150 + 10 * bmi_idx + 2 * goal_idx + act_idx,  # 150,160,170,180 for bmi; small offset for goal/act
            'weight_kg': 40 + 20 * bmi_idx + 3 * goal_idx + act_idx,   # 40,60,80,100 for bmi; small offset for goal/act
            'bmi': 16 + 6 * bmi_idx + goal_idx,  # 16,22,28,34 for bmi; offset for goal
            'mod_act': 5 + 10 * act_idx,         # 5,15,25 for act
            'vig_act': 0 + 10 * act_idx,         # 0,10,20 for act
        }
        # No noise for perfect separability
        base['fitness_goal'] = goal
        base['activity_level'] = act
        base['bmi_category'] = bmi_cat
        # Assign unique template IDs per class (0..N-1)
        base['nutrition_template_id'] = class_idx
        base['workout_template_id'] = class_idx
        return base
    rows = []
    for class_idx, (goal, act, bmi_cat, goal_idx, act_idx, bmi_idx) in enumerate(combos):
        for _ in range(n_per_combo):
            rows.append(perfect_features(goal, act, bmi_cat, goal_idx, act_idx, bmi_idx, class_idx))
    df = pd.DataFrame(rows)
    # Split
    n = len(df)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    n_test = n - n_train - n_val
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train = df.iloc[:n_train].copy(); train['split'] = 'train'
    val = df.iloc[n_train:n_train+n_val].copy(); val['split'] = 'validation'
    test = df.iloc[n_train+n_val:].copy(); test['split'] = 'test'
    return train, val, test

def engineer_features(df):
    # Categorical encodings
    goal_map = {'Fat Loss': 0, 'Muscle Gain': 1, 'Maintenance': 2}
    activity_map_enc = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df['goal_encoded'] = df['fitness_goal'].map(goal_map)
    df['activity_level_encoded'] = df['activity_level'].map(activity_map_enc)
    # Key interaction terms
    df['BMI_Goal_interaction'] = df['bmi'] * df['goal_encoded']
    df['Age_Activity_interaction'] = df['age'] * df['activity_level_encoded']
    df['BMI_Activity_interaction'] = df['bmi'] * df['activity_level_encoded']
    df['Age_Goal_interaction'] = df['age'] * df['goal_encoded']
    # Metabolic Ratios
    df['BMR_per_weight'] = 20  # perfectly separable, set to a constant or unique per class if desired
    df['TDEE_BMR_ratio'] = 2   # perfectly separable, set to a constant or unique per class if desired
    df['activity_efficiency'] = 1.5
    # Health Deviation Scores
    df['BMI_deviation'] = abs(df['bmi'] - 22.5)
    df['weight_height_ratio'] = df['weight_kg'] / df['height_cm']
    df['metabolic_score'] = df['age'] / (df['weight_kg'] + 1)
    # Boolean Classification Flags
    df['high_metabolism'] = 1
    df['very_active'] = (df['activity_level'] == 'High Activity').astype(int)
    df['young_adult'] = (df['age'] <= 30).astype(int)
    df['optimal_BMI'] = ((df['bmi'] >= 18.5) & (df['bmi'] <= 24.9)).astype(int)
    # Gender encoding (for completeness)
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    else:
        df['gender_encoded'] = 0
    # Ensure all model features are present
    feature_columns = [
        'age', 'gender_encoded', 'height_cm', 'weight_kg',
        'bmi', 'goal_encoded', 'activity_level_encoded',
        'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
        'BMR_per_weight', 'TDEE_BMR_ratio', 'activity_efficiency',
        'BMI_deviation', 'weight_height_ratio', 'metabolic_score',
        'high_metabolism', 'very_active', 'young_adult', 'optimal_BMI'
    ]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df

def print_class_summary(df, label, split):
    print(f"{split} {label} distribution:")
    print(df[label].value_counts())

if __name__ == "__main__":
    df = pd.read_csv(CLEANED_PATH)
    df = add_features_and_labels(df)
    numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'mod_act', 'vig_act']
    combo_cols = ['fitness_goal', 'activity_level', 'bmi_category']

    # Synthetic experiment
    synth_train, synth_val, synth_test = create_synthetic_split(df, combo_cols, numeric_cols)
    synth_train.to_csv(f'{OUTPUT_SYNTH}/train.csv', index=False)
    synth_val.to_csv(f'{OUTPUT_SYNTH}/val.csv', index=False)
    synth_test.to_csv(f'{OUTPUT_SYNTH}/test.csv', index=False)
    print('Synthetic experiment splits saved.')
    for split_name, d in [('train', synth_train), ('val', synth_val), ('test', synth_test)]:
        print_class_summary(d, 'nutrition_template_id', split_name)
        print_class_summary(d, 'workout_template_id', split_name)

    # Real experiment
    real_train, real_val, real_test = create_real_split(df)
    real_train.to_csv(f'{OUTPUT_REAL}/train.csv', index=False)
    real_val.to_csv(f'{OUTPUT_REAL}/val.csv', index=False)
    real_test.to_csv(f'{OUTPUT_REAL}/test.csv', index=False)
    print('Real experiment splits saved.')
    for split_name, d in [('train', real_train), ('val', real_val), ('test', real_test)]:
        print_class_summary(d, 'nutrition_template_id', split_name)
        print_class_summary(d, 'workout_template_id', split_name)

    # Fully synthetic, perfectly balanced, noise-free experiment
    fully_synth_train, fully_synth_val, fully_synth_test = create_fully_synthetic_split(combo_cols, numeric_cols, n_per_combo=100)
    fully_synth_train = engineer_features(fully_synth_train)
    fully_synth_val = engineer_features(fully_synth_val)
    fully_synth_test = engineer_features(fully_synth_test)
    fully_synth_train.to_csv(f'{OUTPUT_FULLY_SYNTH}/train.csv', index=False)
    fully_synth_val.to_csv(f'{OUTPUT_FULLY_SYNTH}/val.csv', index=False)
    fully_synth_test.to_csv(f'{OUTPUT_FULLY_SYNTH}/test.csv', index=False)
    print('Fully synthetic, perfectly balanced, noise-free experiment splits saved.')
    for split_name, d in [('train', fully_synth_train), ('val', fully_synth_val), ('test', fully_synth_test)]:
        print_class_summary(d, 'nutrition_template_id', split_name)
        print_class_summary(d, 'workout_template_id', split_name) 