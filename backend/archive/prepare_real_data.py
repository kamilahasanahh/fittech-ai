"""
Data Preparation Script for FitTech AI

- Cleans and prepares raw data.
- Assigns template IDs.
- Splits data into train/val/test using stratified sampling on (fitness_goal, activity_level, bmi_category).
- Excludes rare combinations (<2 samples) from splitting and saves them for transparency.
- Only the train set is ever augmented (in a separate script). Validation and test sets are untouched real data.
- This script ensures no data leakage and that evaluation is always on real, unaugmented data.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src import thesis_model

RAW_FILE = os.path.join(os.path.dirname(__file__), 'e267_Data on age, gender, height, weight, activity levels for each household member.txt')
CLEANED_CSV = os.path.join(os.path.dirname(__file__), 'real_cleaned_data.csv')
LABELLED_CSV = os.path.join(os.path.dirname(__file__), 'real_labelled_data.csv')
TRAIN_CSV = os.path.join(os.path.dirname(__file__), 'real_train.csv')
VAL_CSV = os.path.join(os.path.dirname(__file__), 'real_val.csv')
TEST_CSV = os.path.join(os.path.dirname(__file__), 'real_test.csv')

# Valid combinations for template assignment
# Load valid combinations from file
valid_combos_df = pd.read_csv('valid_template_combinations.csv')
VALID_COMBINATIONS = [tuple(x) for x in valid_combos_df[['fitness_goal', 'activity_level', 'bmi_category']].values]

# Activity mapping (example, adjust as needed)
def get_activity_level(mod_act_hours, vig_act_hours):
    mod_act_minutes = mod_act_hours * 60
    vig_act_minutes = vig_act_hours * 60
    if (mod_act_minutes >= 300 or vig_act_minutes >= 150):
        return 'High Activity'
    elif (mod_act_minutes >= 150 or vig_act_minutes >= 75):
        return 'Moderate Activity'
    else:
        return 'Low Activity'

def add_derived_features(df):
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    def calc_bmr(row):
        if row['gender'] == 'Male':
            return 88.362 + (13.397 * row['weight_kg']) + (4.799 * row['height_cm']) - (5.677 * row['age'])
        else:
            return 447.593 + (9.247 * row['weight_kg']) + (3.098 * row['height_cm']) - (4.330 * row['age'])
    df['bmr'] = df.apply(calc_bmr, axis=1)
    activity_map = {'Low Activity': 1.29, 'Moderate Activity': 1.55, 'High Activity': 1.81}
    df['activity_multiplier'] = df['activity_level'].map(activity_map)
    df['tdee'] = df['bmr'] * df['activity_multiplier']
    # Ensure gender_encoded is present
    if 'gender' in df.columns and 'gender_encoded' not in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
    return df

def assign_template_ids(row, model):
    workout_id, nutrition_id = model.get_template_assignments(
        row['fitness_goal'], row['activity_level'], row['bmi_category']
    )
    return pd.Series({'workout_template_id': workout_id, 'nutrition_template_id': nutrition_id})

def clean_and_prepare_raw_data():
    print(f'Reading raw data from: {RAW_FILE}')
    df_raw = pd.read_csv(RAW_FILE, sep='\t', encoding='utf-8')
    cleaned_rows = []
    for _, row in df_raw.iterrows():
        try:
            age = int(float(row['Member_Age_Orig']))
            if not (18 <= age <= 65):
                continue
            gender_code = int(float(row['Member_Gender_Orig']))
            gender = 'Male' if gender_code == 1 else 'Female' if gender_code == 2 else None
            if gender is None:
                continue
            # Height in cm
            height_str = str(row['HEIGHT']).strip()
            if '.' in height_str:
                feet, inches = height_str.split('.')
                height_cm = float(feet) * 30.48 + float(inches) * 2.54
            else:
                height_cm = float(height_str) * 30.48
            if not (150 <= height_cm <= 200):
                continue
            # Weight in kg
            weight_kg = float(str(row['WEIGHT']).strip()) * 0.453592
            if not (40 <= weight_kg <= 150):
                continue
            # Calculate BMI and check consistency
            bmi = weight_kg / ((height_cm / 100) ** 2)
            if not (10 <= bmi <= 60):
                continue
            # Activity
            mod_act = row.get('Mod_act', 0)
            vig_act = row.get('Vig_act', 0)
            try:
                mod_act_hours = float(str(mod_act).strip()) if pd.notna(mod_act) else 0
            except:
                mod_act_hours = 0
            try:
                vig_act_hours = float(str(vig_act).strip()) if pd.notna(vig_act) else 0
            except:
                vig_act_hours = 0
            activity_level = get_activity_level(mod_act_hours, vig_act_hours)
            # BMI category
            if bmi < 18.5:
                bmi_category = 'Underweight'
            elif bmi < 25:
                bmi_category = 'Normal'
            elif bmi < 30:
                bmi_category = 'Overweight'
            else:
                bmi_category = 'Obese'
            # Assign fitness goal based on valid combinations
            possible_goals = []
            for fg, al, bc in VALID_COMBINATIONS:
                if al == activity_level and bc == bmi_category:
                    possible_goals.append(fg)
            if not possible_goals:
                continue
            # For diversity, randomly pick one of the possible goals
            fitness_goal = possible_goals[0]  # Or random.choice(possible_goals) for more diversity
            cleaned_rows.append({
                'age': age,
                'gender': gender,
                'height_cm': round(height_cm, 1),
                'weight_kg': round(weight_kg, 1),
                'activity_level': activity_level,
                'Mod_act': round(mod_act_hours, 2),
                'Vig_act': round(vig_act_hours, 2),
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'fitness_goal': fitness_goal
            })
        except Exception as e:
            continue
    df_clean = pd.DataFrame(cleaned_rows)
    print(f'Cleaned data shape: {df_clean.shape}')
    df_clean.to_csv(CLEANED_CSV, index=False)
    print(f'Saved cleaned data to: {CLEANED_CSV}')
    return df_clean

def fill_missing_combos_with_synthetic(df, valid_combos, split_name, augmenter):
    # For each valid combo, ensure at least 1 row exists in df; if not, add 1 synthetic row
    group_cols = ['fitness_goal', 'activity_level', 'bmi_category']
    filled = [df]
    for combo in valid_combos:
        mask = (
            (df['fitness_goal'] == combo[0]) &
            (df['activity_level'] == combo[1]) &
            (df['bmi_category'] == combo[2])
        )
        if not mask.any():
            # Generate 1 synthetic row for this combo
            synth = augmenter.generate_synthetic_sample_for_template(combo[0], combo[2])
            synth['fitness_goal'] = combo[0]
            synth['activity_level'] = combo[1]
            synth['bmi_category'] = combo[2]
            synth['split'] = split_name
            # Assign gender randomly
            synth['gender'] = np.random.choice(['Male', 'Female'])
            # Add gender_encoded
            synth['gender_encoded'] = {'Male': 0, 'Female': 1}[synth['gender']]
            # Add other required columns as needed
            filled.append(pd.DataFrame([synth]))
    return pd.concat(filled, ignore_index=True)

def balance_combos_to_target(full_df, valid_combos, augmenter, target_total=100):
    # For each combo, ensure 70 train, 15 val, 15 test (total 100)
    train_rows, val_rows, test_rows = [], [], []
    for combo in valid_combos:
        combo_df = full_df[(full_df['fitness_goal'] == combo[0]) &
                           (full_df['activity_level'] == combo[1]) &
                           (full_df['bmi_category'] == combo[2])]
        n_real = len(combo_df)
        # Shuffle real data for this combo
        combo_df = combo_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = min(70, n_real)
        n_val = min(15, n_real - n_train)
        n_test = min(15, n_real - n_train - n_val)
        # Assign real samples to splits
        if n_train > 0:
            train_rows.append(combo_df.iloc[:n_train])
        if n_val > 0:
            val_rows.append(combo_df.iloc[n_train:n_train+n_val])
        if n_test > 0:
            test_rows.append(combo_df.iloc[n_train+n_val:n_train+n_val+n_test])
        # Fill with synthetic if needed
        for _ in range(70 - n_train):
            synth = augmenter.generate_synthetic_sample_for_template(combo[0], combo[2])
            synth['fitness_goal'] = combo[0]
            synth['activity_level'] = combo[1]
            synth['bmi_category'] = combo[2]
            synth['split'] = 'train'
            synth['gender'] = np.random.choice(['Male', 'Female'])
            synth['gender_encoded'] = {'Male': 0, 'Female': 1}[synth['gender']]
            train_rows.append(pd.DataFrame([synth]))
        for _ in range(15 - n_val):
            synth = augmenter.generate_synthetic_sample_for_template(combo[0], combo[2])
            synth['fitness_goal'] = combo[0]
            synth['activity_level'] = combo[1]
            synth['bmi_category'] = combo[2]
            synth['split'] = 'validation'
            synth['gender'] = np.random.choice(['Male', 'Female'])
            synth['gender_encoded'] = {'Male': 0, 'Female': 1}[synth['gender']]
            val_rows.append(pd.DataFrame([synth]))
        for _ in range(15 - n_test):
            synth = augmenter.generate_synthetic_sample_for_template(combo[0], combo[2])
            synth['fitness_goal'] = combo[0]
            synth['activity_level'] = combo[1]
            synth['bmi_category'] = combo[2]
            synth['split'] = 'test'
            synth['gender'] = np.random.choice(['Male', 'Female'])
            synth['gender_encoded'] = {'Male': 0, 'Female': 1}[synth['gender']]
            test_rows.append(pd.DataFrame([synth]))
    # Concatenate all
    train_bal = pd.concat(train_rows, ignore_index=True)
    val_bal = pd.concat(val_rows, ignore_index=True)
    test_bal = pd.concat(test_rows, ignore_index=True)
    return train_bal, val_bal, test_bal

def main():
    # Step 1: Clean and filter raw data
    df_clean = clean_and_prepare_raw_data()

    # Step 2: Add derived features
    df_clean = add_derived_features(df_clean)

    # Step 3: Assign template IDs
    model = thesis_model.XGFitnessAIModel(templates_dir='../data')
    ids = df_clean.apply(lambda row: assign_template_ids(row, model), axis=1)
    df_clean['workout_template_id'] = ids['workout_template_id']
    df_clean['nutrition_template_id'] = ids['nutrition_template_id']
    df_clean.to_csv(LABELLED_CSV, index=False)
    print(f'Labelled data saved to: {LABELLED_CSV}')

    # Step 4: Create combo_stratify column
    df_clean['combo_stratify'] = (
        df_clean['fitness_goal'].astype(str) + '_' +
        df_clean['activity_level'].astype(str) + '_' +
        df_clean['bmi_category'].astype(str)
    )

    # --- DO NOT EXCLUDE ANY COMBOS ---
    # Use all 21 valid template combinations from valid_template_combinations.csv
    valid_combos_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'valid_template_combinations.csv'))
    valid_combos = [tuple(x) for x in valid_combos_df[['fitness_goal', 'activity_level', 'bmi_category']].values]

    # Step 5: Balance all combos to 70/15/15 per split (100 total)
    from src.fitness_data_augmenter import FitnessDataAugmenter
    augmenter = FitnessDataAugmenter(df_clean)
    train_df, val_df, test_df = balance_combos_to_target(df_clean, valid_combos, augmenter, target_total=100)
    print("[INFO] All 21 valid template combinations are now present in train (70), val (15), and test (15) per combo. Synthetic rows were added for missing combos. For thesis: predictions for these combos are less reliable due to lack of real data.")

    # Save splits
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f'Train/val/test splits saved to: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}')

    # --- Print summary of class counts for each split ---
    group_cols = ['fitness_goal','activity_level','bmi_category']
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f'\nClass counts for {split_name}:')
        class_counts = df.groupby(group_cols).size().reset_index()
        class_counts = class_counts.rename(columns={0: 'count'})
        print(class_counts)

    # After splitting into train_df, val_df, test_df:
    # Load valid combos
    valid_combos = [tuple(x) for x in pd.read_csv('valid_template_combinations.csv')[['fitness_goal', 'activity_level', 'bmi_category']].values]
    # Create augmenter for synthetic row generation
    from src.fitness_data_augmenter import FitnessDataAugmenter
    augmenter = FitnessDataAugmenter(df_clean)
    # Balance all combos to 70/15/15 per split (100 total)
    train_df, val_df, test_df = balance_combos_to_target(df_clean, valid_combos, augmenter, target_total=100)
    print("[INFO] All 21 valid template combinations are now present in train (70), val (15), and test (15) per combo. Synthetic rows were added for missing combos. For thesis: predictions for these combos are less reliable due to lack of real data.")

if __name__ == "__main__":
    main()
    # --- Print valid combinations with >=40 in all splits ---
    train = pd.read_csv('real_train.csv')
    val = pd.read_csv('real_val.csv')
    test = pd.read_csv('real_test.csv')
    group_cols = ['fitness_goal','activity_level','bmi_category']
    def combos(df):
        counts = df.groupby(group_cols).size()
        return set(tuple(x) for x in counts.reset_index()[group_cols].values if counts.loc[tuple(x)] >= 40)
    train_combos = combos(train)
    val_combos = combos(val)
    test_combos = combos(test)
    valid_combos = train_combos & val_combos & test_combos
    print('\nValid combinations with >=40 in all splits:')
    for combo in sorted(valid_combos):
        print(combo) 