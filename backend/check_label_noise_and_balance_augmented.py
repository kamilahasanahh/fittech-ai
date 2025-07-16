import os
import pandas as pd
from collections import Counter
from sklearn.utils import resample

AUGMENTED_DIR = "backend/outputs/augmented"
TRAIN_PATH = os.path.join(AUGMENTED_DIR, "train.csv")
OUTPUT_PATH = os.path.join(AUGMENTED_DIR, "train_balanced.csv")

# Load train split
df = pd.read_csv(TRAIN_PATH)

# Check for label noise: print unique template IDs and their counts
print("\n=== Label Noise Check: nutrition_template_id ===")
print(df['nutrition_template_id'].value_counts().sort_index())
print("\n=== Label Noise Check: workout_template_id ===")
print(df['workout_template_id'].value_counts().sort_index())

# Find rare classes (less than 10% of max count)
min_count = int(0.1 * df['nutrition_template_id'].value_counts().max())
rare_nutrition = df['nutrition_template_id'].value_counts()[lambda x: x < min_count].index.tolist()
min_count_w = int(0.1 * df['workout_template_id'].value_counts().max())
rare_workout = df['workout_template_id'].value_counts()[lambda x: x < min_count_w].index.tolist()
print(f"Rare nutrition_template_id classes: {rare_nutrition}")
print(f"Rare workout_template_id classes: {rare_workout}")

# Balance classes by upsampling rare classes to match the most common class
balanced = []
for col in ['nutrition_template_id', 'workout_template_id']:
    counts = df[col].value_counts()
    max_count = counts.max()
    for class_id in counts.index:
        class_df = df[df[col] == class_id]
        if len(class_df) < max_count:
            class_df = resample(class_df, replace=True, n_samples=max_count, random_state=42)
        balanced.append(class_df)
balanced_df = pd.concat(balanced).drop_duplicates().reset_index(drop=True)

# Print new class counts
print("\n=== After Balancing: nutrition_template_id ===")
print(balanced_df['nutrition_template_id'].value_counts().sort_index())
print("\n=== After Balancing: workout_template_id ===")
print(balanced_df['workout_template_id'].value_counts().sort_index())

# Save balanced split
balanced_df.to_csv(OUTPUT_PATH, index=False)
print(f"Balanced train split saved to {OUTPUT_PATH}") 