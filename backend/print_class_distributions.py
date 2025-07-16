import os
import pandas as pd

EXPERIMENTS = [
    ("REAL", "backend/outputs/real"),
    ("AUGMENTED", "backend/outputs/augmented"),
    ("FULLY_SYNTHETIC", "backend/outputs/fully_synthetic"),
]
SPLITS = ["train", "val", "test"]

for exp_name, exp_dir in EXPERIMENTS:
    print(f"\n=== {exp_name} ===")
    for split in SPLITS:
        path = os.path.join(exp_dir, f"{split}.csv")
        if not os.path.exists(path):
            print(f"  {split}: File not found: {path}")
            continue
        df = pd.read_csv(path)
        print(f"  {split.capitalize()} ({len(df)} rows):")
        for col in ["nutrition_template_id", "workout_template_id"]:
            if col in df.columns:
                counts = df[col].value_counts().sort_index()
                print(f"    {col} class counts:")
                print(counts.to_string())
            else:
                print(f"    {col} not found in columns.") 