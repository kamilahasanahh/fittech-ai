import pandas as pd
import numpy as np
import re

RAW_PATH = r'F:\fittech-ai\backend\e267_Data on age, gender, height, weight, activity levels for each household member.txt'
CLEANED_PATH = r'F:\fittech-ai\backend\cleaned_real_data.csv'

def parse_height_ft_in(height_str):
    if pd.isna(height_str) or height_str == '':
        return np.nan
    # Accept both 5.11 and 5'11" formats
    if isinstance(height_str, (int, float)):
        height_str = str(height_str)
    match = re.match(r'^(\d+)[\.\'](\d+)$', height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 30.48 + inches * 2.54
    try:
        # Try as float: 5.11 means 5 feet 11 inches
        parts = str(height_str).split('.')
        if len(parts) == 2:
            feet = int(parts[0])
            inches = int(parts[1])
            return feet * 30.48 + inches * 2.54
    except Exception:
        pass
    return np.nan

def pounds_to_kg(pounds):
    try:
        return float(pounds) * 0.453592
    except:
        return np.nan

def clean_data():
    df = pd.read_csv(RAW_PATH, sep='\t', dtype=str)
    # Rename columns for clarity
    df = df.rename(columns={
        'Member_Age_Orig': 'age',
        'Member_Gender_Orig': 'gender',
        'HEIGHT': 'height_ft_in',
        'WEIGHT': 'weight_lb',
        'Mod_act': 'mod_act',
        'Vig_act': 'vig_act',
    })
    # Convert types
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['height_cm'] = df['height_ft_in'].apply(parse_height_ft_in)
    df['weight_kg'] = df['weight_lb'].apply(pounds_to_kg)
    # Filter constraints
    df = df[(df['age'] >= 18) & (df['age'] <= 65)]
    df = df[(df['height_cm'] >= 150) & (df['height_cm'] <= 200)]
    df = df[(df['weight_kg'] >= 40) & (df['weight_kg'] <= 150)]
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
    # Drop rows with missing critical values
    df = df.dropna(subset=['age', 'gender', 'height_cm', 'weight_kg', 'bmi'])
    # Save cleaned data
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_PATH}. Rows: {len(df)}")

if __name__ == "__main__":
    clean_data() 