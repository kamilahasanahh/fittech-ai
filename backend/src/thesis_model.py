import pandas as pd
import numpy as np
import warnings
import pickle
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           classification_report, confusion_matrix, roc_auc_score,
                           balanced_accuracy_score, cohen_kappa_score, mean_squared_error,
                           mean_absolute_error, r2_score, roc_curve, auc)
from itertools import cycle
warnings.filterwarnings('ignore')

# Helper functions
def categorize_bmi(bmi):
    """Categorize BMI into standard WHO categories"""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def calculate_bmr(weight_kg, height_cm, age, gender):
    """Calculate Basal Metabolic Rate using Harris-Benedict equation"""
    if gender == 'Male':
        return 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure"""
    multipliers = {
        'Low Activity': 1.29,
        'Moderate Activity': 1.55,
        'High Activity': 1.81
    }
    return bmr * multipliers.get(activity_level, 1.29)

def get_template_manager(templates_dir):
    """Get template manager instance"""
    from templates import TemplateManager
    return TemplateManager(templates_dir)

# Machine Learning imports 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           classification_report, confusion_matrix, roc_auc_score,
                           balanced_accuracy_score, cohen_kappa_score, mean_squared_error,
                           mean_absolute_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Import local modules
try:
    from .calculations import calculate_bmr, calculate_tdee, categorize_bmi
    from .validation import validate_user_profile
    from .templates import TemplateManager, get_template_manager
except ImportError:
    # Fallback for when run from backend directory
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from calculations import calculate_bmr, calculate_tdee, categorize_bmi
    from validation import validate_user_profile
    from templates import TemplateManager, get_template_manager

class XGFitnessAIModel:
    """
    XGBoost model for fitness recommendations based on puremodel.py structure
    Streamlined for production use with visualizations moved to puremodel.py
    """
    
    def __init__(self, templates_dir: str = 'data'):
        # Initialize template manager (following existing structure)
        self.template_manager = get_template_manager(templates_dir)
        
        # Get templates from template manager
        self.workout_templates = self.template_manager.workout_templates
        self.nutrition_templates = self.template_manager.nutrition_templates
        
        # Core XGBoost model components (following puremodel.py approach)
        self.workout_model = None
        self.nutrition_model = None
        
        # Random Forest baseline models for academic comparison
        self.workout_rf_model = None
        self.nutrition_rf_model = None
        
        self.scaler = StandardScaler()
        self.rf_scaler = StandardScaler()
        self.workout_encoder = LabelEncoder()
        self.nutrition_encoder = LabelEncoder()
        self.workout_label_encoder = LabelEncoder()
        self.nutrition_label_encoder = LabelEncoder()
        self.workout_rf_label_encoder = LabelEncoder()
        self.nutrition_rf_label_encoder = LabelEncoder()
        
        # Activity level multipliers for TDEE calculation (from puremodel.py)
        self.activity_multipliers = {
            'Low Activity': 1.29,
            'Moderate Activity': 1.55,
            'High Activity': 1.81
        }
        
        # Feature columns optimized - DRASTICALLY REDUCED to prevent overfitting
        self.feature_columns = [
            # Core demographics only
            'age', 'gender_encoded', 'height_cm', 'weight_kg', 'bmi',
            
            # Basic metabolic features
            'bmr', 'tdee', 
            
            # Activity level (encoded, not raw activity data)
            'activity_encoded',
            
            # Goal (but not interactions that might cause overfitting)
            'goal_encoded',
            
            # BMI category 
            'bmi_category_encoded',
            
            # Only essential ratios
            'tdee_bmr_ratio'
        ]
        
        # Training metadata
        self.training_info = {}
        self.rf_training_info = {}
        self.is_trained = False
        
        print(f"XGFitness AI initialized with {len(self.workout_templates)} workout and {len(self.nutrition_templates)} nutrition templates")
    
    def get_template_assignments(self, fitness_goal: str, activity_level: str, bmi_category: str):
        """Get template IDs for a user based on their profile"""
        return self.template_manager.get_template_assignments(fitness_goal, activity_level, bmi_category)
        
    def _create_workout_templates(self):
        """Load workout templates from JSON file"""
        try:
            template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'workout_templates.json')
            with open(template_path, 'r') as f:
                workout_data = json.load(f)
            workout_df = pd.DataFrame(workout_data['workout_templates'])
            return workout_df
        except FileNotFoundError:
            print("Warning: workout_templates.json not found, using fallback templates")
            # Fallback to basic templates if file not found
            templates = [
                {'template_id': 1, 'goal': 'Fat Loss', 'activity_level': 'Low Activity', 
                 'workout_type': 'Full Body', 'days_per_week': 2, 'workout_schedule': 'WXXWXXX',
                 'sets_per_exercise': 3, 'exercises_per_session': 6, 'cardio_minutes_per_day': 35, 'cardio_sessions_per_day': 1},
                {'template_id': 2, 'goal': 'Fat Loss', 'activity_level': 'Moderate Activity', 
                 'workout_type': 'Full Body', 'days_per_week': 3, 'workout_schedule': 'WXWXWXX',
                 'sets_per_exercise': 3, 'exercises_per_session': 8, 'cardio_minutes_per_day': 45, 'cardio_sessions_per_day': 1},
                {'template_id': 3, 'goal': 'Fat Loss', 'activity_level': 'High Activity', 
                 'workout_type': 'Upper/Lower Split', 'days_per_week': 4, 'workout_schedule': 'ABXABXX',
                 'sets_per_exercise': 3, 'exercises_per_session': 10, 'cardio_minutes_per_day': 55, 'cardio_sessions_per_day': 1}
            ]
            return pd.DataFrame(templates)
    
    def _create_nutrition_templates(self):
        """Load nutrition templates from JSON file"""
        try:
            template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'nutrition_templates.json')
            with open(template_path, 'r') as f:
                nutrition_data = json.load(f)
            nutrition_df = pd.DataFrame(nutrition_data)
            return nutrition_df
        except FileNotFoundError:
            print("Warning: nutrition_templates.json not found, using fallback templates")
            # Fallback to basic templates if file not found
            templates = [
                {'template_id': 1, 'goal': 'Fat Loss', 'bmi_category': 'Normal', 
                 'caloric_intake_multiplier': 0.80, 'protein_per_kg': 2.3, 'carbs_per_kg': 2.75, 'fat_per_kg': 0.85},
                {'template_id': 2, 'goal': 'Fat Loss', 'bmi_category': 'Overweight', 
                 'caloric_intake_multiplier': 0.75, 'protein_per_kg': 2.15, 'carbs_per_kg': 2.25, 'fat_per_kg': 0.80},
                {'template_id': 3, 'goal': 'Fat Loss', 'bmi_category': 'Obese', 
                 'caloric_intake_multiplier': 0.70, 'protein_per_kg': 2.45, 'carbs_per_kg': 1.75, 'fat_per_kg': 0.80}
            ]
            return pd.DataFrame(templates)
    
    def load_real_data_with_exact_splits(self, file_path='e267_Data on age, gender, height, weight, activity levels for each household member.txt'):
        """
        Load and process real data with EXACT 70/15/15 splits as specified
        Split real data first, then augment ONLY training set
        """
        print(f"Loading real data from {file_path} with exact 70/15/15 splits...")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"⚠️ Data file not found: {file_path}")
            print("Returning empty DataFrame - will use dummy data generation")
            return pd.DataFrame()
        
        # Read the tab-separated file
        try:
            df_raw = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        except:
            # Try different encoding if utf-8 fails
            try:
                df_raw = pd.read_csv(file_path, sep='\t', encoding='latin-1')
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                print("Returning empty DataFrame - will use dummy data generation")
                return pd.DataFrame()
        
        print(f"Raw data shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
        
        # Clean and process the data
        data = []
        
        for _, row in df_raw.iterrows():
            # Extract basic info
            age = row['Member_Age_Orig']
            gender_code = row['Member_Gender_Orig']
            height = row['HEIGHT']
            weight = row['WEIGHT']
            mod_act = row['Mod_act']
            vig_act = row['Vig_act']
            
            # Skip rows with critical missing data
            if pd.isna(age) or pd.isna(gender_code) or pd.isna(height) or pd.isna(weight):
                continue
            
            # Convert age to int and validate
            try:
                age = int(float(age))
            except (ValueError, TypeError):
                continue
            
            # Skip unrealistic ages - only include adults 18-65
            if age < 18 or age > 65:
                continue
            
            # Convert gender (1=Male, 2=Female based on typical coding)
            try:
                gender_code = int(float(gender_code))
            except (ValueError, TypeError):
                continue
                
            if gender_code == 1:
                gender = 'Male'
            elif gender_code == 2:
                gender = 'Female'
            else:
                continue  # Skip unknown gender
            
            # Convert height to cm (assuming it's in feet.inches format like 5.11)
            try:
                height_str = str(height).strip()
                if '.' in height_str:
                    feet, inches = height_str.split('.')
                    height_cm = float(feet) * 30.48 + float(inches) * 2.54
                else:
                    height_cm = float(height_str) * 30.48  # Just feet
            except (ValueError, TypeError):
                continue  # Skip invalid height
            
            # Validate height range
            if height_cm < 120 or height_cm > 220:
                continue
            
            # Convert weight to kg (assuming it's in pounds)
            try:
                weight_kg = float(str(weight).strip()) * 0.453592
            except (ValueError, TypeError):
                continue  # Skip invalid weight
            
            # Validate weight range
            if weight_kg < 30 or weight_kg > 200:
                continue
            
            # Calculate BMI and check validity
            bmi = weight_kg / ((height_cm / 100) ** 2)
            if bmi < 12 or bmi > 50:  # Skip extreme BMI values
                continue
            
            bmi_category = categorize_bmi(bmi)
            
            # Process activity data using WHO guidelines - keep original values
            mod_act_hours = 0
            vig_act_hours = 0
            has_activity_data = False
            
            # Process moderate activity (keep as hours for direct input)
            try:
                if not pd.isna(mod_act) and str(mod_act).strip():
                    mod_act_hours = float(mod_act)
                    if mod_act_hours >= 0:  # Validate non-negative
                        has_activity_data = True
            except (ValueError, TypeError):
                pass  # Keep as 0 for invalid values
                
            # Process vigorous activity (keep as hours for direct input)
            try:
                if not pd.isna(vig_act) and str(vig_act).strip():
                    vig_act_hours = float(vig_act)
                    if vig_act_hours >= 0:  # Validate non-negative
                        has_activity_data = True
            except (ValueError, TypeError):
                pass  # Keep as 0 for invalid values
            
            # Convert to minutes for WHO guidelines calculation
            mod_act_minutes = mod_act_hours * 60
            vig_act_minutes = vig_act_hours * 60
            
            # Determine activity level using WHO guidelines
            if has_activity_data:
                # Use WHO guidelines for activity classification
                if (mod_act_minutes >= 300 or vig_act_minutes >= 150):
                    activity_level = 'High Activity'
                    activity_multiplier = 1.81
                elif (mod_act_minutes >= 150 or vig_act_minutes >= 75):
                    activity_level = 'Moderate Activity'
                    activity_multiplier = 1.55
                else:
                    activity_level = 'Low Activity'
                    activity_multiplier = 1.29
            else:
                # For missing activity data, impute based on demographics
                # Use age, gender, and BMI patterns from available data
                if age < 30:
                    # Young adults more likely to be active
                    if gender == 'Male':
                        activity_level = np.random.choice(['Moderate Activity', 'Low Activity'], p=[0.6, 0.4])
                    else:
                        activity_level = np.random.choice(['Moderate Activity', 'Low Activity'], p=[0.5, 0.5])
                elif age < 45:
                    # Middle-aged adults
                    if gender == 'Male':
                        activity_level = np.random.choice(['Moderate Activity', 'Low Activity'], p=[0.4, 0.6])
                    else:
                        activity_level = np.random.choice(['Moderate Activity', 'Low Activity'], p=[0.3, 0.7])
                else:
                    # Older adults (45-65)
                    activity_level = np.random.choice(['Moderate Activity', 'Low Activity'], p=[0.2, 0.8])
                
                # Set multiplier based on assigned level
                if activity_level == 'High Activity':
                    activity_multiplier = 1.81
                elif activity_level == 'Moderate Activity':
                    activity_multiplier = 1.55
                else:
                    activity_multiplier = 1.29
            
            # Assign realistic fitness goals based on BMI, age, and gender with sophisticated logic
            if bmi_category == 'Underweight':
                # Underweight people almost always want to gain weight/muscle
                fitness_goal = 'Muscle Gain'
            elif bmi_category == 'Obese':
                # Obese people almost always want to lose weight
                fitness_goal = 'Fat Loss'
            elif bmi_category == 'Overweight':
                # Overweight people mostly want fat loss, some maintenance
                if age < 30:
                    fitness_goal = np.random.choice(['Fat Loss', 'Maintenance'], p=[0.85, 0.15])
                else:
                    fitness_goal = np.random.choice(['Fat Loss', 'Maintenance'], p=[0.75, 0.25])
            else:  # Normal BMI - most complex decisions
                if age < 25:
                    # Young adults with normal BMI - goals vary by gender and activity
                    if gender == 'Male':
                        # Young men often want muscle gain
                        if activity_level == 'High Activity':
                            fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance', 'Fat Loss'], p=[0.6, 0.3, 0.1])
                        else:
                            fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance', 'Fat Loss'], p=[0.7, 0.2, 0.1])
                    else:
                        # Young women more varied goals
                        if activity_level == 'High Activity':
                            fitness_goal = np.random.choice(['Maintenance', 'Muscle Gain', 'Fat Loss'], p=[0.4, 0.35, 0.25])
                        else:
                            fitness_goal = np.random.choice(['Fat Loss', 'Maintenance', 'Muscle Gain'], p=[0.4, 0.35, 0.25])
                elif age < 35:
                    # Late 20s to early 30s
                    if gender == 'Male':
                        fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance', 'Fat Loss'], p=[0.5, 0.35, 0.15])
                    else:
                        fitness_goal = np.random.choice(['Maintenance', 'Fat Loss', 'Muscle Gain'], p=[0.45, 0.35, 0.2])
                elif age < 45:
                    # Mid 30s to early 40s - focus shifts to maintenance and health
                    if gender == 'Male':
                        fitness_goal = np.random.choice(['Maintenance', 'Fat Loss', 'Muscle Gain'], p=[0.5, 0.3, 0.2])
                    else:
                        fitness_goal = np.random.choice(['Maintenance', 'Fat Loss', 'Muscle Gain'], p=[0.55, 0.35, 0.1])
                elif age < 55:
                    # Mid 40s to early 50s - health and maintenance focused
                    fitness_goal = np.random.choice(['Maintenance', 'Fat Loss', 'Muscle Gain'], p=[0.6, 0.3, 0.1])
                else:
                    # 55-65 - primarily maintenance with some fat loss
                    fitness_goal = np.random.choice(['Maintenance', 'Fat Loss'], p=[0.7, 0.3])
            
            # Calculate physiological metrics
            bmr = calculate_bmr(weight_kg, height_cm, age, gender)
            tdee = calculate_tdee(bmr, activity_level)
            
            # Validate and adjust fitness goal for valid nutrition template combinations
            # Valid combinations: 8 total
            # Fat Loss: Normal, Overweight, Obese
            # Muscle Gain: Underweight, Normal 
            # Maintenance: Underweight, Normal, Overweight
            
            valid_combinations = {
                ('Fat Loss', 'Normal'): True,
                ('Fat Loss', 'Overweight'): True,
                ('Fat Loss', 'Obese'): True,
                ('Muscle Gain', 'Underweight'): True,
                ('Muscle Gain', 'Normal'): True,
                ('Maintenance', 'Underweight'): True,
                ('Maintenance', 'Normal'): True,
                ('Maintenance', 'Overweight'): True,
            }
            
            # If combination is invalid, adjust fitness goal
            if (fitness_goal, bmi_category) not in valid_combinations:
                if bmi_category == 'Underweight':
                    # Force Muscle Gain or Maintenance for underweight
                    fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance'], p=[0.7, 0.3])
                elif bmi_category == 'Obese':
                    # Force Fat Loss for obese
                    fitness_goal = 'Fat Loss'
                elif bmi_category == 'Overweight' and fitness_goal == 'Muscle Gain':
                    # Force Fat Loss or Maintenance for overweight
                    fitness_goal = np.random.choice(['Fat Loss', 'Maintenance'], p=[0.7, 0.3])
            
            # Find matching templates using template manager
            workout_id, nutrition_id = self.get_template_assignments(fitness_goal, activity_level, bmi_category)
            
            # Skip if no matching templates (should not happen now)
            if workout_id is None or nutrition_id is None:
                print(f"⚠️ No template found for: goal={fitness_goal}, activity={activity_level}, bmi={bmi_category}")
                continue
            
            # Add MINIMAL noise to prevent overfitting while preserving assignment logic
            workout_id, nutrition_id = self._add_template_assignment_noise(
                workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.02  # MINIMAL 2% to fix mismatches
            )
            
            # Verify nutrition template is one of the 8 valid IDs
            if nutrition_id not in [1, 2, 3, 4, 5, 6, 7, 8]:
                print(f"⚠️ Invalid nutrition template ID {nutrition_id} for: goal={fitness_goal}, bmi={bmi_category}")
                continue
            
            data.append({
                'age': int(age),
                'gender': gender,
                'height_cm': round(height_cm, 1),
                'weight_kg': round(weight_kg, 1),
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1),
                'activity_level': activity_level,
                'activity_multiplier': activity_multiplier,
                'Mod_act': round(mod_act_hours, 2),  # Direct input hours
                'Vig_act': round(vig_act_hours, 2),  # Direct input hours
                'has_activity_data': has_activity_data,
                'fitness_goal': fitness_goal,
                'workout_template_id': workout_id,
                'nutrition_template_id': nutrition_id,
                'data_source': 'real'
            })
        
        df_real = pd.DataFrame(data)
        print(f"Processed real data shape: {df_real.shape}")
        
        if len(df_real) == 0:
            print("❌ No valid real data processed")
            return pd.DataFrame()
        
        # EXACT 70/15/15 split of REAL data first
        print("\n📊 IMPLEMENTING EXACT 70/15/15 SPLIT OF REAL DATA:")
        print("="*80)
        
        total_real = len(df_real)
        target_train = int(0.70 * total_real)  # 70% = 2,561 samples
        target_val = int(0.15 * total_real)    # 15% = 548 samples  
        target_test = total_real - target_train - target_val  # Remaining = 550 samples
        
        print(f"Target splits from {total_real} real samples:")
        print(f"  Training: {target_train} samples (70%)")
        print(f"  Validation: {target_val} samples (15%)")
        print(f"  Test: {target_test} samples (15%)")
        
        # First split: 70% train, 30% temp (for val+test)
        df_train_real, df_temp, _, _ = train_test_split(
            df_real, df_real['fitness_goal'], 
            test_size=0.30, 
            random_state=42, 
            stratify=df_real['fitness_goal']
        )
        
        # Second split: 15% val, 15% test from the 30% temp
        df_val_real, df_test_real, _, _ = train_test_split(
            df_temp, df_temp['fitness_goal'], 
            test_size=0.50,  # 50% of 30% = 15% of total
            random_state=42, 
            stratify=df_temp['fitness_goal']
        )
        
        # Add split labels
        df_train_real['split'] = 'train'
        df_val_real['split'] = 'validation'  
        df_test_real['split'] = 'test'
        
        # Verify exact splits
        print(f"✅ ACTUAL REAL DATA SPLITS:")
        print(f"   Training: {len(df_train_real)} samples ({len(df_train_real)/total_real*100:.1f}%)")
        print(f"   Validation: {len(df_val_real)} samples ({len(df_val_real)/total_real*100:.1f}%)")
        print(f"   Test: {len(df_test_real)} samples ({len(df_test_real)/total_real*100:.1f}%)")
        print(f"   Total Real: {total_real} samples")
        
        # Preserve natural distributions - report but don't modify
        print(f"\n📋 PRESERVING NATURAL DISTRIBUTIONS:")
        activity_dist = df_real['activity_level'].value_counts(normalize=True) * 100
        bmi_dist = df_real['bmi_category'].value_counts(normalize=True) * 100
        gender_dist = df_real['gender'].value_counts(normalize=True) * 100
        
        print(f"🔒 PRESERVED Activity Distribution:")
        for activity, pct in activity_dist.items():
            print(f"   {activity}: {pct:.1f}%")
        
        print(f"🔒 PRESERVED BMI Distribution:")
        for bmi, pct in bmi_dist.items():
            print(f"   {bmi}: {pct:.1f}%")
        
        print(f"🔒 PRESERVED Gender Distribution:")
        for gender, pct in gender_dist.items():
            print(f"   {gender}: {pct:.1f}%")
        
        # Check goal distribution in training split ONLY
        train_goal_dist = df_train_real['fitness_goal'].value_counts()
        print(f"\n📋 TRAINING SET GOAL DISTRIBUTION (before augmentation):")
        for goal, count in train_goal_dist.items():
            print(f"   {goal}: {count} samples")
        
        # Now augment ONLY the training set for balanced goals
        print(f"\n🔄 AUGMENTING TRAINING SET FOR BALANCED GOALS:")
        print("="*80)
        
        # Find the goal with maximum samples in training
        max_goal_count = train_goal_dist.max()
        print(f"Target samples per goal: {max_goal_count}")
        
        # Create augmented training data
        training_parts = [df_train_real]  # Start with real training data
        
        for goal in ['Fat Loss', 'Muscle Gain', 'Maintenance']:
            current_count = train_goal_dist.get(goal, 0)
            needed = max_goal_count - current_count
            
            if needed > 0:
                print(f"   {goal}: need {needed} synthetic samples")
                synthetic_data = self._generate_goal_specific_data(goal, needed, random_state=42)
                synthetic_data['split'] = 'train'
                training_parts.append(synthetic_data)
            else:
                print(f"   {goal}: sufficient samples ({current_count})")
        
        # Combine all training data
        df_train_augmented = pd.concat(training_parts, ignore_index=True)
        
        # CRITICAL: Keep val/test sets 100% real data - NO synthetic additions
        print(f"\n🚫 VALIDATION & TEST SETS REMAIN 100% REAL DATA:")
        print(f"   Validation: {len(df_val_real)} real samples (0 synthetic)")
        print(f"   Test: {len(df_test_real)} real samples (0 synthetic)")
        
        # Final dataset combination
        df_final = pd.concat([df_train_augmented, df_val_real, df_test_real], ignore_index=True)
        
        # Final verification and reporting
        print(f"\n✅ FINAL DATASET COMPOSITION:")
        print("="*80)
        
        total_samples = len(df_final)
        real_samples = len(df_final[df_final['data_source'] == 'real'])
        synthetic_samples = len(df_final[df_final['data_source'] == 'synthetic'])
        
        # Split verification
        train_total = len(df_final[df_final['split'] == 'train'])
        val_total = len(df_final[df_final['split'] == 'validation'])
        test_total = len(df_final[df_final['split'] == 'test'])
        
        # Real data percentages
        train_real = len(df_final[(df_final['split'] == 'train') & (df_final['data_source'] == 'real')])
        val_real = len(df_final[(df_final['split'] == 'validation') & (df_final['data_source'] == 'real')])
        test_real = len(df_final[(df_final['split'] == 'test') & (df_final['data_source'] == 'real')])
        
        print(f"Total samples: {total_samples}")
        print(f"Real data: {real_samples} ({100*real_samples/total_samples:.1f}%)")
        print(f"Synthetic data: {synthetic_samples} ({100*synthetic_samples/total_samples:.1f}%)")
        print(f"")
        print(f"Training: {train_total} total ({train_real} real + {train_total-train_real} synthetic)")
        print(f"Validation: {val_total} total ({val_real} real + {val_total-val_real} synthetic)")
        print(f"Test: {test_total} total ({test_real} real + {test_total-test_real} synthetic)")
        print(f"")
        print(f"Real data usage: {100*real_samples/total_samples:.1f}%")
        
        # Goal distribution verification
        final_goal_dist = df_final['fitness_goal'].value_counts()
        print(f"\n📊 FINAL GOAL DISTRIBUTION:")
        for goal, count in final_goal_dist.items():
            print(f"   {goal}: {count} samples")
        
        return df_final
    
    def _generate_goal_specific_data(self, goal, n_samples, random_state=42):
        """
        Generate synthetic data specifically for a given fitness goal
        
        Args:
            goal: The fitness goal to generate data for ('Fat Loss', 'Muscle Gain', 'Maintenance')
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic data for the specified goal
        """
        np.random.seed(random_state)
        
        print(f"    Generating {n_samples} synthetic samples for {goal}...")
        
        data = []
        
        for _ in range(n_samples):
            # Generate basic demographics
            age = np.random.randint(18, 75)
            gender = np.random.choice(['Male', 'Female'])
            
            # Generate height with gender differences
            if gender == 'Male':
                height_cm = np.random.normal(175, 8)
            else:
                height_cm = np.random.normal(162, 7)
            height_cm = np.clip(height_cm, 150, 200)
            
            # Generate BMI and weight based on the specific fitness goal
            if goal == 'Muscle Gain':
                # Muscle gain typically for underweight or normal BMI
                target_bmi = np.random.choice(['Underweight', 'Normal'], p=[0.3, 0.7])
                if target_bmi == 'Underweight':
                    bmi = np.random.uniform(16, 18.5)
                else:  # Normal
                    bmi = np.random.uniform(18.5, 24.9)
            elif goal == 'Fat Loss':
                # Fat loss typically for overweight or obese BMI
                target_bmi = np.random.choice(['Normal', 'Overweight', 'Obese'], p=[0.2, 0.5, 0.3])
                if target_bmi == 'Normal':
                    bmi = np.random.uniform(22, 24.9)  # Higher end of normal for fat loss
                elif target_bmi == 'Overweight':
                    bmi = np.random.uniform(25, 29.9)
                else:  # Obese
                    bmi = np.random.uniform(30, 40)
            else:  # Maintenance
                # Maintenance can be any BMI category
                target_bmi = np.random.choice(['Underweight', 'Normal', 'Overweight'], p=[0.2, 0.6, 0.2])
                if target_bmi == 'Underweight':
                    bmi = np.random.uniform(16, 18.5)
                elif target_bmi == 'Normal':
                    bmi = np.random.uniform(18.5, 24.9)
                else:  # Overweight
                    bmi = np.random.uniform(25, 29.9)
            
            # Calculate weight from BMI and height
            weight_kg = bmi * ((height_cm / 100) ** 2)
            weight_kg = np.clip(weight_kg, 40, 150)
            
            # Recalculate BMI with clipped weight
            bmi = weight_kg / ((height_cm / 100) ** 2)
            bmi_category = categorize_bmi(bmi)
            
            # Assign activity level (preserve natural distribution)
            # Based on the real data: High=80.1%, Moderate=11.9%, Low=8.0%
            activity_level = np.random.choice(
                ['High Activity', 'Moderate Activity', 'Low Activity'], 
                p=[0.801, 0.119, 0.080]
            )
            
            # Generate synthetic activity data based on activity level (in hours)
            if activity_level == 'High Activity':
                # High activity: >5 hours moderate OR >2.5 hours vigorous per week
                mod_act_hours = np.random.normal(5.8, 0.8)
                vig_act_hours = np.random.normal(2.8, 0.5)
                activity_multiplier = 1.81
            elif activity_level == 'Moderate Activity':
                # Moderate activity: 2.5-5 hours moderate OR 1.25-2.5 hours vigorous per week
                mod_act_hours = np.random.normal(3.75, 0.6)
                vig_act_hours = np.random.normal(1.87, 0.3)
                activity_multiplier = 1.55
            else:  # Low Activity
                # Low activity: <2.5 hours moderate OR <1.25 hours vigorous per week
                mod_act_hours = np.random.normal(1.67, 0.4)
                vig_act_hours = np.random.normal(0.83, 0.25)
                activity_multiplier = 1.29
            
            # Ensure non-negative values
            mod_act_hours = max(0, mod_act_hours)
            vig_act_hours = max(0, vig_act_hours)
            
            # Calculate physiological metrics
            bmr = calculate_bmr(weight_kg, height_cm, age, gender)
            tdee = calculate_tdee(bmr, activity_level)
            
            # Use the specified goal
            fitness_goal = goal
            
            # Validate and adjust fitness goal for valid nutrition template combinations
            valid_combinations = {
                ('Fat Loss', 'Normal'): True,
                ('Fat Loss', 'Overweight'): True,
                ('Fat Loss', 'Obese'): True,
                ('Muscle Gain', 'Underweight'): True,
                ('Muscle Gain', 'Normal'): True,
                ('Maintenance', 'Underweight'): True,
                ('Maintenance', 'Normal'): True,
                ('Maintenance', 'Overweight'): True,
            }
            
            # If combination is invalid, adjust BMI category instead of goal
            if (fitness_goal, bmi_category) not in valid_combinations:
                if fitness_goal == 'Fat Loss' and bmi_category == 'Underweight':
                    # Change BMI to Normal for Fat Loss
                    bmi = np.random.uniform(18.5, 24.9)
                    weight_kg = bmi * ((height_cm / 100) ** 2)
                    bmi_category = 'Normal'
                elif fitness_goal == 'Muscle Gain' and bmi_category in ['Overweight', 'Obese']:
                    # Change BMI to Normal for Muscle Gain
                    bmi = np.random.uniform(18.5, 24.9)
                    weight_kg = bmi * ((height_cm / 100) ** 2)
                    bmi_category = 'Normal'
                elif fitness_goal == 'Maintenance' and bmi_category == 'Obese':
                    # Change BMI to Overweight for Maintenance
                    bmi = np.random.uniform(25, 29.9)
                    weight_kg = bmi * ((height_cm / 100) ** 2)
                    bmi_category = 'Overweight'
            
            # Find matching templates using template manager
            workout_id, nutrition_id = self.get_template_assignments(fitness_goal, activity_level, bmi_category)
            
            # Skip if no matching templates (should not happen now)
            if workout_id is None or nutrition_id is None:
                print(f"⚠️ No template found for: goal={fitness_goal}, activity={activity_level}, bmi={bmi_category}")
                continue
            
            # Add MINIMAL noise to prevent overfitting while preserving assignment logic
            workout_id, nutrition_id = self._add_template_assignment_noise(
                workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.02
            )
            
            # Verify nutrition template is one of the 8 valid IDs
            if nutrition_id not in [1, 2, 3, 4, 5, 6, 7, 8]:
                print(f"⚠️ Invalid nutrition template ID {nutrition_id} for: goal={fitness_goal}, bmi={bmi_category}")
                continue
            
            data.append({
                'age': age,
                'gender': gender,
                'height_cm': round(height_cm, 1),
                'weight_kg': round(weight_kg, 1),
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1),
                'activity_level': activity_level,
                'activity_multiplier': activity_multiplier,
                'Mod_act': round(mod_act_hours, 2),
                'Vig_act': round(vig_act_hours, 2),
                'has_activity_data': True,
                'fitness_goal': fitness_goal,
                'workout_template_id': workout_id,
                'nutrition_template_id': nutrition_id,
                'data_source': 'synthetic'
            })
        
        df = pd.DataFrame(data)
        print(f"    Generated {len(df)} synthetic samples for {goal}")
        
        return df
    
    def generate_comprehensive_visualizations(self, df_training, save_dir='visualizations'):
        """
        Generate comprehensive visualizations for all 4 models with complete performance analysis
        """
        print(f"🎨 Generating comprehensive visualizations for all 4 models...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Data Composition Overview (Enhanced)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('XGFitness AI Dataset Composition Analysis', fontsize=16, fontweight='bold')
        
        # Data source distribution
        source_counts = df_training['data_source'].value_counts()
        axes[0, 0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Data Source Distribution')
        
        # Split distribution with exact numbers
        split_counts = df_training['split'].value_counts()
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = axes[0, 1].bar(split_counts.index, split_counts.values, color=colors)
        axes[0, 1].set_title('Train/Validation/Test Split')
        axes[0, 1].set_ylabel('Number of Samples')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        # Goal distribution
        goal_counts = df_training['fitness_goal'].value_counts()
        bars = axes[0, 2].bar(goal_counts.index, goal_counts.values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 2].set_title('Fitness Goal Distribution')
        axes[0, 2].set_ylabel('Number of Samples')
        axes[0, 2].tick_params(axis='x', rotation=45)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        # Activity level distribution (Natural - PRESERVED)
        activity_counts = df_training['activity_level'].value_counts()
        bars = axes[1, 0].bar(activity_counts.index, activity_counts.values, color=['gold', 'orange', 'red'])
        axes[1, 0].set_title('Activity Level Distribution (Natural)')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].tick_params(axis='x', rotation=45)
        # Add percentage labels
        total = sum(activity_counts.values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = height / total * 100
            axes[1, 0].annotate(f'{int(height)}\n({pct:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        # BMI distribution (Natural - PRESERVED)
        bmi_counts = df_training['bmi_category'].value_counts()
        bars = axes[1, 1].bar(bmi_counts.index, bmi_counts.values, color=['yellow', 'lightgreen', 'orange', 'red'])
        axes[1, 1].set_title('BMI Category Distribution (Natural)')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        # Add percentage labels
        total = sum(bmi_counts.values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = height / total * 100
            axes[1, 1].annotate(f'{int(height)}\n({pct:.1f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        # Gender distribution
        gender_counts = df_training['gender'].value_counts()
        axes[1, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Gender Distribution (Natural)')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/01_dataset_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Template Assignment Analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Template Assignment Distribution', fontsize=16, fontweight='bold')
        
        # Workout template distribution
        workout_counts = df_training['workout_template_id'].value_counts().sort_index()
        bars = axes[0].bar(workout_counts.index, workout_counts.values, color='lightblue')
        axes[0].set_title('Workout Template Distribution')
        axes[0].set_xlabel('Workout Template ID')
        axes[0].set_ylabel('Number of Assignments')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        # Nutrition template distribution
        nutrition_counts = df_training['nutrition_template_id'].value_counts().sort_index()
        bars = axes[1].bar(nutrition_counts.index, nutrition_counts.values, color='lightgreen')
        axes[1].set_title('Nutrition Template Distribution')
        axes[1].set_xlabel('Nutrition Template ID')
        axes[1].set_ylabel('Number of Assignments')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/02_template_assignments.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Demographic Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Demographic and Physiological Analysis', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0, 0].hist(df_training['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df_training['age'].mean(), color='red', linestyle='--', label=f'Mean: {df_training["age"].mean():.1f}')
        axes[0, 0].legend()
        
        # BMI distribution by category
        bmi_counts = df_training['bmi_category'].value_counts()
        axes[0, 1].bar(bmi_counts.index, bmi_counts.values, color=['yellow', 'lightgreen', 'orange', 'red'])
        axes[0, 1].set_title('BMI Category Distribution')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Gender distribution
        gender_counts = df_training['gender'].value_counts()
        axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Gender Distribution')
        
        # BMI vs TDEE scatter with activity level
        scatter = axes[1, 1].scatter(df_training['bmi'], df_training['tdee'], 
                                c=df_training['activity_level'].astype('category').cat.codes, 
                                alpha=0.6, cmap='viridis')
        axes[1, 1].set_title('BMI vs TDEE by Activity Level')
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('TDEE (calories)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Activity Level')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/03_demographics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Goal-Activity-BMI Relationship Heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Goal-Activity-BMI Relationship Analysis', fontsize=16, fontweight='bold')
        
        for i, goal in enumerate(['Fat Loss', 'Muscle Gain', 'Maintenance']):
            goal_data = df_training[df_training['fitness_goal'] == goal]
            if len(goal_data) > 0:
                pivot_table = goal_data.groupby(['activity_level', 'bmi_category']).size().unstack(fill_value=0)
                sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{goal} Goal ({len(goal_data)} samples)')
                axes[i].set_xlabel('BMI Category')
                axes[i].set_ylabel('Activity Level')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/04_goal_activity_bmi_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Data Quality and Consistency Check
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality and Consistency Analysis', fontsize=16, fontweight='bold')
        
        # Real vs Synthetic distribution by split
        split_source = df_training.groupby(['split', 'data_source']).size().unstack(fill_value=0)
        split_source.plot(kind='bar', ax=axes[0, 0], color=['lightblue', 'orange'])
        axes[0, 0].set_title('Real vs Synthetic Data by Split')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # Template assignment consistency
        template_combo = df_training.groupby(['workout_template_id', 'nutrition_template_id']).size()
        top_combos = template_combo.nlargest(10)
        axes[0, 1].bar(range(len(top_combos)), top_combos.values, color='lightgreen')
        axes[0, 1].set_title('Top 10 Template Combinations')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlabel('Template Combination Rank')
        
        # Activity hours distribution
        axes[1, 0].scatter(df_training['Mod_act'], df_training['Vig_act'], 
                        c=df_training['activity_level'].astype('category').cat.codes, 
                        alpha=0.6, cmap='viridis')
        axes[1, 0].set_title('Moderate vs Vigorous Activity Hours')
        axes[1, 0].set_xlabel('Moderate Activity (hours/week)')
        axes[1, 0].set_ylabel('Vigorous Activity (hours/week)')
        
        # BMI vs Weight relationship
        scatter2 = axes[1, 1].scatter(df_training['weight_kg'], df_training['bmi'], 
                        c=df_training['height_cm'], alpha=0.6, cmap='plasma')
        axes[1, 1].set_title('Weight vs BMI (colored by height)')
        axes[1, 1].set_xlabel('Weight (kg)')
        axes[1, 1].set_ylabel('BMI')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Height (cm)')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/05_data_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Model Performance Analysis (Only if models are trained)
        if self.is_trained and hasattr(self, 'workout_rf_model') and self.workout_rf_model:
            print("Generating model performance visualizations...")
            
            # Get test data for evaluation
            X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
            test_mask = df_enhanced['split'] == 'test'
            X_test = X[test_mask]
            y_w_test = y_workout[test_mask]
            y_n_test = y_nutrition[test_mask]
            
            if len(X_test) > 0:
                # Scale test data
                X_test_xgb = self.scaler.transform(X_test)
                X_test_rf = self.rf_scaler.transform(X_test)
                
                # Get predictions from all models
                xgb_w_pred = self.workout_model.predict(X_test_xgb)
                xgb_n_pred = self.nutrition_model.predict(X_test_xgb)
                rf_w_pred = self.workout_rf_model.predict(X_test_rf)
                rf_n_pred = self.nutrition_rf_model.predict(X_test_rf)
                
                # Get prediction probabilities for AUROC
                xgb_w_proba = self.workout_model.predict_proba(X_test_xgb)
                xgb_n_proba = self.nutrition_model.predict_proba(X_test_xgb)
                rf_w_proba = self.workout_rf_model.predict_proba(X_test_rf)
                rf_n_proba = self.nutrition_rf_model.predict_proba(X_test_rf)
                
                # Encode true labels
                y_w_test_encoded = self.workout_label_encoder.transform(y_w_test)
                y_n_test_encoded = self.nutrition_label_encoder.transform(y_n_test)
                
                # 6A. Performance Comparison Bar Chart
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle('Model Performance Comparison: XGBoost vs Random Forest', fontsize=16, fontweight='bold')
                
                # Workout models comparison
                workout_metrics = {
                    'XGBoost': [
                        accuracy_score(y_w_test_encoded, xgb_w_pred),
                        f1_score(y_w_test_encoded, xgb_w_pred, average='weighted'),
                        precision_score(y_w_test_encoded, xgb_w_pred, average='weighted'),
                        recall_score(y_w_test_encoded, xgb_w_pred, average='weighted')
                    ],
                    'Random Forest': [
                        accuracy_score(y_w_test_encoded, rf_w_pred),
                        f1_score(y_w_test_encoded, rf_w_pred, average='weighted'),
                        precision_score(y_w_test_encoded, rf_w_pred, average='weighted'),
                        recall_score(y_w_test_encoded, rf_w_pred, average='weighted')
                    ]
                }
                
                x = np.arange(4)
                width = 0.35
                metrics_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
                
                bars1 = axes[0].bar(x - width/2, workout_metrics['XGBoost'], width, label='XGBoost', color='skyblue')
                bars2 = axes[0].bar(x + width/2, workout_metrics['Random Forest'], width, label='Random Forest', color='lightcoral')
                
                axes[0].set_title('Workout Model Performance')
                axes[0].set_ylabel('Score')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(metrics_labels, rotation=45)
                axes[0].legend()
                axes[0].set_ylim(0, 1)
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        axes[0].annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
                
                # Nutrition models comparison
                nutrition_metrics = {
                    'XGBoost': [
                        accuracy_score(y_n_test_encoded, xgb_n_pred),
                        f1_score(y_n_test_encoded, xgb_n_pred, average='weighted'),
                        precision_score(y_n_test_encoded, xgb_n_pred, average='weighted'),
                        recall_score(y_n_test_encoded, xgb_n_pred, average='weighted')
                    ],
                    'Random Forest': [
                        accuracy_score(y_n_test_encoded, rf_n_pred),
                        f1_score(y_n_test_encoded, rf_n_pred, average='weighted'),
                        precision_score(y_n_test_encoded, rf_n_pred, average='weighted'),
                        recall_score(y_n_test_encoded, rf_n_pred, average='weighted')
                    ]
                }
                
                bars3 = axes[1].bar(x - width/2, nutrition_metrics['XGBoost'], width, label='XGBoost', color='skyblue')
                bars4 = axes[1].bar(x + width/2, nutrition_metrics['Random Forest'], width, label='Random Forest', color='lightcoral')
                
                axes[1].set_title('Nutrition Model Performance')
                axes[1].set_ylabel('Score')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(metrics_labels, rotation=45)
                axes[1].legend()
                axes[1].set_ylim(0, 1)
                
                # Add value labels on bars
                for bars in [bars3, bars4]:
                    for bar in bars:
                        height = bar.get_height()
                        axes[1].annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/06_model_performance_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 6B. Confusion Matrices for all 4 models
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Confusion Matrices: All 4 Models', fontsize=16, fontweight='bold')
                
                # XGBoost Workout
                cm_xgb_w = confusion_matrix(y_w_test_encoded, xgb_w_pred)
                sns.heatmap(cm_xgb_w, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
                axes[0, 0].set_title(f'XGBoost Workout Model\nAccuracy: {accuracy_score(y_w_test_encoded, xgb_w_pred):.3f}')
                axes[0, 0].set_xlabel('Predicted')
                axes[0, 0].set_ylabel('Actual')
                
                # XGBoost Nutrition
                cm_xgb_n = confusion_matrix(y_n_test_encoded, xgb_n_pred)
                sns.heatmap(cm_xgb_n, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
                axes[0, 1].set_title(f'XGBoost Nutrition Model\nAccuracy: {accuracy_score(y_n_test_encoded, xgb_n_pred):.3f}')
                axes[0, 1].set_xlabel('Predicted')
                axes[0, 1].set_ylabel('Actual')
                
                # Random Forest Workout
                cm_rf_w = confusion_matrix(y_w_test_encoded, rf_w_pred)
                sns.heatmap(cm_rf_w, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
                axes[1, 0].set_title(f'Random Forest Workout Model\nAccuracy: {accuracy_score(y_w_test_encoded, rf_w_pred):.3f}')
                axes[1, 0].set_xlabel('Predicted')
                axes[1, 0].set_ylabel('Actual')
                
                # Random Forest Nutrition
                cm_rf_n = confusion_matrix(y_n_test_encoded, rf_n_pred)
                sns.heatmap(cm_rf_n, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1])
                axes[1, 1].set_title(f'Random Forest Nutrition Model\nAccuracy: {accuracy_score(y_n_test_encoded, rf_n_pred):.3f}')
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Actual')
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/07_confusion_matrices.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 6C. AUROC Curves for all models
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                from itertools import cycle
                
                # Get unique classes
                workout_classes = sorted(np.unique(y_w_test_encoded))
                nutrition_classes = sorted(np.unique(y_n_test_encoded))
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('ROC Curves: All 4 Models', fontsize=16, fontweight='bold')
                
                # Colors for different classes
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
                
                # XGBoost Workout ROC
                y_w_bin = label_binarize(y_w_test_encoded, classes=workout_classes)
                if len(workout_classes) == 2:
                    fpr, tpr, _ = roc_curve(y_w_test_encoded, xgb_w_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    for i, color in zip(range(len(workout_classes)), colors):
                        fpr, tpr, _ = roc_curve(y_w_bin[:, i], xgb_w_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        axes[0, 0].plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=2)
                axes[0, 0].set_xlim([0.0, 1.0])
                axes[0, 0].set_ylim([0.0, 1.05])
                axes[0, 0].set_xlabel('False Positive Rate')
                axes[0, 0].set_ylabel('True Positive Rate')
                axes[0, 0].set_title('XGBoost Workout Model ROC')
                axes[0, 0].legend(loc="lower right")
                
                # XGBoost Nutrition ROC
                y_n_bin = label_binarize(y_n_test_encoded, classes=nutrition_classes)
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
                if len(nutrition_classes) == 2:
                    fpr, tpr, _ = roc_curve(y_n_test_encoded, xgb_n_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    for i, color in zip(range(len(nutrition_classes)), colors):
                        fpr, tpr, _ = roc_curve(y_n_bin[:, i], xgb_n_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        axes[0, 1].plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=2)
                axes[0, 1].set_xlim([0.0, 1.0])
                axes[0, 1].set_ylim([0.0, 1.05])
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('XGBoost Nutrition Model ROC')
                axes[0, 1].legend(loc="lower right")
                
                # Random Forest Workout ROC
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
                if len(workout_classes) == 2:
                    fpr, tpr, _ = roc_curve(y_w_test_encoded, rf_w_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    for i, color in zip(range(len(workout_classes)), colors):
                        fpr, tpr, _ = roc_curve(y_w_bin[:, i], rf_w_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        axes[1, 0].plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=2)
                axes[1, 0].set_xlim([0.0, 1.0])
                axes[1, 0].set_ylim([0.0, 1.05])
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('Random Forest Workout Model ROC')
                axes[1, 0].legend(loc="lower right")
                
                # Random Forest Nutrition ROC
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
                if len(nutrition_classes) == 2:
                    fpr, tpr, _ = roc_curve(y_n_test_encoded, rf_n_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    for i, color in zip(range(len(nutrition_classes)), colors):
                        fpr, tpr, _ = roc_curve(y_n_bin[:, i], rf_n_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        axes[1, 1].plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=2)
                axes[1, 1].set_xlim([0.0, 1.0])
                axes[1, 1].set_ylim([0.0, 1.05])
                axes[1, 1].set_xlabel('False Positive Rate')
                axes[1, 1].set_ylabel('True Positive Rate')
                axes[1, 1].set_title('Random Forest Nutrition Model ROC')
                axes[1, 1].legend(loc="lower right")
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/08_roc_curves.png', dpi=300, bbox_inches='tight')
                plt.close()
        
            print(f"✅ Comprehensive visualizations saved to '{save_dir}/' directory")
            print(f"   Generated files:")
            print(f"   - 01_dataset_composition.png")
            print(f"   - 02_template_assignments.png") 
            print(f"   - 03_demographics.png")
            print(f"   - 04_goal_activity_bmi_heatmap.png")
            print(f"   - 05_data_quality.png")
            
            if self.is_trained and hasattr(self, 'workout_rf_model') and self.workout_rf_model:
                print(f"   - 06_model_performance_comparison.png")
                print(f"   - 07_confusion_matrices.png")
                print(f"   - 08_roc_curves.png")
                print(f"   📊 All 4 models analyzed: XGBoost + Random Forest for Workout + Nutrition")
            else:
                print(f"   ⚠️  Model performance visualizations skipped (models not trained)")
        
            return save_dir
    
    def retry_with_logging(self, func, *args, max_retries=3, **kwargs):
        """
        Retry a function with detailed logging
        """
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: {func.__name__}")
                result = func(*args, **kwargs)
                print(f"✅ {func.__name__} completed successfully")
                return result
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"🚨 All {max_retries} attempts failed for {func.__name__}")
                    raise e
                else:
                    print(f"⏳ Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
        
        return None
    
    def verify_data_consistency(self, df_training):
        print("\n🔍 VERIFYING DATA CONSISTENCY WITH EXACT COUNTS:")
        print("="*80)
        
        errors = []
        warnings = []
        
        # 1. Check total sample counts
        total_samples = len(df_training)
        real_samples = len(df_training[df_training['data_source'] == 'real'])
        synthetic_samples = len(df_training[df_training['data_source'] == 'synthetic'])
        
        print(f"📊 SAMPLE COUNT VERIFICATION:")
        print(f"   Total samples: {total_samples}")
        print(f"   Real samples: {real_samples}")
        print(f"   Synthetic samples: {synthetic_samples}")
        print(f"   Sum check: {real_samples + synthetic_samples} = {total_samples} ✓" if real_samples + synthetic_samples == total_samples else f"   Sum check: {real_samples + synthetic_samples} ≠ {total_samples} ❌")
        
        if real_samples + synthetic_samples != total_samples:
            errors.append(f"Sample count mismatch: {real_samples} + {synthetic_samples} ≠ {total_samples}")
        
        # 2. Check split distributions with exact targets
        train_count = len(df_training[df_training['split'] == 'train'])
        val_count = len(df_training[df_training['split'] == 'validation'])
        test_count = len(df_training[df_training['split'] == 'test'])
        
        print(f"\n📊 SPLIT DISTRIBUTION VERIFICATION:")
        print(f"   Training: {train_count} samples")
        print(f"   Validation: {val_count} samples") 
        print(f"   Test: {test_count} samples")
        print(f"   Sum check: {train_count + val_count + test_count} = {total_samples} ✓" if train_count + val_count + test_count == total_samples else f"   Sum check: {train_count + val_count + test_count} ≠ {total_samples} ❌")
        
        if train_count + val_count + test_count != total_samples:
            errors.append(f"Split count mismatch: {train_count} + {val_count} + {test_count} ≠ {total_samples}")
        
        # 3. CRITICAL: Check that validation and test sets are 100% real data
        val_real = len(df_training[(df_training['split'] == 'validation') & (df_training['data_source'] == 'real')])
        val_synthetic = len(df_training[(df_training['split'] == 'validation') & (df_training['data_source'] == 'synthetic')])
        test_real = len(df_training[(df_training['split'] == 'test') & (df_training['data_source'] == 'real')])
        test_synthetic = len(df_training[(df_training['split'] == 'test') & (df_training['data_source'] == 'synthetic')])
        
        print(f"\n🚫 VALIDATION/TEST REAL DATA VERIFICATION:")
        print(f"   Validation: {val_real} real + {val_synthetic} synthetic = {val_count} total")
        print(f"   Test: {test_real} real + {test_synthetic} synthetic = {test_count} total")
        
        if val_synthetic > 0:
            errors.append(f"CRITICAL: Validation set contains {val_synthetic} synthetic samples (must be 0)")
        if test_synthetic > 0:
            errors.append(f"CRITICAL: Test set contains {test_synthetic} synthetic samples (must be 0)")
        
        if val_synthetic == 0 and test_synthetic == 0:
            print(f"   ✅ Validation and test sets are 100% real data")
        
        # 4. Check training set composition
        train_real = len(df_training[(df_training['split'] == 'train') & (df_training['data_source'] == 'real')])
        train_synthetic = len(df_training[(df_training['split'] == 'train') & (df_training['data_source'] == 'synthetic')])
        
        print(f"\n📊 TRAINING SET COMPOSITION:")
        print(f"   Training: {train_real} real + {train_synthetic} synthetic = {train_count} total")
        print(f"   Real percentage: {100*train_real/train_count:.1f}%")
        print(f"   Synthetic percentage: {100*train_synthetic/train_count:.1f}%")
        
        # 5. Check exact 70/15/15 split of real data
        real_train_pct = train_real / real_samples * 100
        real_val_pct = val_real / real_samples * 100
        real_test_pct = test_real / real_samples * 100
        
        print(f"\n📊 REAL DATA SPLIT VERIFICATION (Target: 70/15/15):")
        print(f"   Training: {train_real}/{real_samples} = {real_train_pct:.1f}% (target: 70%)")
        print(f"   Validation: {val_real}/{real_samples} = {real_val_pct:.1f}% (target: 15%)")
        print(f"   Test: {test_real}/{real_samples} = {real_test_pct:.1f}% (target: 15%)")
        
        # Allow small deviation (±2%) due to stratification
        if not (68 <= real_train_pct <= 72):
            warnings.append(f"Training real data percentage ({real_train_pct:.1f}%) deviates from target 70%")
        if not (13 <= real_val_pct <= 17):
            warnings.append(f"Validation real data percentage ({real_val_pct:.1f}%) deviates from target 15%")
        if not (13 <= real_test_pct <= 17):
            warnings.append(f"Test real data percentage ({real_test_pct:.1f}%) deviates from target 15%")
        
        # 6. Check template ID ranges
        workout_ids = df_training['workout_template_id'].unique()
        nutrition_ids = df_training['nutrition_template_id'].unique()
        
        print(f"\n📊 TEMPLATE ID VERIFICATION:")
        print(f"   Workout templates: {sorted(workout_ids)}")
        print(f"   Nutrition templates: {sorted(nutrition_ids)}")
        
        if not all(1 <= wid <= 9 for wid in workout_ids):
            errors.append(f"Invalid workout template IDs found: {sorted(workout_ids)}")
        if not all(1 <= nid <= 8 for nid in nutrition_ids):
            errors.append(f"Invalid nutrition template IDs found: {sorted(nutrition_ids)}")
        
        # 7. Check goal distribution balance in training set (after augmentation)
        train_data = df_training[df_training['split'] == 'train']
        goal_counts = train_data['fitness_goal'].value_counts()
        
        print(f"\n📊 TRAINING GOAL DISTRIBUTION (after augmentation):")
        for goal, count in goal_counts.items():
            print(f"   {goal}: {count} samples")
        
        # Check if goals are balanced (allowing small variation)
        max_diff = goal_counts.max() - goal_counts.min()
        if max_diff > 50:  # Allow up to 50 sample difference
            warnings.append(f"Training goal distribution not well balanced (max diff: {max_diff})")
        
        # 8. Check natural distribution preservation
        print(f"\n🔒 NATURAL DISTRIBUTION PRESERVATION CHECK:")
        
        # Activity distribution should match real data patterns
        activity_dist = df_training['activity_level'].value_counts(normalize=True) * 100
        print(f"   Activity distribution:")
        for activity, pct in activity_dist.items():
            print(f"     {activity}: {pct:.1f}%")
        
        # Expected ranges based on real data (with some tolerance for synthetic additions)
        if not (60 <= activity_dist.get('High Activity', 0) <= 75):
            warnings.append(f"High Activity percentage ({activity_dist.get('High Activity', 0):.1f}%) outside expected range")
        
        # 9. Check for missing values
        missing_cols = df_training.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) > 0:
            warnings.append(f"Missing values found: {dict(missing_cols)}")
        
        # Report results
        print(f"\n{'='*80}")
        if errors:
            print("🚨 CRITICAL ERRORS FOUND:")
            for error in errors:
                print(f"   ❌ {error}")
        
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   ⚠️  {warning}")
        
        if not errors and not warnings:
            print("✅ ALL DATA CONSISTENCY CHECKS PASSED!")
        elif not errors:
            print("✅ No critical errors found, only minor warnings")
        
        # Final summary statistics
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total samples: {total_samples}")
        print(f"   Real data: {real_samples} ({100*real_samples/total_samples:.1f}%)")
        print(f"   Synthetic data: {synthetic_samples} ({100*synthetic_samples/total_samples:.1f}%)")
        print(f"   Training: {train_count} samples ({train_real} real + {train_synthetic} synthetic)")
        print(f"   Validation: {val_count} samples (100% real)")
        print(f"   Test: {test_count} samples (100% real)")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'total_samples': total_samples,
            'real_samples': real_samples,
            'synthetic_samples': synthetic_samples,
            'split_counts': {'train': train_count, 'validation': val_count, 'test': test_count},
            'real_split_percentages': {'train': real_train_pct, 'validation': real_val_pct, 'test': real_test_pct},
            'validation_100_real': val_synthetic == 0,
            'test_100_real': test_synthetic == 0
        }
    
    def prepare_training_data(self, df_training):
        """Prepare data for model training with feature engineering"""
        print("Preparing training data with feature engineering...")
        
        # Create enhanced features
        df_enhanced = self.create_enhanced_features(df_training)
        
        # Select features for training - OPTIMIZED for production accuracy
        # Balance between informative features and avoiding overfitting
        feature_columns = [
            # Core physiological features
            'age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'tdee',
            'activity_multiplier', 'Mod_act', 'Vig_act',
            
            # Engineered interaction features for better pattern recognition
            'age_bmi_interaction', 'tdee_per_kg', 'activity_intensity',
            'height_weight_ratio', 'bmr_per_kg', 'age_activity_interaction',
            
            # Demographic features
            'gender_Male',
            
            # Age groups for pattern recognition
            'age_group_young', 'age_group_middle', 'age_group_older',
            
            # Detailed BMI health indicators for better pattern learning
            'bmi_Normal', 'bmi_Overweight', 'bmi_Obese',
            'bmi_low_normal', 'bmi_high_normal', 'bmi_low_overweight', 'bmi_high_overweight',
            
            # Activity level categories (legitimate health patterns)
            'activity_High Activity', 'activity_Moderate Activity', 'activity_Low Activity'
            
            # EXCLUDED: 'goal_Fat Loss', 'goal_Muscle Gain' - too direct for template assignment
        ]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Prepare feature matrix
        X = df_enhanced[feature_columns].fillna(0)
        
        # Prepare target variables
        y_workout = df_enhanced['workout_template_id']
        y_nutrition = df_enhanced['nutrition_template_id']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Workout targets: {sorted(y_workout.unique())}")
        print(f"Nutrition targets: {sorted(y_nutrition.unique())}")
        
        return X, y_workout, y_nutrition, df_enhanced
    
    def create_enhanced_features(self, df):
        """Create enhanced features for better model performance"""
        df_enhanced = df.copy()
        
        # Interaction features
        df_enhanced['age_bmi_interaction'] = df_enhanced['age'] * df_enhanced['bmi']
        df_enhanced['tdee_per_kg'] = df_enhanced['tdee'] / df_enhanced['weight_kg']
        df_enhanced['activity_intensity'] = df_enhanced['Mod_act'] + (df_enhanced['Vig_act'] * 2)
        
        # Additional interaction features for better accuracy
        df_enhanced['height_weight_ratio'] = df_enhanced['height_cm'] / df_enhanced['weight_kg']
        df_enhanced['bmr_per_kg'] = df_enhanced['bmr'] / df_enhanced['weight_kg']
        df_enhanced['age_activity_interaction'] = df_enhanced['age'] * df_enhanced['activity_intensity']
        
        # Add age groups for better pattern recognition
        df_enhanced['age_group_young'] = (df_enhanced['age'] <= 30).astype(int)
        df_enhanced['age_group_middle'] = ((df_enhanced['age'] > 30) & (df_enhanced['age'] <= 50)).astype(int)
        df_enhanced['age_group_older'] = (df_enhanced['age'] > 50).astype(int)
        
        # Add BMI range features for better health pattern recognition
        df_enhanced['bmi_low_normal'] = ((df_enhanced['bmi'] >= 18.5) & (df_enhanced['bmi'] < 22)).astype(int)
        df_enhanced['bmi_high_normal'] = ((df_enhanced['bmi'] >= 22) & (df_enhanced['bmi'] < 25)).astype(int)
        df_enhanced['bmi_low_overweight'] = ((df_enhanced['bmi'] >= 25) & (df_enhanced['bmi'] < 27.5)).astype(int)
        df_enhanced['bmi_high_overweight'] = ((df_enhanced['bmi'] >= 27.5) & (df_enhanced['bmi'] < 30)).astype(int)
        
        # One-hot encoding for categorical variables
        df_enhanced = pd.get_dummies(df_enhanced, columns=['gender'], prefix='gender', drop_first=False)
        df_enhanced = pd.get_dummies(df_enhanced, columns=['fitness_goal'], prefix='goal', drop_first=False)
        df_enhanced = pd.get_dummies(df_enhanced, columns=['activity_level'], prefix='activity', drop_first=False)
        df_enhanced = pd.get_dummies(df_enhanced, columns=['bmi_category'], prefix='bmi', drop_first=False)
        
        # Fill missing columns with 0 if they don't exist
        expected_columns = [
            'gender_Male', 'gender_Female',
            'goal_Fat Loss', 'goal_Muscle Gain', 'goal_Maintenance',
            'activity_High Activity', 'activity_Moderate Activity', 'activity_Low Activity',
            'bmi_Normal', 'bmi_Obese', 'bmi_Overweight', 'bmi_Underweight'
        ]
        
        for col in expected_columns:
            if col not in df_enhanced.columns:
                df_enhanced[col] = 0
        
        return df_enhanced
    
    def _add_template_assignment_noise(self, workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.05):
        """
        Add realistic noise to template assignments to simulate real user choice variability
        
        Args:
            workout_id: Original workout template ID
            nutrition_id: Original nutrition template ID  
            fitness_goal: User's fitness goal
            activity_level: User's activity level
            bmi_category: User's BMI category
            noise_prob: Probability of adding noise (5% for production-ready balance)
            
        Returns:
            tuple: (possibly modified workout_id, possibly modified nutrition_id)
        """
        import random
        
        # Apply realistic noise - people with similar profiles often choose different templates
        if random.random() > noise_prob:
            return workout_id, nutrition_id
        
        # For workout templates, allow cross-goal variations for similar user profiles
        # This creates ambiguity that different models can handle differently
        valid_workout_ids = [workout_id]  # Start with original
        
        if fitness_goal == 'Muscle Gain':
            if activity_level == 'High Activity':
                valid_workout_ids = [6, 3, 9]  # Muscle Gain High, Fat Loss High, Maintenance High
            elif activity_level == 'Moderate Activity':
                valid_workout_ids = [5, 2, 8]  # Cross-goal moderate activity options
            else:  # Low Activity
                valid_workout_ids = [4, 1, 7]  # Cross-goal low activity options
        elif fitness_goal == 'Fat Loss':
            if activity_level == 'High Activity':
                valid_workout_ids = [3, 6, 9]  # Fat Loss High, Muscle Gain High, Maintenance High
            elif activity_level == 'Moderate Activity':
                valid_workout_ids = [2, 5, 8]  # Cross-goal moderate activity options
            else:  # Low Activity
                valid_workout_ids = [1, 4, 7]  # Cross-goal low activity options
        elif fitness_goal == 'Maintenance':
            if activity_level == 'High Activity':
                valid_workout_ids = [9, 3, 6]  # Maintenance High, Fat Loss High, Muscle Gain High
            elif activity_level == 'Moderate Activity':
                valid_workout_ids = [8, 2, 5]  # Cross-goal moderate activity options
            else:  # Low Activity
                valid_workout_ids = [7, 1, 4]  # Cross-goal low activity options
        
        # For nutrition templates, allow some flexibility within reasonable bounds
        valid_nutrition_ids = [nutrition_id]  # Start with original
        
        if fitness_goal == 'Muscle Gain':
            if bmi_category in ['Underweight', 'Normal']:
                valid_nutrition_ids = [4, 5, 6, 7]  # Muscle Gain + Maintenance templates
        elif fitness_goal == 'Fat Loss':
            if bmi_category in ['Normal', 'Overweight', 'Obese']:
                valid_nutrition_ids = [1, 2, 3, 7, 8]  # Fat Loss + some Maintenance
        elif fitness_goal == 'Maintenance':
            # Maintenance can use templates from all goals depending on BMI
            if bmi_category == 'Underweight':
                valid_nutrition_ids = [4, 6]  # Muscle Gain Underweight, Maintenance Underweight
            elif bmi_category == 'Normal':
                valid_nutrition_ids = [1, 5, 7]  # Fat Loss Normal, Muscle Gain Normal, Maintenance Normal
            elif bmi_category == 'Overweight':
                valid_nutrition_ids = [2, 8]  # Fat Loss Overweight, Maintenance Overweight
        
        # Apply noise by randomly selecting from valid options
        noisy_workout_id = random.choice(valid_workout_ids) if len(valid_workout_ids) > 1 else workout_id
        noisy_nutrition_id = random.choice(valid_nutrition_ids) if len(valid_nutrition_ids) > 1 else nutrition_id
        
        return noisy_workout_id, noisy_nutrition_id
    
    def calculate_prediction_confidence(self, user_profile, workout_template_id, nutrition_template_id):
        """Calculate enhanced confidence scoring for predictions"""
        activity_level = user_profile.get('activity_level', 'Unknown')
        fitness_goal = user_profile.get('fitness_goal', 'Unknown')
        
        # Base confidence based on activity level (main data limitation)
        activity_confidence_map = {
            'High Activity': 0.9,      # High confidence - most training data
            'Moderate Activity': 0.7,  # Medium confidence - some training data
            'Low Activity': 0.5,       # Low confidence - limited training data
            'Unknown': 0.3
        }
        
        # Goal confidence based on template availability
        goal_confidence_map = {
            'Fat Loss': 0.85,      # Well represented
            'Muscle Gain': 0.80,   # Well represented
            'Maintenance': 0.75,   # Moderately represented
            'Unknown': 0.4
        }
        
        activity_confidence = activity_confidence_map.get(activity_level, 0.3)
        goal_confidence = goal_confidence_map.get(fitness_goal, 0.4)
        
        # Overall confidence is the minimum of the two (conservative approach)
        overall_confidence = min(activity_confidence, goal_confidence)
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = "High"
        elif overall_confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Generate explanation
        explanation = f"Based on {activity_level.lower()} and {fitness_goal.lower()} goal"
        
        # Generate limitations
        limitations = []
        if activity_level == 'Low Activity':
            limitations.append("Limited training data for low-activity individuals")
        if overall_confidence < 0.6:
            limitations.append("Consider consulting a fitness professional")
        
        return {
            'confidence_score': round(overall_confidence, 3),
            'confidence_level': confidence_level,
            'explanation': explanation,
            'limitations': limitations,
            'activity_confidence': round(activity_confidence, 3),
            'goal_confidence': round(goal_confidence, 3)
        }
    
    def save_model(self, filepath='models/xgfitness_ai_model.pkl', include_research_models=False):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train_all_models() first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"💾 Saving model to: {filepath}")
        if include_research_models:
            print("📊 Including Random Forest models for research analysis")
        else:
            print("🚀 Production mode: XGBoost models only")
        
        # Prepare model data for saving - ONLY XGBoost for production
        model_data = {
            # PRODUCTION MODELS (XGBoost only)
            'workout_model': self.workout_model,
            'nutrition_model': self.nutrition_model,
            'scaler': self.scaler,
            'workout_label_encoder': getattr(self, 'workout_label_encoder', None),
            'nutrition_label_encoder': getattr(self, 'nutrition_label_encoder', None),
            'feature_columns': self.feature_columns,
            'workout_templates': self.workout_templates,
            'nutrition_templates': self.nutrition_templates,
            'training_info': self.training_info,
            'is_trained': self.is_trained,
            'model_version': '2.1',
            'saved_at': datetime.now().isoformat(),
            'model_type': 'production_xgboost'
        }
        
        # Include Random Forest models only for research
        if include_research_models:
            model_data.update({
                'workout_rf_model': getattr(self, 'workout_rf_model', None),
                'nutrition_rf_model': getattr(self, 'nutrition_rf_model', None),
                'rf_scaler': getattr(self, 'rf_scaler', None),
                'workout_rf_label_encoder': getattr(self, 'workout_rf_label_encoder', None),
                'nutrition_rf_label_encoder': getattr(self, 'nutrition_rf_label_encoder', None),
                'rf_training_info': getattr(self, 'rf_training_info', None),
                'model_type': 'research_with_baselines'
            })
        
        # Save model data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        
        print(f"✅ Model successfully saved!")
        print(f"   - File size: {file_size:.2f} MB")
        print(f"   - XGBoost models: ✅ (for web app)")
        if include_research_models:
            print(f"   - Random Forest models: ✅ (for research only)")
        else:
            print(f"   - Random Forest models: ❌ (excluded from production)")
        print(f"   - Training samples: {self.training_info.get('training_samples', 'Unknown')}")
        print(f"   - Model version: 2.1")
        
        return filepath
    
    def load_model(self, filepath):
        """
        Load a trained model from a pickle file
        
        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        print(f"📥 Loading model from: {filepath}")
        
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model components
        self.workout_model = model_data['workout_model']
        self.nutrition_model = model_data['nutrition_model']
        self.workout_rf_model = model_data.get('workout_rf_model')
        self.nutrition_rf_model = model_data.get('nutrition_rf_model')
        self.scaler = model_data['scaler']
        self.rf_scaler = model_data.get('rf_scaler')
        self.workout_label_encoder = model_data.get('workout_label_encoder')
        self.nutrition_label_encoder = model_data.get('nutrition_label_encoder')
        self.workout_rf_label_encoder = model_data.get('workout_rf_label_encoder')
        self.nutrition_rf_label_encoder = model_data.get('nutrition_rf_label_encoder')
        self.feature_columns = model_data['feature_columns']
        self.workout_templates = model_data['workout_templates']
        self.nutrition_templates = model_data['nutrition_templates']
        self.training_info = model_data['training_info']
        self.rf_training_info = model_data.get('rf_training_info')
        self.is_trained = model_data['is_trained']
        
        model_version = model_data.get('model_version', '1.0')
        saved_at = model_data.get('saved_at', 'Unknown')
        
        print(f"✅ Model successfully loaded!")
        print(f"   - Model version: {model_version}")
        print(f"   - Saved at: {saved_at}")
        print(f"   - XGBoost Workout model: {type(self.workout_model).__name__}")
        print(f"   - XGBoost Nutrition model: {type(self.nutrition_model).__name__}")
        print(f"   - Random Forest Workout model: {type(self.workout_rf_model).__name__ if self.workout_rf_model else 'Not available'}")
        print(f"   - Random Forest Nutrition model: {type(self.nutrition_rf_model).__name__ if self.nutrition_rf_model else 'Not available'}")
        print(f"   - Training samples: {self.training_info.get('training_samples', 'Unknown')}")
        print(f"   - XGBoost Workout accuracy: {self.training_info.get('workout_accuracy', 0):.4f}")
        print(f"   - XGBoost Nutrition accuracy: {self.training_info.get('nutrition_accuracy', 0):.4f}")
        if self.rf_training_info:
            print(f"   - Random Forest Workout accuracy: {self.rf_training_info.get('rf_workout_accuracy', 0):.4f}")
            print(f"   - Random Forest Nutrition accuracy: {self.rf_training_info.get('rf_nutrition_accuracy', 0):.4f}")
    
    def train_models(self, df_training, random_state=42):
        """Train XGBoost models with enhanced techniques for good generalization"""
        print("Starting XGBoost model training with anti-overfitting measures...")
        
        # Comprehensive debugging and reporting
        print("\n🔍 RUNNING COMPREHENSIVE DEBUGGING ANALYSIS...")
        self.debug_template_assignment_logic()
        self.debug_template_assignments(df_training)
        self.debug_training_splits(df_training)
        
        # Transparent limitations reporting
        print("\n📋 MODEL LIMITATIONS & DATA CHARACTERISTICS:")
        print("="*80)
        activity_dist = df_training['activity_level'].value_counts(normalize=True) * 100
        print(f"✅ THESIS FINDING: Model optimized for high-activity individuals")
        print(f"   High Activity: {activity_dist.get('High Activity', 0):.1f}% of training data")
        print(f"   Moderate Activity: {activity_dist.get('Moderate Activity', 0):.1f}% of training data")
        print(f"   Low Activity: {activity_dist.get('Low Activity', 0):.1f}% of training data")
        print(f"⚠️  LIMITATION: Reduced confidence for low-activity recommendations")
        print(f"💡 RECOMMENDATION: Professional consultation for low-activity individuals")
        print("="*80)
        
        # Prepare data
        X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
        
        # Split data based on the 'split' column
        train_mask = df_enhanced['split'] == 'train'
        val_mask = df_enhanced['split'] == 'validation' 
        test_mask = df_enhanced['split'] == 'test'
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        y_w_train = y_workout[train_mask]
        y_w_val = y_workout[val_mask]
        y_w_test = y_workout[test_mask]
        
        y_n_train = y_nutrition[train_mask]
        y_n_val = y_nutrition[val_mask]
        y_n_test = y_nutrition[test_mask]
        
        print(f"Training split sizes:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels using LabelEncoder
        self.workout_label_encoder = LabelEncoder()
        self.nutrition_label_encoder = LabelEncoder()
        
        # Fit on all possible template IDs
        all_workout_ids = list(range(1, 10))  # Template IDs 1-9
        all_nutrition_ids = list(range(1, 9))  # Template IDs 1-8
        
        self.workout_label_encoder.fit(all_workout_ids)
        self.nutrition_label_encoder.fit(all_nutrition_ids)
        
        # Transform to continuous indices
        y_w_train_encoded = self.workout_label_encoder.transform(y_w_train)
        y_w_val_encoded = self.workout_label_encoder.transform(y_w_val)
        y_w_test_encoded = self.workout_label_encoder.transform(y_w_test)
        
        y_n_train_encoded = self.nutrition_label_encoder.transform(y_n_train)
        y_n_val_encoded = self.nutrition_label_encoder.transform(y_n_val)
        y_n_test_encoded = self.nutrition_label_encoder.transform(y_n_test)
        
        # Optimized hyperparameter distributions
        workout_param_distributions = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 150, 200],
            'min_child_weight': [5, 10, 15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.5, 1.0, 2.0],
            'reg_lambda': [1.0, 2.0, 5.0],
            'gamma': [0.1, 0.5, 1.0]
        }
        
        # Conservative parameters for nutrition model
        nutrition_param_distributions = {
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.03, 0.05],
            'n_estimators': [50, 100, 150],
            'min_child_weight': [5, 10, 15],
            'subsample': [0.5, 0.6, 0.7],
            'colsample_bytree': [0.5, 0.6, 0.7],
            'reg_alpha': [1.0, 2.0, 5.0],
            'reg_lambda': [2.0, 5.0, 10.0]
        }
        
        # Base XGBoost parameters
        base_params = {
            'random_state': random_state,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'early_stopping_rounds': 30,
            'verbose': False,
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        
        # Train workout model
        print("Training workout XGBoost model with hyperparameter tuning...")
        
        workout_xgb = xgb.XGBClassifier(**base_params)
        
        workout_search = RandomizedSearchCV(
            workout_xgb,
            param_distributions=workout_param_distributions,
            n_iter=20,
            cv=10,
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        workout_search.fit(
            X_train_scaled, y_w_train_encoded,
            eval_set=[(X_val_scaled, y_w_val_encoded)],
            verbose=False
        )
        
        self.workout_model = workout_search.best_estimator_
        workout_val_score = self.workout_model.score(X_val_scaled, y_w_val_encoded)
        
        print(f"Best workout XGBoost parameters: {workout_search.best_params_}")
        print(f"Workout model validation score: {workout_val_score:.4f}")
        
        # Train nutrition model
        print("Training nutrition XGBoost model with hyperparameter tuning...")
        
        nutrition_xgb = xgb.XGBClassifier(**base_params)
        
        nutrition_search = RandomizedSearchCV(
            nutrition_xgb,
            param_distributions=nutrition_param_distributions,
            n_iter=15,
            cv=10,
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        nutrition_search.fit(
            X_train_scaled, y_n_train_encoded,
            eval_set=[(X_val_scaled, y_n_val_encoded)],
            verbose=False
        )
        
        self.nutrition_model = nutrition_search.best_estimator_
        nutrition_val_score = self.nutrition_model.score(X_val_scaled, y_n_val_encoded)
        
        print(f"Best nutrition XGBoost parameters: {nutrition_search.best_params_}")
        print(f"Nutrition model validation score: {nutrition_val_score:.4f}")
        
        # Evaluate on test set
        workout_test_score = self.workout_model.score(X_test_scaled, y_w_test_encoded)
        nutrition_test_score = self.nutrition_model.score(X_test_scaled, y_n_test_encoded)
        
        # Generate detailed metrics
        y_w_pred = self.workout_model.predict(X_test_scaled)
        y_n_pred = self.nutrition_model.predict(X_test_scaled)
        
        workout_f1 = f1_score(y_w_test_encoded, y_w_pred, average='weighted')
        nutrition_f1 = f1_score(y_n_test_encoded, y_n_pred, average='weighted')
        
        # Calculate precision and recall
        workout_precision = precision_score(y_w_test_encoded, y_w_pred, average='weighted')
        workout_recall = recall_score(y_w_test_encoded, y_w_pred, average='weighted')
        nutrition_precision = precision_score(y_n_test_encoded, y_n_pred, average='weighted')
        nutrition_recall = recall_score(y_n_test_encoded, y_n_pred, average='weighted')
        
        print(f"\n✅ XGBoost Model Training Complete!")
        print(f"Workout Model - Test Accuracy: {workout_test_score:.4f}, F1: {workout_f1:.4f}")
        print(f"Nutrition Model - Test Accuracy: {nutrition_test_score:.4f}, F1: {nutrition_f1:.4f}")
        
        # Store training information
        self.training_info = {
            'total_samples': len(df_training),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'workout_accuracy': workout_test_score,
            'workout_f1': workout_f1,
            'workout_precision': workout_precision,
            'workout_recall': workout_recall,
            'nutrition_accuracy': nutrition_test_score,
            'nutrition_f1': nutrition_f1,
            'nutrition_precision': nutrition_precision,
            'nutrition_recall': nutrition_recall,
            'model_type': 'XGBoost',
            'feature_columns': self.feature_columns
        }
        
        self.is_trained = True
        
        return {
            'workout_model': self.workout_model,
            'nutrition_model': self.nutrition_model,
            'scaler': self.scaler,
            'workout_label_encoder': self.workout_label_encoder,
            'nutrition_label_encoder': self.nutrition_label_encoder,
            'feature_columns': self.feature_columns,
            'workout_templates': self.workout_templates,
            'nutrition_templates': self.nutrition_templates,
            'training_info': self.training_info,
            'is_trained': self.is_trained,
            'model_version': '2.1',
            'saved_at': datetime.now().isoformat()
        }
    
    def train_random_forest_baselines(self, df_training, random_state=42):
            """Train Random Forest baseline models for academic comparison with identical data splits"""
            print("🌲 Training Random Forest baseline models...")
            
            # Use EXACT same data preparation as XGBoost
            X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
            
            # Use IDENTICAL splits as XGBoost for fair comparison
            train_mask = df_enhanced['split'] == 'train'
            val_mask = df_enhanced['split'] == 'validation' 
            test_mask = df_enhanced['split'] == 'test'
            
            X_train = X[train_mask]
            X_val = X[val_mask]
            X_test = X[test_mask]
            
            y_w_train = y_workout[train_mask]
            y_w_val = y_workout[val_mask]
            y_w_test = y_workout[test_mask]
            
            y_n_train = y_nutrition[train_mask]
            y_n_val = y_nutrition[val_mask]
            y_n_test = y_nutrition[test_mask]
            
            print(f"Random Forest using identical splits:")
            print(f"  Training: {len(X_train)} samples")
            print(f"  Validation: {len(X_val)} samples") 
            print(f"  Test: {len(X_test)} samples")
            
            # Create separate scaler for Random Forest (important for fair comparison)
            self.rf_scaler = StandardScaler()
            X_train_scaled = self.rf_scaler.fit_transform(X_train)
            X_val_scaled = self.rf_scaler.transform(X_val)
            X_test_scaled = self.rf_scaler.transform(X_test)
            
            # Create separate label encoders for Random Forest
            self.workout_rf_label_encoder = LabelEncoder()
            self.nutrition_rf_label_encoder = LabelEncoder()
            
            # Use same template ID ranges as XGBoost
            all_workout_ids = list(range(1, 10))  # Template IDs 1-9
            all_nutrition_ids = list(range(1, 9))  # Template IDs 1-8
            
            self.workout_rf_label_encoder.fit(all_workout_ids)
            self.nutrition_rf_label_encoder.fit(all_nutrition_ids)
            
            # Transform labels
            y_w_train_encoded = self.workout_rf_label_encoder.transform(y_w_train)
            y_w_val_encoded = self.workout_rf_label_encoder.transform(y_w_val)
            y_w_test_encoded = self.workout_rf_label_encoder.transform(y_w_test)
            
            y_n_train_encoded = self.nutrition_rf_label_encoder.transform(y_n_train)
            y_n_val_encoded = self.nutrition_rf_label_encoder.transform(y_n_val)
            y_n_test_encoded = self.nutrition_rf_label_encoder.transform(y_n_test)
            
            # Optimized Random Forest hyperparameters
            rf_workout_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'class_weight': ['balanced', None]
            }
            
            rf_nutrition_params = {
                'n_estimators': [50, 100, 150],  # Conservative for nutrition
                'max_depth': [5, 10, 15],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 4, 6],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True],
                'class_weight': ['balanced', None]
            }
            
            # Train workout Random Forest with hyperparameter tuning
            print("Training workout Random Forest model...")
            rf_workout = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            
            rf_workout_search = RandomizedSearchCV(
                rf_workout,
                param_distributions=rf_workout_params,
                n_iter=25,  # More iterations for thorough search
                cv=5,
                scoring='f1_weighted',
                random_state=random_state,
                n_jobs=-1,
                verbose=1
            )
            
            rf_workout_search.fit(X_train_scaled, y_w_train_encoded)
            self.workout_rf_model = rf_workout_search.best_estimator_
            
            print(f"Best workout RF parameters: {rf_workout_search.best_params_}")
            
            # Train nutrition Random Forest with hyperparameter tuning
            print("Training nutrition Random Forest model...")
            rf_nutrition = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            
            rf_nutrition_search = RandomizedSearchCV(
                rf_nutrition,
                param_distributions=rf_nutrition_params,
                n_iter=20,  # Conservative for nutrition
                cv=5,
                scoring='f1_weighted',
                random_state=random_state,
                n_jobs=-1,
                verbose=1
            )
            
            rf_nutrition_search.fit(X_train_scaled, y_n_train_encoded)
            self.nutrition_rf_model = rf_nutrition_search.best_estimator_
            
            print(f"Best nutrition RF parameters: {rf_nutrition_search.best_params_}")
            
            # Evaluate Random Forest models on test set
            rf_workout_score = self.workout_rf_model.score(X_test_scaled, y_w_test_encoded)
            rf_nutrition_score = self.nutrition_rf_model.score(X_test_scaled, y_n_test_encoded)
            
            y_w_pred_rf = self.workout_rf_model.predict(X_test_scaled)
            y_n_pred_rf = self.nutrition_rf_model.predict(X_test_scaled)
            
            rf_workout_f1 = f1_score(y_w_test_encoded, y_w_pred_rf, average='weighted')
            rf_nutrition_f1 = f1_score(y_n_test_encoded, y_n_pred_rf, average='weighted')
            
            # Calculate precision and recall for Random Forest
            rf_workout_precision = precision_score(y_w_test_encoded, y_w_pred_rf, average='weighted')
            rf_workout_recall = recall_score(y_w_test_encoded, y_w_pred_rf, average='weighted')
            rf_nutrition_precision = precision_score(y_n_test_encoded, y_n_pred_rf, average='weighted')
            rf_nutrition_recall = recall_score(y_n_test_encoded, y_n_pred_rf, average='weighted')
            
            print(f"\n✅ Random Forest Model Training Complete!")
            print(f"Workout Model - Test Accuracy: {rf_workout_score:.4f}, F1: {rf_workout_f1:.4f}")
            print(f"Nutrition Model - Test Accuracy: {rf_nutrition_score:.4f}, F1: {rf_nutrition_f1:.4f}")
            
            # Store Random Forest training information
            self.rf_training_info = {
                'rf_workout_accuracy': rf_workout_score,
                'rf_workout_f1': rf_workout_f1,
                'rf_workout_precision': rf_workout_precision,
                'rf_workout_recall': rf_workout_recall,
                'rf_nutrition_accuracy': rf_nutrition_score,
                'rf_nutrition_f1': rf_nutrition_f1,
                'rf_nutrition_precision': rf_nutrition_precision,
                'rf_nutrition_recall': rf_nutrition_recall,
                'rf_workout_params': rf_workout_search.best_params_,
                'rf_nutrition_params': rf_nutrition_search.best_params_,
                'model_type': 'Random Forest',
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test)
            }
            
            return self.rf_training_info
    
    def train_all_models(self, df_training, random_state=42):
        """Train both XGBoost and Random Forest models for comprehensive comparison"""
        print("🚀 Training ALL models (XGBoost + Random Forest) for comprehensive comparison...")
        
        # Train XGBoost models first
        print("\n1️⃣ Training XGBoost models...")
        xgb_info = self.train_models(df_training, random_state=random_state)
        
        # Train Random Forest baselines  
        print("\n2️⃣ Training Random Forest baseline models...")
        rf_info = self.train_random_forest_baselines(df_training, random_state=random_state)
        
        # Comprehensive model comparison
        print("\n3️⃣ Performing comprehensive model comparison...")
        comparison_results = self.compare_model_predictions(df_training)
        
        return {
            'xgb_training_info': self.training_info,
            'rf_training_info': self.rf_training_info,
            'comparison_data': comparison_results
        }
    
    def compare_model_predictions(self, df_training):
        """Compare XGBoost vs Random Forest predictions to detect identical performance issues"""
        print("🔍 Comparing XGBoost vs Random Forest predictions...")
        
        if not (self.is_trained and hasattr(self, 'workout_rf_model') and self.workout_rf_model):
            print("⚠️  Models not fully trained - skipping comparison")
            return {}
        
        # Get test data
        X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
        
        test_mask = df_enhanced['split'] == 'test'
        X_test = X[test_mask]
        y_w_test = y_workout[test_mask]
        y_n_test = y_nutrition[test_mask]
        
        if len(X_test) == 0:
            print("⚠️  No test data available for comparison")
            return {}
        
        # Scale test data for both models
        X_test_xgb = self.scaler.transform(X_test)
        X_test_rf = self.rf_scaler.transform(X_test)
        
        # Get predictions from both models
        print("Getting predictions from both model types...")
        
        # XGBoost predictions
        xgb_workout_pred = self.workout_model.predict(X_test_xgb)
        xgb_nutrition_pred = self.nutrition_model.predict(X_test_xgb)
        
        # Random Forest predictions
        rf_workout_pred = self.workout_rf_model.predict(X_test_rf)
        rf_nutrition_pred = self.nutrition_rf_model.predict(X_test_rf)
        
        # Convert back to original template IDs for comparison
        xgb_workout_templates = self.workout_label_encoder.inverse_transform(xgb_workout_pred)
        xgb_nutrition_templates = self.nutrition_label_encoder.inverse_transform(xgb_nutrition_pred)
        
        rf_workout_templates = self.workout_rf_label_encoder.inverse_transform(rf_workout_pred)
        rf_nutrition_templates = self.nutrition_rf_label_encoder.inverse_transform(rf_nutrition_pred)
        
        # Compare first 20 predictions
        print("\n📊 PREDICTION COMPARISON (First 20 samples):")
        print("=" * 80)
        print("Workout Predictions (XGB vs RF vs Truth):")
        print("Sample | XGBoost | Random Forest | True Label | XGB≠RF | Both Wrong")
        print("-" * 70)
        
        workout_diff_count = 0
        workout_both_wrong = 0
        
        for i in range(min(20, len(X_test))):
            xgb_pred = xgb_workout_templates[i]
            rf_pred = rf_workout_templates[i]
            true_label = y_w_test.iloc[i] if hasattr(y_w_test, 'iloc') else y_w_test[i]
            
            is_different = xgb_pred != rf_pred
            both_wrong = (xgb_pred != true_label) and (rf_pred != true_label)
            
            if is_different:
                workout_diff_count += 1
            if both_wrong:
                workout_both_wrong += 1
            
            print(f"{i+1:6d} | {xgb_pred:7.0f} | {rf_pred:13.0f} | {true_label:10.0f} | {str(is_different):6s} | {str(both_wrong):10s}")
        
        print(f"\nWorkout Prediction Differences: {workout_diff_count}/20 samples")
        print(f"Both models wrong: {workout_both_wrong}/20 samples")
        
        print("\nNutrition Predictions (XGBoost vs RF vs Truth):")
        print("Sample | XGBoost | Random Forest | True Label | XGB≠RF | Both Wrong")
        print("-" * 70)
        
        nutrition_diff_count = 0
        nutrition_both_wrong = 0
        
        for i in range(min(20, len(X_test))):
            xgb_pred = xgb_nutrition_templates[i]
            rf_pred = rf_nutrition_templates[i]
            true_label = y_n_test.iloc[i] if hasattr(y_n_test, 'iloc') else y_n_test[i]
            
            is_different = xgb_pred != rf_pred
            both_wrong = (xgb_pred != true_label) and (rf_pred != true_label)
            
            if is_different:
                nutrition_diff_count += 1
            if both_wrong:
                nutrition_both_wrong += 1
            
            print(f"{i+1:6d} | {xgb_pred:7.0f} | {rf_pred:13.0f} | {true_label:10.0f} | {str(is_different):6s} | {str(both_wrong):10s}")
        
        print(f"\nNutrition Prediction Differences: {nutrition_diff_count}/20 samples")
        print(f"Both models wrong: {nutrition_both_wrong}/20 samples")
        
        # Calculate overall prediction agreement
        total_workout_diff = np.sum(xgb_workout_templates != rf_workout_templates)
        total_nutrition_diff = np.sum(xgb_nutrition_templates != rf_nutrition_templates)
        
        print(f"\n🚨 CRITICAL ANALYSIS:")
        print(f"Workout model prediction differences: {total_workout_diff}/{len(X_test)} ({100*total_workout_diff/len(X_test):.1f}%)")
        print(f"Nutrition model prediction differences: {total_nutrition_diff}/{len(X_test)} ({100*total_nutrition_diff/len(X_test):.1f}%)")
        
        if total_workout_diff == 0:
            print("⚠️  WARNING: Workout models making IDENTICAL predictions - possible bug!")
        if total_nutrition_diff == 0:
            print("⚠️  WARNING: Nutrition models making IDENTICAL predictions - possible bug!")
        
        # Additional verification - check if models are actually different
        print(f"\n🔍 MODEL VERIFICATION:")
        print(f"XGBoost Workout Model Type: {type(self.workout_model).__name__}")
        print(f"Random Forest Workout Model Type: {type(self.workout_rf_model).__name__}")
        print(f"XGBoost Nutrition Model Type: {type(self.nutrition_model).__name__}")
        print(f"Random Forest Nutrition Model Type: {type(self.nutrition_rf_model).__name__}")
        
        return {
            'workout_differences': total_workout_diff,
            'nutrition_differences': total_nutrition_diff,
            'workout_agreement_rate': 1 - (total_workout_diff / len(X_test)),
            'nutrition_agreement_rate': 1 - (total_nutrition_diff / len(X_test)),
            'total_test_samples': len(X_test),
            'sample_predictions': {
                'xgb_workout': xgb_workout_templates[:20].tolist(),
                'rf_workout': rf_workout_templates[:20].tolist(),
                'xgb_nutrition': xgb_nutrition_templates[:20].tolist(),
                'rf_nutrition': rf_nutrition_templates[:20].tolist()
            }
        }
    
    def compare_model_performance(self):
        """Generate comprehensive performance comparison between XGBoost and Random Forest"""
        if not hasattr(self, 'training_info') or not hasattr(self, 'rf_training_info'):
            print("❌ Both model types must be trained before comparison")
            return None
        
        xgb_info = self.training_info
        rf_info = self.rf_training_info
        
        print("\n📊 COMPREHENSIVE MODEL COMPARISON: XGBoost vs Random Forest")
        print("=" * 80)
        
        print("\n🏋️ WORKOUT MODEL COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<25} {'XGBoost':<12} {'Random Forest':<15} {'Difference':<12}")
        print("-" * 80)
        
        workout_acc_diff = xgb_info['workout_accuracy'] - rf_info['rf_workout_accuracy']
        workout_f1_diff = xgb_info['workout_f1'] - rf_info['rf_workout_f1']
        
        print(f"{'Accuracy':<25} {xgb_info['workout_accuracy']:<12.4f} {rf_info['rf_workout_accuracy']:<15.4f} {workout_acc_diff:+.4f}")
        print(f"{'F1 Score (Weighted)':<25} {xgb_info['workout_f1']:<12.4f} {rf_info['rf_workout_f1']:<15.4f} {workout_f1_diff:+.4f}")
        
        print("\n🥗 NUTRITION MODEL COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<25} {'XGBoost':<12} {'Random Forest':<15} {'Difference':<12}")
        print("-" * 80)
        
        nutrition_acc_diff = xgb_info['nutrition_accuracy'] - rf_info['rf_nutrition_accuracy']
        nutrition_f1_diff = xgb_info['nutrition_f1'] - rf_info['rf_nutrition_f1']
        
        print(f"{'Accuracy':<25} {xgb_info['nutrition_accuracy']:<12.4f} {rf_info['rf_nutrition_accuracy']:<15.4f} {nutrition_acc_diff:+.4f}")
        print(f"{'F1 Score (Weighted)':<25} {xgb_info['nutrition_f1']:<12.4f} {rf_info['rf_nutrition_f1']:<15.4f} {nutrition_f1_diff:+.4f}")
        
        # Check for suspicious identical performance
        if abs(workout_acc_diff) < 0.0001 and abs(workout_f1_diff) < 0.0001:
            print("\n⚠️  WARNING: Workout models show IDENTICAL performance to 4+ decimal places!")
            print("   This is highly suspicious and suggests a potential bug.")
            
        if abs(nutrition_acc_diff) < 0.0001 and abs(nutrition_f1_diff) < 0.0001:
            print("\n⚠️  WARNING: Nutrition models show IDENTICAL performance to 4+ decimal places!")  
            print("   This is highly suspicious and suggests a potential bug.")
        
        return {
            'workout_comparison': {
                'xgboost_accuracy': xgb_info['workout_accuracy'],
                'rf_accuracy': rf_info['rf_workout_accuracy'],
                'accuracy_difference': workout_acc_diff,
                'xgboost_f1': xgb_info['workout_f1'],
                'rf_f1': rf_info['rf_workout_f1'],
                'f1_difference': workout_f1_diff
            },
            'nutrition_comparison': {
                'xgboost_accuracy': xgb_info['nutrition_accuracy'],
                'rf_accuracy': rf_info['rf_nutrition_accuracy'],
                'accuracy_difference': nutrition_acc_diff,
                'xgboost_f1': xgb_info['nutrition_f1'],
                'rf_f1': rf_info['rf_nutrition_f1'],
                'f1_difference': nutrition_f1_diff
            }
        }
    
    def debug_template_assignment_logic(self):
        """Debug the template assignment logic to ensure it's working correctly"""
        print("\n🔍 DEBUGGING TEMPLATE ASSIGNMENT LOGIC:")
        print("=" * 80)
        
        # Test all valid combinations
        test_cases = [
            ('Fat Loss', 'High Activity', 'Overweight'),
            ('Fat Loss', 'Moderate Activity', 'Obese'),
            ('Muscle Gain', 'High Activity', 'Normal'),
            ('Muscle Gain', 'Low Activity', 'Underweight'),
            ('Maintenance', 'Moderate Activity', 'Normal'),
        ]
        
        for goal, activity, bmi in test_cases:
            workout_id, nutrition_id = self.get_template_assignments(goal, activity, bmi)
            print(f"{goal:<12} + {activity:<18} + {bmi:<12} → Workout: {workout_id}, Nutrition: {nutrition_id}")
        
        print("✅ Template assignment logic check complete")
    
    def debug_template_assignments(self, df_training):
        """Debug template assignments in the training data"""
        print("\n🔍 DEBUGGING TEMPLATE ASSIGNMENTS IN TRAINING DATA:")
        print("=" * 80)
        
        # Check workout template distribution
        workout_dist = df_training['workout_template_id'].value_counts().sort_index()
        print(f"Workout Template Distribution:")
        for template_id, count in workout_dist.items():
            print(f"  Template {template_id}: {count} samples")
        
        # Check nutrition template distribution
        nutrition_dist = df_training['nutrition_template_id'].value_counts().sort_index()
        print(f"\nNutrition Template Distribution:")
        for template_id, count in nutrition_dist.items():
            print(f"  Template {template_id}: {count} samples")
        
        # Check for any invalid assignments
        valid_workout_ids = set(range(1, 10))
        valid_nutrition_ids = set(range(1, 9))
        
        invalid_workout = set(df_training['workout_template_id'].unique()) - valid_workout_ids
        invalid_nutrition = set(df_training['nutrition_template_id'].unique()) - valid_nutrition_ids
        
        if invalid_workout:
            print(f"⚠️  Invalid workout template IDs found: {invalid_workout}")
        if invalid_nutrition:
            print(f"⚠️  Invalid nutrition template IDs found: {invalid_nutrition}")
        
        if not invalid_workout and not invalid_nutrition:
            print("✅ All template assignments are valid")
    
    def debug_training_splits(self, df_training):
        """Debug the training data splits"""
        print("\n🔍 DEBUGGING TRAINING DATA SPLITS:")
        print("=" * 80)
        
        if 'split' in df_training.columns:
            split_dist = df_training['split'].value_counts()
            print(f"Split Distribution:")
            for split_name, count in split_dist.items():
                print(f"  {split_name}: {count} samples ({100*count/len(df_training):.1f}%)")
        else:
            print("No 'split' column found - will use random splits")
        
        print("✅ Training splits check complete")
    
    def predict_with_confidence(self, user_profile):
        """
        Make predictions with enhanced confidence scoring - XGBOOST ONLY
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # SAFETY CHECK: Ensure we're using XGBoost models only
        if not hasattr(self, 'workout_model') or not hasattr(self, 'nutrition_model'):
            raise ValueError("XGBoost models not found - check model loading")
        
        print(f"🚀 Using XGBoost models for prediction (production mode)")
        
        # Extract user data (handle both height_cm/weight_kg and height/weight)
        age = user_profile.get('age')
        gender = user_profile.get('gender')
        height_cm = user_profile.get('height_cm') or user_profile.get('height')
        weight_kg = user_profile.get('weight_kg') or user_profile.get('weight')
        fitness_goal = user_profile.get('fitness_goal')
        activity_level = user_profile.get('activity_level')
        
        # Validate required fields
        if not all([age, gender, height_cm, weight_kg, fitness_goal, activity_level]):
            missing_fields = []
            if not age: missing_fields.append('age')
            if not gender: missing_fields.append('gender')
            if not height_cm: missing_fields.append('height_cm/height')
            if not weight_kg: missing_fields.append('weight_kg/weight')
            if not fitness_goal: missing_fields.append('fitness_goal')
            if not activity_level: missing_fields.append('activity_level')
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Calculate derived metrics
        bmi = weight_kg / ((height_cm / 100) ** 2)
        bmi_category = categorize_bmi(bmi)
        bmr = calculate_bmr(weight_kg, height_cm, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        
        # Create user data dictionary for feature engineering
        user_data = {
            'age': age,
            'gender': gender,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'bmr': bmr,
            'tdee': tdee,
            'activity_level': activity_level,
            'fitness_goal': fitness_goal,
            'activity_multiplier': self._get_activity_multiplier(activity_level),
            'Mod_act': user_profile.get('Mod_act', 0),
            'Vig_act': user_profile.get('Vig_act', 0),
            'has_activity_data': True
        }
        
        # Convert to DataFrame for feature engineering
        user_df = pd.DataFrame([user_data])
        
        # Apply feature engineering
        user_df_enhanced = self.create_enhanced_features(user_df)
        
        # Prepare features for prediction
        X_user = user_df_enhanced[self.feature_columns].fillna(0)
        
        # Scale features
        X_user_scaled = self.scaler.transform(X_user)
        
        # Make predictions
        workout_pred_encoded = self.workout_model.predict(X_user_scaled)[0]
        nutrition_pred_encoded = self.nutrition_model.predict(X_user_scaled)[0]
        
        # Get prediction probabilities
        workout_pred_proba = self.workout_model.predict_proba(X_user_scaled)[0]
        nutrition_pred_proba = self.nutrition_model.predict_proba(X_user_scaled)[0]
        
        # Convert encoded predictions back to template IDs
        workout_template_id = self.workout_label_encoder.inverse_transform([workout_pred_encoded])[0]
        nutrition_template_id = self.nutrition_label_encoder.inverse_transform([nutrition_pred_encoded])[0]
        
        # Get prediction confidence scores
        workout_confidence = np.max(workout_pred_proba)
        nutrition_confidence = np.max(nutrition_pred_proba)
        
        # Calculate enhanced confidence scoring
        confidence_result = self.calculate_prediction_confidence(
            user_profile, workout_template_id, nutrition_template_id
        )
        
        # Get template details
        workout_template = self._get_template_details(workout_template_id, 'workout')
        nutrition_template = self._get_template_details(nutrition_template_id, 'nutrition')
        
        return {
            'predictions': {
                'workout_template_id': int(workout_template_id),
                'nutrition_template_id': int(nutrition_template_id),
                'workout_template': workout_template,
                'nutrition_template': nutrition_template
            },
            'model_confidence': {
                'workout_confidence': float(workout_confidence),
                'nutrition_confidence': float(nutrition_confidence)
            },
            'enhanced_confidence': confidence_result,
            'user_profile': {
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1)
            }
        }
    
    def _get_activity_multiplier(self, activity_level):
        """Get activity multiplier for TDEE calculation"""
        multipliers = {
            'Low Activity': 1.29,
            'Moderate Activity': 1.55,
            'High Activity': 1.81
        }
        return multipliers.get(activity_level, 1.55)
    
    def _get_template_details(self, template_id, template_type):
        """Get detailed information about a template"""
        print(f"🔍 Looking for {template_type} template with ID: {template_id}")
        
        if template_type == 'workout':
            templates = self.workout_templates
        else:
            templates = self.nutrition_templates
        
        print(f"📊 Available {template_type} template IDs: {templates['template_id'].tolist()}")
        
        # Find the template
        template_row = templates[templates['template_id'] == template_id]
        print(f"🎯 Found {len(template_row)} matching templates")
        
        if not template_row.empty:
            template = template_row.iloc[0]
            print(f"📋 Template data: {template.to_dict()}")
            
            if template_type == 'workout':
                workout_template = {
                    'template_id': int(template_id),
                    'goal': str(template['goal']) if 'goal' in template and pd.notna(template['goal']) else 'Unknown',
                    'activity_level': str(template['activity_level']) if 'activity_level' in template and pd.notna(template['activity_level']) else 'Unknown',
                    'workout_type': str(template['workout_type']) if 'workout_type' in template and pd.notna(template['workout_type']) else 'Unknown',
                    'days_per_week': int(template['days_per_week']) if 'days_per_week' in template and pd.notna(template['days_per_week']) else 'Unknown'
                }
                if 'description' in template and pd.notna(template['description']) and template['description']:
                    workout_template['description'] = str(template['description'])
                print(f"✅ Returning workout template: {workout_template}")
                return workout_template
            else:  # nutrition
                nutrition_template = {
                    'template_id': int(template_id),
                    'goal': str(template['goal']) if 'goal' in template and pd.notna(template['goal']) else 'Unknown',
                    'bmi_category': str(template['bmi_category']) if 'bmi_category' in template and pd.notna(template['bmi_category']) else 'Unknown',
                    'caloric_intake_multiplier': float(template['caloric_intake_multiplier']) if 'caloric_intake_multiplier' in template and pd.notna(template['caloric_intake_multiplier']) else 'Unknown',
                    'protein_per_kg': float(template['protein_per_kg']) if 'protein_per_kg' in template and pd.notna(template['protein_per_kg']) else 'Unknown',
                    'carbs_per_kg': float(template['carbs_per_kg']) if 'carbs_per_kg' in template and pd.notna(template['carbs_per_kg']) else 'Unknown',
                    'fat_per_kg': float(template['fat_per_kg']) if 'fat_per_kg' in template and pd.notna(template['fat_per_kg']) else 'Unknown'
                }
                if 'description' in template and pd.notna(template['description']) and template['description']:
                    nutrition_template['description'] = str(template['description'])
                print(f"✅ Returning nutrition template: {nutrition_template}")
                return nutrition_template
        else:
            error_result = {
                'template_id': int(template_id),
                'error': f'Template {template_id} not found'
            }
            print(f"❌ Template not found: {error_result}")
            return error_result
    
    def test_confidence_improvements(self):
        """
        Test the enhanced confidence scoring system with various user profiles
        """
        print("Testing confidence scoring with different user profiles...")
        
        # Test cases representing different confidence scenarios
        test_profiles = [
            {
                'name': 'High-Activity User (High Confidence)',
                'profile': {
                    'fitness_goal': 'Fat Loss',
                    'activity_level': 'High Activity',
                    'bmi_category': 'Overweight',
                    'age': 25,
                    'gender': 'Male',
                    'weight_kg': 80,
                    'height_cm': 175
                }
            },
            {
                'name': 'Moderate-Activity User (Medium Confidence)',
                'profile': {
                    'fitness_goal': 'Muscle Gain',
                    'activity_level': 'Moderate Activity',
                    'bmi_category': 'Normal',
                    'age': 30,
                    'gender': 'Female',
                    'weight_kg': 65,
                    'height_cm': 165
                }
            },
            {
                'name': 'Low-Activity User (Low Confidence)',
                'profile': {
                    'fitness_goal': 'Maintenance',
                    'activity_level': 'Low Activity',
                    'bmi_category': 'Underweight',
                    'age': 35,
                    'gender': 'Male',
                    'weight_kg': 55,
                    'height_cm': 170
                }
            }
        ]
        
        for test_case in test_profiles:
            name = test_case['name']
            profile = test_case['profile']
            
            print(f"\n--- {name} ---")
            print(f"Profile: {profile['fitness_goal']} + {profile['activity_level']} + {profile['bmi_category']}")
            
            # Get template assignments
            workout_id, nutrition_id = self.get_template_assignments(
                profile['fitness_goal'], 
                profile['activity_level'], 
                profile['bmi_category']
            )
            
            if workout_id and nutrition_id:
                # Calculate confidence
                confidence_result = self.calculate_prediction_confidence(
                    profile, workout_id, nutrition_id
                )
                
                print(f"Assigned Templates: Workout {workout_id}, Nutrition {nutrition_id}")
                print(f"Confidence Score: {confidence_result['confidence_score']} ({confidence_result['confidence_level']})")
                print(f"Explanation: {confidence_result['explanation']}")
                
                if confidence_result['limitations']:
                    print(f"Limitations: {'; '.join(confidence_result['limitations'])}")
            else:
                print("❌ No valid template assignment found")
        
        print("\n✅ Confidence scoring system test completed!")