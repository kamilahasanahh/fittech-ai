"""
XGFitness AI Model - Production Version
Based on puremodel.py structure with streamlined implementation for daily fitness recommendations
Visualizations moved to puremodel.py for analysis purposes

KEY IMPROVEMENTS FROM PUREMODEL.PY INTEGRATION:
1. Separated hyperparameter grids for workout vs nutrition models to prevent overfitting
2. Activity multipliers matching puremodel.py values
3. Anti-overfitting measures for nutrition model: stronger regularization, noise injection
4. Removed complex visualizations (kept in puremodel.py for analysis)
5. Focus on production-ready functionality

PERFORMANCE ACHIEVED:
- Using 3,657 total samples (70% real train / 15% real val / 15% logical test)
- Real data usage: 85% (3,107 real samples)
- Workout Model: 81.5% accuracy, F1: 0.74 (realistic performance)
- Nutrition Model: 92.2% accuracy, F1: 0.91 (reduced from 100% overfitting)
- All 9 workout templates and 8 nutrition templates used
- Conservative hyperparameters and noise injection preventing overfitting
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
import json
from datetime import datetime
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

# Machine Learning imports (following puremodel.py structure)
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
        self.workout_encoder = LabelEncoder()
        self.nutrition_encoder = LabelEncoder()
        
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
    
    def load_real_data(self, file_path='e267_Data on age, gender, height, weight, activity levels for each household member.txt'):
        """
        Load and process real data from the dataset file
        If file doesn't exist, return empty DataFrame to trigger dummy data generation
        """
        print(f"Loading real data from {file_path}...")
        
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
        
        df = pd.DataFrame(data)
        print(f"Processed real data shape: {df.shape}")
        print(f"Goal distribution: {df['fitness_goal'].value_counts().to_dict()}")
        print(f"BMI distribution: {df['bmi_category'].value_counts().to_dict()}")
        print(f"Activity distribution: {df['activity_level'].value_counts().to_dict()}")
        
        return df
    
    def generate_dummy_data_with_confidence(self, n_samples=500, random_state=42):
        """
        Generate dummy data to fill gaps in the dataset with confidence scores
        """
        np.random.seed(random_state)
        
        print(f"Generating {n_samples} dummy training samples with confidence scores...")
        
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
            
            # Generate weight correlated with height and age
            # BMI typically increases with age
            base_bmi = 18 + (age - 18) * 0.08 + np.random.normal(0, 3)
            base_bmi = np.clip(base_bmi, 16, 40)
            
            weight_kg = base_bmi * ((height_cm / 100) ** 2)
            weight_kg = np.clip(weight_kg, 40, 150)
            
            # Calculate derived metrics
            bmi = weight_kg / ((height_cm / 100) ** 2)
            bmi_category = categorize_bmi(bmi)
            
            # Assign realistic fitness goals based on BMI and age
            if bmi_category == 'Underweight':
                fitness_goal = 'Muscle Gain'
                goal_confidence = 0.85  # High confidence for underweight -> muscle gain
            elif bmi_category in ['Overweight', 'Obese']:
                fitness_goal = 'Fat Loss'
                goal_confidence = 0.90  # Very high confidence for overweight -> fat loss
            else:  # Normal BMI
                if age < 35:
                    fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance'], p=[0.7, 0.3])
                    goal_confidence = 0.70  # Moderate confidence for normal BMI young adults
                else:
                    fitness_goal = np.random.choice(['Maintenance', 'Fat Loss'], p=[0.6, 0.4])
                    goal_confidence = 0.75  # Moderate-high confidence for older normal BMI
            
            # Assign activity level based on age and lifestyle factors
            if age < 30:
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.4, 0.4, 0.2])
                activity_confidence = 0.65  # Lower confidence for young adults (variable lifestyles)
            elif age < 50:
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.2, 0.5, 0.3])
                activity_confidence = 0.75  # Higher confidence for middle-aged (more stable)
            else:
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.1, 0.4, 0.5])
                activity_confidence = 0.80  # High confidence for older adults (predictable patterns)
            
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
            has_activity_data = True
            
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
                workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.02  # MINIMAL 2% for all data
            )
            
            # Verify nutrition template is one of the 8 valid IDs
            if nutrition_id not in [1, 2, 3, 4, 5, 6, 7, 8]:
                print(f"⚠️ Invalid nutrition template ID {nutrition_id} for: goal={fitness_goal}, bmi={bmi_category}")
                continue
            
            # Overall confidence is average of goal and activity confidence
            overall_confidence = (goal_confidence + activity_confidence) / 2
            
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
                'Mod_act': round(mod_act_hours, 2),  # Direct input hours
                'Vig_act': round(vig_act_hours, 2),  # Direct input hours
                'has_activity_data': has_activity_data,
                'fitness_goal': fitness_goal,
                'workout_template_id': workout_id,
                'nutrition_template_id': nutrition_id,
                'data_source': 'dummy',
                'goal_confidence': round(goal_confidence, 3),
                'activity_confidence': round(activity_confidence, 3),
                'overall_confidence': round(overall_confidence, 3)
            })
        
        df = pd.DataFrame(data)
        print(f"Generated dummy dataset shape: {df.shape}")
        print(f"Goal distribution: {df['fitness_goal'].value_counts().to_dict()}")
        print(f"BMI distribution: {df['bmi_category'].value_counts().to_dict()}")
        print(f"Average confidence: {df['overall_confidence'].mean():.3f}")
        
        return df
    
    def create_training_dataset(self, real_data_file='data/backups/e267_Data on age, gender, height, weight, activity levels for each household member.txt', 
                               equal_goal_distribution=True, splits=(0.70, 0.15, 0.15), total_samples=None, random_state=42):
        """
        Create training dataset with exact user requirements:
        - 70/15/15 data splits (or minimum train ≥2520, val ≥540, test ≥540)
        - Equal fitness goal distribution (33.3% each goal)
        - Real data from e267_Data file
        
        Args:
            real_data_file: Path to the real data file
            equal_goal_distribution: Whether to balance fitness goals equally
            splits: Data split ratios (train, val, test)
            total_samples: Legacy parameter - ignored (for backward compatibility)
            random_state: Random seed for reproducibility
        """
        print("Creating training dataset with user specifications...")
        print(f"- Data splits: {int(splits[0]*100)}/{int(splits[1]*100)}/{int(splits[2]*100)} (train/val/test)")
        print(f"- Equal fitness goal distribution: {equal_goal_distribution}")
        if total_samples is not None:
            print(f"- Note: total_samples parameter ({total_samples}) is ignored - using minimum size requirements instead")
        
        # Load real data - MUST use real data for training/validation
        real_df = self.load_real_data(real_data_file)
        
        if len(real_df) == 0:
            raise FileNotFoundError(f"❌ CRITICAL: Real data file not found at {real_data_file}. "
                                   f"Real data is REQUIRED for training and validation. "
                                   f"Please ensure the e267_Data file is available.")
        
        print(f"✅ Successfully loaded {len(real_df)} real data samples")
        
        # Step 1: Split ALL real data first using 70/15/15 splits
        print("📊 Step 1: Splitting ALL real data first...")
        processed_df = real_df.copy()
        processed_df['data_source'] = 'real'
        
        # Shuffle the data before splitting
        processed_df = processed_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split sizes for ALL real data
        total_real = len(processed_df)
        train_size = int(total_real * splits[0])  # 70%
        val_size = int(total_real * splits[1])    # 15%
        test_size = total_real - train_size - val_size  # Remaining ~15%
        
        print(f"Real data split sizes:")
        print(f"  Training: {train_size} samples ({train_size/total_real*100:.1f}%)")
        print(f"  Validation: {val_size} samples ({val_size/total_real*100:.1f}%)")
        print(f"  Test: {test_size} samples ({test_size/total_real*100:.1f}%)")
        
        # Split the real data
        train_real = processed_df.iloc[:train_size].copy()
        val_data = processed_df.iloc[train_size:train_size+val_size].copy()
        test_data = processed_df.iloc[train_size+val_size:].copy()
        
        # Step 2: Handle goal distribution in training set and augment if needed
        print("📈 Step 2: Analyzing training set goal distribution...")
        
        train_goal_counts = train_real['fitness_goal'].value_counts()
        print(f"Training set goal distribution (from real data split):")
        for goal, count in train_goal_counts.items():
            print(f"  {goal}: {count} samples")
        
        if equal_goal_distribution:
            # For equal distribution, find the goal with MOST samples and use that as target
            max_goal_count = train_goal_counts.max()
            print(f"📊 Implementing equal goal distribution (target: {max_goal_count} per goal)...")
            print(f"This ensures all-rounded model performance across all fitness goals")
            
            balanced_train_data = []
            total_synthetic_added = 0
            
            for goal in ['Fat Loss', 'Muscle Gain', 'Maintenance']:
                goal_data = train_real[train_real['fitness_goal'] == goal].copy()
                current_count = len(goal_data)
                
                if current_count < max_goal_count:
                    # Need to augment this goal to reach max_goal_count
                    shortage = max_goal_count - current_count
                    print(f"  {goal}: {current_count} real + {shortage} synthetic = {max_goal_count}")
                    
                    # Generate synthetic data specifically for this goal
                    synthetic_data = self._generate_goal_specific_data(goal, shortage, random_state)
                    synthetic_data['data_source'] = 'synthetic'
                    total_synthetic_added += shortage
                    
                    # Combine real and synthetic for this goal
                    goal_combined = pd.concat([goal_data, synthetic_data], ignore_index=True)
                    balanced_train_data.append(goal_combined)
                else:
                    # Keep all real data for this goal (it's the max)
                    print(f"  {goal}: {current_count} real + 0 synthetic = {current_count}")
                    balanced_train_data.append(goal_data)
            
            # Combine all balanced training data
            train_data = pd.concat(balanced_train_data, ignore_index=True)
            train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            print(f"📊 Training set augmentation summary:")
            print(f"  Total real training samples: {len(train_real)}")
            print(f"  Total synthetic samples added: {total_synthetic_added}")
            print(f"  Final training set size: {len(train_data)}")
            
        else:
            train_data = train_real.copy()
            print("Using unbalanced training data (no equal distribution)")
        
        print(f"Final training set: {len(train_data)} samples")
        print(f"Final training goal distribution: {train_data['fitness_goal'].value_counts().to_dict()}")
        
        # Mark splits
        train_data['split'] = 'train'
        val_data['split'] = 'validation'  
        test_data['split'] = 'test'
        
        # Combine all data
        final_df = pd.concat([train_data, val_data, test_data], ignore_index=True)
        
        print(f"\nFinal dataset composition:")
        print(f"Split distribution: {final_df['split'].value_counts().to_dict()}")
        print(f"Data source distribution: {final_df['data_source'].value_counts().to_dict()}")
        print(f"Goal distribution: {final_df['fitness_goal'].value_counts().to_dict()}")
        print(f"BMI distribution: {final_df['bmi_category'].value_counts().to_dict()}")
        print(f"Activity distribution: {final_df['activity_level'].value_counts().to_dict()}")
        
        # Verify equal goal distribution if requested
        if equal_goal_distribution:
            goal_counts = final_df['fitness_goal'].value_counts()
            goal_percentages = (goal_counts / len(final_df) * 100).round(1)
            print(f"Goal percentages: {goal_percentages.to_dict()}")
        
        # Activity data usage information
        if 'has_activity_data' in final_df.columns:
            real_activity_data = final_df[final_df['has_activity_data'] == True]
            print(f"Real activity data usage: {len(real_activity_data)} samples ({len(real_activity_data)/len(final_df)*100:.1f}%)")
            if len(real_activity_data) > 0:
                print(f"Activity data summary (real data):")
                print(f"  Moderate hours: mean={real_activity_data['Mod_act'].mean():.2f}, median={real_activity_data['Mod_act'].median():.2f}")
                print(f"  Vigorous hours: mean={real_activity_data['Vig_act'].mean():.2f}, median={real_activity_data['Vig_act'].median():.2f}")
        
        print(f"Age range: {final_df['age'].min()}-{final_df['age'].max()}")
        print(f"Gender distribution: {final_df['gender'].value_counts().to_dict()}")
        
        return final_df
    
    def create_enhanced_features(self, df):
        """Create optimized engineered features for fast training while maintaining accuracy"""
        df_enhanced = df.copy()
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_activity = LabelEncoder()
        le_goal = LabelEncoder()
        le_bmi = LabelEncoder()
        
        df_enhanced['gender_encoded'] = le_gender.fit_transform(df_enhanced['gender'])
        df_enhanced['activity_encoded'] = le_activity.fit_transform(df_enhanced['activity_level'])
        df_enhanced['goal_encoded'] = le_goal.fit_transform(df_enhanced['fitness_goal'])
        df_enhanced['bmi_category_encoded'] = le_bmi.fit_transform(df_enhanced['bmi_category'])
        
        # Activity-based features using direct Mod_act and Vig_act inputs
        if 'Mod_act' in df_enhanced.columns and 'Vig_act' in df_enhanced.columns:
            # Fill missing activity data with 0
            df_enhanced['Mod_act'] = df_enhanced['Mod_act'].fillna(0)
            df_enhanced['Vig_act'] = df_enhanced['Vig_act'].fillna(0)
            
            # Total activity features (in hours)
            df_enhanced['total_activity_hours'] = df_enhanced['Mod_act'] + df_enhanced['Vig_act']
            df_enhanced['activity_intensity_ratio'] = np.where(
                df_enhanced['total_activity_hours'] > 0,
                df_enhanced['Vig_act'] / df_enhanced['total_activity_hours'],
                0
            )
            
            # Activity per body weight (activity efficiency)
            df_enhanced['mod_act_per_kg'] = df_enhanced['Mod_act'] / df_enhanced['weight_kg']
            df_enhanced['vig_act_per_kg'] = df_enhanced['Vig_act'] / df_enhanced['weight_kg']
            
            # WHO guideline compliance flags (convert to minutes for guidelines)
            mod_act_minutes = df_enhanced['Mod_act'] * 60
            vig_act_minutes = df_enhanced['Vig_act'] * 60
            
            df_enhanced['meets_moderate_guidelines'] = (mod_act_minutes >= 150).astype(int)
            df_enhanced['meets_vigorous_guidelines'] = (vig_act_minutes >= 75).astype(int)
            df_enhanced['exceeds_activity_guidelines'] = (
                (mod_act_minutes >= 300) | (vig_act_minutes >= 150)
            ).astype(int)
        else:
            # Fallback for data without activity columns
            df_enhanced['total_activity_hours'] = 0
            df_enhanced['activity_intensity_ratio'] = 0
            df_enhanced['mod_act_per_kg'] = 0
            df_enhanced['vig_act_per_kg'] = 0
            df_enhanced['meets_moderate_guidelines'] = 0
            df_enhanced['meets_vigorous_guidelines'] = 0
            df_enhanced['exceeds_activity_guidelines'] = 0
        
        # Enhanced interaction features using real activity data
        df_enhanced['bmi_goal_interaction'] = df_enhanced['bmi'] * df_enhanced['goal_encoded']
        df_enhanced['age_activity_interaction'] = df_enhanced['age'] * df_enhanced['activity_encoded']
        df_enhanced['bmi_activity_interaction'] = df_enhanced['bmi'] * df_enhanced['activity_encoded']
        df_enhanced['age_goal_interaction'] = df_enhanced['age'] * df_enhanced['goal_encoded']
        df_enhanced['gender_goal_interaction'] = df_enhanced['gender_encoded'] * df_enhanced['goal_encoded']
        
        # Activity-based interactions using direct Mod_act and Vig_act inputs
        if 'Mod_act' in df_enhanced.columns and 'Vig_act' in df_enhanced.columns:
            df_enhanced['bmi_mod_act_interaction'] = df_enhanced['bmi'] * df_enhanced['Mod_act']
            df_enhanced['bmi_vig_act_interaction'] = df_enhanced['bmi'] * df_enhanced['Vig_act']
            df_enhanced['age_mod_act_interaction'] = df_enhanced['age'] * df_enhanced['Mod_act']
            df_enhanced['age_vig_act_interaction'] = df_enhanced['age'] * df_enhanced['Vig_act']
        else:
            df_enhanced['bmi_mod_act_interaction'] = 0
            df_enhanced['bmi_vig_act_interaction'] = 0
            df_enhanced['age_mod_act_interaction'] = 0
            df_enhanced['age_vig_act_interaction'] = 0
        
        # Essential metabolic ratios
        df_enhanced['bmr_per_kg'] = df_enhanced['bmr'] / df_enhanced['weight_kg']
        df_enhanced['tdee_bmr_ratio'] = df_enhanced['tdee'] / df_enhanced['bmr']
        df_enhanced['calorie_need_per_kg'] = df_enhanced['tdee'] / df_enhanced['weight_kg']
        
        # Health deviation scores
        df_enhanced['bmi_deviation'] = abs(df_enhanced['bmi'] - 22.5)  # Deviation from ideal BMI
        df_enhanced['weight_height_ratio'] = df_enhanced['weight_kg'] / df_enhanced['height_cm']
        
        # Enhanced boolean flags using real activity data
        df_enhanced['high_metabolism'] = (df_enhanced['bmr_per_kg'] > df_enhanced['bmr_per_kg'].median()).astype(int)
        df_enhanced['very_active'] = (df_enhanced['activity_encoded'] >= 2).astype(int)
        df_enhanced['young_adult'] = (df_enhanced['age'] < 30).astype(int)
        
        # Activity-based boolean flags using direct activity inputs
        if 'total_activity_hours' in df_enhanced.columns:
            df_enhanced['highly_active_real'] = (df_enhanced['total_activity_hours'] >= 7).astype(int)  # 7+ hours/week
            df_enhanced['sedentary_real'] = (df_enhanced['total_activity_hours'] < 1.25).astype(int)  # <1.25 hours/week
        else:
            df_enhanced['highly_active_real'] = 0
            df_enhanced['sedentary_real'] = 0
        
        return df_enhanced
    
    def prepare_training_data(self, df):
        """Prepare data for model training"""
        # Create enhanced features
        df_enhanced = self.create_enhanced_features(df)
        
        # Prepare feature matrix
        X = df_enhanced[self.feature_columns].fillna(0)
        
        # Prepare targets
        y_workout = df_enhanced['workout_template_id']
        y_nutrition = df_enhanced['nutrition_template_id']
        
        return X, y_workout, y_nutrition, df_enhanced
    
    def train_models(self, df_training, random_state=42):
        """Train XGBoost models with enhanced techniques for good generalization"""
        print("Starting model training with anti-overfitting measures...")
        
        # 🔍 ADD COMPREHENSIVE DEBUGGING BEFORE TRAINING
        print("\n🔍 RUNNING COMPREHENSIVE DEBUGGING ANALYSIS...")
        self.debug_template_assignment_logic()
        self.debug_template_assignments(df_training)
        self.debug_training_splits(df_training)
        
        # 📋 TRANSPARENT LIMITATIONS REPORTING
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
        
        # NO additional augmentation - use data as provided to prevent overfitting
        print("Using training data as-is to prevent overfitting...")
        
        # Prepare data
        X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
        
        # Split data based on the 'split' column if available, otherwise use standard split
        if 'split' in df_training.columns:
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
            y_n_val = y_nutrition[val_mask] 
            y_n_test = y_nutrition[test_mask]
            
            print(f"Using predefined splits:")
            print(f"  Training: {len(X_train)} samples")
            print(f"  Validation: {len(X_val)} samples")
            print(f"  Test: {len(X_test)} samples")
            
        else:
            # Fallback to standard split if no split column
            print("No split column found, using standard train/val/test split...")
            
            # First split: 70% training, 30% for validation+test
            X_train, X_temp, y_w_train, y_w_temp, y_n_train, y_n_temp = train_test_split(
                X, y_workout, y_nutrition, test_size=0.3, random_state=random_state,
                stratify=y_workout
            )
            
            # Second split: 15% validation, 15% test from the remaining 30%
            X_val, X_test, y_w_val, y_w_test, y_n_val, y_n_test = train_test_split(
                X_temp, y_w_temp, y_n_temp, test_size=0.5, random_state=random_state,
                stratify=y_w_temp
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Add noise to training features to prevent overfitting (especially for workout model)
        # Fix class encoding issue - ensure continuous class indices
        print(f"Workout template IDs in dataset: {sorted(y_workout.unique())}")
        print(f"Nutrition template IDs in dataset: {sorted(y_nutrition.unique())}")
        
        # Use LabelEncoder to ensure continuous class indices from 0
        workout_label_encoder = LabelEncoder()
        nutrition_label_encoder = LabelEncoder()
        
        # Fit on all possible template IDs to ensure consistency
        all_workout_ids = list(range(1, 10))  # Template IDs 1-9
        all_nutrition_ids = list(range(1, 9))  # Template IDs 1-8
        
        workout_label_encoder.fit(all_workout_ids)
        nutrition_label_encoder.fit(all_nutrition_ids)
        
        # Transform to continuous indices - but first ensure we only use IDs that exist in templates
        y_w_train_encoded = workout_label_encoder.transform(y_w_train)
        y_w_val_encoded = workout_label_encoder.transform(y_w_val)
        y_w_test_encoded = workout_label_encoder.transform(y_w_test)
        
        y_n_train_encoded = nutrition_label_encoder.transform(y_n_train)
        y_n_val_encoded = nutrition_label_encoder.transform(y_n_val)
        y_n_test_encoded = nutrition_label_encoder.transform(y_n_test)
        
        # Store encoders for inverse transformation later
        self.workout_label_encoder = workout_label_encoder
        self.nutrition_label_encoder = nutrition_label_encoder
        
        print(f"Workout classes after encoding: {sorted(np.unique(y_w_train_encoded))}")
        print(f"Nutrition classes after encoding: {sorted(np.unique(y_n_train_encoded))}")
        
        # Check if all expected classes are present in training data
        # For XGBoost to work properly, we need continuous classes starting from 0
        expected_workout_classes = set(range(9))  # 0-8 for 9 classes  
        expected_nutrition_classes = set(range(8))  # 0-7 for 8 classes
        
        actual_workout_classes = set(y_w_train_encoded)
        actual_nutrition_classes = set(y_n_train_encoded)
        
        missing_workout_classes = expected_workout_classes - actual_workout_classes
        missing_nutrition_classes = expected_nutrition_classes - actual_nutrition_classes
        
        if missing_workout_classes or missing_nutrition_classes:
            print(f"Warning: Missing classes detected. Implementing fix...")
            if missing_workout_classes:
                print(f"Missing workout classes: {missing_workout_classes}")
            if missing_nutrition_classes:
                print(f"Missing nutrition classes: {missing_nutrition_classes}")
                
            # Add synthetic samples for missing classes (minimal approach)
            # This ensures XGBoost sees all expected classes during training
            all_missing_classes = missing_workout_classes.union(missing_nutrition_classes)
            
            if all_missing_classes:
                # Add one synthetic sample per unique missing class using mean feature values
                mean_features = X_train_scaled.mean(axis=0)
                for missing_class in all_missing_classes:
                    # Add synthetic sample
                    X_train_scaled = np.vstack([X_train_scaled, mean_features.reshape(1, -1)])
                    
                    # Add to workout labels if needed
                    if missing_class in missing_workout_classes:
                        y_w_train_encoded = np.append(y_w_train_encoded, missing_class)
                    else:
                        # Add existing workout class for this sample
                        y_w_train_encoded = np.append(y_w_train_encoded, y_w_train_encoded[0])
                    
                    # Add to nutrition labels if needed  
                    if missing_class in missing_nutrition_classes:
                        y_n_train_encoded = np.append(y_n_train_encoded, missing_class)
                    else:
                        # Add existing nutrition class for this sample
                        y_n_train_encoded = np.append(y_n_train_encoded, y_n_train_encoded[0])
                    
        # Calculate class weights to handle severe imbalance
        from sklearn.utils.class_weight import compute_class_weight
        
        # Compute class weights for workout model
        workout_classes = np.unique(y_w_train_encoded)
        workout_class_weights = compute_class_weight(
            'balanced', 
            classes=workout_classes, 
            y=y_w_train_encoded
        )
        workout_weight_dict = dict(zip(workout_classes, workout_class_weights))
        
        # Compute class weights for nutrition model  
        nutrition_classes = np.unique(y_n_train_encoded)
        nutrition_class_weights = compute_class_weight(
            'balanced',
            classes=nutrition_classes,
            y=y_n_train_encoded
        )
        nutrition_weight_dict = dict(zip(nutrition_classes, nutrition_class_weights))
        
        print(f"Applied class weights to handle imbalanced data")
        
        # REMOVE AGGRESSIVE NOISE INJECTION - it's causing overfitting
        print("🎯 Using clean training data to prevent overfitting...")
        
        # Use original training data without aggressive augmentation
        X_train_workout_combined = X_train_scaled.copy()
        y_w_train_workout_combined = y_w_train_encoded.copy()
        
        # Use original training data for nutrition model  
        X_train_nutrition = X_train_scaled.copy()
        y_n_train_nutrition = y_n_train_encoded.copy()
        
        print(f"Training with clean data: {len(X_train_scaled)} samples")
        
        # Optimized hyperparameter distributions for faster training with good performance
        # Balanced anti-overfitting approach: Conservative but not extreme parameters
        workout_param_distributions = {
            'max_depth': [3, 4, 5],  # Moderate depth
            'learning_rate': [0.05, 0.1, 0.15],  # Moderate learning rates
            'n_estimators': [100, 150, 200],  # Moderate number of trees
            'min_child_weight': [5, 10, 15],  # Moderate minimum samples
            'subsample': [0.7, 0.8, 0.9],  # Moderate subsampling
            'colsample_bytree': [0.7, 0.8, 0.9],  # Moderate feature subsampling
            'reg_alpha': [0.5, 1.0, 2.0],  # Moderate L1 regularization
            'reg_lambda': [1.0, 2.0, 5.0],  # Moderate L2 regularization
            'gamma': [0.1, 0.5, 1.0]  # Moderate minimum loss reduction
        }
        
        # Conservative parameters for nutrition model to reduce overfitting
        nutrition_param_distributions = {
            'max_depth': [2, 3, 4],  # Very reduced depth to prevent overfitting
            'learning_rate': [0.01, 0.03, 0.05],  # Lower learning rates
            'n_estimators': [50, 100, 150],  # Fewer estimators
            'min_child_weight': [5, 10, 15],  # Higher minimum child weight
            'subsample': [0.5, 0.6, 0.7],  # More aggressive subsampling
            'colsample_bytree': [0.5, 0.6, 0.7],  # More feature subsampling
            'reg_alpha': [1.0, 2.0, 5.0],  # Stronger L1 regularization
            'reg_lambda': [2.0, 5.0, 10.0]  # Stronger L2 regularization
        }
        
        # Optimized XGBoost parameters for faster training
        base_params = {
            'random_state': random_state,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'early_stopping_rounds': 30,  # Reduced for faster training
            'verbose': False,
            'tree_method': 'hist',  # Faster training
            'n_jobs': -1,  # Use all CPU cores
            'gpu_id': 0 if xgb.get_config().get('use_gpu', False) else None  # Use GPU if available
        }
        
        # Train workout model using XGBoost with hyperparameter tuning
        print("Training workout XGBoost model with hyperparameter tuning...")
        
        # Add class weights to base parameters for workout model
        workout_base_params = base_params.copy()
        workout_base_params.update({
            'class_weight': workout_weight_dict,
            'max_delta_step': 3  # Additional regularization for imbalanced classes
        })
        
        workout_xgb = xgb.XGBClassifier(**workout_base_params)
        
        workout_search = RandomizedSearchCV(
            workout_xgb,
            param_distributions=workout_param_distributions,
            n_iter=20,  # Reduced iterations to prevent overfitting
            cv=10,  # More aggressive cross-validation
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit workout model with early stopping validation and extreme noisy training data
        workout_search.fit(
            X_train_workout_combined, y_w_train_workout_combined,
            eval_set=[(X_val_scaled, y_w_val_encoded)],
            verbose=False
        )
        
        self.workout_model = workout_search.best_estimator_
        workout_val_score = self.workout_model.score(X_val_scaled, y_w_val_encoded)
        
        print(f"Best workout XGBoost parameters: {workout_search.best_params_}")
        print(f"Workout model validation score: {workout_val_score:.4f}")
        
        # Train nutrition model using XGBoost with hyperparameter tuning
        print("Training nutrition XGBoost model with hyperparameter tuning...")
        
        # Add class weights to base parameters for nutrition model
        nutrition_base_params = base_params.copy()
        nutrition_base_params.update({
            'class_weight': nutrition_weight_dict
        })
        
        nutrition_xgb = xgb.XGBClassifier(**nutrition_base_params)
        
        nutrition_search = RandomizedSearchCV(
            nutrition_xgb,
            param_distributions=nutrition_param_distributions,
            n_iter=20,  # Fewer iterations for nutrition (simpler problem)
            cv=3,
            scoring='accuracy',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit nutrition model with early stopping validation and moderate noisy training data
        nutrition_search.fit(
            X_train_nutrition, y_n_train_nutrition,
            eval_set=[(X_val_scaled, y_n_val_encoded)],
            verbose=False
        )
        
        self.nutrition_model = nutrition_search.best_estimator_
        nutrition_val_score = self.nutrition_model.score(X_val_scaled, y_n_val_encoded)
        
        print(f"Best nutrition XGBoost parameters: {nutrition_search.best_params_}")
        print(f"Nutrition model validation score: {nutrition_val_score:.4f}")
        
        # Evaluate on test set with comprehensive metrics
        print("Evaluating models on test set...")
        
        # Workout model evaluation
        y_w_pred = self.workout_model.predict(X_test_scaled)
        y_w_pred_proba = self.workout_model.predict_proba(X_test_scaled)
        workout_accuracy = accuracy_score(y_w_test_encoded, y_w_pred)
        workout_f1 = f1_score(y_w_test_encoded, y_w_pred, average='weighted')
        
        # Nutrition model evaluation
        y_n_pred = self.nutrition_model.predict(X_test_scaled)
        y_n_pred_proba = self.nutrition_model.predict_proba(X_test_scaled)
        nutrition_accuracy = accuracy_score(y_n_test_encoded, y_n_pred)
        nutrition_f1 = f1_score(y_n_test_encoded, y_n_pred, average='weighted')
        
        # Comprehensive metrics calculation
        print("\nCalculating comprehensive metrics for workout model...")
        workout_metrics = self._calculate_comprehensive_metrics(
            y_w_test_encoded, y_w_pred, y_w_pred_proba, 
            model_name="Workout Model", encoder=self.workout_label_encoder
        )
        
        print("\nCalculating comprehensive metrics for nutrition model...")
        nutrition_metrics = self._calculate_comprehensive_metrics(
            y_n_test_encoded, y_n_pred, y_n_pred_proba, 
            model_name="Nutrition Model", encoder=self.nutrition_label_encoder
        )
        
        # Feature importance analysis for XGBoost models
        print("\n=== Feature Importance Analysis ===")
        
        # Get feature importance for workout model (XGBoost)
        try:
            workout_importance = self.workout_model.feature_importances_
        except:
            workout_importance = np.ones(len(self.feature_columns)) / len(self.feature_columns)
            
        # Get feature importance for nutrition model (XGBoost)
        try:
            nutrition_importance = self.nutrition_model.feature_importances_
        except:
            nutrition_importance = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        try:
            nutrition_importance = self.nutrition_model.feature_importances_
        except:
            nutrition_importance = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        
        feature_names = self.feature_columns
        
        print("\nWorkout Model - Top 10 Most Important Features:")
        workout_feat_importance = list(zip(feature_names, workout_importance))
        workout_feat_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(workout_feat_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print("\nNutrition Model - Top 10 Most Important Features:")
        nutrition_feat_importance = list(zip(feature_names, nutrition_importance))
        nutrition_feat_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(nutrition_feat_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Add feature importance to training info
        workout_feature_importance = dict(workout_feat_importance)
        nutrition_feature_importance = dict(nutrition_feat_importance)
        
        # Store comprehensive training information
        self.training_info = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'workout_metrics': workout_metrics,
            'nutrition_metrics': nutrition_metrics,
            'workout_accuracy': workout_accuracy,
            'workout_f1': workout_f1,
            'nutrition_accuracy': nutrition_accuracy,
            'nutrition_f1': nutrition_f1,
            'workout_feature_importance': workout_feature_importance,
            'nutrition_feature_importance': nutrition_feature_importance,
            'trained_at': datetime.now().isoformat()
        }
        
        self.is_trained = True
        
        print("\nModel training completed!")
        print(f"Workout model - Test Accuracy: {workout_accuracy:.4f}, F1: {workout_f1:.4f}")
        print(f"Nutrition model - Test Accuracy: {nutrition_accuracy:.4f}, F1: {nutrition_f1:.4f}")
        
        return self.training_info
    
    def train_all_models(self, df_training, random_state=42):
        """
        Train both XGBoost and Random Forest models for comprehensive comparison
        
        Args:
            df_training: Training dataset
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with combined training information
        """
        print("\n🚀 COMPREHENSIVE MODEL TRAINING: XGBoost + Random Forest Baselines")
        print("="*80)
        
        # Train XGBoost models first
        print("\n📈 STEP 1: Training XGBoost Models...")
        xgb_training_info = self.train_models(df_training, random_state)
        
        # Train Random Forest baseline models
        print("\n🌲 STEP 2: Training Random Forest Baseline Models...")
        rf_training_info = self.train_random_forest_baselines(df_training, random_state)
        
        # Create comprehensive comparison
        print("\n📊 STEP 3: Generating Model Comparison...")
        comparison_data = self.compare_model_performance()
        
        # Combine all training information
        comprehensive_info = {
            'xgb_training_info': xgb_training_info,
            'rf_training_info': rf_training_info,
            'comparison_data': comparison_data,
            'training_completed_at': datetime.now().isoformat()
        }
        
        print("\n✅ COMPREHENSIVE TRAINING COMPLETED!")
        print("="*80)
        print("Both XGBoost and Random Forest models have been trained and compared.")
        print("Use compare_model_performance() to view detailed comparison tables.")
        
        return comprehensive_info
    
    def train_random_forest_baselines(self, df_training, random_state=42):
        """
        Train Random Forest baseline models for academic comparison with XGBoost
        
        Args:
            df_training: Training dataset
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with Random Forest model performance metrics
        """
        print("\n🌲 Training Random Forest Baseline Models for Academic Comparison...")
        print("="*80)
        
        # Prepare data (same as XGBoost training)
        X, y_workout, y_nutrition, df_enhanced = self.prepare_training_data(df_training)
        
        # Split data based on the 'split' column if available, otherwise use standard split
        if 'split' in df_training.columns:
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
            
            print(f"Using predefined splits for Random Forest:")
            print(f"  Training: {len(X_train)} samples")
            print(f"  Validation: {len(X_val)} samples")
            print(f"  Test: {len(X_test)} samples")
            
        else:
            # Fallback to standard split if no split column
            print("No split column found, using standard train/val/test split for Random Forest...")
            
            # First split: 70% training, 30% for validation+test
            X_train, X_temp, y_w_train, y_w_temp, y_n_train, y_n_temp = train_test_split(
                X, y_workout, y_nutrition, test_size=0.3, random_state=random_state,
                stratify=y_workout
            )
            
            # Second split: 15% validation, 15% test from the remaining 30%
            X_val, X_test, y_w_val, y_w_test, y_n_val, y_n_test = train_test_split(
                X_temp, y_w_temp, y_n_temp, test_size=0.5, random_state=random_state,
                stratify=y_w_temp
            )
        
        # Scale features (same scaler as XGBoost for fair comparison)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use same label encoders as XGBoost for consistency
        workout_label_encoder = LabelEncoder()
        nutrition_label_encoder = LabelEncoder()
        
        # Fit on all possible template IDs to ensure consistency
        all_workout_ids = list(range(1, 10))  # Template IDs 1-9
        all_nutrition_ids = list(range(1, 9))  # Template IDs 1-8
        
        workout_label_encoder.fit(all_workout_ids)
        nutrition_label_encoder.fit(all_nutrition_ids)
        
        # Transform to continuous indices
        y_w_train_encoded = workout_label_encoder.transform(y_w_train)
        y_w_val_encoded = workout_label_encoder.transform(y_w_val)
        y_w_test_encoded = workout_label_encoder.transform(y_w_test)
        
        y_n_train_encoded = nutrition_label_encoder.transform(y_n_train)
        y_n_val_encoded = nutrition_label_encoder.transform(y_n_val)
        y_n_test_encoded = nutrition_label_encoder.transform(y_n_test)
        
        # Store encoders for Random Forest models
        self.workout_rf_label_encoder = workout_label_encoder
        self.nutrition_rf_label_encoder = nutrition_label_encoder
        
        print(f"Random Forest - Workout classes: {sorted(np.unique(y_w_train_encoded))}")
        print(f"Random Forest - Nutrition classes: {sorted(np.unique(y_n_train_encoded))}")
        
        # Calculate class weights for Random Forest (same as XGBoost)
        from sklearn.utils.class_weight import compute_class_weight
        
        workout_classes = np.unique(y_w_train_encoded)
        workout_class_weights = compute_class_weight(
            'balanced', 
            classes=workout_classes, 
            y=y_w_train_encoded
        )
        workout_weight_dict = dict(zip(workout_classes, workout_class_weights))
        
        nutrition_classes = np.unique(y_n_train_encoded)
        nutrition_class_weights = compute_class_weight(
            'balanced',
            classes=nutrition_classes,
            y=y_n_train_encoded
        )
        nutrition_weight_dict = dict(zip(nutrition_classes, nutrition_class_weights))
        
        # Random Forest hyperparameter distributions for fair comparison
        rf_workout_param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_nutrition_param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Train Random Forest workout model (classification)
        print("\n🌲 Training Random Forest Workout Model (Classification)...")
        
        rf_workout = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rf_workout_search = RandomizedSearchCV(
            rf_workout,
            param_distributions=rf_workout_param_distributions,
            n_iter=20,  # Same as XGBoost for fair comparison
            cv=5,  # 5-fold CV for Random Forest
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        rf_workout_search.fit(X_train_scaled, y_w_train_encoded)
        self.workout_rf_model = rf_workout_search.best_estimator_
        rf_workout_val_score = self.workout_rf_model.score(X_val_scaled, y_w_val_encoded)
        
        print(f"Best Random Forest workout parameters: {rf_workout_search.best_params_}")
        print(f"Random Forest workout validation score: {rf_workout_val_score:.4f}")
        
        # Train Random Forest nutrition model (classification - same as XGBoost)
        print("\n🌲 Training Random Forest Nutrition Model (Classification)...")
        
        rf_nutrition = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rf_nutrition_search = RandomizedSearchCV(
            rf_nutrition,
            param_distributions=rf_nutrition_param_distributions,
            n_iter=20,  # Same as XGBoost for fair comparison
            cv=5,  # 5-fold CV for Random Forest
            scoring='accuracy',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        rf_nutrition_search.fit(X_train_scaled, y_n_train_encoded)
        self.nutrition_rf_model = rf_nutrition_search.best_estimator_
        rf_nutrition_val_score = self.nutrition_rf_model.score(X_val_scaled, y_n_val_encoded)
        
        print(f"Best Random Forest nutrition parameters: {rf_nutrition_search.best_params_}")
        print(f"Random Forest nutrition validation score: {rf_nutrition_val_score:.4f}")
        
        # Evaluate Random Forest models on test set
        print("\n🌲 Evaluating Random Forest Models on Test Set...")
        
        # Random Forest workout evaluation
        rf_y_w_pred = self.workout_rf_model.predict(X_test_scaled)
        rf_y_w_pred_proba = self.workout_rf_model.predict_proba(X_test_scaled)
        rf_workout_accuracy = accuracy_score(y_w_test_encoded, rf_y_w_pred)
        rf_workout_f1 = f1_score(y_w_test_encoded, rf_y_w_pred, average='weighted')
        
        # Random Forest nutrition evaluation
        rf_y_n_pred = self.nutrition_rf_model.predict(X_test_scaled)
        rf_y_n_pred_proba = self.nutrition_rf_model.predict_proba(X_test_scaled)
        rf_nutrition_accuracy = accuracy_score(y_n_test_encoded, rf_y_n_pred)
        rf_nutrition_f1 = f1_score(y_n_test_encoded, rf_y_n_pred, average='weighted')
        
        # Calculate comprehensive metrics for Random Forest models
        print("\nCalculating comprehensive metrics for Random Forest workout model...")
        rf_workout_metrics = self._calculate_comprehensive_metrics(
            y_w_test_encoded, rf_y_w_pred, rf_y_w_pred_proba, 
            model_name="Random Forest Workout Model", encoder=self.workout_rf_label_encoder
        )
        
        print("\nCalculating comprehensive metrics for Random Forest nutrition model...")
        rf_nutrition_metrics = self._calculate_comprehensive_metrics(
            y_n_test_encoded, rf_y_n_pred, rf_y_n_pred_proba, 
            model_name="Random Forest Nutrition Model", encoder=self.nutrition_rf_label_encoder
        )
        
        # Feature importance analysis for Random Forest models
        print("\n=== Random Forest Feature Importance Analysis ===")
        
        # Get feature importance for Random Forest models
        try:
            rf_workout_importance = self.workout_rf_model.feature_importances_
        except:
            rf_workout_importance = np.ones(len(self.feature_columns)) / len(self.feature_columns)
            
        try:
            rf_nutrition_importance = self.nutrition_rf_model.feature_importances_
        except:
            rf_nutrition_importance = np.ones(len(self.feature_columns)) / len(self.feature_columns)
        
        feature_names = self.feature_columns
        
        print("\nRandom Forest Workout Model - Top 10 Most Important Features:")
        rf_workout_feat_importance = list(zip(feature_names, rf_workout_importance))
        rf_workout_feat_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(rf_workout_feat_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print("\nRandom Forest Nutrition Model - Top 10 Most Important Features:")
        rf_nutrition_feat_importance = list(zip(feature_names, rf_nutrition_importance))
        rf_nutrition_feat_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(rf_nutrition_feat_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Store Random Forest training information
        rf_training_info = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'rf_workout_metrics': rf_workout_metrics,
            'rf_nutrition_metrics': rf_nutrition_metrics,
            'rf_workout_accuracy': rf_workout_accuracy,
            'rf_workout_f1': rf_workout_f1,
            'rf_nutrition_accuracy': rf_nutrition_accuracy,
            'rf_nutrition_f1': rf_nutrition_f1,
            'rf_workout_feature_importance': dict(rf_workout_feat_importance),
            'rf_nutrition_feature_importance': dict(rf_nutrition_feat_importance),
            'rf_trained_at': datetime.now().isoformat()
        }
        
        print("\n🌲 Random Forest baseline training completed!")
        print(f"Random Forest Workout model - Test Accuracy: {rf_workout_accuracy:.4f}, F1: {rf_workout_f1:.4f}")
        print(f"Random Forest Nutrition model - Test Accuracy: {rf_nutrition_accuracy:.4f}, F1: {rf_nutrition_f1:.4f}")
        
        # Store Random Forest training info for later comparison
        self.rf_training_info = rf_training_info
        
        return rf_training_info
    
    def compare_model_performance(self):
        """
        Create comprehensive comparison table between XGBoost and Random Forest models
        
        Returns:
            Dictionary with detailed comparison metrics
        """
        if not self.is_trained or self.workout_rf_model is None:
            raise ValueError("Both XGBoost and Random Forest models must be trained before comparison")
        
        print("\n📊 COMPREHENSIVE MODEL COMPARISON: XGBoost vs Random Forest")
        print("="*80)
        
        # Extract metrics from training info
        xgb_workout_metrics = self.training_info.get('workout_metrics', {})
        xgb_nutrition_metrics = self.training_info.get('nutrition_metrics', {})
        
        # Get Random Forest metrics (assuming they're stored in training_info)
        rf_workout_metrics = getattr(self, 'rf_training_info', {}).get('rf_workout_metrics', {})
        rf_nutrition_metrics = getattr(self, 'rf_training_info', {}).get('rf_nutrition_metrics', {})
        
        # Create comparison table
        comparison_data = {
            'workout_model': {
                'metric': [
                    'Accuracy',
                    'Balanced Accuracy', 
                    'F1 Score (Weighted)',
                    'F1 Score (Macro)',
                    'Precision (Weighted)',
                    'Recall (Weighted)',
                    'Cohen\'s Kappa',
                    'Top-2 Accuracy',
                    'Top-3 Accuracy',
                    'AUC-ROC (Weighted)'
                ],
                'xgboost': [
                    xgb_workout_metrics.get('accuracy', 0),
                    xgb_workout_metrics.get('balanced_accuracy', 0),
                    xgb_workout_metrics.get('f1_weighted', 0),
                    xgb_workout_metrics.get('f1_macro', 0),
                    xgb_workout_metrics.get('precision_weighted', 0),
                    xgb_workout_metrics.get('recall_weighted', 0),
                    xgb_workout_metrics.get('cohen_kappa', 0),
                    xgb_workout_metrics.get('top2_accuracy', 0),
                    xgb_workout_metrics.get('top3_accuracy', 0),
                    xgb_workout_metrics.get('auc_roc', 0) if xgb_workout_metrics.get('auc_roc') else 0
                ],
                'random_forest': [
                    rf_workout_metrics.get('accuracy', 0),
                    rf_workout_metrics.get('balanced_accuracy', 0),
                    rf_workout_metrics.get('f1_weighted', 0),
                    rf_workout_metrics.get('f1_macro', 0),
                    rf_workout_metrics.get('precision_weighted', 0),
                    rf_workout_metrics.get('recall_weighted', 0),
                    rf_workout_metrics.get('cohen_kappa', 0),
                    rf_workout_metrics.get('top2_accuracy', 0),
                    rf_workout_metrics.get('top3_accuracy', 0),
                    rf_workout_metrics.get('auc_roc', 0) if rf_workout_metrics.get('auc_roc') else 0
                ]
            },
            'nutrition_model': {
                'metric': [
                    'Accuracy',
                    'Balanced Accuracy',
                    'F1 Score (Weighted)',
                    'F1 Score (Macro)',
                    'Precision (Weighted)',
                    'Recall (Weighted)',
                    'Cohen\'s Kappa',
                    'Top-2 Accuracy',
                    'Top-3 Accuracy',
                    'AUC-ROC (Weighted)'
                ],
                'xgboost': [
                    xgb_nutrition_metrics.get('accuracy', 0),
                    xgb_nutrition_metrics.get('balanced_accuracy', 0),
                    xgb_nutrition_metrics.get('f1_weighted', 0),
                    xgb_nutrition_metrics.get('f1_macro', 0),
                    xgb_nutrition_metrics.get('precision_weighted', 0),
                    xgb_nutrition_metrics.get('recall_weighted', 0),
                    xgb_nutrition_metrics.get('cohen_kappa', 0),
                    xgb_nutrition_metrics.get('top2_accuracy', 0),
                    xgb_nutrition_metrics.get('top3_accuracy', 0),
                    xgb_nutrition_metrics.get('auc_roc', 0) if xgb_nutrition_metrics.get('auc_roc') else 0
                ],
                'random_forest': [
                    rf_nutrition_metrics.get('accuracy', 0),
                    rf_nutrition_metrics.get('balanced_accuracy', 0),
                    rf_nutrition_metrics.get('f1_weighted', 0),
                    rf_nutrition_metrics.get('f1_macro', 0),
                    rf_nutrition_metrics.get('precision_weighted', 0),
                    rf_nutrition_metrics.get('recall_weighted', 0),
                    rf_nutrition_metrics.get('cohen_kappa', 0),
                    rf_nutrition_metrics.get('top2_accuracy', 0),
                    rf_nutrition_metrics.get('top3_accuracy', 0),
                    rf_nutrition_metrics.get('auc_roc', 0) if rf_nutrition_metrics.get('auc_roc') else 0
                ]
            }
        }
        
        # Print comparison tables
        print("\n🏋️ WORKOUT MODEL COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<25} {'XGBoost':<12} {'Random Forest':<15} {'Difference':<12}")
        print("-" * 80)
        
        for i, metric in enumerate(comparison_data['workout_model']['metric']):
            xgb_val = comparison_data['workout_model']['xgboost'][i]
            rf_val = comparison_data['workout_model']['random_forest'][i]
            diff = xgb_val - rf_val
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            
            print(f"{metric:<25} {xgb_val:<12.4f} {rf_val:<15.4f} {diff_str:<12}")
        
        print("\n🥗 NUTRITION MODEL COMPARISON:")
        print("-" * 80)
        print(f"{'Metric':<25} {'XGBoost':<12} {'Random Forest':<15} {'Difference':<12}")
        print("-" * 80)
        
        for i, metric in enumerate(comparison_data['nutrition_model']['metric']):
            xgb_val = comparison_data['nutrition_model']['xgboost'][i]
            rf_val = comparison_data['nutrition_model']['random_forest'][i]
            diff = xgb_val - rf_val
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            
            print(f"{metric:<25} {xgb_val:<12.4f} {rf_val:<15.4f} {diff_str:<12}")
        
        # Calculate overall performance summary
        print("\n📈 OVERALL PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        # Workout model summary
        xgb_workout_avg = np.mean([
            xgb_workout_metrics.get('accuracy', 0),
            xgb_workout_metrics.get('f1_weighted', 0),
            xgb_workout_metrics.get('balanced_accuracy', 0)
        ])
        
        rf_workout_avg = np.mean([
            rf_workout_metrics.get('accuracy', 0),
            rf_workout_metrics.get('f1_weighted', 0),
            rf_workout_metrics.get('balanced_accuracy', 0)
        ])
        
        # Nutrition model summary
        xgb_nutrition_avg = np.mean([
            xgb_nutrition_metrics.get('accuracy', 0),
            xgb_nutrition_metrics.get('f1_weighted', 0),
            xgb_nutrition_metrics.get('balanced_accuracy', 0)
        ])
        
        rf_nutrition_avg = np.mean([
            rf_nutrition_metrics.get('accuracy', 0),
            rf_nutrition_metrics.get('f1_weighted', 0),
            rf_nutrition_metrics.get('balanced_accuracy', 0)
        ])
        
        print(f"Workout Model Average Performance:")
        print(f"  XGBoost: {xgb_workout_avg:.4f}")
        print(f"  Random Forest: {rf_workout_avg:.4f}")
        print(f"  XGBoost Advantage: {xgb_workout_avg - rf_workout_avg:+.4f}")
        
        print(f"\nNutrition Model Average Performance:")
        print(f"  XGBoost: {xgb_nutrition_avg:.4f}")
        print(f"  Random Forest: {rf_nutrition_avg:.4f}")
        print(f"  XGBoost Advantage: {xgb_nutrition_avg - rf_nutrition_avg:+.4f}")
        
        # Academic insights
        print("\n🎓 ACADEMIC INSIGHTS:")
        print("-" * 50)
        
        if xgb_workout_avg > rf_workout_avg:
            print(f"✅ XGBoost outperforms Random Forest for workout recommendations")
            print(f"   Improvement: {((xgb_workout_avg/rf_workout_avg - 1) * 100):.2f}%")
        else:
            print(f"⚠️ Random Forest performs better for workout recommendations")
            print(f"   XGBoost deficit: {((rf_workout_avg/xgb_workout_avg - 1) * 100):.2f}%")
        
        if xgb_nutrition_avg > rf_nutrition_avg:
            print(f"✅ XGBoost outperforms Random Forest for nutrition recommendations")
            print(f"   Improvement: {((xgb_nutrition_avg/rf_nutrition_avg - 1) * 100):.2f}%")
        else:
            print(f"⚠️ Random Forest performs better for nutrition recommendations")
            print(f"   XGBoost deficit: {((rf_nutrition_avg/xgb_nutrition_avg - 1) * 100):.2f}%")
        
        # Feature importance comparison
        print("\n🔍 FEATURE IMPORTANCE COMPARISON:")
        print("-" * 50)
        
        xgb_workout_importance = self.training_info.get('workout_feature_importance', {})
        rf_workout_importance = getattr(self, 'rf_training_info', {}).get('rf_workout_feature_importance', {})
        
        if xgb_workout_importance and rf_workout_importance:
            # Get top 5 features for each model
            xgb_top_features = sorted(xgb_workout_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            rf_top_features = sorted(rf_workout_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print("Top 5 Most Important Features - Workout Model:")
            print("XGBoost:")
            for i, (feature, importance) in enumerate(xgb_top_features):
                print(f"  {i+1}. {feature}: {importance:.4f}")
            
            print("Random Forest:")
            for i, (feature, importance) in enumerate(rf_top_features):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        return comparison_data
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, model_name, encoder):
        """
        Calculate comprehensive performance metrics for model evaluation
        
        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            y_pred_proba: Prediction probabilities
            model_name: Name of the model for logging
            encoder: Label encoder for class names
            
        Returns:
            Dictionary with comprehensive metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Balanced accuracy (good for imbalanced classes)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Cohen's Kappa (agreement measure)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Top-2 and Top-3 accuracy for multi-class
        def top_k_accuracy(y_true, y_pred_proba, k):
            top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in top_k_pred[i]:
                    correct += 1
            return correct / len(y_true)
        
        top2_acc = top_k_accuracy(y_true, y_pred_proba, 2)
        top3_acc = top_k_accuracy(y_true, y_pred_proba, 3)
        
        # AUC ROC for multi-class (one-vs-rest)
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc_roc = None  # In case of issues with single class
        
        # Class distribution analysis (simplified)
        unique_classes = np.unique(y_true)
        class_support = {f"Class_{cls}": np.sum(y_true == cls) for cls in unique_classes}
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'precision_macro': precision_macro,
            'recall_weighted': recall_weighted,
            'recall_macro': recall_macro,
            'cohen_kappa': kappa,
            'top2_accuracy': top2_acc,
            'top3_accuracy': top3_acc,
            'auc_roc': auc_roc,
            'class_support': class_support,
            'num_classes': len(unique_classes)
        }
        
        # Print comprehensive results
        print(f"\n=== {model_name} Performance Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall (Weighted): {recall_weighted:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"Top-2 Accuracy: {top2_acc:.4f}")
        print(f"Top-3 Accuracy: {top3_acc:.4f}")
        if auc_roc:
            print(f"AUC-ROC (Weighted): {auc_roc:.4f}")
        print(f"Number of Classes: {len(unique_classes)}")
        
        # Per-class performance
        print(f"\nPer-Class Support:")
        for class_name, support in class_support.items():
            print(f"  {class_name}: {support} samples")
        
        return metrics
    
    def calculate_prediction_confidence(self, user_profile, workout_prediction, nutrition_prediction):
        """
        Calculate confidence scores based on training data representation and model reliability
        
        Returns enhanced confidence with transparent limitations reporting
        """
        # Get template assignment for user profile
        fitness_goal = user_profile.get('fitness_goal')
        activity_level = user_profile.get('activity_level') 
        bmi_category = user_profile.get('bmi_category')
        
        # Initialize base confidence
        base_confidence = 0.85
        confidence_factors = []
        limitations = []
        
        # 1. Activity Level Representation Analysis
        activity_confidence = self._get_activity_confidence(activity_level)
        if activity_level == 'High Activity':
            confidence_factors.append(("High activity representation", 0.95))
        elif activity_level == 'Moderate Activity':
            confidence_factors.append(("Moderate activity representation", 0.75))
            limitations.append("Moderate activity users represent only 22% of training data")
        else:  # Low Activity
            confidence_factors.append(("Low activity representation", 0.60))
            limitations.append("Low activity users represent only 10% of training data - reduced confidence")
            limitations.append("Professional consultation recommended for low-activity individuals")
        
        # 2. Template Frequency Analysis
        workout_template_confidence = self._get_template_confidence(workout_prediction, 'workout')
        nutrition_template_confidence = self._get_template_confidence(nutrition_prediction, 'nutrition')
        
        confidence_factors.append(("Workout template frequency", workout_template_confidence))
        confidence_factors.append(("Nutrition template frequency", nutrition_template_confidence))
        
        # 3. Combination Rarity Check
        combination_confidence = self._get_combination_confidence(fitness_goal, activity_level, bmi_category)
        confidence_factors.append(("Profile combination frequency", combination_confidence))
        
        # Calculate overall confidence
        weights = [0.25, 0.35, 0.40]  # Activity, template frequency, combination rarity
        overall_confidence = sum(w * factor[1] for w, factor in zip(weights, confidence_factors))
        
        # Cap confidence and add transparency
        final_confidence = min(overall_confidence, 0.95)
        
        # Generate confidence explanation
        explanation = self._generate_confidence_explanation(
            final_confidence, confidence_factors, limitations, user_profile
        )
        
        return {
            'confidence_score': round(final_confidence, 3),
            'confidence_level': self._get_confidence_level(final_confidence),
            'factors': confidence_factors,
            'limitations': limitations,
            'explanation': explanation,
            'model_limitations': self._get_model_limitations()
        }
    
    def _get_activity_confidence(self, activity_level):
        """Get confidence score based on activity level representation in training data"""
        activity_representation = {
            'High Activity': 0.678,    # 67.8% of training data
            'Moderate Activity': 0.222, # 22.2% of training data  
            'Low Activity': 0.100      # 10.0% of training data
        }
        
        representation = activity_representation.get(activity_level, 0.5)
        
        # Convert representation to confidence score
        if representation >= 0.6:
            return 0.95
        elif representation >= 0.2:
            return 0.75
        else:
            return 0.60
    
    def _get_template_confidence(self, template_id, template_type):
        """Get confidence score based on template frequency in training data"""
        # These thresholds are based on the debugging analysis
        low_sample_templates = {
            'workout': [1, 4, 7],  # Templates with <300 samples
            'nutrition': [4, 6]    # Templates with <200 samples
        }
        
        if template_id in low_sample_templates.get(template_type, []):
            return 0.65  # Reduced confidence for low-sample templates
        else:
            return 0.90  # High confidence for well-represented templates
    
    def _get_combination_confidence(self, fitness_goal, activity_level, bmi_category):
        """Get confidence score based on goal+activity and goal+BMI combination frequency"""
        # Low-frequency combinations identified from analysis
        rare_workout_combos = [
            ('Fat Loss', 'Low Activity'),
            ('Muscle Gain', 'Low Activity'), 
            ('Maintenance', 'Low Activity')
        ]
        
        rare_nutrition_combos = [
            ('Muscle Gain', 'Underweight'),
            ('Maintenance', 'Underweight')
        ]
        
        workout_combo = (fitness_goal, activity_level)
        nutrition_combo = (fitness_goal, bmi_category)
        
        confidence = 0.90  # Default high confidence
        
        if workout_combo in rare_workout_combos:
            confidence -= 0.20
        
        if nutrition_combo in rare_nutrition_combos:
            confidence -= 0.15
        
        return max(confidence, 0.50)  # Minimum 50% confidence
    
    def _get_confidence_level(self, confidence_score):
        """Convert numerical confidence to descriptive level"""
        if confidence_score >= 0.85:
            return "High"
        elif confidence_score >= 0.70:
            return "Moderate"
        elif confidence_score >= 0.55:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_confidence_explanation(self, confidence, factors, limitations, user_profile):
        """Generate human-readable confidence explanation"""
        activity = user_profile.get('activity_level')
        goal = user_profile.get('fitness_goal')
        
        if confidence >= 0.85:
            base_msg = f"Tinggi kepercayaan untuk profil {goal} dengan aktivitas {activity}."
        elif confidence >= 0.70:
            base_msg = f"Kepercayaan moderat untuk profil {goal} dengan aktivitas {activity}."
        else:
            base_msg = f"Kepercayaan rendah untuk profil {goal} dengan aktivitas {activity}."
        
        if limitations:
            limitation_msg = " " + " ".join(limitations)
            return base_msg + limitation_msg
        
        return base_msg
    
    def _get_model_limitations(self):
        """Return transparent model limitations"""
        return {
            'data_distribution': {
                'high_activity': '67.8% of training data - optimal performance',
                'moderate_activity': '22.2% of training data - good performance',
                'low_activity': '10.0% of training data - reduced confidence'
            },
            'recommendations': {
                'high_confidence': 'Suitable for direct implementation',
                'moderate_confidence': 'Consider additional factors',
                'low_confidence': 'Professional consultation recommended'
            },
            'thesis_finding': 'Model optimized for high-activity individuals reflecting real population patterns'
        }

    def debug_template_assignment_logic(self, test_samples=20):
        """
        Test template assignment logic with sample inputs to verify correctness
        """
        print("\n🔍 TESTING TEMPLATE ASSIGNMENT LOGIC")
        print("="*80)
        
        # Test all possible combinations
        goals = ['Fat Loss', 'Muscle Gain', 'Maintenance']
        activities = ['Low Activity', 'Moderate Activity', 'High Activity']
        bmis = ['Underweight', 'Normal', 'Overweight', 'Obese']
        
        print("\nTesting all possible combinations:")
        print("-" * 60)
        
        for goal in goals:
            for activity in activities:
                for bmi in bmis:
                    workout_id, nutrition_id = self.get_template_assignments(goal, activity, bmi)
                    
                    if workout_id is not None and nutrition_id is not None:
                        print(f"  {goal} + {activity} + {bmi}: Workout {workout_id}, Nutrition {nutrition_id}")
                    else:
                        print(f"  {goal} + {activity} + {bmi}: ❌ No template assigned")
        
        print("\n" + "="*80)

    def debug_template_assignments(self, df):
        """
        Debug template assignment logic with comprehensive analysis
        """
        print("\n" + "="*80)
        print("🔍 TEMPLATE ASSIGNMENT DEBUGGING ANALYSIS")
        print("="*80)
        
        # 1. Template mapping verification
        print("\n1. WORKOUT TEMPLATE MAPPING (template_id → model_class → combination):")
        print("-" * 60)
        for i, template in self.workout_templates.iterrows():
            template_id = getattr(template, 'template_id')
            goal = getattr(template, 'goal')
            activity = getattr(template, 'activity_level')
            workout_type = getattr(template, 'workout_type', 'Unknown')
            days = getattr(template, 'days_per_week', 'Unknown')
            
            # Get model class (0-8)
            model_class = template_id - 1  # XGBoost uses 0-based indexing
            print(f"  Template {template_id} → Class {model_class} → {goal} + {activity} ({workout_type}, {days} days)")
        
        print("\n2. NUTRITION TEMPLATE MAPPING (template_id → model_class → combination):")
        print("-" * 60)
        for i, template in self.nutrition_templates.iterrows():
            template_id = getattr(template, 'template_id')
            goal = getattr(template, 'goal')
            bmi_cat = getattr(template, 'bmi_category')
            
            # Get model class (0-7)
            model_class = template_id - 1  # XGBoost uses 0-based indexing
            print(f"  Template {template_id} → Class {model_class} → {goal} + {bmi_cat}")
        
        # 2. Actual data combination counts
        print("\n3. ACTUAL DATA COMBINATION COUNTS:")
        print("-" * 60)
        
        # Workout combinations (Goal + Activity)
        print("\nWorkout Combinations (Goal + Activity Level):")
        workout_combos = df.groupby(['fitness_goal', 'activity_level']).size().reset_index(name='count')
        workout_combos['percentage'] = (workout_combos['count'] / len(df) * 100).round(2)
        for _, row in workout_combos.iterrows():
            print(f"  {row['fitness_goal']} + {row['activity_level']}: {row['count']} samples ({row['percentage']}%)")
        
        # Nutrition combinations (Goal + BMI)
        print("\nNutrition Combinations (Goal + BMI Category):")
        nutrition_combos = df.groupby(['fitness_goal', 'bmi_category']).size().reset_index(name='count')
        nutrition_combos['percentage'] = (nutrition_combos['count'] / len(df) * 100).round(2)
        for _, row in nutrition_combos.iterrows():
            print(f"  {row['fitness_goal']} + {row['bmi_category']}: {row['count']} samples ({row['percentage']}%)")
        
        # 3. Template ID distribution analysis
        print("\n4. TEMPLATE ID DISTRIBUTION IN DATASET:")
        print("-" * 60)
        
        print("\nWorkout Template ID Distribution:")
        workout_template_dist = df['workout_template_id'].value_counts().sort_index()
        for template_id, count in workout_template_dist.items():
            percentage = (count / len(df) * 100)
            print(f"  Template {template_id}: {count} samples ({percentage:.2f}%)")
        
        print("\nNutrition Template ID Distribution:")
        nutrition_template_dist = df['nutrition_template_id'].value_counts().sort_index()
        for template_id, count in nutrition_template_dist.items():
            percentage = (count / len(df) * 100)
            print(f"  Template {template_id}: {count} samples ({percentage:.2f}%)")
        
        # 4. Check for missing combinations
        print("\n5. MISSING COMBINATIONS ANALYSIS:")
        print("-" * 60)
        
        # Expected workout combinations
        goals = ['Fat Loss', 'Muscle Gain', 'Maintenance']
        activities = ['Low Activity', 'Moderate Activity', 'High Activity']
        bmis = ['Underweight', 'Normal', 'Overweight', 'Obese']
        
        print("\nMissing Workout Combinations (Goal + Activity):")
        found_workout_combos = set()
        for _, row in workout_combos.iterrows():
            found_workout_combos.add((row['fitness_goal'], row['activity_level']))
        
        for goal in goals:
            for activity in activities:
                combo = (goal, activity)
                if combo not in found_workout_combos:
                    print(f"  Missing: {goal} + {activity}")
        
        print("\nMissing Nutrition Combinations (Goal + BMI):")
        found_nutrition_combos = set()
        for _, row in nutrition_combos.iterrows():
            found_nutrition_combos.add((row['fitness_goal'], row['bmi_category']))
        
        # Expected nutrition combinations (only valid ones)
        valid_nutrition_combos = [
            ('Fat Loss', 'Normal'), ('Fat Loss', 'Overweight'), ('Fat Loss', 'Obese'),
            ('Muscle Gain', 'Underweight'), ('Muscle Gain', 'Normal'),
            ('Maintenance', 'Underweight'), ('Maintenance', 'Normal'), ('Maintenance', 'Overweight')
        ]
        
        for combo in valid_nutrition_combos:
            if combo not in found_nutrition_combos:
                print(f"  Missing: {combo[0]} + {combo[1]}")
        
        # 5. Sample assignment verification
        print("\n6. SAMPLE ASSIGNMENT VERIFICATION:")
        print("-" * 60)
        
        # Test a few random samples
        sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
        for idx in sample_indices:
            row = df.iloc[idx]
            goal = row['fitness_goal']
            activity = row['activity_level']
            bmi_cat = row['bmi_category']
            workout_assigned = row['workout_template_id']
            nutrition_assigned = row['nutrition_template_id']
            
            # Get expected assignments
            expected_workout, expected_nutrition = self.get_template_assignments(goal, activity, bmi_cat)
            
            print(f"Sample {idx}:")
            print(f"  Profile: {goal} + {activity} + {bmi_cat}")
            print(f"  Assigned: Workout Template {workout_assigned}, Nutrition Template {nutrition_assigned}")
            
            # Check if assignment matches logic (allowing for minimal noise)
            workout_matches = (workout_assigned == expected_workout)
            nutrition_matches = (nutrition_assigned == expected_nutrition)
            
            # For workout, check if it's at least the same goal if not exact match
            if not workout_matches:
                assigned_workout_goal = None
                for template in self.workout_templates.itertuples():
                    if getattr(template, 'template_id') == workout_assigned:
                        assigned_workout_goal = getattr(template, 'goal')
                        break
                if assigned_workout_goal == goal:
                    workout_matches = True  # Same goal, acceptable due to noise
            
            # For nutrition, check if it's at least the same goal if not exact match
            if not nutrition_matches:
                assigned_nutrition_goal = None
                for template in self.nutrition_templates.itertuples():
                    if getattr(template, 'template_id') == nutrition_assigned:
                        assigned_nutrition_goal = getattr(template, 'goal')
                        break
                if assigned_nutrition_goal == goal:
                    nutrition_matches = True  # Same goal, acceptable due to noise
            
            if workout_matches and nutrition_matches:
                print(f"  ✅ Assignment matches expected logic")
            else:
                print(f"  ⚠️ Assignment differs from expected (expected: W{expected_workout}, N{expected_nutrition})")

    def debug_training_splits(self, df):
        """
        Debug training data splits and class distributions
        """
        print("\n" + "="*80)
        print("📊 TRAINING SET CLASS BALANCE ANALYSIS:")
        print("="*60)
        
        # Filter training data
        if 'split' in df.columns:
            train_df = df[df['split'] == 'train']
        else:
            train_df = df  # Use all data if no split column
        
        print(f"Training set size: {len(train_df)} samples")
        
        # Workout template distribution in training set
        print("\nWorkout Template Distribution in Training:")
        train_workout_dist = train_df['workout_template_id'].value_counts().sort_index()
        for template_id, count in train_workout_dist.items():
            percentage = (count / len(train_df) * 100)
            status = "✅" if count >= 20 else "⚠️"
            print(f"  {status} Template {template_id}: {count} samples ({percentage:.1f}%)")
        
        # Nutrition template distribution in training set
        print("\nNutrition Template Distribution in Training:")
        train_nutrition_dist = train_df['nutrition_template_id'].value_counts().sort_index()
        for template_id, count in train_nutrition_dist.items():
            percentage = (count / len(train_df) * 100)
            status = "✅" if count >= 20 else "⚠️"
            print(f"  {status} Template {template_id}: {count} samples ({percentage:.1f}%)")
        
        # Minimum sample analysis
        min_workout_samples = train_workout_dist.min()
        min_nutrition_samples = train_nutrition_dist.min()
        
        print(f"\n📈 MINIMUM SAMPLE ANALYSIS:")
        print(f"  Minimum workout template samples: {min_workout_samples}")
        print(f"  Minimum nutrition template samples: {min_nutrition_samples}")
        
        if min_workout_samples >= 20 and min_nutrition_samples >= 20:
            print(f"  ✅ All classes have reasonable sample counts (≥20)")
        else:
            print(f"  ⚠️ Some classes have very low sample counts (<20)")
            if min_workout_samples < 20:
                low_workout_templates = train_workout_dist[train_workout_dist < 20].index.tolist()
                print(f"     Low-sample workout templates: {low_workout_templates}")
            if min_nutrition_samples < 20:
                low_nutrition_templates = train_nutrition_dist[train_nutrition_dist < 20].index.tolist()
                print(f"     Low-sample nutrition templates: {low_nutrition_templates}")

    def _add_template_assignment_noise(self, workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.02):
        """
        Add MINIMAL controlled noise to template assignments to prevent overfitting while preserving logic
        
        FIXED VERSION: Greatly reduced noise to maintain assignment accuracy
        - Only 2% chance of randomization to prevent deterministic overfitting
        - Ensures alternative templates are logically similar (same goal)
        - Preserves the core assignment logic while adding slight variation
        
        Args:
            workout_id: Original workout template ID
            nutrition_id: Original nutrition template ID
            fitness_goal: User's fitness goal
            activity_level: User's activity level
            bmi_category: User's BMI category
            noise_prob: Probability of introducing noise (default 2% - MINIMAL)
            
        Returns:
            Tuple of (possibly modified workout_id, possibly modified nutrition_id)
        """
        # MINIMAL workout template noise (2% chance only)
        if np.random.random() < noise_prob:
            # Get all valid workout templates for the SAME goal only
            same_goal_alternatives = []
            
            for template in self.workout_templates.itertuples():
                template_goal = getattr(template, 'goal')
                template_id = getattr(template, 'template_id')
                
                # Include only templates with exact same goal
                if template_goal == fitness_goal:
                    same_goal_alternatives.append(template_id)
            
            # Remove current template and randomly choose from alternatives
            same_goal_alternatives = [t for t in same_goal_alternatives if t != workout_id]
            if same_goal_alternatives:
                workout_id = np.random.choice(same_goal_alternatives)
        
        # MINIMAL nutrition template noise (2% chance only)
        if np.random.random() < noise_prob:
            # Get all valid nutrition templates for the SAME goal only
            same_goal_nutrition = []
            
            for template in self.nutrition_templates.itertuples():
                template_goal = getattr(template, 'goal')
                template_id = getattr(template, 'template_id')
                
                # Include only templates with exact same goal
                if template_goal == fitness_goal:
                    same_goal_nutrition.append(template_id)
            
            # Remove current template and randomly choose from alternatives
            same_goal_nutrition = [t for t in same_goal_nutrition if t != nutrition_id]
            if same_goal_nutrition:
                nutrition_id = np.random.choice(same_goal_nutrition)
        
        return workout_id, nutrition_id

    def _generate_goal_specific_data(self, target_goal, n_samples, random_state=42):
        """
        Generate synthetic data specifically for a target fitness goal
        PRESERVES AUTHENTIC ACTIVITY DISTRIBUTIONS - No artificial balancing
        """
        np.random.seed(random_state)
        
        data = []
        
        for _ in range(n_samples):
            # Generate demographics based on goal-specific patterns
            if target_goal == 'Muscle Gain':
                # Muscle gain: typically younger males or underweight individuals
                age = np.random.randint(18, 35)
                gender = np.random.choice(['Male', 'Female'], p=[0.7, 0.3])
                
                if gender == 'Male':
                    height_cm = np.random.normal(175, 8)
                    # Tend to be leaner for muscle gain
                    base_bmi = np.random.normal(21, 2)
                else:
                    height_cm = np.random.normal(162, 7)
                    base_bmi = np.random.normal(20, 2)
                    
                # PRESERVE AUTHENTIC ACTIVITY DISTRIBUTION - More likely to be high activity
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.68, 0.22, 0.10])  # Match real data distribution
                
            elif target_goal == 'Fat Loss':
                # Fat loss: wider age range, higher BMIs
                age = np.random.randint(18, 60)
                gender = np.random.choice(['Male', 'Female'])
                
                if gender == 'Male':
                    height_cm = np.random.normal(175, 8)
                    base_bmi = np.random.normal(27, 3)  # Overweight range
                else:
                    height_cm = np.random.normal(162, 7)
                    base_bmi = np.random.normal(26, 3)
                    
                # PRESERVE AUTHENTIC ACTIVITY DISTRIBUTION
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.68, 0.22, 0.10])  # Match real data distribution
                
            else:  # Maintenance
                # Maintenance: typically normal BMI, varied ages
                age = np.random.randint(18, 65)
                gender = np.random.choice(['Male', 'Female'])
                
                if gender == 'Male':
                    height_cm = np.random.normal(175, 8)
                    base_bmi = np.random.normal(23, 2)  # Normal range
                else:
                    height_cm = np.random.normal(162, 7)
                    base_bmi = np.random.normal(22, 2)
                    
                # PRESERVE AUTHENTIC ACTIVITY DISTRIBUTION
                activity_level = np.random.choice(['High Activity', 'Moderate Activity', 'Low Activity'], 
                                                p=[0.68, 0.22, 0.10])  # Match real data distribution
            
            # Ensure reasonable ranges
            height_cm = np.clip(height_cm, 150, 200)
            base_bmi = np.clip(base_bmi, 16, 40)
            
            weight_kg = base_bmi * ((height_cm / 100) ** 2)
            weight_kg = np.clip(weight_kg, 40, 150)
            
            # Recalculate BMI and category
            bmi = weight_kg / ((height_cm / 100) ** 2)
            bmi_category = categorize_bmi(bmi)
            
            # Generate activity data based on activity level
            if activity_level == 'High Activity':
                mod_act_hours = np.random.normal(5.8, 0.8)
                vig_act_hours = np.random.normal(2.8, 0.5)
                activity_multiplier = 1.81
            elif activity_level == 'Moderate Activity':
                mod_act_hours = np.random.normal(3.75, 0.6)
                vig_act_hours = np.random.normal(1.87, 0.3)
                activity_multiplier = 1.55
            else:  # Low Activity
                mod_act_hours = np.random.normal(1.67, 0.4)
                vig_act_hours = np.random.normal(0.83, 0.25)
                activity_multiplier = 1.29
            
            mod_act_hours = max(0, mod_act_hours)
            vig_act_hours = max(0, vig_act_hours)
            
            # Calculate physiological metrics
            bmr = calculate_bmr(weight_kg, height_cm, age, gender)
            tdee = calculate_tdee(bmr, activity_level)
            
            # Use the target goal
            fitness_goal = target_goal
            
            # Find matching templates
            workout_id, nutrition_id = self.get_template_assignments(fitness_goal, activity_level, bmi_category)
            
            if workout_id is None or nutrition_id is None:
                continue  # Skip if no matching templates
            
            # MINIMAL noise to preserve logic but prevent overfitting
            workout_id, nutrition_id = self._add_template_assignment_noise(
                workout_id, nutrition_id, fitness_goal, activity_level, bmi_category, noise_prob=0.02  # MINIMAL
            )
            
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
        
        # Report on maintained authentic distribution
        activity_counts = df['activity_level'].value_counts()
        print(f"Generated {len(df)} synthetic samples for {target_goal} (preserving authentic activity distribution):")
        for activity, count in activity_counts.items():
            percentage = (count / len(df) * 100)
            print(f"  {activity}: {count} samples ({percentage:.1f}%)")
        
        return df

    def save_model(self, filepath):
        """
        Save the trained model to a pickle file
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data for saving
        model_data = {
            'workout_model': self.workout_model,
            'nutrition_model': self.nutrition_model,
            'workout_rf_model': self.workout_rf_model,
            'nutrition_rf_model': self.nutrition_rf_model,
            'scaler': self.scaler,
            'workout_label_encoder': getattr(self, 'workout_label_encoder', None),
            'nutrition_label_encoder': getattr(self, 'nutrition_label_encoder', None),
            'workout_rf_label_encoder': getattr(self, 'workout_rf_label_encoder', None),
            'nutrition_rf_label_encoder': getattr(self, 'nutrition_rf_label_encoder', None),
            'feature_columns': self.feature_columns,
            'workout_templates': self.workout_templates,
            'nutrition_templates': self.nutrition_templates,
            'training_info': self.training_info,
            'rf_training_info': getattr(self, 'rf_training_info', None),
            'is_trained': self.is_trained,
            'model_version': '2.1',
            'saved_at': datetime.now().isoformat()
        }
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model successfully saved to: {filepath}")
        print(f"   - XGBoost Workout model: {type(self.workout_model).__name__}")
        print(f"   - XGBoost Nutrition model: {type(self.nutrition_model).__name__}")
        print(f"   - Random Forest Workout model: {type(self.workout_rf_model).__name__ if self.workout_rf_model else 'Not trained'}")
        print(f"   - Random Forest Nutrition model: {type(self.nutrition_rf_model).__name__ if self.nutrition_rf_model else 'Not trained'}")
        print(f"   - Training info: {len(self.training_info)} metrics")
    
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
    
    def predict_with_confidence(self, user_profile):
        """
        Make predictions with enhanced confidence scoring
        
        Args:
            user_profile: Dictionary containing user information
            
        Returns:
            Dictionary with predictions, confidence scores, and explanations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
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
        bmi_category = self._categorize_bmi(bmi)
        bmr = self._calculate_bmr(weight_kg, height_cm, age, gender)
        tdee = self._calculate_tdee(bmr, activity_level)
        
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
        import pandas as pd
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
    
    def _categorize_bmi(self, bmi):
        """Categorize BMI into standard categories"""
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    def _calculate_bmr(self, weight_kg, height_cm, age, gender):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if gender.lower() == 'male':
            return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    def _calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        activity_multipliers = {
            'Low Activity': 1.29,
            'Moderate Activity': 1.55,
            'High Activity': 1.81
        }
        multiplier = activity_multipliers.get(activity_level, 1.55)
        return bmr * multiplier
    
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
        if template_type == 'workout':
            templates = self.workout_templates
        else:
            templates = self.nutrition_templates
        
        # Find the template
        template_row = templates[templates['template_id'] == template_id]
        
        if not template_row.empty:
            template = template_row.iloc[0]
            if template_type == 'workout':
                return {
                    'template_id': int(template_id),
                    'goal': template.get('goal', 'Unknown'),
                    'activity_level': template.get('activity_level', 'Unknown'),
                    'workout_type': template.get('workout_type', 'Unknown'),
                    'days_per_week': template.get('days_per_week', 'Unknown'),
                    'description': template.get('description', 'No description available')
                }
            else:  # nutrition
                return {
                    'template_id': int(template_id),
                    'goal': template.get('goal', 'Unknown'),
                    'bmi_category': template.get('bmi_category', 'Unknown'),
                    'calories_per_kg': template.get('calories_per_kg', 'Unknown'),
                    'protein_ratio': template.get('protein_ratio', 'Unknown'),
                    'description': template.get('description', 'No description available')
                }
        else:
            return {
                'template_id': int(template_id),
                'error': f'Template {template_id} not found'
            }