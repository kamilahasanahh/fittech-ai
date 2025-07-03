import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import Config
from ..utils.calculations import calculate_bmr, calculate_tdee, categorize_bmi
from ..utils.validation import validate_user_profile

logger = logging.getLogger(__name__)


class ThesisAlignedXGBoostSystem:
    """
    XGBoost system aligned with thesis specifications:
    - 75 total templates: 15 workout + 60 nutrition
    - No experience level parameter
    - Exact TDEE multipliers from thesis
    - Production-ready with logging and error handling
    """
    
    def __init__(self):
        self.workout_model: Optional[xgb.XGBClassifier] = None
        self.nutrition_model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = []
        self.is_trained = False
        
        # Template structure matching thesis exactly
        self.workout_templates = self.create_workout_templates()  # 15 templates
        self.nutrition_templates = self.create_nutrition_templates()  # 60 templates
        
        logger.info("Initialized ThesisAlignedXGBoostSystem")
        logger.info(f"Workout templates: {len(self.workout_templates)}")
        logger.info(f"Nutrition templates: {len(self.nutrition_templates)}")
        
    def create_workout_templates(self) -> pd.DataFrame:
        """Create 15 workout templates: 3 goals × 5 activity levels"""
        templates = []
        template_id = 0
        
        goals = ['Muscle Gain', 'Fat Loss', 'Maintenance']
        activities = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active']
        
        base_values = {
            'Muscle Gain': {'volume': 14, 'frequency': 4, 'cardio': 90},
            'Fat Loss': {'volume': 10, 'frequency': 5, 'cardio': 180},
            'Maintenance': {'volume': 8, 'frequency': 3, 'cardio': 120}
        }
        
        activity_multipliers = {
            'Sedentary': 0.8,
            'Lightly Active': 0.9,
            'Moderately Active': 1.0,
            'Very Active': 1.1,
            'Extremely Active': 1.2
        }
        
        for goal in goals:
            for activity in activities:
                template = {
                    'template_id': template_id,
                    'fitness_goal': goal,
                    'activity_level': activity,
                    'training_volume': int(base_values[goal]['volume'] * activity_multipliers[activity]),
                    'training_frequency': base_values[goal]['frequency'],
                    'cardio_volume': int(base_values[goal]['cardio'] * activity_multipliers[activity])
                }
                templates.append(template)
                template_id += 1
                
        return pd.DataFrame(templates)
    
    def create_nutrition_templates(self) -> pd.DataFrame:
        """Create 60 nutrition templates: 3 goals × 4 BMI categories × 5 activity levels"""
        templates = []
        template_id = 0
        
        goals = ['Muscle Gain', 'Fat Loss', 'Maintenance']
        bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        activities = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active']
        
        # Base caloric adjustments (exact from thesis)
        caloric_adjustments = {
            'Muscle Gain': 1.10,
            'Fat Loss': 0.80,
            'Maintenance': 1.0
        }
        
        # BMI-based adjustments
        bmi_adjustments = {
            'Underweight': {'cal': 1.05, 'protein': 1.1},
            'Normal': {'cal': 1.0, 'protein': 1.0},
            'Overweight': {'cal': 0.95, 'protein': 1.1},
            'Obese': {'cal': 0.90, 'protein': 1.2}
        }
        
        # Protein requirements g/kg (thesis values)
        protein_requirements = {
            'Muscle Gain': 2.0,
            'Fat Loss': 2.2,
            'Maintenance': 1.6
        }
        
        # Carb requirements g/kg (thesis Table 2.6)
        carb_requirements = {
            'Muscle Gain': {'Sedentary': 4, 'Lightly Active': 5, 'Moderately Active': 6, 'Very Active': 7, 'Extremely Active': 7},
            'Fat Loss': {'Sedentary': 2, 'Lightly Active': 2.5, 'Moderately Active': 3, 'Very Active': 3.5, 'Extremely Active': 4},
            'Maintenance': {'Sedentary': 3, 'Lightly Active': 4, 'Moderately Active': 5, 'Very Active': 5, 'Extremely Active': 6}
        }
        
        for goal in goals:
            for bmi_cat in bmi_categories:
                for activity in activities:
                    # Calculate multipliers using thesis specifications
                    cal_mult = caloric_adjustments[goal] * bmi_adjustments[bmi_cat]['cal']
                    protein_mult = protein_requirements[goal] * bmi_adjustments[bmi_cat]['protein']
                    carb_per_kg = carb_requirements[goal][activity]
                    
                    template = {
                        'template_id': template_id,
                        'fitness_goal': goal,
                        'bmi_category': bmi_cat,
                        'activity_level': activity,
                        'caloric_multiplier': cal_mult,
                        'protein_per_kg': protein_mult,
                        'carbs_per_kg': carb_per_kg,
                        'fat_percentage': 0.25  # 25% of remaining calories
                    }
                    templates.append(template)
                    template_id += 1
                    
        return pd.DataFrame(templates)
    
    def generate_training_data(self, n_samples: int = 15000) -> pd.DataFrame:
        """Generate comprehensive training data with template assignments"""
        logger.info(f"Generating {n_samples} training samples...")
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            if i % 1000 == 0:
                logger.info(f"Generated {i}/{n_samples} samples")
                
            # Generate realistic user data (same logic as your original)
            age = max(18, min(65, int(np.random.normal(35, 12))))
            gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
            
            if gender == 'Male':
                height = max(150, min(200, np.random.normal(175, 8)))
            else:
                height = max(150, min(200, np.random.normal(162, 7)))
            
            # BMI-based weight generation
            bmi_dist = np.random.choice(['underweight', 'normal', 'overweight', 'obese'], p=[0.05, 0.55, 0.30, 0.10])
            
            if bmi_dist == 'underweight':
                target_bmi = np.random.uniform(16, 18.4)
            elif bmi_dist == 'normal':
                target_bmi = np.random.uniform(18.5, 24.9)
            elif bmi_dist == 'overweight':
                target_bmi = np.random.uniform(25, 29.9)
            else:
                target_bmi = np.random.uniform(30, 40)
            
            weight = target_bmi * (height / 100) ** 2
            
            # Activity level (age-influenced)
            age_factor = (65 - age) / 47
            activity_probs = [0.15 + 0.1 * (1 - age_factor), 0.25, 0.35 - 0.05 * (1 - age_factor), 0.20 * age_factor, 0.05 * age_factor]
            activity_probs = np.array(activity_probs) / np.sum(activity_probs)
            
            activity_level = np.random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active'], p=activity_probs)
            
            # Fitness goal (BMI-influenced)
            if target_bmi < 20:
                goal_probs = [0.7, 0.2, 0.1]
            elif target_bmi < 25:
                goal_probs = [0.4, 0.4, 0.2]
            elif target_bmi < 30:
                goal_probs = [0.2, 0.3, 0.5]
            else:
                goal_probs = [0.1, 0.1, 0.8]
                
            fitness_goal = np.random.choice(['Muscle Gain', 'Maintenance', 'Fat Loss'], p=goal_probs)
            
            # Target weight
            if fitness_goal == 'Muscle Gain':
                target_weight = weight + np.random.uniform(3, 9)
            elif fitness_goal == 'Fat Loss':
                max_loss = min(weight * 0.25, 20)
                target_weight = weight - np.random.uniform(5, max_loss)
            else:
                target_weight = weight + np.random.uniform(-2, 2)
            
            target_weight = max(45, min(150, target_weight))
            
            # Calculate metrics
            bmr = calculate_bmr(weight, height, age, gender)
            tdee = calculate_tdee(bmr, activity_level)
            bmi = weight / (height / 100) ** 2
            bmi_category = categorize_bmi(bmi)
            
            data.append({
                'age': int(age),
                'gender': gender,
                'height': round(height, 1),
                'weight': round(weight, 1),
                'target_weight': round(target_weight, 1),
                'fitness_goal': fitness_goal,
                'activity_level': activity_level,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1),
                'bmi': round(bmi, 1),
                'bmi_category': bmi_category
            })
        
        user_data = pd.DataFrame(data)
        
        # Assign template IDs
        workout_template_ids = []
        nutrition_template_ids = []
        
        for _, user in user_data.iterrows():
            # Find workout template
            workout_match = self.workout_templates[
                (self.workout_templates['fitness_goal'] == user['fitness_goal']) &
                (self.workout_templates['activity_level'] == user['activity_level'])
            ]
            workout_template_ids.append(workout_match.iloc[0]['template_id'])
            
            # Find nutrition template
            nutrition_match = self.nutrition_templates[
                (self.nutrition_templates['fitness_goal'] == user['fitness_goal']) &
                (self.nutrition_templates['bmi_category'] == user['bmi_category']) &
                (self.nutrition_templates['activity_level'] == user['activity_level'])
            ]
            nutrition_template_ids.append(nutrition_match.iloc[0]['template_id'])
        
        user_data['workout_template_id'] = workout_template_ids
        user_data['nutrition_template_id'] = nutrition_template_ids
        
        logger.info(f"Successfully generated {len(user_data)} training samples")
        return user_data
    
    def prepare_features(self, data: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """Prepare features for model training - no experience level"""
        feature_cols = [
            'age', 'height', 'weight', 'target_weight',
            'bmr', 'tdee', 'bmi', 'gender', 'fitness_goal', 'activity_level', 
            'bmi_category'
        ]
        
        X = data[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'fitness_goal', 'activity_level', 'bmi_category']
        
        for col in categorical_cols:
            if fit_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])
        
        # Scale numerical features
        numerical_cols = ['age', 'height', 'weight', 'target_weight', 'bmr', 'tdee', 'bmi']
        
        if fit_encoders:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        if fit_encoders:
            self.feature_names = X.columns.tolist()
        
        return X
    
    def train_models(self) -> pd.DataFrame:
        """Train separate XGBoost models for workout and nutrition recommendations"""
        logger.info("Training XGBoost models aligned with thesis specifications...")
        
        try:
            # Generate training data
            training_data = self.generate_training_data()
            
            # Prepare features
            X = self.prepare_features(training_data, fit_encoders=True)
            
            # Split data
            X_train, X_test, y_train_workout, y_test_workout = train_test_split(
                X, training_data['workout_template_id'], test_size=0.2, random_state=42
            )
            _, _, y_train_nutrition, y_test_nutrition = train_test_split(
                X, training_data['nutrition_template_id'], test_size=0.2, random_state=42
            )
            
            # Train workout model
            logger.info("Training workout recommendation model...")
            self.workout_model = xgb.XGBClassifier(**Config.MODEL_CONFIG.to_dict())
            self.workout_model.fit(X_train, y_train_workout)
            
            # Train nutrition model
            logger.info("Training nutrition recommendation model...")
            self.nutrition_model = xgb.XGBClassifier(**Config.MODEL_CONFIG.to_dict())
            self.nutrition_model.fit(X_train, y_train_nutrition)
            
            # Evaluate models
            logger.info("Evaluating models...")
            self.evaluate_models(X_test, y_test_workout, y_test_nutrition)
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            return training_data
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test_workout: pd.Series, y_test_nutrition: pd.Series) -> Dict[str, float]:
        """Evaluate both models using thesis metrics"""
        
        # Workout model evaluation
        workout_pred = self.workout_model.predict(X_test)
        workout_accuracy = accuracy_score(y_test_workout, workout_pred)
        
        # Nutrition model evaluation
        nutrition_pred = self.nutrition_model.predict(X_test)
        nutrition_accuracy = accuracy_score(y_test_nutrition, nutrition_pred)
        
        logger.info("=== Model Evaluation ===")
        logger.info(f"Workout Model Accuracy: {workout_accuracy:.4f}")
        logger.info(f"Nutrition Model Accuracy: {nutrition_accuracy:.4f}")
        logger.info(f"Average Accuracy: {(workout_accuracy + nutrition_accuracy) / 2:.4f}")
        
        return {
            'workout_accuracy': workout_accuracy,
            'nutrition_accuracy': nutrition_accuracy,
            'average_accuracy': (workout_accuracy + nutrition_accuracy) / 2
        }
    
    def predict_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Predict recommendations for a user profile"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        try:
            # Validate input
            validated_profile = validate_user_profile(user_profile)
            
            # Calculate derived metrics
            bmr = calculate_bmr(
                validated_profile['weight'], 
                validated_profile['height'], 
                validated_profile['age'], 
                validated_profile['gender']
            )
            
            tdee = calculate_tdee(bmr, validated_profile['activity_level'])
            bmi = validated_profile['weight'] / (validated_profile['height'] / 100) ** 2
            bmi_category = categorize_bmi(bmi)
            
            # Prepare input features
            input_data = pd.DataFrame([{
                'age': validated_profile['age'],
                'height': validated_profile['height'],
                'weight': validated_profile['weight'],
                'target_weight': validated_profile['target_weight'],
                'bmr': bmr,
                'tdee': tdee,
                'bmi': bmi,
                'gender': validated_profile['gender'],
                'fitness_goal': validated_profile['fitness_goal'],
                'activity_level': validated_profile['activity_level'],
                'bmi_category': bmi_category
            }])
            
            # Prepare features
            X = self.prepare_features(input_data, fit_encoders=False)
            
            # Make predictions
            workout_template_id = self.workout_model.predict(X)[0]
            nutrition_template_id = self.nutrition_model.predict(X)[0]
            
            # Get template details
            workout_template = self.workout_templates[
                self.workout_templates['template_id'] == workout_template_id
            ].iloc[0]
            
            nutrition_template = self.nutrition_templates[
                self.nutrition_templates['template_id'] == nutrition_template_id
            ].iloc[0]
            
            # Calculate specific nutrition values
            daily_calories = int(tdee * nutrition_template['caloric_multiplier'])
            daily_protein = int(validated_profile['weight'] * nutrition_template['protein_per_kg'])
            daily_carbs = int(validated_profile['weight'] * nutrition_template['carbs_per_kg'])
            
            # Calculate fat from remaining calories
            protein_calories = daily_protein * 4
            carb_calories = daily_carbs * 4
            remaining_calories = daily_calories - protein_calories - carb_calories
            daily_fat = max(20, int(remaining_calories / 9))
            
            # Structure the response (compatible with your existing frontend)
            result = {
                'exercise_recommendation': {
                    'template_id': int(workout_template_id),
                    'training_volume': int(workout_template['training_volume']),
                    'training_frequency': int(workout_template['training_frequency']),
                    'cardio_volume': int(workout_template['cardio_volume'])
                },
                'nutrition_recommendation': {
                    'template_id': int(nutrition_template_id),
                    'daily_calories': daily_calories,
                    'daily_protein': daily_protein,
                    'daily_carbs': daily_carbs,
                    'daily_fat': daily_fat
                },
                'calculated_metrics': {
                    'bmr': int(bmr),
                    'tdee': int(tdee),
                    'bmi': round(bmi, 1),
                    'bmi_category': bmi_category
                },
                'template_info': {
                    'total_templates': 75,
                    'workout_templates': 15,
                    'nutrition_templates': 60,
                    'thesis_aligned': True
                }
            }
            
            logger.info(f"Generated recommendations for user: {validated_profile['age']}yo {validated_profile['gender']}")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting recommendations: {str(e)}")
            raise
    
    def save_models(self, filepath_base: str = None) -> None:
        """Save the trained models and preprocessing objects"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained models")
            
        filepath_base = filepath_base or Config.MODEL_PATH
        
        try:
            Config.create_directories()
            
            # Save models
            joblib.dump(self.workout_model, f'{filepath_base}_workout.pkl')
            joblib.dump(self.nutrition_model, f'{filepath_base}_nutrition.pkl')
            
            # Save preprocessing objects
            joblib.dump(self.scaler, f'{filepath_base}_scaler.pkl')
            joblib.dump(self.label_encoders, f'{filepath_base}_encoders.pkl')
            
            # Save templates
            self.workout_templates.to_csv(f'{filepath_base}_workout_templates.csv', index=False)
            self.nutrition_templates.to_csv(f'{filepath_base}_nutrition_templates.csv', index=False)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'trained_at': datetime.now().isoformat(),
                'thesis_aligned': True,
                'total_templates': 75,
                'workout_templates': 15,
                'nutrition_templates': 60,
                'model_config': Config.MODEL_CONFIG.__dict__,
                'template_structure': {
                    'workout': '3 goals × 5 activity levels = 15',
                    'nutrition': '3 goals × 4 BMI categories × 5 activity levels = 60'
                }
            }
            
            with open(f'{filepath_base}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Thesis-aligned models saved to {filepath_base}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath_base: str = None) -> None:
        """Load the trained models and preprocessing objects"""
        filepath_base = filepath_base or Config.MODEL_PATH
        
        try:
            self.workout_model = joblib.load(f'{filepath_base}_workout.pkl')
            self.nutrition_model = joblib.load(f'{filepath_base}_nutrition.pkl')
            self.scaler = joblib.load(f'{filepath_base}_scaler.pkl')
            self.label_encoders = joblib.load(f'{filepath_base}_encoders.pkl')
            
            # Load templates
            self.workout_templates = pd.read_csv(f'{filepath_base}_workout_templates.csv')
            self.nutrition_templates = pd.read_csv(f'{filepath_base}_nutrition_templates.csv')
            
            # Load metadata
            with open(f'{filepath_base}_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.is_trained = True
            
            logger.info("Thesis-aligned models loaded successfully!")
            logger.info(f"Model trained at: {metadata['trained_at']}")
            logger.info(f"Total templates: {metadata['total_templates']}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise