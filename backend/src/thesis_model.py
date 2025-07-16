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
import logging
warnings.filterwarnings('ignore')

# Helper functions
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
from sklearn.utils.multiclass import unique_labels

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
        self.workout_templates = getattr(self.template_manager, 'workout_templates', pd.DataFrame())
        if self.workout_templates is None:
            self.workout_templates = pd.DataFrame()
        self.nutrition_templates = getattr(self.template_manager, 'nutrition_templates', pd.DataFrame())
        if self.nutrition_templates is None:
            self.nutrition_templates = pd.DataFrame()
        
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
        
        # Feature columns - ENHANCED with engineered features
        self.feature_columns = [
            'age', 'gender_encoded', 'height_cm', 'weight_kg',
            'bmi', 'goal_encoded', 'activity_level_encoded',
            'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
            'BMR_per_weight', 'TDEE_BMR_ratio', 'activity_efficiency',
            'BMI_deviation', 'weight_height_ratio', 'metabolic_score',
            'high_metabolism', 'very_active', 'young_adult', 'optimal_BMI'
        ]
        
        # Training metadata
        self.training_info = {}
        self.rf_training_info = {}
        self.is_trained = False
        
        print(f"XGFitness AI initialized with {len(self.workout_templates)} workout and {len(self.nutrition_templates)} nutrition templates")
    
    def train_all_models(self, training_data, random_state=42):
        """
        Train XGBoost models for workout and nutrition template prediction.
        Returns a flat dictionary with training metrics and info for all splits and models.
        """
        import xgboost as xgb
        import numpy as np
        import pandas as pd
        from sklearn.utils.class_weight import compute_class_weight
        # Prepare features and targets
        feature_cols = [
            'age', 'gender_encoded', 'height_cm', 'weight_kg',
            'bmi', 'goal_encoded', 'activity_level_encoded',
            'BMI_Goal_interaction', 'Age_Activity_interaction', 'BMI_Activity_interaction', 'Age_Goal_interaction',
            'BMR_per_weight', 'TDEE_BMR_ratio', 'activity_efficiency',
            'BMI_deviation', 'weight_height_ratio', 'metabolic_score',
            'high_metabolism', 'very_active', 'young_adult', 'optimal_BMI'
        ]
        # If any feature is missing, add it as zeros
        for col in feature_cols:
            if col not in training_data.columns:
                training_data[col] = 0
        X = training_data[feature_cols].fillna(0) if training_data is not None else pd.DataFrame()
        split = training_data['split'] if training_data is not None and 'split' in training_data else pd.Series(dtype=str)
        # Only use template IDs for the training split
        y_workout = pd.Series(dtype=int)
        y_nutrition = pd.Series(dtype=int)
        if 'workout_template_id' in training_data.columns:
            y_workout = training_data.loc[split == 'train', 'workout_template_id'].astype(int)
        if 'nutrition_template_id' in training_data.columns:
            y_nutrition = training_data.loc[split == 'train', 'nutrition_template_id'].astype(int)
        # Split features
        X_train = X[split == 'train'] if not X.empty else pd.DataFrame()
        X_val = X[split == 'validation'] if not X.empty else pd.DataFrame()
        X_test = X[split == 'test'] if not X.empty else pd.DataFrame()
        y_w_train = y_workout if not y_workout.empty else pd.Series(dtype=int)
        y_n_train = y_nutrition if not y_nutrition.empty else pd.Series(dtype=int)
        # For validation, use the template IDs from the validation split
        if 'workout_template_id' in training_data.columns:
            y_w_val = training_data.loc[split == 'validation', 'workout_template_id'].astype(int)
        else:
            y_w_val = pd.Series(dtype=int)
        if 'nutrition_template_id' in training_data.columns:
            y_n_val = training_data.loc[split == 'validation', 'nutrition_template_id'].astype(int)
        else:
            y_n_val = pd.Series(dtype=int)
        # For test, use the template IDs from the test split
        if 'workout_template_id' in training_data.columns:
            y_w_test = training_data.loc[split == 'test', 'workout_template_id'].astype(int)
        else:
            y_w_test = pd.Series(dtype=int)
        if 'nutrition_template_id' in training_data.columns:
            y_n_test = training_data.loc[split == 'test', 'nutrition_template_id'].astype(int)
        else:
            y_n_test = pd.Series(dtype=int)
        # Fit the scaler on the training features
        if not X_train.empty:
            self.scaler.fit(X_train)
        # Guarantee pandas Series for y_workout and y_nutrition
        if not isinstance(y_workout, pd.Series):
            y_workout = pd.Series(y_workout)
        if not isinstance(y_nutrition, pd.Series):
            y_nutrition = pd.Series(y_nutrition)
        # Guarantee DataFrame type for len() safety
        if training_data is None or not isinstance(training_data, pd.DataFrame):
            training_data = pd.DataFrame()
        if X_train is None or not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame()
        if X_val is None or not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame()
        if X_test is None or not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame()
        # Map workout_template_id and nutrition_template_id to 0-based indices using TRAINING SET ONLY
        workout_id_map = {id_: idx for idx, id_ in enumerate(sorted(y_w_train.unique()))}
        nutrition_id_map = {id_: idx for idx, id_ in enumerate(sorted(y_n_train.unique()))}
        # Store reverse maps for decoding predictions later
        self.workout_id_reverse_map = {v: k for k, v in workout_id_map.items()}
        self.nutrition_id_reverse_map = {v: k for k, v in nutrition_id_map.items()}
        # Encode each split
        y_w_train_encoded = y_w_train.map(workout_id_map)
        y_w_val_encoded = y_w_val.map(workout_id_map)
        y_w_test_encoded = y_w_test.map(workout_id_map)
        y_n_train_encoded = y_n_train.map(nutrition_id_map)
        y_n_val_encoded = y_n_val.map(nutrition_id_map)
        y_n_test_encoded = y_n_test.map(nutrition_id_map)
        # --- Data Leakage Check (after encoding) ---
        for col in feature_cols:
            if col in training_data.columns:
                try:
                    if 'nutrition_template_id' in training_data.columns:
                        s1 = pd.Series(training_data[col])
                        s2 = pd.Series(training_data['nutrition_template_id'])
                        corr_nutrition = s1.corr(s2)
                    else:
                        corr_nutrition = 0
                except Exception:
                    corr_nutrition = 0
                try:
                    if 'workout_template_id' in training_data.columns:
                        s1 = pd.Series(training_data[col])
                        s2 = pd.Series(training_data['workout_template_id'])
                        corr_workout = s1.corr(s2)
                    else:
                        corr_workout = 0
                except Exception:
                    corr_workout = 0
                if abs(corr_nutrition) == 1.0:
                    print(f"⚠️ Data leakage warning: Feature '{col}' is perfectly correlated with nutrition_template_id!")
                if abs(corr_workout) == 1.0:
                    print(f"⚠️ Data leakage warning: Feature '{col}' is perfectly correlated with workout_template_id!")
        # --- Compute class weights for sample_weight ---
        def get_sample_weights(y):
            classes = np.unique(y)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, weights))
            return np.array([class_weight_dict[label] for label in y])
        w_workout = get_sample_weights(y_w_train)
        w_nutrition = get_sample_weights(y_n_train)
        # --- XGBoost hyperparameters: DRAMATICALLY INCREASED REGULARIZATION ---
        xgb_param_grid = {
            'n_estimators': [50],
            'max_depth': [2],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [50],
            'gamma': [0.2, 0.3, 0.5],
            'reg_alpha': [20.0],
            'reg_lambda': [50.0]
        }
        # Optimize XGBoost Workout Model
        xgb_workout_base = xgb.XGBClassifier(
            random_state=random_state, 
            n_jobs=-1, 
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=50
        )
        xgb_workout_search = RandomizedSearchCV(
            xgb_workout_base,
            xgb_param_grid,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        xgb_workout_search.fit(
            X_train, 
            y_w_train_encoded, 
            sample_weight=w_workout,
            eval_set=[(X_val, y_w_val_encoded)],
            verbose=False
        )
        xgb_workout = xgb_workout_search.best_estimator_
        print(f"✅ XGBoost Workout optimized - Best F1: {xgb_workout_search.best_score_:.4f}")
        # Optimize XGBoost Nutrition Model
        xgb_nutrition_base = xgb.XGBClassifier(
            random_state=random_state, 
            n_jobs=-1, 
            tree_method='hist',
            eval_metric='mlogloss',
            early_stopping_rounds=50
        )
        xgb_nutrition_search = RandomizedSearchCV(
            xgb_nutrition_base,
            xgb_param_grid,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        xgb_nutrition_search.fit(
            X_train, 
            y_n_train_encoded, 
            sample_weight=w_nutrition,
            eval_set=[(X_val, y_n_val_encoded)],
            verbose=False
        )
        xgb_nutrition = xgb_nutrition_search.best_estimator_
        print(f"✅ XGBoost Nutrition optimized - Best F1: {xgb_nutrition_search.best_score_:.4f}")
        # Train final models with best parameters
        xgb_workout.fit(X_train, y_w_train_encoded, sample_weight=w_workout, eval_set=[(X_val, y_w_val_encoded)], verbose=False)
        xgb_nutrition.fit(X_train, y_n_train_encoded, sample_weight=w_nutrition, eval_set=[(X_val, y_n_val_encoded)], verbose=False)
        # Store optimized models
        self.workout_model = xgb_workout
        self.nutrition_model = xgb_nutrition
        # Metrics helper
        def get_metrics(model, X, y):
            y_pred = model.predict(X)
            return {
                'accuracy': accuracy_score(y, y_pred),
                'f1': f1_score(y, y_pred, average='weighted'),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
            }
        # XGBoost metrics (workout)
        xgb_workout_train = get_metrics(xgb_workout, X_train, y_w_train_encoded)
        xgb_workout_val = get_metrics(xgb_workout, X_val, y_w_val_encoded)
        xgb_workout_test = get_metrics(xgb_workout, X_test, y_w_test_encoded)
        # XGBoost metrics (nutrition)
        xgb_nutrition_train = get_metrics(xgb_nutrition, X_train, y_n_train_encoded)
        xgb_nutrition_val = get_metrics(xgb_nutrition, X_val, y_n_val_encoded)
        xgb_nutrition_test = get_metrics(xgb_nutrition, X_test, y_n_test_encoded)
        # Ensure all are DataFrames before calling len()
        if training_data is None:
            training_data = pd.DataFrame()
        if X_train is None:
            X_train = pd.DataFrame()
        if X_val is None:
            X_val = pd.DataFrame()
        if X_test is None:
            X_test = pd.DataFrame()
        # Return flat dictionary for CLI printing
        return {
            # Sample counts
            'total_samples': len(training_data),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            # XGBoost Workout
            'workout_accuracy': xgb_workout_test['accuracy'],
            'workout_f1': xgb_workout_test['f1'],
            'workout_precision': xgb_workout_test['precision'],
            'workout_recall': xgb_workout_test['recall'],
            'workout_train_accuracy': xgb_workout_train['accuracy'],
            'workout_val_accuracy': xgb_workout_val['accuracy'],
            # XGBoost Nutrition
            'nutrition_accuracy': xgb_nutrition_test['accuracy'],
            'nutrition_f1': xgb_nutrition_test['f1'],
            'nutrition_precision': xgb_nutrition_test['precision'],
            'nutrition_recall': xgb_nutrition_test['recall'],
            'nutrition_train_accuracy': xgb_nutrition_train['accuracy'],
            'nutrition_val_accuracy': xgb_nutrition_val['accuracy'],
        }

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
            template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nutrition_templates.json')
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
    
    def load_real_data_balanced(self, train_csv='real_train.csv', val_csv='real_val.csv', test_csv='real_test.csv'):
        """
        Load and concatenate pre-split train, validation, and test CSVs. No cleaning, augmentation, or splitting is performed here.
        """
        import os
        import pandas as pd
        # Paths relative to backend/src
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, train_csv)
        val_path = os.path.join(base_dir, val_csv)
        test_path = os.path.join(base_dir, test_csv)
        print(f"Loading train from: {train_path}")
        print(f"Loading val from: {val_path}")
        print(f"Loading test from: {test_path}")
        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            print(f"❌ One or more split CSVs not found.")
            return pd.DataFrame()
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        df_train['split'] = 'train'
        df_val['split'] = 'validation'
        df_test['split'] = 'test'
        df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)
        print(f"Final dataset shape: {df_final.shape}")
        return df_final

    def save_model(self, file_path, include_research_models=False):
        """
        Save the trained models, encoders, and template manager to disk using pickle.
        If include_research_models is True, also save the Random Forest models.
        """
        model_data = {
            'workout_model': self.workout_model,
            'nutrition_model': self.nutrition_model,
            'scaler': self.scaler,
            'workout_encoder': self.workout_encoder,
            'nutrition_encoder': self.nutrition_encoder,
            'workout_label_encoder': self.workout_label_encoder,
            'nutrition_label_encoder': self.nutrition_label_encoder,
            'template_manager': self.template_manager,
            'workout_id_reverse_map': getattr(self, 'workout_id_reverse_map', None),
            'nutrition_id_reverse_map': getattr(self, 'nutrition_id_reverse_map', None),
        }
        if include_research_models:
            model_data.update({
                'workout_rf_model': self.workout_rf_model,
                'nutrition_rf_model': self.nutrition_rf_model,
                'rf_scaler': self.rf_scaler,
                'workout_rf_label_encoder': self.workout_rf_label_encoder,
                'nutrition_rf_label_encoder': self.nutrition_rf_label_encoder,
            })
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {file_path} (include_research_models={include_research_models})")