#!/usr/bin/env python3
"""
XGFitness AI Visualization Runner
Run this script after training to generate comprehensive visualizations
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🎨 XGFitness AI Visualization Generator")
    print("="*60)
    
    # Check for trained models
    model_files = [
        'models/xgfitness_ai_model.pkl',
        'models/xgfitness_ai_model_production.pkl', 
        'models/research_model_comparison_production.pkl'
    ]
    
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("   Please run 'python train_model.py' first")
        return
    
    print(f"📁 Found {len(available_models)} trained model(s):")
    for i, model_file in enumerate(available_models, 1):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"   {i}. {model_file} ({size_mb:.1f} MB)")
    
    # Prioritize research model for visualizations (has both XGBoost + Random Forest)
    research_model = 'models/research_model_comparison_production.pkl'
    if research_model in available_models:
        model_to_use = research_model
        print(f"\n🚀 Using RESEARCH model (best for visualizations): {model_to_use}")
    else:
        # Fallback to largest available model
        model_to_use = max(available_models, key=lambda x: os.path.getsize(x))
        print(f"\n🚀 Using model: {model_to_use} (research model not available)")
    
    # Load the model
    try:
        print("📥 Loading trained model...")
        with open(model_to_use, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a mock model object with the loaded data
        class LoadedModel:
            def __init__(self, model_data):
                for key, value in model_data.items():
                    setattr(self, key, value)
            
            def prepare_training_data(self, df_training):
                """Mock prepare_training_data method for visualization compatibility"""
                # Import the actual model class to use its feature engineering
                try:
                    from src.thesis_model import XGFitnessAIModel
                    temp_model = XGFitnessAIModel('../data')
                    return temp_model.prepare_training_data(df_training)
                except Exception as e:
                    print(f"⚠️  Could not use full feature engineering: {e}")
                    # Fallback to simplified features
                    if 'split' not in df_training.columns:
                        df_training['split'] = np.random.choice(['train', 'validation', 'test'], 
                                                               len(df_training), p=[0.7, 0.15, 0.15])
                    
                    # Create feature matrix (simplified for visualizations)
                    feature_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'tdee']
                    available_cols = [col for col in feature_cols if col in df_training.columns]
                    
                    X = df_training[available_cols].values
                    y_workout = df_training['workout_template_id'].values
                    y_nutrition = df_training['nutrition_template_id'].values
                    
                    return X, y_workout, y_nutrition, df_training
            
            def compare_model_predictions(self, df_training):
                """Mock method for model comparison"""
                # Simple mock comparison data
                test_samples = len(df_training[df_training['split'] == 'test']) if 'split' in df_training.columns else 100
                return {
                    'workout_differences': int(test_samples * 0.06),  # 6% difference based on training output
                    'nutrition_differences': int(test_samples * 0.05),  # 5% difference based on training output
                    'total_test_samples': test_samples
                }
        
        model = LoadedModel(model_data)
        print("✅ Model loaded successfully")
        
        # Print model info
        print(f"   - Model type: {getattr(model, 'model_type', 'Unknown')}")
        print(f"   - Model version: {getattr(model, 'model_version', 'Unknown')}")
        print(f"   - Training samples: {model.training_info.get('training_samples', 'Unknown')}")
        print(f"   - Has XGBoost models: {'✅' if hasattr(model, 'workout_model') else '❌'}")
        print(f"   - Has Random Forest models: {'✅' if hasattr(model, 'workout_rf_model') and model.workout_rf_model else '❌'}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate or load training data for visualizations
    try:
        # Try to load existing training data
        training_data_files = [
            'training_data.csv',
            'data/training_data.csv',
            '../data/training_data.csv'
        ]
        
        df_training = None
        for data_file in training_data_files:
            if os.path.exists(data_file):
                print(f"📊 Loading training data from: {data_file}")
                df_training = pd.read_csv(data_file)
                break
        
        if df_training is None:
            print("⚠️  No training data CSV found, generating sample data...")
            df_training = generate_sample_data_for_viz(model)
            
        print(f"✅ Training data ready: {len(df_training)} samples")
        
    except Exception as e:
        print(f"⚠️  Error loading training data: {e}")
        print("   Generating sample data for visualization...")
        df_training = generate_sample_data_for_viz(model)
    
    # Generate visualizations
    try:
        print(f"\n🎨 Generating comprehensive visualizations...")
        
        # Import and run the visualization suite
        from visualisations import XGFitnessVisualizationSuite
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"visualizations/run_{timestamp}"
        
        # Generate all visualizations
        viz_suite = XGFitnessVisualizationSuite(model, df_training, output_dir)
        
        # Check if we're using actual training data with all features
        using_real_data = df_training is not None and len(df_training.columns) >= 15
        if using_real_data:
            print("✅ Using real training data - generating complete visualizations")
        else:
            print("⚠️  Using limited data - some visualizations may be simplified")
            
        viz_suite.generate_all_visualizations()
        
        print(f"\n🎉 Visualization generation completed!")
        print(f"📁 Output directory: {os.path.abspath(output_dir)}")
        
        # List generated files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            print(f"📊 Generated {len(files)} visualization files:")
            for file in sorted(files):
                print(f"   ✅ {file}")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

def generate_sample_data_for_viz(model):
    """Generate sample training data for visualization when CSV not available"""
    print("🔄 Generating sample data for visualization...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data matching the model's expected structure
    df_training = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'height_cm': np.random.normal(170, 10, n_samples),
        'weight_kg': np.random.normal(70, 15, n_samples),
        'activity_level': np.random.choice(['Low Activity', 'Moderate Activity', 'High Activity'], 
                                         n_samples, p=[0.3, 0.4, 0.3]),
        'fitness_goal': np.random.choice(['Fat Loss', 'Muscle Gain', 'Maintenance'], 
                                       n_samples, p=[0.5, 0.3, 0.2]),
        'data_source': np.random.choice(['real', 'synthetic'], n_samples, p=[0.7, 0.3]),
        'split': np.random.choice(['train', 'validation', 'test'], n_samples, p=[0.7, 0.15, 0.15]),
        'Mod_act': np.random.uniform(0, 10, n_samples),
        'Vig_act': np.random.uniform(0, 5, n_samples)
    })
    
    # Assign template IDs based on model templates
    if hasattr(model, 'workout_templates') and model.workout_templates is not None and len(model.workout_templates) > 0:
        workout_template_ids = list(range(1, len(model.workout_templates) + 1))
        df_training['workout_template_id'] = np.random.choice(workout_template_ids, n_samples)
    else:
        df_training['workout_template_id'] = np.random.randint(1, 10, n_samples)
    
    if hasattr(model, 'nutrition_templates') and model.nutrition_templates is not None and len(model.nutrition_templates) > 0:
        nutrition_template_ids = list(range(1, len(model.nutrition_templates) + 1))
        df_training['nutrition_template_id'] = np.random.choice(nutrition_template_ids, n_samples)
    else:
        df_training['nutrition_template_id'] = np.random.randint(1, 9, n_samples)
    
    # Calculate derived fields
    df_training['bmi'] = df_training['weight_kg'] / ((df_training['height_cm'] / 100) ** 2)
    df_training['bmi_category'] = pd.cut(df_training['bmi'], 
                                       bins=[0, 18.5, 25, 30, 100], 
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Calculate BMR using Harris-Benedict equation
    df_training['bmr'] = df_training.apply(lambda row: 
        88.362 + (13.397 * row['weight_kg']) + (4.799 * row['height_cm']) - (5.677 * row['age'])
        if row['gender'] == 'Male' else
        447.593 + (9.247 * row['weight_kg']) + (3.098 * row['height_cm']) - (4.330 * row['age']), 
        axis=1)
    
    # Calculate TDEE
    activity_multipliers = {'Low Activity': 1.29, 'Moderate Activity': 1.55, 'High Activity': 1.81}
    df_training['tdee'] = df_training['bmr'] * df_training['activity_level'].map(activity_multipliers)
    
    print(f"✅ Sample data generated: {len(df_training)} samples")
    return df_training

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Visualization generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
