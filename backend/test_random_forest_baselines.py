#!/usr/bin/env python3
"""
Test script for Random Forest baseline models in XGFitness AI
Demonstrates training both XGBoost and Random Forest models for academic comparison
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from thesis_model import XGFitnessAIModel

def test_random_forest_baselines():
    """
    Test the Random Forest baseline functionality
    """
    print("🌲 Testing Random Forest Baseline Models for Academic Comparison")
    print("="*80)
    
    # Initialize the model
    model = XGFitnessAIModel(templates_dir='../data')
    
    try:
        # Create training dataset
        print("\n📊 Creating training dataset...")
        df_training = model.create_training_dataset(
            real_data_file='e267_Data on age, gender, height, weight, activity levels for each household member.txt',
            equal_goal_distribution=True,
            splits=(0.70, 0.15, 0.15),
            random_state=42
        )
        
        print(f"✅ Training dataset created with {len(df_training)} samples")
        print(f"   - Training: {len(df_training[df_training['split'] == 'train'])} samples")
        print(f"   - Validation: {len(df_training[df_training['split'] == 'validation'])} samples")
        print(f"   - Test: {len(df_training[df_training['split'] == 'test'])} samples")
        
        # Train all models (XGBoost + Random Forest)
        print("\n🚀 Training all models for comprehensive comparison...")
        comprehensive_info = model.train_all_models(df_training, random_state=42)
        
        # Save the trained models
        print("\n💾 Saving trained models...")
        model.save_model('models/xgfitness_ai_with_baselines.pkl')
        
        # Test model comparison
        print("\n📊 Testing model comparison functionality...")
        comparison_data = model.compare_model_performance()
        
        # Test prediction with both models
        print("\n🔮 Testing predictions with both models...")
        test_user = {
            'age': 25,
            'gender': 'Male',
            'height_cm': 175,
            'weight_kg': 70,
            'fitness_goal': 'Muscle Gain',
            'activity_level': 'High Activity',
            'Mod_act': 5.0,
            'Vig_act': 2.5
        }
        
        # XGBoost prediction
        xgb_prediction = model.predict_with_confidence(test_user)
        
        # Random Forest prediction (if available)
        if model.workout_rf_model and model.nutrition_rf_model:
            # Create user data for Random Forest prediction
            user_data = {
                'age': test_user['age'],
                'gender': test_user['gender'],
                'height_cm': test_user['height_cm'],
                'weight_kg': test_user['weight_kg'],
                'bmi': test_user['weight_kg'] / ((test_user['height_cm'] / 100) ** 2),
                'bmi_category': model._categorize_bmi(test_user['weight_kg'] / ((test_user['height_cm'] / 100) ** 2)),
                'bmr': model._calculate_bmr(test_user['weight_kg'], test_user['height_cm'], test_user['age'], test_user['gender']),
                'tdee': model._calculate_tdee(model._calculate_bmr(test_user['weight_kg'], test_user['height_cm'], test_user['age'], test_user['gender']), test_user['activity_level']),
                'activity_level': test_user['activity_level'],
                'fitness_goal': test_user['fitness_goal'],
                'Mod_act': test_user['Mod_act'],
                'Vig_act': test_user['Vig_act']
            }
            
            # Prepare features for Random Forest
            user_df = pd.DataFrame([user_data])
            user_df_enhanced = model.create_enhanced_features(user_df)
            X_user = user_df_enhanced[model.feature_columns].fillna(0)
            X_user_scaled = model.scaler.transform(X_user)
            
            # Random Forest predictions
            rf_workout_pred = model.workout_rf_model.predict(X_user_scaled)[0]
            rf_nutrition_pred = model.nutrition_rf_model.predict(X_user_scaled)[0]
            
            # Convert back to template IDs
            rf_workout_template_id = model.workout_rf_label_encoder.inverse_transform([rf_workout_pred])[0]
            rf_nutrition_template_id = model.nutrition_rf_label_encoder.inverse_transform([rf_nutrition_pred])[0]
            
            print(f"\n📋 Prediction Comparison for Test User:")
            print(f"User Profile: {test_user['age']}yo {test_user['gender']}, {test_user['height_cm']}cm, {test_user['weight_kg']}kg")
            print(f"Goal: {test_user['fitness_goal']}, Activity: {test_user['activity_level']}")
            print(f"\nXGBoost Predictions:")
            print(f"  Workout Template: {xgb_prediction['predictions']['workout_template_id']}")
            print(f"  Nutrition Template: {xgb_prediction['predictions']['nutrition_template_id']}")
            print(f"  XGBoost Confidence: {xgb_prediction['enhanced_confidence']['confidence_score']}")
            print(f"\nRandom Forest Predictions:")
            print(f"  Workout Template: {rf_workout_template_id}")
            print(f"  Nutrition Template: {rf_nutrition_template_id}")
            
            # Check if predictions match
            workout_match = xgb_prediction['predictions']['workout_template_id'] == rf_workout_template_id
            nutrition_match = xgb_prediction['predictions']['nutrition_template_id'] == rf_nutrition_template_id
            
            print(f"\nPrediction Agreement:")
            print(f"  Workout Templates: {'✅ Match' if workout_match else '❌ Different'}")
            print(f"  Nutrition Templates: {'✅ Match' if nutrition_match else '❌ Different'}")
        
        print("\n✅ Random Forest baseline testing completed successfully!")
        print("\n🎓 Academic Comparison Summary:")
        print("="*50)
        print("The system now provides:")
        print("1. XGBoost models (primary recommendation system)")
        print("2. Random Forest baseline models (academic comparison)")
        print("3. Comprehensive performance comparison tables")
        print("4. Feature importance analysis for both models")
        print("5. Prediction agreement analysis")
        print("\nThis enables rigorous academic evaluation of XGBoost vs Random Forest")
        print("performance for fitness recommendation systems.")
        
        return comprehensive_info
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the test
    result = test_random_forest_baselines()
    
    if result:
        print(f"\n🎉 Test completed successfully!")
        print(f"Models saved to: models/xgfitness_ai_with_baselines.pkl")
    else:
        print(f"\n❌ Test failed!")
        sys.exit(1) 