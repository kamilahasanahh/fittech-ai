#!/usr/bin/env python3
"""
XGFitness AI - Comprehensive Test Suite
Tests model training, predictions, and validates all requirements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from thesis_model import XGFitnessAIModel

def test_model_training():
    """Test model training with small dataset"""
    print("=== Testing Model Training ===")
    
    model = XGFitnessAIModel('../data')  # Use main data directory
    
    try:
        # Train with small dataset for testing
        training_data = model.create_training_dataset(
            real_data_file='../e267_Data on age, gender, height, weight, activity levels for each household member.txt',
            total_samples=500
        )
        
        # Train models
        results = model.train_models(training_data)
        
        print(f"✅ Training successful!")
        print(f"   Workout accuracy: {results['workout_accuracy']:.3f}")
        print(f"   Nutrition accuracy: {results['nutrition_accuracy']:.3f}")
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def test_predictions(model):
    """Test model predictions with various user types"""
    print("\n=== Testing Predictions ===")
    
    test_users = [
        {
            'name': 'Normal Weight Fat Loss',
            'age': 28, 'gender': 'Female', 'height': 165, 'weight': 60,
            'fitness_goal': 'Fat Loss', 'activity_level': 'Moderate Activity'
        },
        {
            'name': 'Underweight Muscle Gain',
            'age': 22, 'gender': 'Male', 'height': 180, 'weight': 55,
            'fitness_goal': 'Muscle Gain', 'activity_level': 'Low Activity'
        },
        {
            'name': 'Overweight Maintenance',
            'age': 35, 'gender': 'Female', 'height': 160, 'weight': 75,
            'fitness_goal': 'Maintenance', 'activity_level': 'High Activity'
        }
    ]
    
    for user in test_users:
        print(f"\nTesting: {user['name']}")
        user_data = {k: v for k, v in user.items() if k != 'name'}
        
        try:
            result = model.predict_with_confidence(user_data)
            
            if result.get('success'):
                print(f"  ✅ BMI: {result['user_metrics']['bmi']} ({result['user_metrics']['bmi_category']})")
                print(f"  ✅ Workout Template: {result['workout_recommendation']['template_id']}")
                print(f"  ✅ Nutrition Template: {result['nutrition_recommendation']['template_id']}")
                print(f"  ✅ Confidence: {result['confidence_scores']['confidence_level']}")
            else:
                print(f"  ❌ Prediction failed: {result.get('error')}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def validate_requirements():
    """Validate all implementation requirements"""
    print("\n=== Validating Requirements ===")
    
    model = XGFitnessAIModel()
    
    # Check feature engineering
    required_features = [
        'bmi_goal_interaction', 'age_activity_interaction', 'bmi_activity_interaction', 'age_goal_interaction',  # Interactions
        'bmr_per_kg', 'tdee_bmr_ratio',  # Metabolic ratios
        'bmi_deviation', 'weight_height_ratio',  # Health deviations
        'high_metabolism', 'very_active', 'young_adult'  # Boolean flags
    ]
    
    print("Feature Engineering:")
    for feature in required_features:
        if feature in model.feature_columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - MISSING")
    
    # Check model components
    print("\nModel Components:")
    print(f"  ✅ RandomizedSearchCV: {'✅' if 'RandomizedSearchCV' in str(model.train_models.__code__.co_names) else '❌'}")
    print(f"  ✅ StandardScaler: {'✅' if hasattr(model, 'scaler') else '❌'}")
    print(f"  ✅ Templates: {len(model.workout_templates)} workout, {len(model.nutrition_templates)} nutrition")
    
    # Check data requirements
    print("\nData Requirements:")
    try:
        real_data = model.load_real_data('../e267_Data on age, gender, height, weight, activity levels for each household member.txt')
        print(f"  ✅ Real data: {len(real_data)} samples (ages {real_data['age'].min()}-{real_data['age'].max()})")
        
        dummy_data = model.generate_dummy_data_with_confidence(n_samples=10)
        has_confidence = 'overall_confidence' in dummy_data.columns
        print(f"  ✅ Dummy data with confidence: {'✅' if has_confidence else '❌'}")
        
    except Exception as e:
        print(f"  ❌ Data loading error: {e}")

def main():
    """Run comprehensive test suite"""
    print("🏃‍♀️ XGFitness AI - Comprehensive Test Suite 🏃‍♂️")
    print("=" * 50)
    
    # Test training
    model = test_model_training()
    
    if model:
        # Test predictions
        test_predictions(model)
        
        # Save model for future use
        try:
            model.save_model('models/xgfitness_test_model.pkl')
            print(f"\n✅ Model saved successfully")
        except Exception as e:
            print(f"\n❌ Model save failed: {e}")
    
    # Validate requirements (independent of training)
    validate_requirements()
    
    print("\n🎉 Test suite completed!")

if __name__ == "__main__":
    main()
