#!/usr/bin/env python3
"""Quick test script for the optimized model"""

import pickle
from src.thesis_model import XGFitnessAIModel

def test_model():
    # Load the trained model
    try:
        with open('models/xgfitness_ai_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test prediction
    test_user = {
        'age': 25,
        'gender': 'Male',
        'height': 180,
        'weight': 75,
        'activity_level': 'Moderate Activity',
        'fitness_goal': 'Muscle Gain'
    }
    
    try:
        result = model.predict_with_confidence(test_user)
        print("✅ Prediction successful")
        print(f"Workout Template: {result['workout_recommendation']['template_id']}")
        print(f"Nutrition Template: {result['nutrition_recommendation']['template_id']}")
        print(f"Overall Confidence: {result['confidence_scores']['overall_confidence']:.3f}")
        
        # Show template details
        workout = result['workout_recommendation']
        nutrition = result['nutrition_recommendation']
        
        print(f"\n📊 Workout: {workout['workout_type']} - {workout['days_per_week']} days/week")
        print(f"🍎 Nutrition: {nutrition['target_calories']} calories, {nutrition['target_protein']}g protein")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    test_model()
