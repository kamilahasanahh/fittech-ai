"""
Test model from backend directory
"""
import sys
import os

# Add src to path
sys.path.append('src')

from thesis_model import XGFitnessAIModel

def test_model():
    try:
        model = XGFitnessAIModel('../data')
        model.load_model('models/xgfitness_ai_model.pkl')
        
        test_data = {
            "age": 25,
            "gender": "Male", 
            "height": 175,
            "weight": 70,
            "activity_level": "Moderate Activity",
            "fitness_goal": "Muscle Gain"
        }
        
        result = model.predict_with_confidence(test_data)
        print("Prediction result:")
        print(f"Success: {result.get('success')}")
        if result.get('success'):
            print(f"Workout ID: {result['workout_recommendation']['template_id']}")
            print(f"Nutrition ID: {result['nutrition_recommendation']['template_id']}")
            print(f"Confidence: {result['confidence_scores']['overall_confidence']:.3f}")
        else:
            print(f"Error: {result.get('error')}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
