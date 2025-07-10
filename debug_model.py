"""
Debug script to test model prediction directly
"""
import sys
import os
import traceback

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from thesis_model import XGFitnessAIModel

def test_model_directly():
    """Test the model directly without Flask"""
    try:
        print("Initializing model...")
        model = XGFitnessAIModel('data')  # Use main data directory
        
        # Try to load existing model
        model_path = 'backend/models/xgfitness_ai_model.pkl'
        if os.path.exists(model_path):
            model.load_model(model_path)
            print("Model loaded successfully")
        else:
            print("No existing model found")
            return
        
        # Test prediction
        test_data = {
            "age": 25,
            "gender": "Male",
            "height": 175,
            "weight": 70,
            "activity_level": "Moderate Activity",
            "fitness_goal": "Muscle Gain"
        }
        
        print("Making prediction...")
        result = model.predict_with_confidence(test_data)
        
        print("Prediction result:")
        print(f"Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"Workout template ID: {result.get('workout_recommendation', {}).get('template_id')}")
            print(f"Nutrition template ID: {result.get('nutrition_recommendation', {}).get('template_id')}")
            print(f"Overall confidence: {result.get('confidence_scores', {}).get('overall_confidence')}")
        else:
            print(f"Error: {result.get('error')}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_model_directly()
