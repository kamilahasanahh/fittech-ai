"""
Test script to verify backend API functionality
"""
import requests
import json

def test_backend():
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print("Health check:", response.status_code)
        if response.status_code == 200:
            print("Health response:", json.dumps(response.json(), indent=2))
        else:
            print("Health error:", response.text)
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test prediction endpoint
    test_data = {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity",
        "fitness_goal": "Muscle Gain"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        print("\nPrediction test:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("Prediction success:", result.get('success', False))
            if result.get('success'):
                print("Workout template ID:", result.get('workout_recommendation', {}).get('template_id'))
                print("Nutrition template ID:", result.get('nutrition_recommendation', {}).get('template_id'))
                print("Overall confidence:", result.get('confidence_scores', {}).get('overall_confidence'))
            else:
                print("Prediction error:", result.get('error'))
        else:
            print("Prediction error:", response.text)
    except Exception as e:
        print(f"Prediction test failed: {e}")

if __name__ == "__main__":
    test_backend()
