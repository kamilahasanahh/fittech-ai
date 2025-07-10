#!/usr/bin/env python3
"""Simple test script for the updated model - no pickle loading"""

from src.thesis_model import XGFitnessAIModel

def test_model_instantiation():
    """Test that the model can be instantiated with new import structure"""
    try:
        print("Creating XGFitness AI model...")
        model = XGFitnessAIModel(templates_dir='../data')
        print("✅ Model instantiated successfully")
        
        # Test template access
        templates = model.template_manager.get_all_templates()
        print(f"✅ Templates loaded: {len(templates['workout'])} workout, {len(templates['nutrition'])} nutrition")
        
        # Test template assignment
        workout_id, nutrition_id = model.get_template_assignments('Muscle Gain', 'Moderate Activity', 'Normal')
        print(f"✅ Template assignment works: Workout={workout_id}, Nutrition={nutrition_id}")
        
        # Get template details
        workout_template = model.template_manager.get_workout_template(workout_id)
        nutrition_template = model.template_manager.get_nutrition_template(nutrition_id)
        
        if workout_template and nutrition_template:
            print(f"✅ Template details retrieved successfully")
            print(f"   Workout keys: {list(workout_template.keys())}")
            print(f"   Nutrition keys: {list(nutrition_template.keys())}")
            print(f"   Workout: {workout_template.get('workout_type', 'N/A')}, {workout_template.get('days_per_week', 'N/A')} days/week")
            print(f"   Nutrition: {nutrition_template.get('caloric_intake', 'N/A')} cal multiplier, {nutrition_template.get('protein_per_kg', 'N/A')}g protein/kg")
        else:
            print("❌ Failed to retrieve template details")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_instantiation()
