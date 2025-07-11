#!/usr/bin/env python3
"""
XGFitness AI Model Training Script
Trains the enhanced XGFitness model with improved confidence scoring
"""

import os
import sys
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from thesis_model import XGFitnessAIModel

def main():
    """Train the XGFitness AI model with enhanced confidence scoring"""
    print("=" * 60)
    print("🏋️ XGFitness AI Model Training")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize model
    print("📚 Initializing XGFitness AI Model...")
    model = XGFitnessAIModel(templates_dir='../data')
    print("✅ Model initialized successfully")
    print()
    
    # Create training dataset
    print("🔍 Creating training dataset...")
    training_data = model.create_training_dataset(
        real_data_file='e267_Data on age, gender, height, weight, activity levels for each household member.txt',
        total_samples=2000,
        random_state=42
    )
    print(f"✅ Training dataset created: {len(training_data)} samples")
    print()
    
    # Train models
    print("🚀 Starting model training...")
    start_time = time.time()
    
    training_info = model.train_models(training_data, random_state=42)
    
    training_time = time.time() - start_time
    print(f"✅ Model training completed in {training_time:.2f} seconds")
    print()
    
    # Test enhanced confidence scoring system
    print("🎯 Testing Enhanced Confidence Scoring System...")
    try:
        model.test_confidence_improvements()
    except Exception as e:
        print(f"⚠️ Enhanced confidence testing failed: {e}")
        print("Continuing with basic confidence testing...")
    print()
    
    # Display training results
    print("📊 Training Results Summary:")
    print("-" * 40)
    print(f"Total samples: {training_info['total_samples']}")
    print(f"Training samples: {training_info['training_samples']}")
    print(f"Validation samples: {training_info['validation_samples']}")
    print(f"Test samples: {training_info['test_samples']}")
    print()
    print(f"Workout Model Performance:")
    print(f"  - Accuracy: {training_info['workout_accuracy']:.4f}")
    print(f"  - F1 Score: {training_info['workout_f1']:.4f}")
    print()
    print(f"Nutrition Model Performance:")
    print(f"  - Accuracy: {training_info['nutrition_accuracy']:.4f}")
    print(f"  - F1 Score: {training_info['nutrition_f1']:.4f}")
    print()
    
    # Save the trained model
    print("💾 Saving trained model...")
    model_path = 'models/xgfitness_ai_model.pkl'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
    print()
    
    # Test the enhanced confidence scoring system
    print("🧪 Testing Enhanced Confidence Scoring System...")
    model.test_confidence_improvements()
    print()
    
    # Test with real prediction examples
    print("🎯 Testing Real Predictions with Enhanced Confidence...")
    test_examples = [
        {
            'name': 'Typical Indonesian Male',
            'data': {
                'age': 28,
                'gender': 'Male',
                'height': 170,
                'weight': 75,
                'activity_level': 'Moderate Activity',
                'fitness_goal': 'Fat Loss'
            }
        },
        {
            'name': 'Young Indonesian Female',
            'data': {
                'age': 24,
                'gender': 'Female',
                'height': 160,
                'weight': 55,
                'activity_level': 'Low Activity',
                'fitness_goal': 'Maintenance'
            }
        }
    ]
    
    for example in test_examples:
        print(f"\n🧑‍🤝‍🧑 Testing: {example['name']}")
        try:
            result = model.predict_with_confidence(example['data'])
            if result['success']:
                conf = result['confidence_scores']
                print(f"  📊 Confidence Scores:")
                print(f"    Overall: {conf['overall_confidence']:.3f} ({conf['confidence_level']})")
                print(f"    Workout: {conf['workout_confidence']:.3f}")
                print(f"    Nutrition: {conf['nutrition_confidence']:.3f}")
                print(f"  💬 Message: {conf['confidence_message']}")
                
                # Show actual recommendations
                workout = result.get('workout_recommendation', {})
                nutrition = result.get('nutrition_recommendation', {})
                
                print(f"  🏋️ Workout Plan:")
                print(f"    Type: {workout.get('workout_type', 'N/A')}")
                print(f"    Days/Week: {workout.get('days_per_week', 'N/A')}")
                print(f"    Schedule: {workout.get('workout_schedule', 'N/A')}")
                print(f"    Sets per Exercise: {workout.get('sets_per_exercise', 'N/A')}")
                print(f"    Exercises per Session: {workout.get('exercises_per_session', 'N/A')}")
                print(f"    Cardio Minutes/Day: {workout.get('cardio_minutes_per_day', 'N/A')}")
                print(f"    Cardio Sessions/Day: {workout.get('cardio_sessions_per_day', 'N/A')}")
                
                print(f"  🥗 Nutrition Plan:")
                # Calculate actual values based on user data
                user_weight = example['data']['weight']
                calories = nutrition.get('target_calories', 0)  # Fixed: use target_calories
                protein_per_kg = nutrition.get('protein_per_kg', 0)
                carbs_per_kg = nutrition.get('carbs_per_kg', 0)
                fat_per_kg = nutrition.get('fat_per_kg', 0)
                
                protein_grams = protein_per_kg * user_weight
                carbs_grams = carbs_per_kg * user_weight
                fat_grams = fat_per_kg * user_weight
                
                print(f"    Caloric Intake: {calories:.0f} kcal")
                print(f"    Protein: {protein_grams:.0f}g ({protein_per_kg:.1f}g/kg)")
                print(f"    Carbohydrates: {carbs_grams:.0f}g ({carbs_per_kg:.1f}g/kg)")
                print(f"    Fats: {fat_grams:.0f}g ({fat_per_kg:.1f}g/kg)")
            else:
                print(f"  ❌ Error: {result['error']}")
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_time:.2f} seconds")
    print()
    print("🚀 Your model is now ready with enhanced confidence scoring!")
    print("   - More realistic confidence scores")
    print("   - Multi-factor confidence calculation")
    print("   - User-friendly explanations in Indonesian")
    print("   - Actionable improvement tips")
    print()
    print("📱 You can now restart your Flask app to use the new model:")
    print("   python app.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
