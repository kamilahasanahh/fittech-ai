## train_model.py

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from thesis_model import XGFitnessAIModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train the XGFitness AI models."""
    logger.info("=== Training XGFitness AI Models ===")
    logger.info("Template Structure:")
    logger.info("- Workout Templates: 9 (3 goals × 3 activity levels)")
    logger.info("- Nutrition Templates: 8 (3 goals × 4 BMI categories - specific combinations)")
    logger.info("- Total Templates: 17")
    logger.info("- Model: XGBoost with separate workout & nutrition models")
    logger.info("- Split: 70% real train / 15% real val / 15% logical test")
    
    try:
        # Create necessary directories
        directories = ['models', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("Created necessary directories")
        
        # Initialize and train models
        system = XGFitnessAIModel('../data')  # Use main data directory
        
        # Use maximum available real data with 70/15/15 split
        training_data = system.create_training_dataset(
            real_data_file='../e267_Data on age, gender, height, weight, activity levels for each household member.txt',
            total_samples=3657  # 70% train (2561) + 15% val (548) + 15% test (548)
        )
        training_results = system.train_models(training_data)
        
        # Save the models
        system.save_model('models/xgfitness_ai_model.pkl')
        
        logger.info("XGFitness AI model training completed successfully!")
        logger.info(f"Training samples: {len(training_data)}")
        logger.info(f"Models saved to: models/xgfitness_ai_model.pkl")
        
        # Log comprehensive performance metrics
        if 'workout_metrics' in training_results:
            workout_metrics = training_results['workout_metrics']
            logger.info(f"\n=== Workout Model Performance ===")
            logger.info(f"Accuracy: {workout_metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {workout_metrics['f1_weighted']:.4f}")
            logger.info(f"Balanced Accuracy: {workout_metrics['balanced_accuracy']:.4f}")
            logger.info(f"Top-2 Accuracy: {workout_metrics['top2_accuracy']:.4f}")
            logger.info(f"Cohen's Kappa: {workout_metrics['cohen_kappa']:.4f}")
            
        if 'nutrition_metrics' in training_results:
            nutrition_metrics = training_results['nutrition_metrics']
            logger.info(f"\n=== Nutrition Model Performance ===")
            logger.info(f"Accuracy: {nutrition_metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {nutrition_metrics['f1_weighted']:.4f}")
            logger.info(f"Balanced Accuracy: {nutrition_metrics['balanced_accuracy']:.4f}")
            logger.info(f"Top-2 Accuracy: {nutrition_metrics['top2_accuracy']:.4f}")
            logger.info(f"Cohen's Kappa: {nutrition_metrics['cohen_kappa']:.4f}")
        
        # Test with example user
        test_user = {
            'age': 25,
            'gender': 'Male',
            'height': 175,
            'weight': 70,
            'fitness_goal': 'Muscle Gain',
            'activity_level': 'Moderate Activity'
        }
        
        logger.info("\n=== Testing XGFitness AI Models ===")
        logger.info(f"Test user: {test_user['age']}yo {test_user['gender']}, {test_user['fitness_goal']}")
        
        recommendations = system.predict_with_confidence(test_user)
        workout_rec = recommendations['workout_recommendation']
        nutrition_rec = recommendations['nutrition_recommendation']
        confidence = recommendations['confidence_scores']
        
        logger.info(f"Confidence Level: {confidence['confidence_level']}")
        logger.info(f"Overall Confidence: {confidence['overall_confidence']:.3f}")
        logger.info(f"Workout Template ID: {workout_rec['template_id']}")
        logger.info(f"Nutrition Template ID: {nutrition_rec['template_id']}")
        
        # Test with multiple users to verify model functionality
        test_users = [
            {
                'age': 35,
                'gender': 'Female',
                'height': 165,
                'weight': 75,
                'fitness_goal': 'Fat Loss',
                'activity_level': 'Low Activity'
            },
            {
                'age': 45,
                'gender': 'Male',
                'height': 180,
                'weight': 80,
                'fitness_goal': 'Maintenance',
                'activity_level': 'High Activity'
            }
        ]
        
        logger.info("\n=== Additional Test Cases ===")
        for i, user in enumerate(test_users, 2):
            logger.info(f"\nTest User {i}: {user['age']}yo {user['gender']}, {user['fitness_goal']}, {user['activity_level']}")
            recommendations = system.predict_with_confidence(user)
            workout_rec = recommendations['workout_recommendation']
            nutrition_rec = recommendations['nutrition_recommendation']
            confidence = recommendations['confidence_scores']
            
            logger.info(f"  Workout: Template {workout_rec['template_id']} - Confidence: {confidence['confidence_level']}")
            logger.info(f"  Nutrition: Template {nutrition_rec['template_id']} - Overall: {confidence['overall_confidence']:.3f}")
        
        logger.info("\nXGFitness AI Training Complete")
        logger.info("Model ready for daily fitness predictions!")
        logger.info("Enhanced features and confidence scoring implemented!")
        
        # Display template distribution
        logger.info(f"\n=== Template Distribution Analysis ===")
        workout_distribution = training_data['workout_template_id'].value_counts().sort_index()
        nutrition_distribution = training_data['nutrition_template_id'].value_counts().sort_index()
        
        logger.info(f"Workout templates used: {len(workout_distribution)} out of 9")
        logger.info(f"Nutrition templates used: {len(nutrition_distribution)} out of 8")
        logger.info(f"Min workout samples per template: {workout_distribution.min()}")
        logger.info(f"Max workout samples per template: {workout_distribution.max()}")
        logger.info(f"Min nutrition samples per template: {nutrition_distribution.min()}")
        logger.info(f"Max nutrition samples per template: {nutrition_distribution.max()}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()