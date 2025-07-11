#!/usr/bin/env python3
"""
Test script for the comprehensive visualization suite
Demonstrates how to generate all visualizations after model training
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualizations'))

from thesis_model import XGFitnessAIModel
from model_visualization_suite import create_comprehensive_visualizations

def test_visualization_suite():
    """
    Test the complete visualization pipeline
    """
    print("🎨 Testing Comprehensive Visualization Suite")
    print("="*60)
    
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
        
        # Train all models (XGBoost + Random Forest)
        print("\n🚀 Training all models for visualization...")
        comprehensive_info = model.train_all_models(df_training, random_state=42)
        
        # Prepare data for visualization
        print("\n📈 Preparing visualization data...")
        model_results = prepare_visualization_data(model, comprehensive_info, df_training)
        
        # Generate all visualizations
        print("\n🎨 Generating comprehensive visualizations...")
        create_comprehensive_visualizations(model_results, df_training, 'visualizations/')
        
        print("\n✅ Visualization pipeline completed successfully!")
        print("\n📁 Generated visualizations in the following directories:")
        print("  - visualizations/comparisons/")
        print("  - visualizations/confusion_matrices/")
        print("  - visualizations/roc_curves/")
        print("  - visualizations/feature_importance/")
        print("  - visualizations/data_analysis/")
        print("  - visualizations/xgboost_analysis/")
        print("  - visualizations/diagnostics/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_visualization_data(model, comprehensive_info, df_training):
    """
    Prepare data structure for visualization suite
    
    Args:
        model: Trained XGFitnessAIModel instance
        comprehensive_info: Results from train_all_models
        df_training: Training dataset
        
    Returns:
        Dictionary with all data needed for visualizations
    """
    # Extract metrics from comprehensive info
    xgb_info = comprehensive_info.get('xgb_training_info', {})
    rf_info = comprehensive_info.get('rf_training_info', {})
    comparison_data = comprehensive_info.get('comparison_data', {})
    
    # Prepare metrics dictionary for performance comparison
    metrics_dict = {
        'workout': {
            'accuracy': {
                'XGBoost': xgb_info.get('workout_accuracy', 0),
                'Random Forest': rf_info.get('rf_workout_accuracy', 0)
            },
            'f1_weighted': {
                'XGBoost': xgb_info.get('workout_f1', 0),
                'Random Forest': rf_info.get('rf_workout_f1', 0)
            },
            'precision_weighted': {
                'XGBoost': xgb_info.get('workout_metrics', {}).get('precision_weighted', 0),
                'Random Forest': rf_info.get('rf_workout_metrics', {}).get('precision_weighted', 0)
            },
            'recall_weighted': {
                'XGBoost': xgb_info.get('workout_metrics', {}).get('recall_weighted', 0),
                'Random Forest': rf_info.get('rf_workout_metrics', {}).get('recall_weighted', 0)
            },
            'balanced_accuracy': {
                'XGBoost': xgb_info.get('workout_metrics', {}).get('balanced_accuracy', 0),
                'Random Forest': rf_info.get('rf_workout_metrics', {}).get('balanced_accuracy', 0)
            },
            'cohen_kappa': {
                'XGBoost': xgb_info.get('workout_metrics', {}).get('cohen_kappa', 0),
                'Random Forest': rf_info.get('rf_workout_metrics', {}).get('cohen_kappa', 0)
            }
        },
        'nutrition': {
            'accuracy': {
                'XGBoost': xgb_info.get('nutrition_accuracy', 0),
                'Random Forest': rf_info.get('rf_nutrition_accuracy', 0)
            },
            'f1_weighted': {
                'XGBoost': xgb_info.get('nutrition_f1', 0),
                'Random Forest': rf_info.get('rf_nutrition_f1', 0)
            },
            'precision_weighted': {
                'XGBoost': xgb_info.get('nutrition_metrics', {}).get('precision_weighted', 0),
                'Random Forest': rf_info.get('rf_nutrition_metrics', {}).get('precision_weighted', 0)
            },
            'recall_weighted': {
                'XGBoost': xgb_info.get('nutrition_metrics', {}).get('recall_weighted', 0),
                'Random Forest': rf_info.get('rf_nutrition_metrics', {}).get('recall_weighted', 0)
            },
            'balanced_accuracy': {
                'XGBoost': xgb_info.get('nutrition_metrics', {}).get('balanced_accuracy', 0),
                'Random Forest': rf_info.get('rf_nutrition_metrics', {}).get('balanced_accuracy', 0)
            },
            'cohen_kappa': {
                'XGBoost': xgb_info.get('nutrition_metrics', {}).get('cohen_kappa', 0),
                'Random Forest': rf_info.get('rf_nutrition_metrics', {}).get('cohen_kappa', 0)
            }
        }
    }
    
    # Prepare feature importance data
    importance_dict = {
        'XGBoost Workout': xgb_info.get('workout_feature_importance', {}),
        'XGBoost Nutrition': xgb_info.get('nutrition_feature_importance', {}),
        'Random Forest Workout': rf_info.get('rf_workout_feature_importance', {}),
        'Random Forest Nutrition': rf_info.get('rf_nutrition_feature_importance', {})
    }
    
    # Prepare class labels
    class_labels_dict = {
        'xgboost_workout': [f'Template {i}' for i in range(1, 10)],
        'xgboost_nutrition': [f'Template {i}' for i in range(1, 9)],
        'rf_workout': [f'Template {i}' for i in range(1, 10)],
        'rf_nutrition': [f'Template {i}' for i in range(1, 9)]
    }
    
    # Prepare number of classes
    n_classes_dict = {
        'xgboost_workout': 9,
        'xgboost_nutrition': 8,
        'rf_workout': 9,
        'rf_nutrition': 8
    }
    
    # For demonstration, create sample prediction data
    # In a real scenario, you would use actual test predictions
    test_mask = df_training['split'] == 'test'
    if test_mask.sum() > 0:
        test_data = df_training[test_mask]
        
        # Prepare features for test data
        X_test, y_w_test, y_n_test, df_test_enhanced = model.prepare_training_data(test_data)
        X_test_scaled = model.scaler.transform(X_test)
        
        # Get predictions from both models
        y_w_pred_xgb = model.workout_model.predict(X_test_scaled)
        y_n_pred_xgb = model.nutrition_model.predict(X_test_scaled)
        y_w_pred_rf = model.workout_rf_model.predict(X_test_scaled)
        y_n_pred_rf = model.nutrition_rf_model.predict(X_test_scaled)
        
        # Get prediction probabilities
        y_w_score_xgb = model.workout_model.predict_proba(X_test_scaled)
        y_n_score_xgb = model.nutrition_model.predict_proba(X_test_scaled)
        y_w_score_rf = model.workout_rf_model.predict_proba(X_test_scaled)
        y_n_score_rf = model.nutrition_rf_model.predict_proba(X_test_scaled)
        
        # Prepare true labels (encoded)
        y_w_true_encoded = model.workout_label_encoder.transform(y_w_test)
        y_n_true_encoded = model.nutrition_label_encoder.transform(y_n_test)
        
        y_true_dict = {
            'xgboost_workout': y_w_true_encoded,
            'xgboost_nutrition': y_n_true_encoded,
            'rf_workout': y_w_true_encoded,
            'rf_nutrition': y_n_true_encoded
        }
        
        y_pred_dict = {
            'xgboost_workout': y_w_pred_xgb,
            'xgboost_nutrition': y_n_pred_xgb,
            'rf_workout': y_w_pred_rf,
            'rf_nutrition': y_n_pred_rf
        }
        
        y_score_dict = {
            'xgboost_workout': y_w_score_xgb,
            'xgboost_nutrition': y_n_score_xgb,
            'rf_workout': y_w_score_rf,
            'rf_nutrition': y_n_score_rf
        }
    else:
        # Fallback if no test data
        y_true_dict = {}
        y_pred_dict = {}
        y_score_dict = {}
    
    # Prepare diagnostics data
    diagnostics_data = {
        'prediction_confidence': np.random.random(100),  # Sample confidence scores
        'cv_scores': [np.random.random(5) for _ in range(4)]  # Sample CV scores
    }
    
    # Compile all data
    model_results = {
        'comparison_data': metrics_dict,
        'feature_importance': importance_dict,
        'feature_names': model.feature_columns,
        'class_labels': class_labels_dict,
        'n_classes': n_classes_dict,
        'y_true': y_true_dict,
        'y_pred': y_pred_dict,
        'y_score': y_score_dict,
        'xgboost_model': model.workout_model,  # For XGBoost analysis
        'evals_result': None,  # Would contain evaluation results from training
        'diagnostics': diagnostics_data
    }
    
    return model_results

def create_sample_visualizations():
    """
    Create sample visualizations with dummy data for demonstration
    """
    print("🎨 Creating sample visualizations with dummy data...")
    
    # Create sample data
    sample_metrics = {
        'workout': {
            'accuracy': {'XGBoost': 0.85, 'Random Forest': 0.82},
            'f1_weighted': {'XGBoost': 0.84, 'Random Forest': 0.81},
            'precision_weighted': {'XGBoost': 0.85, 'Random Forest': 0.82},
            'recall_weighted': {'XGBoost': 0.84, 'Random Forest': 0.81},
            'balanced_accuracy': {'XGBoost': 0.83, 'Random Forest': 0.80},
            'cohen_kappa': {'XGBoost': 0.82, 'Random Forest': 0.79}
        },
        'nutrition': {
            'accuracy': {'XGBoost': 0.92, 'Random Forest': 0.89},
            'f1_weighted': {'XGBoost': 0.91, 'Random Forest': 0.88},
            'precision_weighted': {'XGBoost': 0.92, 'Random Forest': 0.89},
            'recall_weighted': {'XGBoost': 0.91, 'Random Forest': 0.88},
            'balanced_accuracy': {'XGBoost': 0.90, 'Random Forest': 0.87},
            'cohen_kappa': {'XGBoost': 0.89, 'Random Forest': 0.86}
        }
    }
    
    # Create sample feature importance as arrays (not dictionaries)
    sample_importance = {
        'XGBoost Workout': np.random.random(10),
        'Random Forest Workout': np.random.random(10)
    }
    
    sample_data = pd.DataFrame({
        'fitness_goal': np.random.choice(['Fat Loss', 'Muscle Gain', 'Maintenance'], 1000),
        'bmi_category': np.random.choice(['Underweight', 'Normal', 'Overweight', 'Obese'], 1000),
        'activity_level': np.random.choice(['Low Activity', 'Moderate Activity', 'High Activity'], 1000),
        'age': np.random.randint(18, 65, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'split': np.random.choice(['train', 'validation', 'test'], 1000)
    })
    
    # Import and use visualization functions
    from model_visualization_suite import (
        plot_performance_comparison, plot_feature_importance, 
        plot_data_distribution
    )
    
    # Create sample visualizations
    plot_performance_comparison(sample_metrics, ['XGBoost', 'Random Forest'])
    plot_feature_importance(sample_importance, [f'feature_{i}' for i in range(10)], 
                           ['XGBoost Workout', 'Random Forest Workout'])
    plot_data_distribution(sample_data)
    
    print("✅ Sample visualizations created!")

if __name__ == "__main__":
    print("🎨 XGFitness AI Visualization Suite Test")
    print("="*50)
    
    # Test with sample data first
    print("\n1. Testing with sample data...")
    create_sample_visualizations()
    
    # Test with real model training
    print("\n2. Testing with real model training...")
    success = test_visualization_suite()
    
    if success:
        print("\n🎉 All visualization tests completed successfully!")
        print("\n📋 Summary of generated visualizations:")
        print("  ✅ Performance comparison charts")
        print("  ✅ Confusion matrices")
        print("  ✅ ROC curves")
        print("  ✅ Feature importance plots")
        print("  ✅ Data distribution analysis")
        print("  ✅ XGBoost diagnostics")
        print("  ✅ Model comparison tables")
        print("\n📁 All plots saved as high-quality PNG files in visualizations/ directory")
    else:
        print("\n❌ Some visualization tests failed. Check the error messages above.") 