"""
Enhanced Flask API for XGFitness AI system
Provides comprehensive fitness recommendations with confidence scoring
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
from datetime import datetime
import logging
import numpy as np

# Add src directory to path for imports
backend_dir = os.path.dirname(__file__)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

# Also add to PYTHONPATH for pickle
import sys
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from thesis_model import XGFitnessAIModel
from validation import validate_api_request_data, create_validation_summary, get_validation_rules
from calculations import calculate_bmr, calculate_tdee, categorize_bmi
from templates import get_template_manager

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

def initialize_model():
    """Initialize or load the XGFitness AI model"""
    global model
    
    try:
        model = XGFitnessAIModel('../data')  # Use main data directory
        
        # Try to load existing model
        model_path = 'models/xgfitness_ai_model.pkl'
        if os.path.exists(model_path):
            model.load_model(model_path)
            logger.info("Loaded existing trained model")
        else:
            # Train new model if none exists
            logger.info("No existing model found, training new model...")
            os.makedirs('models', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            training_data = model.create_training_dataset(total_samples=2000)
            model.train_models(training_data)
            model.save_model(model_path)
            
            # Save templates using template manager
            model.template_manager.save_all_templates()
            
            logger.info("Model trained and saved successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'FitTech AI API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Get fitness recommendations',
            '/health': 'GET - Check API health',
            '/templates': 'GET - Get available templates',
            '/validation-rules': 'GET - Get input validation rules',
            '/calculate-metrics': 'POST - Calculate BMI, BMR, TDEE'
        },
        'documentation': 'Send POST request to /predict with user data'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model
    
    status = {
        'status': 'healthy' if model and model.is_trained else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'model_trained': model.is_trained if model else False
    }
    
    if model and model.is_trained:
        # Convert NumPy types to native Python types for JSON serialization
        training_info = convert_numpy_types(model.training_info)
        status.update(training_info)
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for fitness recommendations
    
    Expected JSON input:
    {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity",
        "fitness_goal": "Muscle Gain"
    }
    """
    global model
    
    try:
        # Check if model is available
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available. Please try again later.',
                'code': 'MODEL_UNAVAILABLE'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'code': 'NO_DATA'
            }), 400
        
        # Validate input data
        try:
            clean_data = validate_api_request_data(data)
            validation_warnings = []
        except Exception as validation_error:
            return jsonify({
                'success': False,
                'error': 'Invalid input data',
                'validation_error': str(validation_error),
                'code': 'VALIDATION_ERROR'
            }), 400
        
        # Make prediction
        prediction_result = model.predict_with_confidence(clean_data)
        
        # Add validation warnings if any
        if validation_warnings:
            prediction_result['validation_warnings'] = validation_warnings
        
        # Add request metadata
        prediction_result['request_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model.training_info.get('training_date', 'Unknown'),
            'api_version': '1.0.0'
        }
        
        # Convert NumPy types to native Python types for JSON serialization
        prediction_result = convert_numpy_types(prediction_result)
        return jsonify(prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction',
            'code': 'PREDICTION_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/calculate-metrics', methods=['POST'])
def calculate_metrics():
    """
    Calculate basic fitness metrics (BMI, BMR, TDEE) without full recommendations
    
    Expected JSON input:
    {
        "age": 25,
        "gender": "Male",
        "height": 175,
        "weight": 70,
        "activity_level": "Moderate Activity"
    }
    """
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Basic validation for required fields
        required_fields = ['age', 'gender', 'height', 'weight', 'activity_level']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Calculate metrics
        age = int(data['age'])
        gender = data['gender']
        height = float(data['height'])
        weight = float(data['weight'])
        activity_level = data['activity_level']
        
        # Validate ranges
        if not (18 <= age <= 100):
            return jsonify({'success': False, 'error': 'Age must be between 18 and 100'}), 400
        if not (120 <= height <= 250):
            return jsonify({'success': False, 'error': 'Height must be between 120 and 250 cm'}), 400
        if not (30 <= weight <= 300):
            return jsonify({'success': False, 'error': 'Weight must be between 30 and 300 kg'}), 400
        
        bmi = weight / ((height / 100) ** 2)
        bmi_category = categorize_bmi(bmi)
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        
        return jsonify({
            'success': True,
            'metrics': {
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
                'bmr': round(bmr, 1),
                'tdee': round(tdee, 1)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error calculating metrics'
        }), 500

@app.route('/templates', methods=['GET'])
def get_templates():
    """Get all available workout and nutrition templates"""
    global model
    
    try:
        if not model:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        return jsonify({
            'success': True,
            'workout_templates': model.template_manager.workout_templates.to_dict('records'),
            'nutrition_templates': model.template_manager.nutrition_templates.to_dict('records'),
            'template_count': {
                'workout': len(model.template_manager.workout_templates),
                'nutrition': len(model.template_manager.nutrition_templates)
            }
        })
        
    except Exception as e:
        logger.error(f"Templates error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving templates'
        }), 500

@app.route('/validation-rules', methods=['GET'])
def get_validation_rules_endpoint():
    """Get validation rules for frontend form validation"""
    try:
        rules = get_validation_rules()
        return jsonify({
            'success': True,
            'validation_rules': rules
        })
    except Exception as e:
        logger.error(f"Validation rules error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving validation rules'
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information and training statistics"""
    global model
    
    try:
        if not model or not model.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 503
        
        return jsonify({
            'success': True,
            'model_info': model.training_info,
            'feature_count': len(model.feature_columns),
            'template_counts': {
                'workout_templates': len(model.template_manager.workout_templates),
                'nutrition_templates': len(model.template_manager.nutrition_templates)
            }
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error retrieving model information'
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the model with new parameters (admin endpoint)
    """
    global model
    
    try:
        # This could be protected with authentication in production
        data = request.get_json() or {}
        
        n_samples = data.get('n_samples', 2000)
        
        if not (500 <= n_samples <= 10000):
            return jsonify({
                'success': False,
                'error': 'n_samples must be between 500 and 10000'
            }), 400
        
        logger.info(f"Starting model retraining with {n_samples} samples...")
        
        # Reinitialize model
        model = XGFitnessAIModel('../data')  # Use main data directory
        
        # Generate new training data
        training_data = model.create_training_dataset(
            real_data_file='../e267_Data on age, gender, height, weight, activity levels for each household member.txt',
            total_samples=n_samples
        )
        
        # Train model
        training_results = model.train_models(training_data)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model.save_model('models/xgfitness_ai_model.pkl')
        
        logger.info("Model retraining completed successfully")
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'training_results': {
                'workout_accuracy': training_results['workout_accuracy'],
                'nutrition_accuracy': training_results['nutrition_accuracy'],
                'training_samples': n_samples
            },
            'model_info': model.training_info
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Error during model retraining'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }), 500

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # NumPy scalar
        return obj.item()
    else:
        return obj

def create_app():
    """Application factory"""
    # Initialize model
    if not initialize_model():
        logger.error("Failed to initialize model. API may not function properly.")
    
    return app

if __name__ == '__main__':
    # Initialize model
    if initialize_model():
        print("FitTech AI Model initialized successfully")
        print("Starting Flask API server...")
        
        # Print available endpoints
        print("\nAvailable endpoints:")
        print("GET  /               - API documentation")
        print("GET  /health         - Health check")
        print("POST /predict        - Get fitness recommendations")
        print("POST /calculate-metrics - Calculate BMI, BMR, TDEE")
        print("GET  /templates      - Get available templates")
        print("GET  /validation-rules - Get validation rules")
        print("GET  /model-info     - Get model information")
        print("POST /retrain        - Retrain model (admin)")
        
        # Print example request
        print("\nExample request to /predict:")
        print("""{
    "age": 25,
    "gender": "Male",
    "height": 175,
    "weight": 70,
    "activity_level": "Moderate Activity",
    "fitness_goal": "Muscle Gain"
}""")
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize FitTech AI model. Exiting.")
        sys.exit(1)