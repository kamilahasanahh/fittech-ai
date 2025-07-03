import logging
import sys
import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.model.thesis_model import ThesisAlignedXGBoostSystem
from src.utils.validation import ValidationError

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'fitness_app.log'))
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure CORS for your React frontend
CORS(app)

# Global model instance
model = None


def initialize_model():
    """Initialize the model on startup."""
    global model
    try:
        Config.create_directories()
        model = ThesisAlignedXGBoostSystem()
        
        # Try to load existing model
        if os.path.exists(f'{Config.MODEL_PATH}_workout.pkl'):
            model.load_models()
            logger.info("Existing model loaded successfully")
        else:
            logger.warning("No existing model found. Train the model first!")
            
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        model = None


# Initialize model when app starts
with app.app_context():
    initialize_model()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and model.is_trained,
        'thesis_aligned': True,
        'total_templates': 75 if model else 0
    })

@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify(routes)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_redirect():
    return redirect(url_for('predict'), code=307)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate fitness recommendations.
    Compatible with your existing frontend API calls.
    """
    try:
        if model is None or not model.is_trained:
            return jsonify({'error': 'Model not available. Please train the model first.'}), 503

        # Get request data
        user_data = request.get_json()
        if not user_data:
            return jsonify({'error': 'No data provided'}), 400

        # Generate recommendations
        recommendations = model.predict_recommendations(user_data)
        
        logger.info(f"Generated recommendations for user: {user_data.get('age')}yo {user_data.get('gender')}")
        
        return jsonify({
            'success': True,
            'data': recommendations,
            'thesis_aligned': True,
            "message": "Prediction endpoint working"
        })

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({'error': f'Validation error: {str(e)}'}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train a new model (admin endpoint).
    Use this to train your thesis-aligned model.
    """
    try:
        global model
        
        # Basic admin check (enhance this for production)
        admin_key = request.headers.get('X-Admin-Key')
        if admin_key != 'your-secret-admin-key':  # Change this!
            return jsonify({'error': 'Unauthorized'}), 401

        logger.info("Starting model training...")
        
        model = ThesisAlignedXGBoostSystem()
        training_data = model.train_models()
        model.save_models()
        
        return jsonify({
            'success': True,
            'message': 'Thesis-aligned model trained successfully',
            'training_samples': len(training_data),
            'total_templates': 75,
            'workout_templates': 15,
            'nutrition_templates': 60
        })

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500
    
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"})

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get template information for debugging/inspection."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    return jsonify({
        'workout_templates': model.workout_templates.to_dict('records'),
        'nutrition_templates': model.nutrition_templates.to_dict('records'),
        'total_templates': len(model.workout_templates) + len(model.nutrition_templates)
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information and statistics."""
    if model is None or not model.is_trained:
        return jsonify({'error': 'Model not available'}), 503
        
    return jsonify({
        'thesis_aligned': True,
        'total_templates': 75,
        'template_breakdown': {
            'workout_templates': 15,
            'nutrition_templates': 60,
            'structure': {
                'workout': '3 goals × 5 activity levels',
                'nutrition': '3 goals × 4 BMI categories × 5 activity levels'
            }
        },
        'features_used': len(model.feature_names),
        'is_trained': model.is_trained
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.DEBUG
    )