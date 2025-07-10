"""
Enhanced system configuration for FitTech AI
Centralized configuration management with environment support
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Model Configuration
    MODEL_VERSION = "1.0.0"
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/fittech_ai_model.pkl')
    TEMPLATES_PATH = os.getenv('TEMPLATES_PATH', 'data/')
    
    # Training Configuration
    DEFAULT_TRAINING_SAMPLES = int(os.getenv('TRAINING_SAMPLES', '2000'))
    TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
    
    # XGBoost Model Parameters
    XGBOOST_PARAMS = {
        'max_depth': int(os.getenv('XGB_MAX_DEPTH', '4')),
        'learning_rate': float(os.getenv('XGB_LEARNING_RATE', '0.05')),
        'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', '200')),
        'min_child_weight': int(os.getenv('XGB_MIN_CHILD_WEIGHT', '5')),
        'subsample': float(os.getenv('XGB_SUBSAMPLE', '0.7')),
        'colsample_bytree': float(os.getenv('XGB_COLSAMPLE_BYTREE', '0.7')),
        'reg_alpha': float(os.getenv('XGB_REG_ALPHA', '0.5')),
        'reg_lambda': float(os.getenv('XGB_REG_LAMBDA', '2.0')),
        'early_stopping_rounds': int(os.getenv('XGB_EARLY_STOPPING', '20'))
    }
    
    # Validation Configuration
    VALIDATION_RULES = {
        'age': {'min': 18, 'max': 100},
        'height': {'min': 120, 'max': 250, 'unit': 'cm'},
        'weight': {'min': 30, 'max': 300, 'unit': 'kg'},
        'bmi': {'underweight': 18.5, 'normal': 25, 'overweight': 30}
    }
    
    # Confidence Scoring Thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': float(os.getenv('CONFIDENCE_HIGH', '0.8')),
        'medium': float(os.getenv('CONFIDENCE_MEDIUM', '0.6')),
        'low': float(os.getenv('CONFIDENCE_LOW', '0.4'))
    }
    
    # Activity Level Multipliers (for TDEE calculation)
    ACTIVITY_MULTIPLIERS = {
        'Low Activity': 1.29,
        'Moderate Activity': 1.55,
        'High Activity': 1.81
    }
    
    # Valid Input Options
    VALID_GENDERS = ['Male', 'Female']
    VALID_ACTIVITY_LEVELS = ['Low Activity', 'Moderate Activity', 'High Activity']
    VALID_FITNESS_GOALS = ['Fat Loss', 'Muscle Gain', 'Maintenance']
    VALID_BMI_CATEGORIES = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    # API Configuration
    API_VERSION = "1.0.0"
    MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '1024'))  # KB
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100 per hour')
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    ENABLE_CORS = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/fittech_ai.log')
    
    # Database Configuration (for future use)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///fittech_ai.db')
    
    # Feature Engineering Configuration
    FEATURE_COLUMNS = [
        'age', 'gender_encoded', 'height_cm', 'weight_kg', 'bmi',
        'bmr', 'tdee', 'activity_encoded', 'goal_encoded', 'bmi_category_encoded',
        'bmi_goal_interaction', 'age_activity_interaction', 'bmr_per_kg',
        'tdee_bmr_ratio', 'bmi_deviation', 'high_metabolism',
        'very_active', 'young_adult'
    ]

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    
    # Use smaller dataset for faster training in development
    DEFAULT_TRAINING_SAMPLES = 1000
    
    # More verbose logging
    LOG_LEVEL = 'DEBUG'
    
    # Allow all origins in development
    ALLOWED_ORIGINS = ['*']

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    
    # Larger dataset for better accuracy in production
    DEFAULT_TRAINING_SAMPLES = 5000
    
    # More conservative model parameters
    XGBOOST_PARAMS = {
        **Config.XGBOOST_PARAMS,
        'learning_rate': 0.03,  # Lower learning rate
        'n_estimators': 300,    # More estimators
        'reg_alpha': 1.0,       # More regularization
        'reg_lambda': 3.0
    }
    
    # Stricter security
    ENABLE_CORS = False
    ALLOWED_ORIGINS = []  # Should be set via environment variable
    
    # Production logging
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    
    # Small dataset for fast tests
    DEFAULT_TRAINING_SAMPLES = 200
    
    # Fast model parameters for testing
    XGBOOST_PARAMS = {
        **Config.XGBOOST_PARAMS,
        'n_estimators': 50,
        'early_stopping_rounds': 5
    }
    
    # Test database
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration mapping
config_mapping = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(environment: str = None) -> Config:
    """
    Get configuration based on environment
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Config: Configuration object
    """
    if environment is None:
        environment = os.getenv('FLASK_ENV', 'development')
    
    config_class = config_mapping.get(environment, DevelopmentConfig)
    return config_class()

def get_template_definitions() -> Dict[str, Any]:
    """
    Get template definitions for workout and nutrition plans
    Note: Templates are now managed by the TemplateManager class
    
    Returns:
        dict: Template definitions from CSV files
    """
    from templates import get_template_manager
    
    template_manager = get_template_manager()
    templates = template_manager.get_all_templates()
    
    return {
        'workout_templates': templates['workout'].to_dict('records'),
        'nutrition_templates': templates['nutrition'].to_dict('records')
    }

def get_model_hyperparameters(environment: str = None) -> Dict[str, Any]:
    """
    Get optimized hyperparameters based on environment
    
    Args:
        environment: Environment name
        
    Returns:
        dict: Hyperparameters for model training
    """
    config = get_config(environment)
    
    return {
        'xgboost': config.XGBOOST_PARAMS,
        'training': {
            'test_size': config.TEST_SIZE,
            'random_state': config.RANDOM_STATE,
            'n_samples': config.DEFAULT_TRAINING_SAMPLES
        },
        'validation': config.VALIDATION_RULES,
        'confidence': config.CONFIDENCE_THRESHOLDS
    }

def validate_configuration() -> Dict[str, Any]:
    """
    Validate current configuration settings
    
    Returns:
        dict: Validation results
    """
    config = get_config()
    issues = []
    warnings = []
    
    # Validate paths
    if not os.path.exists(os.path.dirname(config.MODEL_PATH)):
        issues.append(f"Model directory does not exist: {os.path.dirname(config.MODEL_PATH)}")
    
    if not os.path.exists(config.TEMPLATES_PATH):
        warnings.append(f"Templates directory does not exist: {config.TEMPLATES_PATH}")
    
    # Validate hyperparameters
    if config.XGBOOST_PARAMS['learning_rate'] <= 0 or config.XGBOOST_PARAMS['learning_rate'] > 1:
        issues.append("Learning rate must be between 0 and 1")
    
    if config.XGBOOST_PARAMS['max_depth'] < 1 or config.XGBOOST_PARAMS['max_depth'] > 20:
        warnings.append("Max depth should typically be between 1 and 20")
    
    # Validate confidence thresholds
    thresholds = config.CONFIDENCE_THRESHOLDS
    if not (0 < thresholds['low'] < thresholds['medium'] < thresholds['high'] <= 1):
        issues.append("Confidence thresholds must be in ascending order between 0 and 1")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'config_summary': {
            'environment': os.getenv('FLASK_ENV', 'development'),
            'model_path': config.MODEL_PATH,
            'training_samples': config.DEFAULT_TRAINING_SAMPLES,
            'debug_mode': config.DEBUG
        }
    }

# Environment setup helper
def setup_environment():
    """Setup environment directories and files"""
    config = get_config()
    
    # Create necessary directories
    directories = [
        os.path.dirname(config.MODEL_PATH),
        config.TEMPLATES_PATH,
        os.path.dirname(config.LOG_FILE),
        'data',
        'models',
        'logs'
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Test configuration
    print("FitTech AI Configuration Test")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    # Validate configuration
    validation = validate_configuration()
    
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has issues:")
        for issue in validation['issues']:
            print(f"  • {issue}")
    
    if validation['warnings']:
        print("\n⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    print(f"\nConfiguration Summary:")
    for key, value in validation['config_summary'].items():
        print(f"  {key}: {value}")
    
    # Test different environments
    print(f"\nEnvironment Configurations:")
    for env in ['development', 'production', 'testing']:
        config = get_config(env)
        print(f"  {env}:")
        print(f"    Training samples: {config.DEFAULT_TRAINING_SAMPLES}")
        print(f"    Debug mode: {config.DEBUG}")
        print(f"    Learning rate: {config.XGBOOST_PARAMS['learning_rate']}")
    
    print(f"\nTemplate Summary:")
    templates = get_template_definitions()
    print(f"  Workout templates: {len(templates['workout_templates'])}")
    print(f"  Nutrition templates: {len(templates['nutrition_templates'])}")