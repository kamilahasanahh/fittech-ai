import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration matching your thesis specs."""
    
    # XGBoost parameters optimized for your use case
    max_depth: int = 7
    learning_rate: float = 0.08
    n_estimators: int = 312
    min_child_weight: int = 3
    subsample: float = 0.85
    colsample_bytree: float = 0.72
    random_state: int = 42
    n_jobs: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XGBoost."""
        return {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }

class Config:
    """Main configuration class."""
    
    # Environment
    ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = ENV == 'development'
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Model paths
    MODEL_PATH = os.path.join(MODEL_DIR, 'thesis_aligned_model')
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    
    # CORS settings for your React frontend
    CORS_ORIGINS = [
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",
        # Add your production URLs here
    ]
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model configurations
    MODEL_CONFIG = ModelConfig()
    
    # Validation ranges
    VALIDATION_RANGES = {
        'age': (18, 80),
        'height': (140, 220),
        'weight': (40, 200),
        'target_weight': (40, 200)
    }
    
    VALID_CATEGORIES = {
        'gender': ['Male', 'Female'],
        'fitness_goal': ['Muscle Gain', 'Fat Loss', 'Maintenance'],
        'activity_level': [
            'Sedentary', 'Lightly Active', 'Moderately Active',
            'Very Active', 'Extremely Active'
        ]
    }

    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for directory in [cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)