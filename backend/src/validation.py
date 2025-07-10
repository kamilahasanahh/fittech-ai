## validation.py

from typing import Dict, Any
try:
    from .config import Config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from config import Config

class ValidationError(Exception):
    """Custom validation error."""
    pass

def validate_user_profile(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate user profile input.
    
    Args:
        user_profile: Dictionary containing user data
        
    Returns:
        Validated and cleaned user profile
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = [
        'age', 'gender', 'height', 'weight',
        'fitness_goal', 'activity_level'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in user_profile]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
    
    # Validate ranges
    validation_ranges = {
        'age': (Config.VALIDATION_RULES['age']['min'], Config.VALIDATION_RULES['age']['max']),
        'height': (Config.VALIDATION_RULES['height']['min'], Config.VALIDATION_RULES['height']['max']),
        'weight': (Config.VALIDATION_RULES['weight']['min'], Config.VALIDATION_RULES['weight']['max'])
    }
    
    for field, (min_val, max_val) in validation_ranges.items():
        if field in user_profile:
            value = user_profile[field]
            if not isinstance(value, (int, float)) or not min_val <= value <= max_val:
                raise ValidationError(
                    f"{field} must be between {min_val} and {max_val}, got {value}"
                )
    
    # Validate categories
    valid_categories = {
        'gender': ['Male', 'Female'],
        'fitness_goal': ['Fat Loss', 'Muscle Gain', 'Maintenance'],
        'activity_level': ['Low Activity', 'Moderate Activity', 'High Activity']
    }
    
    for field, valid_values in valid_categories.items():
        if field in user_profile:
            value = user_profile[field]
            if value not in valid_values:
                raise ValidationError(
                    f"{field} must be one of {valid_values}, got {value}"
                )
    
    # Return validated profile (no changes needed for thesis model)
    return user_profile.copy()

def validate_api_request_data(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate API request data for fitness recommendations.
    
    Args:
        request_data: Dictionary containing API request data
        
    Returns:
        Validated and cleaned request data
        
    Raises:
        ValidationError: If validation fails
    """
    # Use the existing validate_user_profile function
    return validate_user_profile(request_data)

def create_validation_summary(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a validation summary for the user profile.
    
    Args:
        user_profile: User profile dictionary
        
    Returns:
        Validation summary with status and messages
    """
    try:
        validated_profile = validate_user_profile(user_profile)
        return {
            'valid': True,
            'profile': validated_profile,
            'messages': ['All validation checks passed']
        }
    except ValidationError as e:
        return {
            'valid': False,
            'profile': user_profile,
            'messages': [str(e)]
        }
    except Exception as e:
        return {
            'valid': False,
            'profile': user_profile,
            'messages': [f'Validation error: {str(e)}']
        }

def get_validation_rules() -> Dict[str, Any]:
    """
    Get validation rules for API documentation.
    
    Returns:
        Dictionary containing validation rules
    """
    return {
        'required_fields': [
            'age', 'gender', 'height', 'weight', 
            'fitness_goal', 'activity_level'
        ],
        'validation_ranges': {
            'age': {'min': Config.VALIDATION_RULES['age']['min'], 'max': Config.VALIDATION_RULES['age']['max']},
            'height': {'min': Config.VALIDATION_RULES['height']['min'], 'max': Config.VALIDATION_RULES['height']['max']},
            'weight': {'min': Config.VALIDATION_RULES['weight']['min'], 'max': Config.VALIDATION_RULES['weight']['max']}
        },
        'valid_categories': {
            'gender': ['Male', 'Female'],
            'fitness_goal': ['Fat Loss', 'Muscle Gain', 'Maintenance'],
            'activity_level': ['Low Activity', 'Moderate Activity', 'High Activity']
        }
    }