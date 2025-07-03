from typing import Dict, Any
from ..config import Config

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
        'age', 'gender', 'height', 'weight', 'target_weight',
        'fitness_goal', 'activity_level'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in user_profile]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
    
    # Validate ranges
    for field, (min_val, max_val) in Config.VALIDATION_RANGES.items():
        if field in user_profile:
            value = user_profile[field]
            if not isinstance(value, (int, float)) or not min_val <= value <= max_val:
                raise ValidationError(
                    f"{field} must be between {min_val} and {max_val}, got {value}"
                )
    
    # Validate categories
    for field, valid_values in Config.VALID_CATEGORIES.items():
        if field in user_profile:
            value = user_profile[field]
            if value not in valid_values:
                raise ValidationError(
                    f"{field} must be one of {valid_values}, got {value}"
                )
    
    # Return validated profile (no changes needed for thesis model)
    return user_profile.copy()