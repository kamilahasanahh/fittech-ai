"""
Template Management Module for XGFitness AI
Handles loading and management of workout and nutrition templates
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple

class TemplateManager:
    """
    Manages workout and nutrition templates for the XGFitness AI system
    """
    
    def __init__(self, templates_dir: str = 'data'):
        """
        Initialize template manager
        
        Args:
            templates_dir: Directory containing template CSV files
        """
        self.templates_dir = templates_dir
        self.workout_templates = None
        self.nutrition_templates = None
        self._workout_lookup = {}
        self._nutrition_lookup = {}
        
        # Load templates
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load templates from JSON files"""
        try:
            # Load workout templates
            workout_path = os.path.join(self.templates_dir, 'workout_templates.json')
            if os.path.exists(workout_path):
                with open(workout_path, 'r') as f:
                    workout_data = json.load(f)
                # Handle both direct array and wrapped object formats
                if isinstance(workout_data, list):
                    self.workout_templates = pd.DataFrame(workout_data)
                else:
                    self.workout_templates = pd.DataFrame(workout_data['workout_templates'])
            else:
                # Create default templates if file doesn't exist
                self.workout_templates = self._create_default_workout_templates()
                self.save_workout_templates()
            
            # Load nutrition templates
            # Try both locations: data/nutrition_templates.json (backend) and data/nutrition/nutrition_templates.json (main)
            nutrition_path = os.path.join(self.templates_dir, 'nutrition_templates.json')
            if not os.path.exists(nutrition_path):
                nutrition_path = os.path.join(self.templates_dir, 'nutrition', 'nutrition_templates.json')
            
            if os.path.exists(nutrition_path):
                print(f"Loading nutrition templates from: {nutrition_path}")
                with open(nutrition_path, 'r') as f:
                    nutrition_data = json.load(f)
                # Handle both direct array and wrapped object formats
                if isinstance(nutrition_data, list):
                    print(f"Loading {len(nutrition_data)} nutrition templates from direct array")
                    self.nutrition_templates = pd.DataFrame(nutrition_data)
                else:
                    print(f"Loading nutrition templates from wrapped object")
                    self.nutrition_templates = pd.DataFrame(nutrition_data['nutrition_templates'])
            else:
                print("No nutrition templates file found, using defaults")
                # Create default templates if file doesn't exist
                self.nutrition_templates = self._create_default_nutrition_templates()
                self.save_nutrition_templates()
            
            # Build lookup dictionaries for faster access
            self._build_lookups()
            
            print(f"Templates loaded: {len(self.workout_templates)} workout, {len(self.nutrition_templates)} nutrition")
            
        except Exception as e:
            print(f"Error loading templates: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to default templates
            self.workout_templates = self._create_default_workout_templates()
            self.nutrition_templates = self._create_default_nutrition_templates()
            self._build_lookups()
    
    def _create_default_workout_templates(self) -> pd.DataFrame:
        """Create default workout templates matching the new CSV structure"""
        templates = [
            {'template_id': 1, 'goal': 'Fat Loss', 'activity_level': 'Low Activity', 
             'workout_type': 'Full Body', 'days_per_week': 2, 'workout_schedule': 'WXXWXXX',
             'sets_per_exercise': 3, 'exercises_per_session': 6, 'cardio_minutes_per_day': 35, 'cardio_sessions_per_day': 1},
            {'template_id': 2, 'goal': 'Fat Loss', 'activity_level': 'Moderate Activity', 
             'workout_type': 'Full Body', 'days_per_week': 3, 'workout_schedule': 'WXWXWXX',
             'sets_per_exercise': 3, 'exercises_per_session': 8, 'cardio_minutes_per_day': 45, 'cardio_sessions_per_day': 1},
            {'template_id': 3, 'goal': 'Fat Loss', 'activity_level': 'High Activity', 
             'workout_type': 'Upper/Lower Split', 'days_per_week': 4, 'workout_schedule': 'ABXABXX',
             'sets_per_exercise': 3, 'exercises_per_session': 10, 'cardio_minutes_per_day': 55, 'cardio_sessions_per_day': 1},
            {'template_id': 4, 'goal': 'Muscle Gain', 'activity_level': 'Low Activity', 
             'workout_type': 'Full Body', 'days_per_week': 2, 'workout_schedule': 'WXXWXXX',
             'sets_per_exercise': 4, 'exercises_per_session': 6, 'cardio_minutes_per_day': 18, 'cardio_sessions_per_day': 0},
            {'template_id': 5, 'goal': 'Muscle Gain', 'activity_level': 'Moderate Activity', 
             'workout_type': 'Upper/Lower Split', 'days_per_week': 4, 'workout_schedule': 'ABXABXX',
             'sets_per_exercise': 4, 'exercises_per_session': 8, 'cardio_minutes_per_day': 25, 'cardio_sessions_per_day': 1},
            {'template_id': 6, 'goal': 'Muscle Gain', 'activity_level': 'High Activity', 
             'workout_type': 'Push/Pull/Legs', 'days_per_week': 3, 'workout_schedule': 'ABCX',
             'sets_per_exercise': 4, 'exercises_per_session': 10, 'cardio_minutes_per_day': 30, 'cardio_sessions_per_day': 1},
            {'template_id': 7, 'goal': 'Maintenance', 'activity_level': 'Low Activity', 
             'workout_type': 'Full Body', 'days_per_week': 2, 'workout_schedule': 'WXXWXXX',
             'sets_per_exercise': 3, 'exercises_per_session': 4, 'cardio_minutes_per_day': 30, 'cardio_sessions_per_day': 0},
            {'template_id': 8, 'goal': 'Maintenance', 'activity_level': 'Moderate Activity', 
             'workout_type': 'Upper/Lower Split', 'days_per_week': 4, 'workout_schedule': 'ABXABXX',
             'sets_per_exercise': 3, 'exercises_per_session': 6, 'cardio_minutes_per_day': 40, 'cardio_sessions_per_day': 1},
            {'template_id': 9, 'goal': 'Maintenance', 'activity_level': 'High Activity', 
             'workout_type': 'Push/Pull/Legs', 'days_per_week': 3, 'workout_schedule': 'ABCX',
             'sets_per_exercise': 3, 'exercises_per_session': 8, 'cardio_minutes_per_day': 50, 'cardio_sessions_per_day': 1}
        ]
        return pd.DataFrame(templates)
    
    def _create_default_nutrition_templates(self) -> pd.DataFrame:
        """Create default nutrition templates matching the new CSV structure"""
        templates = [
            {'template_id': 1, 'goal': 'Fat Loss', 'bmi_category': 'Normal', 
             'caloric_intake_multiplier': 0.80, 'protein_per_kg': 2.3, 'carbs_per_kg': 2.75, 'fat_per_kg': 0.85},
            {'template_id': 2, 'goal': 'Fat Loss', 'bmi_category': 'Overweight', 
             'caloric_intake_multiplier': 0.75, 'protein_per_kg': 2.15, 'carbs_per_kg': 2.25, 'fat_per_kg': 0.80},
            {'template_id': 3, 'goal': 'Fat Loss', 'bmi_category': 'Obese', 
             'caloric_intake_multiplier': 0.70, 'protein_per_kg': 2.45, 'carbs_per_kg': 1.75, 'fat_per_kg': 0.80},
            {'template_id': 4, 'goal': 'Muscle Gain', 'bmi_category': 'Underweight', 
             'caloric_intake_multiplier': 1.15, 'protein_per_kg': 2.3, 'carbs_per_kg': 4.75, 'fat_per_kg': 1.0},
            {'template_id': 5, 'goal': 'Muscle Gain', 'bmi_category': 'Normal', 
             'caloric_intake_multiplier': 1.10, 'protein_per_kg': 2.1, 'carbs_per_kg': 4.25, 'fat_per_kg': 0.95},
            {'template_id': 6, 'goal': 'Maintenance', 'bmi_category': 'Underweight', 
             'caloric_intake_multiplier': 1.00, 'protein_per_kg': 1.8, 'carbs_per_kg': 3.25, 'fat_per_kg': 0.90},
            {'template_id': 7, 'goal': 'Maintenance', 'bmi_category': 'Normal', 
             'caloric_intake_multiplier': 0.95, 'protein_per_kg': 1.8, 'carbs_per_kg': 3.25, 'fat_per_kg': 0.85},
            {'template_id': 8, 'goal': 'Maintenance', 'bmi_category': 'Overweight', 
             'caloric_intake_multiplier': 0.90, 'protein_per_kg': 1.8, 'carbs_per_kg': 3.25, 'fat_per_kg': 0.80}
        ]
        return pd.DataFrame(templates)
    
    def _build_lookups(self) -> None:
        """Build lookup dictionaries for faster template assignment"""
        self._workout_lookup = {}
        self._nutrition_lookup = {}
        
        # Build workout lookup: (goal, activity_level) -> template_id
        for _, row in self.workout_templates.iterrows():
            key = (row['goal'], row['activity_level'])
            self._workout_lookup[key] = row['template_id']
        
        # Build nutrition lookup: (goal, bmi_category) -> template_id
        for _, row in self.nutrition_templates.iterrows():
            key = (row['goal'], row['bmi_category'])
            self._nutrition_lookup[key] = row['template_id']
    
    def get_workout_template(self, template_id: int) -> Optional[Dict]:
        """
        Get workout template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            dict or None: Template data
        """
        template = self.workout_templates[self.workout_templates['template_id'] == template_id]
        if len(template) > 0:
            return template.iloc[0].to_dict()
        return None
    
    def get_nutrition_template(self, template_id: int) -> Optional[Dict]:
        """
        Get nutrition template by ID
        
        Args:
            template_id: Template ID
            
        Returns:
            dict or None: Template data
        """
        template = self.nutrition_templates[self.nutrition_templates['template_id'] == template_id]
        if len(template) > 0:
            return template.iloc[0].to_dict()
        return None
    
    def find_workout_template(self, goal: str, activity_level: str) -> Optional[int]:
        """
        Find workout template ID by goal and activity level
        
        Args:
            goal: Fitness goal (Fat Loss, Muscle Gain, Maintenance)
            activity_level: Activity level (Low Activity, Moderate Activity, High Activity)
            
        Returns:
            int or None: Template ID
        """
        key = (goal, activity_level)
        return self._workout_lookup.get(key)
    
    def find_nutrition_template(self, goal: str, bmi_category: str) -> Optional[int]:
        """
        Find nutrition template ID by goal and BMI category
        
        Args:
            goal: Fitness goal (Fat Loss, Muscle Gain, Maintenance)
            bmi_category: BMI category (Underweight, Normal, Overweight, Obese)
            
        Returns:
            int or None: Template ID
        """
        key = (goal, bmi_category)
        return self._nutrition_lookup.get(key)
    
    def get_template_assignments(self, goal: str, activity_level: str, bmi_category: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get both workout and nutrition template IDs for a user profile
        
        Args:
            goal: Fitness goal
            activity_level: Activity level
            bmi_category: BMI category
            
        Returns:
            tuple: (workout_template_id, nutrition_template_id)
        """
        workout_id = self.find_workout_template(goal, activity_level)
        nutrition_id = self.find_nutrition_template(goal, bmi_category)
        return workout_id, nutrition_id
    
    def get_all_templates(self) -> Dict[str, pd.DataFrame]:
        """
        Get all templates
        
        Returns:
            dict: Dictionary with 'workout' and 'nutrition' DataFrames
        """
        return {
            'workout': self.workout_templates.copy(),
            'nutrition': self.nutrition_templates.copy()
        }
    
    def get_template_summary(self) -> Dict[str, any]:
        """
        Get summary of all templates
        
        Returns:
            dict: Template summary statistics
        """
        return {
            'workout_count': len(self.workout_templates),
            'nutrition_count': len(self.nutrition_templates),
            'workout_goals': self.workout_templates['goal'].unique().tolist(),
            'nutrition_goals': self.nutrition_templates['goal'].unique().tolist(),
            'activity_levels': self.workout_templates['activity_level'].unique().tolist(),
            'bmi_categories': self.nutrition_templates['bmi_category'].unique().tolist(),
            'total_combinations': len(self._workout_lookup) + len(self._nutrition_lookup)
        }
    
    def save_workout_templates(self) -> None:
        """Save workout templates to JSON"""
        os.makedirs(self.templates_dir, exist_ok=True)
        workout_path = os.path.join(self.templates_dir, 'workout_templates.json')
        workout_data = self.workout_templates.to_dict('records')
        with open(workout_path, 'w') as f:
            json.dump(workout_data, f, indent=2)
        print(f"Workout templates saved to {workout_path}")
    
    def save_nutrition_templates(self) -> None:
        """Save nutrition templates to JSON"""
        os.makedirs(self.templates_dir, exist_ok=True)
        nutrition_path = os.path.join(self.templates_dir, 'nutrition_templates.json')
        nutrition_data = self.nutrition_templates.to_dict('records')
        with open(nutrition_path, 'w') as f:
            json.dump(nutrition_data, f, indent=2)
        print(f"Nutrition templates saved to {nutrition_path}")
    
    def save_all_templates(self) -> None:
        """Save all templates to CSV files"""
        self.save_workout_templates()
        self.save_nutrition_templates()
    
    def add_workout_template(self, template_data: Dict) -> bool:
        """
        Add a new workout template
        
        Args:
            template_data: Template data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            # Validate required fields
            required_fields = ['goal', 'activity_level', 'sets_per_day', 'sessions_per_day', 
                             'cardio_minutes_per_day', 'cardio_sessions_per_day']
            
            for field in required_fields:
                if field not in template_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Assign new ID
            new_id = self.workout_templates['template_id'].max() + 1 if len(self.workout_templates) > 0 else 1
            template_data['template_id'] = new_id
            
            # Add description if not provided
            if 'description' not in template_data:
                template_data['description'] = f"Custom {template_data['goal']} template for {template_data['activity_level']}"
            
            # Add to DataFrame
            new_template = pd.DataFrame([template_data])
            self.workout_templates = pd.concat([self.workout_templates, new_template], ignore_index=True)
            
            # Rebuild lookups
            self._build_lookups()
            
            print(f"Added workout template with ID {new_id}")
            return True
            
        except Exception as e:
            print(f"Error adding workout template: {e}")
            return False
    
    def add_nutrition_template(self, template_data: Dict) -> bool:
        """
        Add a new nutrition template
        
        Args:
            template_data: Template data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            # Validate required fields
            required_fields = ['goal', 'bmi_category', 'tdee_multiplier', 'protein_per_kg', 
                             'carbs_per_kg', 'fat_per_kg']
            
            for field in required_fields:
                if field not in template_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Assign new ID
            new_id = self.nutrition_templates['template_id'].max() + 1 if len(self.nutrition_templates) > 0 else 1
            template_data['template_id'] = new_id
            
            # Add description if not provided
            if 'description' not in template_data:
                template_data['description'] = f"Custom {template_data['goal']} nutrition template for {template_data['bmi_category']}"
            
            # Add to DataFrame
            new_template = pd.DataFrame([template_data])
            self.nutrition_templates = pd.concat([self.nutrition_templates, new_template], ignore_index=True)
            
            # Rebuild lookups
            self._build_lookups()
            
            print(f"Added nutrition template with ID {new_id}")
            return True
            
        except Exception as e:
            print(f"Error adding nutrition template: {e}")
            return False
    
    def validate_templates(self) -> Dict[str, List[str]]:
        """
        Validate all templates for completeness and consistency
        
        Returns:
            dict: Validation results with any issues found
        """
        issues = []
        warnings = []
        
        # Check workout templates
        required_workout_fields = ['template_id', 'goal', 'activity_level', 'sets_per_day', 
                                 'sessions_per_day', 'cardio_minutes_per_day', 'cardio_sessions_per_day']
        
        for _, template in self.workout_templates.iterrows():
            for field in required_workout_fields:
                if pd.isna(template.get(field)):
                    issues.append(f"Workout template {template.get('template_id', 'Unknown')}: Missing {field}")
            
            # Validate ranges
            if template.get('sets_per_day', 0) < 0.1:
                warnings.append(f"Workout template {template.get('template_id')}: Very low sets per day")
            if template.get('sessions_per_day', 0) > 1:
                warnings.append(f"Workout template {template.get('template_id')}: More than 1 session per day")
        
        # Check nutrition templates
        required_nutrition_fields = ['template_id', 'goal', 'bmi_category', 'caloric_intake', 
                                   'protein_per_kg', 'carbs_per_kg', 'fat_per_kg']
        
        for _, template in self.nutrition_templates.iterrows():
            for field in required_nutrition_fields:
                if pd.isna(template.get(field)):
                    issues.append(f"Nutrition template {template.get('template_id', 'Unknown')}: Missing {field}")
            
            # Validate ranges
            if template.get('caloric_intake', 1.0) < 0.5 or template.get('caloric_intake', 1.0) > 2.0:
                warnings.append(f"Nutrition template {template.get('template_id')}: Extreme caloric intake multiplier")
        
        # Check coverage (all combinations should exist)
        goals = ['Fat Loss', 'Muscle Gain', 'Maintenance']
        activity_levels = ['Low Activity', 'Moderate Activity', 'High Activity']
        bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        
        # Check workout coverage
        for goal in goals:
            for activity in activity_levels:
                if (goal, activity) not in self._workout_lookup:
                    issues.append(f"Missing workout template for {goal} + {activity}")
        
        # Check nutrition coverage (not all combinations are required)
        expected_nutrition_combinations = [
            ('Fat Loss', 'Normal'), ('Fat Loss', 'Overweight'), ('Fat Loss', 'Obese'),
            ('Muscle Gain', 'Underweight'), ('Muscle Gain', 'Normal'),
            ('Maintenance', 'Underweight'), ('Maintenance', 'Normal'), ('Maintenance', 'Overweight')
        ]
        
        for goal, bmi_cat in expected_nutrition_combinations:
            if (goal, bmi_cat) not in self._nutrition_lookup:
                issues.append(f"Missing nutrition template for {goal} + {bmi_cat}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'template_counts': {
                'workout': len(self.workout_templates),
                'nutrition': len(self.nutrition_templates)
            }
        }

# Global template manager instance
_template_manager = None

def get_template_manager(templates_dir: str = 'data') -> TemplateManager:
    """
    Get global template manager instance (singleton pattern)
    
    Args:
        templates_dir: Directory containing template files
        
    Returns:
        TemplateManager: Global template manager instance
    """
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager(templates_dir)
    return _template_manager

# Convenience functions for backward compatibility
def load_workout_templates(templates_dir: str = 'data') -> pd.DataFrame:
    """Load workout templates - backward compatibility function"""
    manager = get_template_manager(templates_dir)
    return manager.workout_templates.copy()

def load_nutrition_templates(templates_dir: str = 'data') -> pd.DataFrame:
    """Load nutrition templates - backward compatibility function"""
    manager = get_template_manager(templates_dir)
    return manager.nutrition_templates.copy()

if __name__ == "__main__":
    # Test template manager
    print("Testing Template Manager")
    print("=" * 40)
    
    # Initialize template manager
    manager = TemplateManager('data')
    
    # Get summary
    summary = manager.get_template_summary()
    print(f"Template Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate templates
    validation = manager.validate_templates()
    if validation['valid']:
        print("\nAll templates are valid")
    else:
        print(f"\nTemplate validation issues:")
        for issue in validation['issues']:
            print(f"  • {issue}")
    
    if validation['warnings']:
        print(f"\n⚠️ Template warnings:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    
    # Test template lookup
    print(f"\nTesting template lookup:")
    workout_id = manager.find_workout_template('Muscle Gain', 'Moderate Activity')
    nutrition_id = manager.find_nutrition_template('Muscle Gain', 'Normal')
    print(f"  Muscle Gain + Moderate Activity → Workout Template {workout_id}")
    print(f"  Muscle Gain + Normal BMI → Nutrition Template {nutrition_id}")
    
    # Test getting template details
    if workout_id:
        workout_template = manager.get_workout_template(workout_id)
        print(f"  Workout Template {workout_id}: {workout_template['sessions_per_week']} sessions/week")
    
    if nutrition_id:
        nutrition_template = manager.get_nutrition_template(nutrition_id)
        print(f"  Nutrition Template {nutrition_id}: {nutrition_template['protein_per_kg']}g protein/kg")