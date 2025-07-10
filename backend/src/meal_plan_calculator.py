"""
Meal Plan Calculator Module for XGFitness AI
Generates scaled meal plans based on individual calorie and macro needs
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import random
from math import ceil

class MealPlanCalculator:
    """
    Calculates and scales meal plans based on individual nutrition targets
    """
    
    def __init__(self, meal_plans_path: str = 'data/meals/meal_plans.json'):
        """
        Initialize meal plan calculator
        
        Args:
            meal_plans_path: Path to meal plans JSON file
        """
        self.meal_plans_path = meal_plans_path
        self.meal_data = None
        self.load_meal_plans()
    
    def load_meal_plans(self):
        """Load meal plan data from JSON"""
        try:
            # Try multiple possible paths
            possible_paths = [
                self.meal_plans_path,
                os.path.join('..', self.meal_plans_path),
                os.path.join('data', 'meals', 'meal_plans.json'),
                'meal_plans.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        self.meal_data = json.load(f)
                    print(f"Loaded meal plans from {path}")
                    return
            
            print(f"Warning: Meal plans file not found in any of these paths: {possible_paths}")
            self.meal_data = None
        except Exception as e:
            print(f"Error loading meal plans: {e}")
            self.meal_data = None
    
    def calculate_daily_meal_plan(self, 
                                target_calories: int, 
                                target_protein: int, 
                                target_carbs: int, 
                                target_fat: int,
                                preferences: Optional[Dict] = None) -> Dict:
        """
        Generate a complete daily meal plan scaled to individual needs
        
        Args:
            target_calories: Daily calorie target
            target_protein: Daily protein target (g)
            target_carbs: Daily carbohydrate target (g)
            target_fat: Daily fat target (g)
            preferences: Optional preferences for meal selection
            
        Returns:
            Dictionary with scaled daily meal plan
        """
        if self.meal_data is None:
            return {
                'success': False,
                'error': 'Meal plan data not available'
            }
        
        try:
            # Select meals (random or based on preferences)
            selected_meals = self._select_meals(preferences)
            
            # Calculate meal calorie targets
            meal_targets = {
                'sarapan': target_calories * 0.25,  # 25% for breakfast
                'makan_siang': target_calories * 0.40,  # 40% for lunch
                'makan_malam': target_calories * 0.30,  # 30% for dinner
                'snack': target_calories * 0.05  # 5% for snacks
            }
            
            scaled_meals = {}
            total_scaled_calories = 0
            total_scaled_protein = 0
            total_scaled_carbs = 0
            total_scaled_fat = 0
            
            # Scale each meal
            for meal_type, meal_template in selected_meals.items():
                if meal_template:
                    scaled_meal = self._scale_meal(meal_template, meal_targets[meal_type])
                    scaled_meals[meal_type] = scaled_meal
                    
                    # Add to totals
                    total_scaled_calories += scaled_meal['scaled_calories']
                    total_scaled_protein += scaled_meal['scaled_protein']
                    total_scaled_carbs += scaled_meal['scaled_carbs']
                    total_scaled_fat += scaled_meal['scaled_fat']
            
            # Calculate accuracy
            calorie_accuracy = (total_scaled_calories / target_calories) * 100
            protein_accuracy = (total_scaled_protein / target_protein) * 100
            carb_accuracy = (total_scaled_carbs / target_carbs) * 100
            fat_accuracy = (total_scaled_fat / target_fat) * 100
            
            return {
                'success': True,
                'daily_meal_plan': scaled_meals,
                'nutrition_summary': {
                    'total_calories': round(total_scaled_calories),
                    'total_protein': round(total_scaled_protein, 1),
                    'total_carbs': round(total_scaled_carbs, 1),
                    'total_fat': round(total_scaled_fat, 1)
                },
                'targets': {
                    'calories': target_calories,
                    'protein': target_protein,
                    'carbs': target_carbs,
                    'fat': target_fat
                },
                'accuracy': {
                    'calories': round(calorie_accuracy, 1),
                    'protein': round(protein_accuracy, 1),
                    'carbs': round(carb_accuracy, 1),
                    'fat': round(fat_accuracy, 1)
                },
                'meal_distribution': {
                    'sarapan_calories': round(meal_targets['sarapan']),
                    'makan_siang_calories': round(meal_targets['makan_siang']),
                    'makan_malam_calories': round(meal_targets['makan_malam']),
                    'snack_calories': round(meal_targets['snack'])
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error calculating meal plan: {str(e)}'
            }
    
    def _select_meals(self, preferences: Optional[Dict] = None) -> Dict:
        """Select meals for the day based on preferences or randomly"""
        if not self.meal_data:
            return {}
        
        selected = {}
        
        # Select breakfast
        sarapan_options = self.meal_data['meal_templates']['sarapan']
        selected['sarapan'] = random.choice(sarapan_options) if sarapan_options else None
        
        # Select lunch
        makan_siang_options = self.meal_data['meal_templates']['makan_siang']
        selected['makan_siang'] = random.choice(makan_siang_options) if makan_siang_options else None
        
        # Select dinner
        makan_malam_options = self.meal_data['meal_templates']['makan_malam']
        selected['makan_malam'] = random.choice(makan_malam_options) if makan_malam_options else None
        
        # Select snack
        snack_options = self.meal_data['snack_templates']
        selected['snack'] = random.choice(snack_options) if snack_options else None
        
        return selected
    
    def _scale_meal(self, meal_template: Dict, target_calories: float) -> Dict:
        """Scale a meal template to meet target calories"""
        base_calories = meal_template['total_base_calories']
        scaling_factor = target_calories / base_calories
        
        scaled_foods = []
        scaled_calories = 0
        scaled_protein = 0
        scaled_carbs = 0
        scaled_fat = 0
        
        for food in meal_template['foods']:
            scaled_amount = food['base_amount'] * scaling_factor
            
            # Calculate scaled nutrition values
            food_calories = scaled_amount * food['calories_per_unit']
            food_protein = scaled_amount * food['protein_per_unit']
            food_carbs = scaled_amount * food['carbs_per_unit']
            food_fat = scaled_amount * food['fat_per_unit']
            
            scaled_foods.append({
                'nama': food['nama'],
                'amount': round(scaled_amount),
                'unit': food['unit'],
                'calories': round(food_calories),
                'protein': round(food_protein, 1),
                'carbs': round(food_carbs, 1),
                'fat': round(food_fat, 1)
            })
            
            # Add to meal totals
            scaled_calories += food_calories
            scaled_protein += food_protein
            scaled_carbs += food_carbs
            scaled_fat += food_fat
        
        return {
            'meal_id': meal_template['id'],
            'meal_name': meal_template['name'],
            'description': meal_template['description'],
            'scaling_factor': round(scaling_factor, 2),
            'foods': scaled_foods,
            'scaled_calories': scaled_calories,
            'scaled_protein': scaled_protein,
            'scaled_carbs': scaled_carbs,
            'scaled_fat': scaled_fat
        }
    
    def get_meal_options(self, meal_type: str) -> List[Dict]:
        """Get available meal options for a specific meal type"""
        if not self.meal_data:
            return []
        
        if meal_type == 'snack':
            return self.meal_data.get('snack_templates', [])
        else:
            return self.meal_data.get('meal_templates', {}).get(meal_type, [])
    
    def calculate_single_meal(self, meal_id: str, target_calories: float) -> Dict:
        """Calculate a single scaled meal by ID"""
        if not self.meal_data:
            return {'success': False, 'error': 'Meal data not available'}
        
        # Find the meal template
        meal_template = None
        for meal_type in ['sarapan', 'makan_siang', 'makan_malam']:
            for meal in self.meal_data['meal_templates'][meal_type]:
                if meal['id'] == meal_id:
                    meal_template = meal
                    break
            if meal_template:
                break
        
        # Check snacks too
        if not meal_template:
            for snack in self.meal_data['snack_templates']:
                if snack['id'] == meal_id:
                    meal_template = snack
                    break
        
        if not meal_template:
            return {'success': False, 'error': f'Meal with ID {meal_id} not found'}
        
        scaled_meal = self._scale_meal(meal_template, target_calories)
        scaled_meal['success'] = True
        return scaled_meal
    
    def generate_weekly_meal_plan(self, 
                                daily_calories: int, 
                                daily_protein: int, 
                                daily_carbs: int, 
                                daily_fat: int) -> Dict:
        """Generate a 7-day meal plan with variety"""
        if not self.meal_data:
            return {'success': False, 'error': 'Meal data not available'}
        
        weekly_plan = {}
        days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        
        for day in days:
            daily_plan = self.calculate_daily_meal_plan(
                daily_calories, daily_protein, daily_carbs, daily_fat
            )
            if daily_plan['success']:
                weekly_plan[day] = daily_plan
        
        return {
            'success': True,
            'weekly_meal_plan': weekly_plan,
            'summary': {
                'total_daily_calories': daily_calories,
                'total_daily_protein': daily_protein,
                'total_daily_carbs': daily_carbs,
                'total_daily_fat': daily_fat
            }
        }

# Global instance for easy import
meal_plan_calculator = MealPlanCalculator()
