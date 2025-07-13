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
                                preferences: Optional[Dict] = None,
                                max_food_adjustment: float = 0.4) -> Dict:
        """
        Generate a complete daily meal plan scaled to individual needs using Flexible V2 method
        
        Args:
            target_calories: Daily calorie target
            target_protein: Daily protein target (g)
            target_carbs: Daily carbohydrate target (g)
            target_fat: Daily fat target (g)
            preferences: Optional preferences for meal selection
            max_food_adjustment: Maximum adjustment for individual foods (default: 0.4 = ±40%)
            
        Returns:
            Dictionary with scaled daily meal plan using flexible v2 method
        """
        if self.meal_data is None:
            return {
                'success': False,
                'error': 'Meal plan data not available'
            }
        
        # Always use flexible v2 method for best accuracy and results
        return self.calculate_daily_meal_plan_flexible_v2(
            target_calories, target_protein, target_carbs, target_fat, 
            preferences, max_food_adjustment, use_optimized_distribution=True
        )
    
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

    def _select_meals_smart(self, target_protein_pct: float, target_carb_pct: float, target_fat_pct: float, preferences: Optional[Dict] = None) -> Dict:
        """
        Smart meal selection based on target macro percentages
        Chooses meals that best match the desired macro ratios
        """
        if not self.meal_data:
            return {}
        
        def calculate_macro_score(meal, target_p_pct, target_c_pct, target_f_pct):
            """Calculate how well a meal matches target macro percentages"""
            total_cals = meal['total_base_calories']
            if total_cals == 0:
                return float('inf')  # Avoid division by zero
            
            # Calculate meal's macro percentages
            p_pct = (meal['total_base_protein'] * 4 / total_cals) * 100
            c_pct = (meal['total_base_carbs'] * 4 / total_cals) * 100
            f_pct = (meal['total_base_fat'] * 9 / total_cals) * 100
            
            # Calculate distance from target (lower is better)
            p_diff = abs(p_pct - target_p_pct)
            c_diff = abs(c_pct - target_c_pct)
            f_diff = abs(f_pct - target_f_pct)
            
            # Weighted score (protein weighted higher for fitness goals)
            score = (p_diff * 1.5) + c_diff + f_diff
            return score
        
        selected = {}
        
        # Select best breakfast
        sarapan_options = self.meal_data['meal_templates']['sarapan']
        if sarapan_options:
            best_breakfast = min(sarapan_options, 
                               key=lambda m: calculate_macro_score(m, target_protein_pct, target_carb_pct, target_fat_pct))
            selected['sarapan'] = best_breakfast
        
        # Select best lunch  
        makan_siang_options = self.meal_data['meal_templates']['makan_siang']
        if makan_siang_options:
            best_lunch = min(makan_siang_options,
                           key=lambda m: calculate_macro_score(m, target_protein_pct, target_carb_pct, target_fat_pct))
            selected['makan_siang'] = best_lunch
        
        # Select best dinner
        makan_malam_options = self.meal_data['meal_templates']['makan_malam']
        if makan_malam_options:
            best_dinner = min(makan_malam_options,
                            key=lambda m: calculate_macro_score(m, target_protein_pct, target_carb_pct, target_fat_pct))
            selected['makan_malam'] = best_dinner
        
        # Select best snack
        snack_options = self.meal_data['snack_templates']
        if snack_options:
            best_snack = min(snack_options,
                           key=lambda m: calculate_macro_score(m, target_protein_pct, target_carb_pct, target_fat_pct))
            selected['snack'] = best_snack
        
        return selected

    def calculate_daily_meal_plan_smart(self, 
                                      target_calories: int, 
                                      target_protein: int, 
                                      target_carbs: int, 
                                      target_fat: int,
                                      preferences: Optional[Dict] = None) -> Dict:
        """
        Generate daily meal plan with smart meal selection for better macro matching
        """
        if self.meal_data is None:
            return {
                'success': False,
                'error': 'Meal plan data not available'
            }
        
        try:
            # Calculate target macro percentages
            target_protein_pct = (target_protein * 4 / target_calories) * 100
            target_carb_pct = (target_carbs * 4 / target_calories) * 100  
            target_fat_pct = (target_fat * 9 / target_calories) * 100
            
            # Use smart meal selection
            selected_meals = self._select_meals_smart(target_protein_pct, target_carb_pct, target_fat_pct, preferences)
            
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
                'method': 'smart_selection',
                'target_macros': {
                    'protein_pct': round(target_protein_pct, 1),
                    'carb_pct': round(target_carb_pct, 1), 
                    'fat_pct': round(target_fat_pct, 1)
                },
                'selected_meals_info': {
                    'breakfast': selected_meals.get('sarapan', {}).get('name', 'None'),
                    'lunch': selected_meals.get('makan_siang', {}).get('name', 'None'),
                    'dinner': selected_meals.get('makan_malam', {}).get('name', 'None'),
                    'snack': selected_meals.get('snack', {}).get('name', 'None')
                },
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
                'error': f'Error calculating smart meal plan: {str(e)}'
            }
    
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
    
    def _scale_meal_flexible(self, meal_template: Dict, target_calories: float, 
                           target_protein: float = None, target_carbs: float = None, 
                           target_fat: float = None, max_adjustment: float = 0.5) -> Dict:
        """
        Flexible meal scaling with individual food item adjustments for better macro matching
        
        Args:
            meal_template: Base meal template
            target_calories: Target calorie amount for this meal
            target_protein: Optional target protein (g) for better protein matching
            target_carbs: Optional target carbs (g) for better carb matching  
            target_fat: Optional target fat (g) for better fat matching
            max_adjustment: Maximum adjustment factor for individual foods (0.5 = ±50%)
            
        Returns:
            Scaled meal with flexible adjustments
        """
        base_calories = meal_template['total_base_calories']
        base_scaling_factor = target_calories / base_calories
        
        # Start with base scaling
        scaled_foods = []
        
        # Calculate target macros if not provided (proportional to calories)
        if target_protein is None:
            target_protein = (meal_template['total_base_protein'] / base_calories) * target_calories
        if target_carbs is None:
            target_carbs = (meal_template['total_base_carbs'] / base_calories) * target_calories
        if target_fat is None:
            target_fat = (meal_template['total_base_fat'] / base_calories) * target_calories
        
        # Categorize foods by macro dominance
        protein_foods = []
        carb_foods = []
        fat_foods = []
        balanced_foods = []
        
        for food in meal_template['foods']:
            base_amount = food['base_amount'] * base_scaling_factor
            
            # Calculate macro percentages for this food
            food_calories = base_amount * food['calories_per_unit']
            if food_calories > 0:
                protein_pct = (food['protein_per_unit'] * base_amount * 4) / food_calories * 100
                carb_pct = (food['carbs_per_unit'] * base_amount * 4) / food_calories * 100
                fat_pct = (food['fat_per_unit'] * base_amount * 9) / food_calories * 100
                
                # Categorize by dominant macro (>40% of calories)
                if protein_pct > 40:
                    protein_foods.append((food, base_amount))
                elif carb_pct > 40:
                    carb_foods.append((food, base_amount))
                elif fat_pct > 40:
                    fat_foods.append((food, base_amount))
                else:
                    balanced_foods.append((food, base_amount))
            else:
                balanced_foods.append((food, base_amount))
        
        # Apply smart adjustments
        adjusted_foods = []
        
        # Calculate current totals with base scaling
        current_protein = sum(food['protein_per_unit'] * amount for food, amount in 
                            protein_foods + carb_foods + fat_foods + balanced_foods)
        current_carbs = sum(food['carbs_per_unit'] * amount for food, amount in 
                          protein_foods + carb_foods + fat_foods + balanced_foods)
        current_fat = sum(food['fat_per_unit'] * amount for food, amount in 
                        protein_foods + carb_foods + fat_foods + balanced_foods)
        
        # Calculate adjustment factors for each macro category
        protein_adjustment = self._calculate_safe_adjustment(
            current_protein, target_protein, max_adjustment)
        carb_adjustment = self._calculate_safe_adjustment(
            current_carbs, target_carbs, max_adjustment)
        fat_adjustment = self._calculate_safe_adjustment(
            current_fat, target_fat, max_adjustment)
        
        # Apply adjustments to categorized foods
        all_foods = [
            (protein_foods, protein_adjustment, 'protein'),
            (carb_foods, carb_adjustment, 'carb'),
            (fat_foods, fat_adjustment, 'fat'),
            (balanced_foods, 1.0, 'balanced')  # Keep balanced foods as-is
        ]
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for food_group, adjustment, category in all_foods:
            for food, base_amount in food_group:
                # Apply category-specific adjustment
                final_amount = base_amount * adjustment
                
                # Ensure minimum reasonable portions
                min_amount = food['base_amount'] * base_scaling_factor * 0.3
                max_amount = food['base_amount'] * base_scaling_factor * (1 + max_adjustment)
                final_amount = max(min_amount, min(final_amount, max_amount))
                
                # Calculate final nutrition values
                food_calories = final_amount * food['calories_per_unit']
                food_protein = final_amount * food['protein_per_unit']
                food_carbs = final_amount * food['carbs_per_unit']
                food_fat = final_amount * food['fat_per_unit']
                
                adjusted_foods.append({
                    'nama': food['nama'],
                    'amount': round(final_amount),
                    'unit': food['unit'],
                    'calories': round(food_calories),
                    'protein': round(food_protein, 1),
                    'carbs': round(food_carbs, 1),
                    'fat': round(food_fat, 1),
                    'category': category,
                    'adjustment_factor': round(adjustment, 2),
                    'base_scaled_amount': round(base_amount)
                })
                
                total_calories += food_calories
                total_protein += food_protein
                total_carbs += food_carbs
                total_fat += food_fat
        
        return {
            'meal_id': meal_template['id'],
            'meal_name': meal_template['name'],
            'description': meal_template['description'],
            'method': 'flexible_adjustment',
            'base_scaling_factor': round(base_scaling_factor, 2),
            'macro_adjustments': {
                'protein_adjustment': round(protein_adjustment, 2),
                'carb_adjustment': round(carb_adjustment, 2),
                'fat_adjustment': round(fat_adjustment, 2)
            },
            'foods': adjusted_foods,
            'scaled_calories': total_calories,
            'scaled_protein': total_protein,
            'scaled_carbs': total_carbs,
            'scaled_fat': total_fat,
            'targets_met': {
                'calories': abs(total_calories - target_calories) < 25,
                'protein': abs(total_protein - target_protein) < 5,
                'carbs': abs(total_carbs - target_carbs) < 8,
                'fat': abs(total_fat - target_fat) < 3
            }
        }
    
    def _calculate_safe_adjustment(self, current: float, target: float, max_adjustment: float) -> float:
        """Calculate safe adjustment factor within limits"""
        if current == 0:
            return 1.0
        
        ideal_adjustment = target / current
        
        # Limit adjustment to max_adjustment range
        min_factor = 1.0 - max_adjustment
        max_factor = 1.0 + max_adjustment
        
        return max(min_factor, min(ideal_adjustment, max_factor))
    
    def calculate_daily_meal_plan_flexible(self, 
                                         target_calories: int, 
                                         target_protein: int, 
                                         target_carbs: int, 
                                         target_fat: int,
                                         preferences: Optional[Dict] = None,
                                         max_food_adjustment: float = 0.4) -> Dict:
        """
        Generate daily meal plan with flexible individual food adjustments
        
        Args:
            target_calories: Daily calorie target
            target_protein: Daily protein target (g)
            target_carbs: Daily carbohydrate target (g)
            target_fat: Daily fat target (g)
            preferences: Optional preferences for meal selection
            max_food_adjustment: Maximum adjustment for individual foods (0.4 = ±40%)
            
        Returns:
            Dictionary with flexibly adjusted daily meal plan
        """
        if self.meal_data is None:
            return {
                'success': False,
                'error': 'Meal plan data not available'
            }
        
        try:
            # Calculate target macro percentages for smart selection
            target_protein_pct = (target_protein * 4 / target_calories) * 100
            target_carb_pct = (target_carbs * 4 / target_calories) * 100  
            target_fat_pct = (target_fat * 9 / target_calories) * 100
            
            # Use smart meal selection
            selected_meals = self._select_meals_smart(target_protein_pct, target_carb_pct, target_fat_pct, preferences)
            
            # Calculate meal targets (calories and macros)
            meal_calorie_targets = {
                'sarapan': target_calories * 0.25,
                'makan_siang': target_calories * 0.40,
                'makan_malam': target_calories * 0.30,
                'snack': target_calories * 0.05
            }
            
            meal_protein_targets = {
                'sarapan': target_protein * 0.25,
                'makan_siang': target_protein * 0.40,
                'makan_malam': target_protein * 0.30,
                'snack': target_protein * 0.05
            }
            
            meal_carb_targets = {
                'sarapan': target_carbs * 0.25,
                'makan_siang': target_carbs * 0.40,
                'makan_malam': target_carbs * 0.30,
                'snack': target_carbs * 0.05
            }
            
            meal_fat_targets = {
                'sarapan': target_fat * 0.25,
                'makan_siang': target_fat * 0.40,
                'makan_malam': target_fat * 0.30,
                'snack': target_fat * 0.05
            }
            
            scaled_meals = {}
            total_scaled_calories = 0
            total_scaled_protein = 0
            total_scaled_carbs = 0
            total_scaled_fat = 0
            
            # Scale each meal with flexible adjustments
            for meal_type, meal_template in selected_meals.items():
                if meal_template:
                    scaled_meal = self._scale_meal_flexible(
                        meal_template, 
                        meal_calorie_targets[meal_type],
                        meal_protein_targets[meal_type],
                        meal_carb_targets[meal_type],
                        meal_fat_targets[meal_type],
                        max_food_adjustment
                    )
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
            
            # Calculate differences
            calorie_diff = total_scaled_calories - target_calories
            protein_diff = total_scaled_protein - target_protein
            carb_diff = total_scaled_carbs - target_carbs
            fat_diff = total_scaled_fat - target_fat
            
            return {
                'success': True,
                'method': 'flexible_adjustment',
                'max_food_adjustment': max_food_adjustment,
                'selected_meals_info': {
                    'breakfast': selected_meals.get('sarapan', {}).get('name', 'None'),
                    'lunch': selected_meals.get('makan_siang', {}).get('name', 'None'),
                    'dinner': selected_meals.get('makan_malam', {}).get('name', 'None'),
                    'snack': selected_meals.get('snack', {}).get('name', 'None')
                },
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
                'differences': {
                    'calories': round(calorie_diff, 1),
                    'protein': round(protein_diff, 1),
                    'carbs': round(carb_diff, 1),
                    'fat': round(fat_diff, 1)
                },
                'accuracy': {
                    'calories': round(calorie_accuracy, 1),
                    'protein': round(protein_accuracy, 1),
                    'carbs': round(carb_accuracy, 1),
                    'fat': round(fat_accuracy, 1)
                },
                'targets_met': {
                    'overall': abs(calorie_diff) < 30 and abs(protein_diff) < 8 and abs(carb_diff) < 12 and abs(fat_diff) < 5,
                    'calories': abs(calorie_diff) < 30,
                    'protein': abs(protein_diff) < 8,
                    'carbs': abs(carb_diff) < 12,
                    'fat': abs(fat_diff) < 5
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error calculating flexible meal plan: {str(e)}'
            }
    
    def _calculate_optimized_meal_targets(self, total_calories: int, total_protein: int, 
                                        total_carbs: int, total_fat: int) -> Dict:
        """
        Calculate optimized meal targets based on typical meal macro distributions
        Different meals have different optimal macro patterns
        """
        
        # Meal distribution percentages (customizable)
        distributions = {
            'sarapan': {
                'calories': 0.25,
                'protein': 0.20,  # Breakfast typically lower protein
                'carbs': 0.30,    # Higher carbs for energy
                'fat': 0.25
            },
            'makan_siang': {
                'calories': 0.40,
                'protein': 0.45,  # Lunch highest protein
                'carbs': 0.40,    # Moderate carbs
                'fat': 0.35       # Moderate fat
            },
            'makan_malam': {
                'calories': 0.30,
                'protein': 0.30,  # Moderate protein
                'carbs': 0.25,    # Lower carbs (evening)
                'fat': 0.35       # Higher fat for satiety
            },
            'snack': {
                'calories': 0.05,
                'protein': 0.05,  # Light protein
                'carbs': 0.05,    # Light carbs
                'fat': 0.05       # Light fat
            }
        }
        
        meal_targets = {}
        
        for meal_type, dist in distributions.items():
            meal_targets[meal_type] = {
                'calories': total_calories * dist['calories'],
                'protein': total_protein * dist['protein'],
                'carbs': total_carbs * dist['carbs'],
                'fat': total_fat * dist['fat']
            }
        
        return meal_targets

    def calculate_daily_meal_plan_flexible_v2(self, 
                                             target_calories: int, 
                                             target_protein: int, 
                                             target_carbs: int, 
                                             target_fat: int,
                                             preferences: Optional[Dict] = None,
                                             max_food_adjustment: float = 0.4,
                                             use_optimized_distribution: bool = True) -> Dict:
        """
        Enhanced flexible meal planning with optimized meal target distribution
        """
        if self.meal_data is None:
            return {
                'success': False,
                'error': 'Meal plan data not available'
            }
        
        try:
            # Calculate target macro percentages for smart selection
            target_protein_pct = (target_protein * 4 / target_calories) * 100
            target_carb_pct = (target_carbs * 4 / target_calories) * 100  
            target_fat_pct = (target_fat * 9 / target_calories) * 100
            
            # Use smart meal selection
            selected_meals = self._select_meals_smart(target_protein_pct, target_carb_pct, target_fat_pct, preferences)
            
            # Calculate meal targets using optimized distribution
            if use_optimized_distribution:
                meal_targets = self._calculate_optimized_meal_targets(
                    target_calories, target_protein, target_carbs, target_fat
                )
            else:
                # Fallback to proportional distribution
                meal_targets = {
                    'sarapan': {
                        'calories': target_calories * 0.25,
                        'protein': target_protein * 0.25,
                        'carbs': target_carbs * 0.25,
                        'fat': target_fat * 0.25
                    },
                    'makan_siang': {
                        'calories': target_calories * 0.40,
                        'protein': target_protein * 0.40,
                        'carbs': target_carbs * 0.40,
                        'fat': target_fat * 0.40
                    },
                    'makan_malam': {
                        'calories': target_calories * 0.30,
                        'protein': target_protein * 0.30,
                        'carbs': target_carbs * 0.30,
                        'fat': target_fat * 0.30
                    },
                    'snack': {
                        'calories': target_calories * 0.05,
                        'protein': target_protein * 0.05,
                        'carbs': target_carbs * 0.05,
                        'fat': target_fat * 0.05
                    }
                }
            
            scaled_meals = {}
            total_scaled_calories = 0
            total_scaled_protein = 0
            total_scaled_carbs = 0
            total_scaled_fat = 0
            
            # Scale each meal with flexible adjustments
            for meal_type, meal_template in selected_meals.items():
                if meal_template and meal_type in meal_targets:
                    targets = meal_targets[meal_type]
                    scaled_meal = self._scale_meal_flexible(
                        meal_template, 
                        targets['calories'],
                        targets['protein'],
                        targets['carbs'],
                        targets['fat'],
                        max_food_adjustment
                    )
                    scaled_meals[meal_type] = scaled_meal
                    
                    # Add to totals
                    total_scaled_calories += scaled_meal['scaled_calories']
                    total_scaled_protein += scaled_meal['scaled_protein']
                    total_scaled_carbs += scaled_meal['scaled_carbs']
                    total_scaled_fat += scaled_meal['scaled_fat']
            
            # Calculate accuracy and differences
            calorie_accuracy = (total_scaled_calories / target_calories) * 100
            protein_accuracy = (total_scaled_protein / target_protein) * 100
            carb_accuracy = (total_scaled_carbs / target_carbs) * 100
            fat_accuracy = (total_scaled_fat / target_fat) * 100
            
            calorie_diff = total_scaled_calories - target_calories
            protein_diff = total_scaled_protein - target_protein
            carb_diff = total_scaled_carbs - target_carbs
            fat_diff = total_scaled_fat - target_fat
            
            return {
                'success': True,
                'method': 'flexible_v2',
                'max_food_adjustment': max_food_adjustment,
                'optimized_distribution': use_optimized_distribution,
                'selected_meals_info': {
                    'breakfast': selected_meals.get('sarapan', {}).get('name', 'None'),
                    'lunch': selected_meals.get('makan_siang', {}).get('name', 'None'),
                    'dinner': selected_meals.get('makan_malam', {}).get('name', 'None'),
                    'snack': selected_meals.get('snack', {}).get('name', 'None')
                },
                'meal_targets': {
                    meal_type: {
                        'calories': round(targets['calories']),
                        'protein': round(targets['protein'], 1),
                        'carbs': round(targets['carbs'], 1),
                        'fat': round(targets['fat'], 1)
                    } for meal_type, targets in meal_targets.items()
                },
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
                'differences': {
                    'calories': round(calorie_diff, 1),
                    'protein': round(protein_diff, 1),
                    'carbs': round(carb_diff, 1),
                    'fat': round(fat_diff, 1)
                },
                'accuracy': {
                    'calories': round(calorie_accuracy, 1),
                    'protein': round(protein_accuracy, 1),
                    'carbs': round(carb_accuracy, 1),
                    'fat': round(fat_accuracy, 1)
                },
                'targets_met': {
                    'overall': abs(calorie_diff) < 50 and abs(protein_diff) < 10 and abs(carb_diff) < 15 and abs(fat_diff) < 8,
                    'calories': abs(calorie_diff) < 50,
                    'protein': abs(protein_diff) < 10,
                    'carbs': abs(carb_diff) < 15,
                    'fat': abs(fat_diff) < 8
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error calculating flexible v2 meal plan: {str(e)}'
            }

    # ...existing code...
