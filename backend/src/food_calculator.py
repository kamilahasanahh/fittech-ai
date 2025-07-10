"""
Food Calculator Module for XGFitness AI
Calculates specific food amounts to meet nutrition targets
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional

# Import the new meal plan calculator
try:
    from .meal_plan_calculator import meal_plan_calculator
except ImportError:
    try:
        from meal_plan_calculator import meal_plan_calculator
    except ImportError:
        meal_plan_calculator = None
        print("Warning: Meal plan calculator not available")

class FoodCalculator:
    """
    Calculates food examples to meet specific nutrition targets
    Now integrated with meal plan system
    """
    
    def __init__(self, templates_dir: str = 'data'):
        """
        Initialize food calculator
        
        Args:
            templates_dir: Directory containing nutrition CSV files
        """
        self.templates_dir = templates_dir
        self.nutrition_data = None
        self.meal_calculator = meal_plan_calculator
        self.load_nutrition_data()
    
    def load_nutrition_data(self):
        """Load nutrition data from CSV"""
        try:
            # Try multiple possible paths for the nutrition file
            possible_paths = [
                os.path.join(self.templates_dir, 'nutrition', 'nutrition_macro_summary.json'),
                os.path.join('..', self.templates_dir, 'nutrition', 'nutrition_macro_summary.json'),
                os.path.join('data', 'nutrition', 'nutrition_macro_summary.json'),
                os.path.join('data', 'nutrition_database.json'),
                'nutrition_macro_summary.json'
            ]
            
            for nutrition_path in possible_paths:
                if os.path.exists(nutrition_path):
                    with open(nutrition_path, 'r', encoding='utf-8') as f:
                        nutrition_data = json.load(f)
                    self.nutrition_data = pd.DataFrame(nutrition_data['nutrition_database'])
                    print(f"Loaded nutrition data: {len(self.nutrition_data)} foods from {nutrition_path}")
                    return
            
            print(f"Warning: Nutrition JSON file not found in any of these paths: {possible_paths}")
            self.nutrition_data = None
        except Exception as e:
            print(f"Error loading nutrition data: {e}")
            self.nutrition_data = None
    
    def calculate_food_examples(self, target_protein: int, target_carbs: int, target_fat: int) -> Dict:
        """
        Calculate food examples to meet nutrition targets
        
        Args:
            target_protein: Target protein in grams
            target_carbs: Target carbohydrates in grams
            target_fat: Target fat in grams
            
        Returns:
            Dictionary with food examples and calculations
        """
        if self.nutrition_data is None:
            return {
                'success': False,
                'error': 'Nutrition data not available',
                'food_examples': []
            }
        
        try:
            # Define food categories based on macro content
            protein_foods = ['Ayam Goreng tanpa Pelapis (Kulit Dimakan)', 'Ikan Panggang', 'Tempe Goreng', 'Tahu Goreng']
            carb_foods = ['Nasi Putih (Butir-Sedang, Dimasak)', 'Roti Gandum', 'Mie Telur (Ditambah, Masak)']
            
            food_examples = []
            total_calories = 0
            
            # Calculate main protein source (70% of target protein)
            protein_target = target_protein * 0.7
            protein_food_data = self.nutrition_data[self.nutrition_data['Nama'].isin(protein_foods)]
            
            if len(protein_food_data) > 0:
                selected_protein = protein_food_data.iloc[0]  # Take first available
                protein_per_100g = selected_protein['Protein (g)']
                amount_needed = (protein_target / protein_per_100g) * 100
                calories_contributed = (amount_needed/100) * selected_protein['Kalori (kkal)']
                
                food_examples.append({
                    'nama': selected_protein['Nama'],
                    'jumlah': f"{int(amount_needed)}g",
                    'target': 'protein',
                    'kontribusi': f"{int(protein_target)}g protein",
                    'kalori': int(calories_contributed)
                })
                total_calories += calories_contributed
            
            # Calculate main carb source (60% of target carbs)
            carb_target = target_carbs * 0.6
            carb_food_data = self.nutrition_data[self.nutrition_data['Nama'].isin(carb_foods)]
            
            if len(carb_food_data) > 0:
                selected_carb = carb_food_data.iloc[0]  # Take first available
                carb_per_100g = selected_carb['Karbohidrat (g)']
                amount_needed = (carb_target / carb_per_100g) * 100
                calories_contributed = (amount_needed/100) * selected_carb['Kalori (kkal)']
                
                food_examples.append({
                    'nama': selected_carb['Nama'],
                    'jumlah': f"{int(amount_needed)}g",
                    'target': 'carbs',
                    'kontribusi': f"{int(carb_target)}g karbohidrat",
                    'kalori': int(calories_contributed)
                })
                total_calories += calories_contributed
            
            # Calculate secondary protein source (30% of target protein)
            remaining_protein = target_protein * 0.3
            remaining_protein_foods = [f for f in protein_foods if f != (selected_protein['Nama'] if len(protein_food_data) > 0 else '')]
            
            if remaining_protein_foods:
                secondary_protein_data = self.nutrition_data[self.nutrition_data['Nama'] == remaining_protein_foods[0]]
                if len(secondary_protein_data) > 0:
                    secondary_protein = secondary_protein_data.iloc[0]
                    protein_per_100g = secondary_protein['Protein (g)']
                    amount_needed = (remaining_protein / protein_per_100g) * 100
                    calories_contributed = (amount_needed/100) * secondary_protein['Kalori (kkal)']
                    
                    food_examples.append({
                        'nama': secondary_protein['Nama'],
                        'jumlah': f"{int(amount_needed)}g",
                        'target': 'protein',
                        'kontribusi': f"{int(remaining_protein)}g protein",
                        'kalori': int(calories_contributed)
                    })
                    total_calories += calories_contributed
            
            # Add healthy fat example (40% of target fat)
            fat_target = target_fat * 0.4
            # Find food with highest fat content for this example
            if len(self.nutrition_data) > 0:
                fat_food = self.nutrition_data.loc[self.nutrition_data['Lemak (g)'].idxmax()]
                fat_per_100g = fat_food['Lemak (g)']
                
                if fat_per_100g > 0:
                    amount_needed = (fat_target / fat_per_100g) * 100
                    calories_contributed = (amount_needed/100) * fat_food['Kalori (kkal)']
                    
                    food_examples.append({
                        'nama': fat_food['Nama'],
                        'jumlah': f"{int(amount_needed)}g",
                        'target': 'fat',
                        'kontribusi': f"{int(fat_target)}g lemak",
                        'kalori': int(calories_contributed)
                    })
                    total_calories += calories_contributed
            
            return {
                'success': True,
                'food_examples': food_examples,
                'total_calculated_calories': int(total_calories),
                'nutrition_breakdown': {
                    'protein_from_examples': int(target_protein),
                    'carbs_from_examples': int(carb_target),
                    'fat_from_examples': int(fat_target)
                },
                'note': 'Contoh perhitungan makanan untuk memenuhi target nutrisi. Variasikan dengan makanan lain sesuai selera.'
            }
            
        except Exception as e:
            return {
                'success': False, 
                'error': f'Error calculating food examples: {str(e)}',
                'food_examples': []
            }
    
    def get_meal_plan_recommendations(self, target_calories: int, target_protein: int, 
                                    target_carbs: int, target_fat: int) -> Dict:
        """
        Get comprehensive meal plan recommendations using the new meal plan system
        
        Args:
            target_calories: Target calories per day
            target_protein: Target protein in grams
            target_carbs: Target carbohydrates in grams
            target_fat: Target fat in grams
            
        Returns:
            Dictionary with meal plan recommendations
        """
        if self.meal_calculator is None:
            # Fallback to old system
            return self.calculate_food_examples(target_protein, target_carbs, target_fat)
        
        try:
            # Generate daily meal plan
            daily_plan = self.meal_calculator.calculate_daily_meal_plan(
                target_calories, target_protein, target_carbs, target_fat
            )
            
            if daily_plan['success']:
                # Format for compatibility with existing API
                formatted_response = {
                    'success': True,
                    'meal_plan_type': 'comprehensive',
                    'daily_meal_plan': daily_plan['daily_meal_plan'],
                    'nutrition_summary': daily_plan['nutrition_summary'],
                    'accuracy': daily_plan['accuracy'],
                    'meal_distribution': daily_plan['meal_distribution'],
                    'note': 'Rencana makan harian lengkap yang disesuaikan dengan target nutrisi Anda. '
                           'Porsi dapat disesuaikan sesuai kebutuhan dan selera.'
                }
                
                # Add legacy food_examples format for backward compatibility
                food_examples = []
                for meal_type, meal_data in daily_plan['daily_meal_plan'].items():
                    for food in meal_data['foods']:
                        food_examples.append({
                            'nama': food['nama'],
                            'jumlah': f"{food['amount']}{food['unit']}",
                            'target': meal_type,
                            'kontribusi': f"{food['calories']} kcal",
                            'kalori': food['calories']
                        })
                
                formatted_response['food_examples'] = food_examples
                formatted_response['total_calculated_calories'] = daily_plan['nutrition_summary']['total_calories']
                
                return formatted_response
            else:
                return daily_plan
                
        except Exception as e:
            # Fallback to old system if new system fails
            print(f"Meal plan system error, falling back to legacy system: {e}")
            return self.calculate_food_examples(target_protein, target_carbs, target_fat)
    
    def get_weekly_meal_plan(self, target_calories: int, target_protein: int, 
                           target_carbs: int, target_fat: int) -> Dict:
        """
        Generate a 7-day meal plan
        
        Args:
            target_calories: Target calories per day
            target_protein: Target protein in grams
            target_carbs: Target carbohydrates in grams
            target_fat: Target fat in grams
            
        Returns:
            Dictionary with weekly meal plan
        """
        if self.meal_calculator is None:
            return {
                'success': False,
                'error': 'Meal plan calculator not available'
            }
        
        return self.meal_calculator.generate_weekly_meal_plan(
            target_calories, target_protein, target_carbs, target_fat
        )
    
    def get_meal_options_by_type(self, meal_type: str) -> List[Dict]:
        """
        Get available meal options for a specific meal type
        
        Args:
            meal_type: Type of meal (sarapan, makan_siang, makan_malam, snack)
            
        Returns:
            List of available meal options
        """
        if self.meal_calculator is None:
            return []
        
        return self.meal_calculator.get_meal_options(meal_type)
    
    def scale_specific_meal(self, meal_id: str, target_calories: float) -> Dict:
        """
        Scale a specific meal to target calories
        
        Args:
            meal_id: ID of the meal to scale
            target_calories: Target calories for the meal
            
        Returns:
            Dictionary with scaled meal data
        """
        if self.meal_calculator is None:
            return {
                'success': False,
                'error': 'Meal plan calculator not available'
            }
        
        return self.meal_calculator.calculate_single_meal(meal_id, target_calories)

# Global instance for easy import
food_calculator = FoodCalculator()
