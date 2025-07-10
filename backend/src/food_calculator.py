"""
Food Calculator Module for XGFitness AI
Calculates specific food amounts to meet nutrition targets
"""

import pandas as pd
import os
from typing import Dict, List, Optional

class FoodCalculator:
    """
    Calculates food examples to meet specific nutrition targets
    """
    
    def __init__(self, templates_dir: str = 'data'):
        """
        Initialize food calculator
        
        Args:
            templates_dir: Directory containing nutrition CSV files
        """
        self.templates_dir = templates_dir
        self.nutrition_data = None
        self.load_nutrition_data()
    
    def load_nutrition_data(self):
        """Load nutrition data from CSV"""
        try:
            # Try multiple possible paths for the nutrition file
            possible_paths = [
                os.path.join(self.templates_dir, 'nutrition_macro_summary.csv'),
                os.path.join('..', self.templates_dir, 'nutrition_macro_summary.csv'),
                os.path.join('data', 'nutrition_macro_summary.csv'),
                'nutrition_macro_summary.csv'
            ]
            
            for nutrition_path in possible_paths:
                if os.path.exists(nutrition_path):
                    self.nutrition_data = pd.read_csv(nutrition_path)
                    print(f"Loaded nutrition data: {len(self.nutrition_data)} foods from {nutrition_path}")
                    return
            
            print(f"Warning: Nutrition file not found in any of these paths: {possible_paths}")
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

# Global instance for easy import
food_calculator = FoodCalculator()
