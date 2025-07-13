#!/usr/bin/env python3
"""
Debug script to test meal plan calculations
"""
import sys
import os
import json

# Add the backend src directory to path
backend_src_path = os.path.join(os.path.dirname(__file__), 'backend', 'src')
sys.path.insert(0, backend_src_path)

from meal_plan_calculator import MealPlanCalculator

def debug_meal_calculation():
    # Test with the values from the example
    target_calories = 2503
    target_protein = 161
    target_carbs = 126
    target_fat = 70
    
    print(f"Target: {target_calories} kcal, {target_protein}g protein, {target_carbs}g carbs, {target_fat}g fat")
    
    # Load meal data
    meal_data_path = os.path.join(os.path.dirname(__file__), 'data', 'meals', 'meal_plans.json')
    with open(meal_data_path, 'r', encoding='utf-8') as f:
        meal_data = json.load(f)
    
    calculator = MealPlanCalculator(meal_data)
    
    # Calculate meal plan
    result = calculator.calculate_daily_meal_plan(
        target_calories, target_protein, target_carbs, target_fat
    )
    
    if result['success']:
        print("\n=== MEAL PLAN CALCULATION RESULTS ===")
        print(f"Method: {result['method']}")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Total: {result['nutrition_summary']['total_calories']} kcal")
        print(f"Protein: {result['nutrition_summary']['total_protein']}g")
        print(f"Carbs: {result['nutrition_summary']['total_carbs']}g")
        print(f"Fat: {result['nutrition_summary']['total_fat']}g")
        
        print("\n=== MEAL BREAKDOWN ===")
        for meal_type, meal_data in result['daily_meal_plan'].items():
            print(f"\n{meal_type.upper()}: {meal_data['meal_name']}")
            print(f"Target: {meal_data.get('target_calories', 'N/A')} kcal")
            print(f"Actual: {meal_data['scaled_calories']:.0f} kcal, {meal_data['scaled_protein']:.1f}g protein")
            
            for food in meal_data['foods']:
                print(f"  - {food['nama']}: {food['amount']}g = {food['calories']} kcal, {food['protein']}g protein, {food['carbs']}g carbs, {food['fat']}g fat")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    debug_meal_calculation()
