#!/usr/bin/env python3
"""
Debug script to test meal plan calculations
"""
import sys
import os
import json

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.meal_plan_calculator import MealPlanCalculator

def debug_meal_calculation():
    # Test with the values from the example
    target_calories = 2503
    target_protein = 161
    target_carbs = 126
    target_fat = 70
    
    print(f"Target: {target_calories} kcal, {target_protein}g protein, {target_carbs}g carbs, {target_fat}g fat")
    
    # Load meal data using the correct path
    meal_data_path = os.path.join('..', 'data', 'meals', 'meal_plans.json')
    
    calculator = MealPlanCalculator(meal_data_path)
    
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
        
        print("\n=== MEAL TARGETS ===")
        if 'meal_targets' in result:
            for meal_type, targets in result['meal_targets'].items():
                print(f"{meal_type}: {targets['calories']:.0f} kcal, {targets['protein']:.1f}g protein, {targets['carbs']:.1f}g carbs, {targets['fat']:.1f}g fat")
        
        print("\n=== MEAL BREAKDOWN ===")
        for meal_type, meal_data in result['daily_meal_plan'].items():
            target_info = result['meal_targets'].get(meal_type, {})
            print(f"\n{meal_type.upper()}: {meal_data['meal_name']}")
            print(f"Target: {target_info.get('calories', 'N/A')} kcal, {target_info.get('protein', 'N/A')}g protein, {target_info.get('carbs', 'N/A')}g carbs, {target_info.get('fat', 'N/A')}g fat")
            print(f"Actual: {meal_data['scaled_calories']:.0f} kcal, {meal_data['scaled_protein']:.1f}g protein, {meal_data['scaled_carbs']:.1f}g carbs, {meal_data['scaled_fat']:.1f}g fat")
            
            for food in meal_data['foods']:
                print(f"  - {food['nama']}: {food['amount']}g = {food['calories']} kcal, {food['protein']}g protein, {food['carbs']}g carbs, {food['fat']}g fat")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    debug_meal_calculation()
