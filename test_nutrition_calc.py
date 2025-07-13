#!/usr/bin/env python3
"""
Test script to verify nutrition calculations for a 70kg, 20-year-old man, 170cm, Fat Loss goal, High Activity
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from calculations import calculate_complete_nutrition_plan

# Test profile: 70kg, 20-year-old man, 170cm, fat loss goal, high activity
user_profile = {
    "age": 20,
    "gender": "Male", 
    "height": 170,
    "weight": 70,
    "activity_level": "High Activity",
    "fitness_goal": "Fat Loss"
}

# Template 1: Fat Loss + Normal BMI
nutrition_template = {
    "template_id": 1,
    "goal": "Fat Loss",
    "bmi_category": "Normal",
    "caloric_intake_multiplier": 0.8,
    "protein_per_kg": 2.3,
    "carbs_per_kg": 1.8,
    "fat_per_kg": 1.0
}

print("=== Test User Profile ===")
print(f"Age: {user_profile['age']}")
print(f"Gender: {user_profile['gender']}")
print(f"Height: {user_profile['height']} cm")
print(f"Weight: {user_profile['weight']} kg")
print(f"Activity Level: {user_profile['activity_level']}")
print(f"Fitness Goal: {user_profile['fitness_goal']}")

print("\n=== Expected Template (ID 1) ===")
print(f"Template ID: {nutrition_template['template_id']}")
print(f"Goal: {nutrition_template['goal']}")
print(f"BMI Category: {nutrition_template['bmi_category']}")
print(f"Caloric Multiplier: {nutrition_template['caloric_intake_multiplier']}")
print(f"Protein per kg: {nutrition_template['protein_per_kg']}")
print(f"Carbs per kg: {nutrition_template['carbs_per_kg']}")
print(f"Fat per kg: {nutrition_template['fat_per_kg']}")

print("\n=== Expected Calculations ===")
print(f"Expected Protein: {user_profile['weight']} kg × {nutrition_template['protein_per_kg']} = {user_profile['weight'] * nutrition_template['protein_per_kg']}g")
print(f"Expected Carbs: {user_profile['weight']} kg × {nutrition_template['carbs_per_kg']} = {user_profile['weight'] * nutrition_template['carbs_per_kg']}g")
print(f"Expected Fat: {user_profile['weight']} kg × {nutrition_template['fat_per_kg']} = {user_profile['weight'] * nutrition_template['fat_per_kg']}g")

# Manual calculation to understand the issue
print("\n=== Manual Calculation Debug ===")
# BMR calculation (Male): 88.362 + (13.397 × weight) + (4.799 × height) - (5.677 × age)
bmr = 88.362 + (13.397 * user_profile['weight']) + (4.799 * user_profile['height']) - (5.677 * user_profile['age'])
print(f"BMR: {bmr:.1f}")

# Activity multiplier for High Activity = 1.81
tdee = bmr * 1.81
print(f"TDEE: {tdee:.1f}")

# Target calories = TDEE × caloric_multiplier
target_calories = tdee * nutrition_template['caloric_intake_multiplier']
print(f"Target Calories: {target_calories:.1f}")

# Template macros (should be the final answer)
template_protein = user_profile['weight'] * nutrition_template['protein_per_kg']
template_carbs = user_profile['weight'] * nutrition_template['carbs_per_kg']
template_fat = user_profile['weight'] * nutrition_template['fat_per_kg']

print(f"Template Protein: {template_protein}g")
print(f"Template Carbs: {template_carbs}g")
print(f"Template Fat: {template_fat}g")

# Calculate macro calories
template_protein_cals = template_protein * 4
template_carb_cals = template_carbs * 4
template_fat_cals = template_fat * 9
template_total_cals = template_protein_cals + template_carb_cals + template_fat_cals

print(f"Template Total Calories: {template_total_cals}")
print(f"Target Calories: {target_calories:.1f}")
print(f"Difference: {abs(template_total_cals - target_calories):.1f} (threshold: 50)")

if abs(template_total_cals - target_calories) > 50:
    scaling_factor = target_calories / template_total_cals
    print(f"❌ SCALING APPLIED! Factor: {scaling_factor:.3f}")
    print(f"Scaled Protein: {template_protein * scaling_factor:.1f}g")
    print(f"Scaled Carbs: {template_carbs * scaling_factor:.1f}g")
    print(f"Scaled Fat: {template_fat * scaling_factor:.1f}g")
else:
    print("✅ No scaling needed, using template values as-is")

print("\n=== Backend Calculation Result ===")
result = calculate_complete_nutrition_plan(user_profile, nutrition_template)

if result['success']:
    calculations = result['calculations']
    macros = calculations['macronutrients']
    
    print(f"✅ Success: {result['success']}")
    print(f"BMR: {calculations['bmr']}")
    print(f"TDEE: {calculations['tdee']}")
    print(f"BMI: {calculations['bmi']} ({calculations['bmi_category']})")
    print(f"Target Calories: {calculations['target_calories']}")
    print(f"Calculation Weight: {calculations['calculation_weight']}")
    
    print(f"\n📊 Macronutrients:")
    print(f"Protein: {macros['protein_g']}g")
    print(f"Carbs: {macros['carbs_g']}g") 
    print(f"Fat: {macros['fat_g']}g")
    
    print(f"\n🔍 Verification:")
    print(f"Protein per kg: {macros['protein_g'] / user_profile['weight']:.1f}g/kg (should be {nutrition_template['protein_per_kg']})")
    print(f"Carbs per kg: {macros['carbs_g'] / user_profile['weight']:.1f}g/kg (should be {nutrition_template['carbs_per_kg']})")
    print(f"Fat per kg: {macros['fat_g'] / user_profile['weight']:.1f}g/kg (should be {nutrition_template['fat_per_kg']})")
    
    # Check if values match expected
    expected_protein = user_profile['weight'] * nutrition_template['protein_per_kg']
    expected_carbs = user_profile['weight'] * nutrition_template['carbs_per_kg']
    expected_fat = user_profile['weight'] * nutrition_template['fat_per_kg']
    
    protein_match = abs(macros['protein_g'] - expected_protein) < 1
    carbs_match = abs(macros['carbs_g'] - expected_carbs) < 1  
    fat_match = abs(macros['fat_g'] - expected_fat) < 1
    
    print(f"\n✅ Results:")
    print(f"Protein {'✓' if protein_match else '✗'}: {macros['protein_g']}g vs expected {expected_protein}g")
    print(f"Carbs {'✓' if carbs_match else '✗'}: {macros['carbs_g']}g vs expected {expected_carbs}g")
    print(f"Fat {'✓' if fat_match else '✗'}: {macros['fat_g']}g vs expected {expected_fat}g")
    
else:
    print(f"❌ Error: {result['error']}")
