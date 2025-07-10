# Fitness Template Information Guide

## Overview
This guide explains how to use the workout and nutrition templates for personalized fitness planning. The templates are designed to work together, providing comprehensive guidance for different goals, activity levels, and body compositions.

## Workout Templates

### Template Structure
```
template_id,goal,activity_level,workout_type,days_per_week,workout_schedule,sets_per_exercise,exercises_per_session,cardio_minutes_per_day,cardio_sessions_per_day
```

### Workout Template Data
| ID | Goal | Activity Level | Workout Type | Days/Week | Schedule | Sets/Exercise | Exercises/Session | Cardio Min/Day | Cardio Sessions/Day |
|----|------|---------------|--------------|-----------|----------|---------------|-------------------|----------------|-------------------|
| 1 | Fat Loss | Low Activity | Full Body | 2 | WXXWXXX | 3 | 6 | 35 | 1 |
| 2 | Fat Loss | Moderate Activity | Full Body | 3 | WXWXWXX | 3 | 8 | 45 | 1 |
| 3 | Fat Loss | High Activity | Upper/Lower Split | 4 | ABXABXX | 3 | 10 | 55 | 1 |
| 4 | Muscle Gain | Low Activity | Full Body | 2 | WXXWXXX | 4 | 6 | 18 | 0 |
| 5 | Muscle Gain | Moderate Activity | Upper/Lower Split | 4 | ABXABXX | 4 | 8 | 25 | 1 |
| 6 | Muscle Gain | High Activity | Push/Pull/Legs | 3 | ABCX | 4 | 10 | 30 | 1 |
| 7 | Maintenance | Low Activity | Full Body | 2 | WXXWXXX | 3 | 4 | 30 | 0 |
| 8 | Maintenance | Moderate Activity | Upper/Lower Split | 4 | ABXABXX | 3 | 6 | 40 | 1 |
| 9 | Maintenance | High Activity | Push/Pull/Legs | 3 | ABCX | 3 | 8 | 50 | 1 |

### Workout Type Definitions

#### Full Body Workouts
- **Structure**: Train all major muscle groups in each session
- **Best for**: Beginners, time-limited individuals
- **Muscle Groups**: Legs, chest, back, shoulders, arms, core

#### Upper/Lower Split
- **A (Upper)**: Chest, back, shoulders, arms
- **B (Lower)**: Legs, glutes, core
- **Best for**: Intermediate trainees with moderate time availability

#### Push/Pull/Legs Split
- **A (Push)**: Chest, shoulders, triceps
- **B (Pull)**: Back, biceps
- **C (Legs)**: Quads, hamstrings, glutes, calves
- **Best for**: Advanced trainees with high time availability

### Schedule Notation
- **W** = Workout day
- **X** = Rest day
- **A/B/C** = Different workout sessions in split routines

## Nutrition Templates

### Template Structure
```
template_id,goal,bmi_category,caloric_intake,protein_per_kg,carbs_per_kg,fat_per_kg
```

### Nutrition Template Data
| ID | Goal | BMI Category | Caloric Intake | Protein/kg | Carbs/kg | Fat/kg |
|----|------|--------------|----------------|------------|----------|--------|
| 1 | Fat Loss | Normal | 0.80 | 2.3 | 2.75 | 0.85 |
| 2 | Fat Loss | Overweight | 0.75 | 2.15 | 2.25 | 0.80 |
| 3 | Fat Loss | Obese | 0.70 | 2.45 | 1.75 | 0.80 |
| 4 | Muscle Gain | Underweight | 1.15 | 2.3 | 4.75 | 1.0 |
| 5 | Muscle Gain | Normal | 1.10 | 2.1 | 4.25 | 0.95 |
| 6 | Maintenance | Underweight | 1.00 | 1.8 | 3.25 | 0.90 |
| 7 | Maintenance | Normal | 0.95 | 1.8 | 3.25 | 0.85 |
| 8 | Maintenance | Overweight | 0.90 | 1.8 | 3.25 | 0.80 |

### Nutrition Multipliers Explanation

**All nutrition values are multipliers that should be applied to calculated values:**

#### Caloric Intake Multiplier
- Applied to **Basal Metabolic Rate (BMR) × Activity Factor**
- **Example**: If BMR × Activity Factor = 2000 calories, and multiplier = 0.80, then target calories = 1600

#### Macronutrient Multipliers (per kg body weight)
- **Protein/kg**: Grams of protein per kilogram of body weight
- **Carbs/kg**: Grams of carbohydrates per kilogram of body weight  
- **Fat/kg**: Grams of fat per kilogram of body weight

**Example Calculation for 70kg individual:**
- Protein: 70kg × 2.3 = 161g protein
- Carbs: 70kg × 2.75 = 192.5g carbs
- Fat: 70kg × 0.85 = 59.5g fat

## How to Use Templates

### Step 1: Determine User Profile
1. **Goal**: Fat Loss, Muscle Gain, or Maintenance
2. **Activity Level**: Low, Moderate, or High
3. **BMI Category**: Underweight, Normal, Overweight, or Obese

### Step 2: Select Workout Template
Match user's goal and activity level to appropriate workout template (1-9)

### Step 3: Select Nutrition Template
Match user's goal and BMI category to appropriate nutrition template (1-8)

### Step 4: Apply Calculations
1. Calculate BMR using standard formula
2. Determine activity factor
3. Apply caloric multiplier to BMR × Activity Factor
4. Calculate macronutrients using body weight multipliers

## Template Logic

### Fat Loss Templates
- **Lower caloric intake** (0.70-0.80 multiplier)
- **Higher protein** to preserve muscle mass
- **Moderate to high cardio** volume
- **Higher training frequency** for increased calorie burn

### Muscle Gain Templates
- **Higher caloric intake** (1.10-1.15 multiplier)
- **Higher carbohydrates** for energy and recovery
- **Higher training volume** with adequate recovery
- **Minimal cardio** to preserve calories for muscle growth

### Maintenance Templates
- **Balanced caloric intake** (0.90-1.00 multiplier)
- **Moderate macronutrient distribution**
- **Sustainable training frequency**
- **Moderate cardio** for health benefits

### BMI-Specific Adjustments
- **Underweight**: Higher calories and carbs
- **Normal**: Balanced approach
- **Overweight**: Moderate caloric restriction
- **Obese**: Aggressive caloric restriction with higher protein

## Implementation Notes

1. **Progressive Adjustment**: Start with template values and adjust based on individual response
2. **Monitoring**: Track progress and modify multipliers as needed
3. **Sustainability**: Choose templates that match lifestyle and preferences
4. **Medical Clearance**: Consult healthcare providers before starting any new program