## calculations.py

def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
    """Calculate BMR using Harris-Benedict equation from thesis"""
    if gender == 'Male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Calculate TDEE based on activity level - exact thesis values"""
    multipliers = {
        'Low Activity': 1.29,
        'Moderate Activity': 1.55,
        'High Activity': 1.81
    }
    return bmr * multipliers[activity_level]

def categorize_bmi(bmi: float) -> str:
    """Categorize BMI value according to thesis"""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'