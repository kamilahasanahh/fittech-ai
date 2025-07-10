// Meal Plan Service for fetching organized meal plans from backend API
class MealPlanService {
  constructor() {
    this.baseURL = 'http://localhost:5000'; // Backend API URL
  }

  async generateDailyMealPlan(targetCalories, targetProtein, targetCarbs, targetFat) {
    try {
      const response = await fetch(`${this.baseURL}/meal-plan`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_calories: targetCalories,
          target_protein: targetProtein,
          target_carbs: targetCarbs,
          target_fat: targetFat
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error generating meal plan:', error);
      
      // Return fallback data structure if API fails
      return {
        success: false,
        error: error.message,
        fallback: true
      };
    }
  }

  async getMealOptions(mealType) {
    try {
      const response = await fetch(`${this.baseURL}/meal-options/${mealType}`);
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching meal options:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async scaleMeal(mealId, targetCalories) {
    try {
      const response = await fetch(`${this.baseURL}/scale-meal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          meal_id: mealId,
          target_calories: targetCalories
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error scaling meal:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // Transform backend meal plan to frontend format
  transformMealPlanToFrontend(backendMealPlan) {
    if (!backendMealPlan.success || !backendMealPlan.meal_plan) {
      return [];
    }

    const mealPlan = backendMealPlan.meal_plan.daily_meal_plan;
    const suggestions = [];

    // Map backend meal types to frontend format
    const mealMapping = {
      'sarapan': { name: '🌅 Sarapan', type: 'sarapan' },
      'makan_siang': { name: '☀️ Makan Siang', type: 'makan_siang' },
      'makan_malam': { name: '🌙 Makan Malam', type: 'makan_malam' },
      'snack': { name: '🍪 Camilan', type: 'camilan' }
    };

    Object.entries(mealMapping).forEach(([backendType, frontendConfig]) => {
      const meal = mealPlan[backendType];
      if (meal) {
        const transformedFoods = meal.foods.map(food => ({
          name: food.nama,
          grams: food.amount,
          actualCalories: food.calories,
          actualProtein: food.protein,
          actualCarbs: food.carbs,
          actualFat: food.fat
        }));

        suggestions.push({
          meal: frontendConfig.name,
          mealType: frontendConfig.type,
          targetCalories: Math.round(meal.scaled_calories),
          targetProtein: Math.round(meal.scaled_protein),
          foods: transformedFoods,
          mealName: meal.meal_name,
          description: meal.description
        });
      }
    });

    return suggestions;
  }
}

// Create singleton instance
const mealPlanService = new MealPlanService();

export { mealPlanService };
