// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class ApiService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('❌ API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Health check
  async healthCheck() {
    try {
      const response = await this.api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  // Get recommendations
  async getRecommendations(userData) {
    try {
      console.log('🔄 API: Starting getRecommendations with data:', userData);
      console.log('🔄 API: Base URL:', API_BASE_URL);
      
      // Validate user data
      this.validateUserData(userData);
      console.log('✅ API: User data validation passed');
      
      console.log('🔄 API: Making POST request to /predict...');
      const response = await this.api.post('/predict', userData);
      console.log('✅ API: Response received:', response.status, response.data);
      
      // Check if response has the expected structure (predictions, model_confidence, etc.)
      if (response.data && response.data.predictions) {
        console.log('✅ API: Valid predictions response, returning data');
        return response.data;
      } else if (response.data && response.data.error) {
        console.error('❌ API: Backend returned error:', response.data.error);
        throw new Error(response.data.error);
      } else {
        console.error('❌ API: Unexpected response structure:', response.data);
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      console.error('❌ API: Error in getRecommendations:', error);
      if (error.response) {
        console.error('❌ API: Error response data:', error.response.data);
        console.error('❌ API: Error response status:', error.response.status);
        const errorMessage = error.response.data?.error || 'Server error occurred';
        throw new Error(`Recommendation failed: ${errorMessage}`);
      } else if (error.request) {
        console.error('❌ API: No response received:', error.request);
        throw new Error('Backend server is not responding. Please make sure the Flask server is running on port 5000.');
      } else {
        console.error('❌ API: Request setup error:', error.message);
        throw new Error(`Request failed: ${error.message}`);
      }
    }
  }

  // Get workout templates
  async getWorkoutTemplates() {
    try {
      const response = await this.api.get('/templates');
      return response.data.workout_templates || [];
    } catch (error) {
      throw new Error(`Failed to fetch workout templates: ${error.message}`);
    }
  }

  // Get nutrition templates
  async getNutritionTemplates() {
    try {
      const response = await this.api.get('/templates');
      return response.data.nutrition_templates || [];
    } catch (error) {
      throw new Error(`Failed to fetch nutrition templates: ${error.message}`);
    }
  }

  // Get all templates (both workout and nutrition) in one call
  async getAllTemplates() {
    try {
      const response = await this.api.get('/templates');
      if (response.data.success) {
        return {
          success: true,
          workoutTemplates: response.data.workout_templates || [],
          nutritionTemplates: response.data.nutrition_templates || [],
          templateCount: response.data.template_count || { workout: 0, nutrition: 0 }
        };
      } else {
        throw new Error('API returned unsuccessful response');
      }
    } catch (error) {
      throw new Error(`Failed to fetch templates: ${error.message}`);
    }
  }

  // Get system information
  async getSystemInfo() {
    try {
      const response = await this.api.get('/system/info');
      return response.data.system_info;
    } catch (error) {
      throw new Error(`Failed to fetch system info: ${error.message}`);
    }
  }

  // Retrain models
  async retrainModels() {
    try {
      const response = await this.api.post('/retrain');
      return response.data;
    } catch (error) {
      throw new Error(`Model retraining failed: ${error.message}`);
    }
  }

  // Get meal plan
  async getMealPlan(targetCalories, targetProtein, targetCarbs, targetFat, preferences = {}) {
    try {
      const response = await this.api.post('/meal-plan', {
        target_calories: targetCalories,
        target_protein: targetProtein,
        target_carbs: targetCarbs,
        target_fat: targetFat,
        preferences: preferences
      });
      
      if (response.data.success) {
        return response.data.meal_plan;
      } else {
        throw new Error(response.data.error || 'Failed to generate meal plan');
      }
    } catch (error) {
      if (error.response) {
        const errorMessage = error.response.data?.error || 'Server error occurred';
        throw new Error(`Meal plan generation failed: ${errorMessage}`);
      } else if (error.request) {
        throw new Error('Backend server is not responding. Please make sure the Flask server is running on port 5000.');
      } else {
        throw new Error(`Request failed: ${error.message}`);
      }
    }
  }

  // Get weekly meal plan
  async getWeeklyMealPlan(targetCalories, targetProtein, targetCarbs, targetFat, preferences = {}) {
    try {
      const response = await this.api.post('/weekly-meal-plan', {
        target_calories: targetCalories,
        target_protein: targetProtein,
        target_carbs: targetCarbs,
        target_fat: targetFat,
        preferences: preferences
      });
      
      if (response.data.success) {
        return response.data.weekly_plan;
      } else {
        throw new Error(response.data.error || 'Failed to generate weekly meal plan');
      }
    } catch (error) {
      if (error.response) {
        const errorMessage = error.response.data?.error || 'Server error occurred';
        throw new Error(`Weekly meal plan generation failed: ${errorMessage}`);
      } else if (error.request) {
        throw new Error('Backend server is not responding. Please make sure the Flask server is running on port 5000.');
      } else {
        throw new Error(`Request failed: ${error.message}`);
      }
    }
  }

  // Validate user data before sending
  validateUserData(userData) {
    const required = ['age', 'gender', 'height', 'weight', 'fitness_goal', 'activity_level'];
    const missing = required.filter(field => !userData[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate ranges
    if (userData.age < 18 || userData.age > 65) {
      throw new Error('Age must be between 18 and 65');
    }

    if (userData.height < 150 || userData.height > 200) {
      throw new Error('Height must be between 150 and 200 cm');
    }

    if (userData.weight < 45 || userData.weight > 150) {
      throw new Error('Weight must be between 45 and 150 kg');
    }

    if (!['Male', 'Female'].includes(userData.gender)) {
      throw new Error('Gender must be Male or Female');
    }

    if (!['Muscle Gain', 'Fat Loss', 'Maintenance'].includes(userData.fitness_goal)) {
      throw new Error('Invalid fitness goal');
    }

    if (!['Low Activity', 'Moderate Activity', 'High Activity'].includes(userData.activity_level)) {
      throw new Error('Invalid activity level');
    }

    return true;
  }
}

export const apiService = new ApiService();

// Helper functions for calculations (client-side)
export const calculateBMI = (weight, height) => {
  const heightInMeters = height / 100;
  return weight / (heightInMeters * heightInMeters);
};

export const categorizeBMI = (bmi) => {
  if (bmi < 18.5) return 'Underweight';
  if (bmi < 25) return 'Normal';
  if (bmi < 30) return 'Overweight';
  return 'Obese';
};

export const calculateBMR = (weight, height, age, gender) => {
  if (gender === 'Male') {
    return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age);
  } else {
    return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age);
  }
};

export const calculateTDEE = (bmr, activityLevel) => {
  const multipliers = {
    'Sedentary': 1.2,
    'Lightly Active': 1.375,
    'Moderately Active': 1.55,
    'Very Active': 1.725,
    'Extremely Active': 1.9
  };
  return bmr * multipliers[activityLevel];
};

// Constants
export const FITNESS_GOALS = [
  { value: 'Muscle Gain', label: 'Muscle Gain', description: 'Build muscle mass and strength' },
  { value: 'Fat Loss', label: 'Fat Loss', description: 'Reduce body fat while preserving muscle' },
  { value: 'Maintenance', label: 'Maintenance', description: 'Maintain current physique and health' }
];

export const ACTIVITY_LEVELS = [
  { value: 'Low Activity', label: 'Low Activity', description: 'Minimal exercise or sedentary lifestyle', multiplier: 1.29 },
  { value: 'Moderate Activity', label: 'Moderate Activity', description: 'Regular exercise 3-5 days/week', multiplier: 1.55 },
  { value: 'High Activity', label: 'High Activity', description: 'Intense exercise 6-7 days/week or very active job', multiplier: 1.81 }
];