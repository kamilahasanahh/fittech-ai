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
      // Validate user data
      this.validateUserData(userData);
      
      const response = await this.api.post('/predict', userData);
      
      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'Failed to get recommendations');
      }
    } catch (error) {
      if (error.response) {
        const errorMessage = error.response.data?.error || 'Server error occurred';
        throw new Error(`Recommendation failed: ${errorMessage}`);
      } else if (error.request) {
        throw new Error('Backend server is not responding. Please make sure the Flask server is running on port 5000.');
      } else {
        throw new Error(`Request failed: ${error.message}`);
      }
    }
  }

  // Get workout templates
  async getWorkoutTemplates() {
    try {
      const response = await this.api.get('/templates/workout');
      return response.data.templates;
    } catch (error) {
      throw new Error(`Failed to fetch workout templates: ${error.message}`);
    }
  }

  // Get nutrition templates
  async getNutritionTemplates() {
    try {
      const response = await this.api.get('/templates/nutrition');
      return response.data.templates;
    } catch (error) {
      throw new Error(`Failed to fetch nutrition templates: ${error.message}`);
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