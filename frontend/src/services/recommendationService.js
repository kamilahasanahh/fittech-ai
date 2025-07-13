// frontend/src/services/recommendationService.js
import { db } from './firebaseConfig';
import { doc, setDoc, getDoc, collection, query, orderBy, limit, getDocs, where } from 'firebase/firestore';

export const recommendationService = {
  // Save a new recommendation with timestamp and complete display data
  async saveRecommendation(userId, userData, recommendations, userProfile = null, displayData = null) {
    try {
      const timestamp = new Date();
      const recommendationId = `${userId}_${timestamp.getTime()}`;
      
      // Create comprehensive recommendation data that matches what users see
      const recommendationData = {
        id: recommendationId,
        userId: userId,
        userData: userData,
        recommendations: recommendations,
        userProfile: userProfile, // Store calculated BMI, BMR, TDEE, etc.
        displayData: displayData, // Store any formatted display data
        createdAt: timestamp,
        date: timestamp.toISOString().split('T')[0], // YYYY-MM-DD format
        dateTime: timestamp.toISOString(),
        isActive: true, // Mark as the current active recommendation
        
        // Store calculated nutrition values for quick access
        calculatedNutrition: this.calculateNutritionFromTemplate(userData, userProfile, recommendations),
        
        // Store workout summary for quick access
        workoutSummary: this.extractWorkoutSummary(recommendations),
        
        // Store template data for offline access
        templateData: {
          nutritionTemplate: recommendations.predictions?.nutrition_template || recommendations.nutrition_recommendation,
          workoutTemplate: recommendations.predictions?.workout_template || recommendations.workout_recommendation
        },
        
        // Metadata for tracking
        version: '2.0', // Version for future compatibility
        source: 'web_app' // Source of recommendation
      };

      // Save the recommendation
      const recommendationRef = doc(db, 'recommendations', recommendationId);
      await setDoc(recommendationRef, recommendationData);

      // Update user's current recommendation reference
      const userRef = doc(db, 'users', userId);
      await setDoc(userRef, {
        currentRecommendationId: recommendationId,
        lastRecommendationDate: timestamp,
        updatedAt: timestamp
      }, { merge: true });

      // Set previous recommendations as inactive
      await this.setOtherRecommendationsInactive(userId, recommendationId);

      console.log('✅ Comprehensive recommendation saved:', recommendationId);
      return recommendationData;
    } catch (error) {
      console.error('Error saving recommendation:', error);
      throw error;
    }
  },

  // Calculate nutrition values from template data
  calculateNutritionFromTemplate(userData, userProfile, recommendations) {
    try {
      const nutrition = recommendations.nutrition_recommendation || 
                       recommendations.predictions?.nutrition_template ||
                       recommendations.nutrition_template;
      
      if (!nutrition || !userProfile) return null;
      
      const userWeight = parseFloat(userData.weight) || 70;
      const tdee = userProfile.tdee || userProfile.total_daily_energy_expenditure || 2000;
      
      // Use template data if available, otherwise fallback to direct values
      const templateId = nutrition.template_id || recommendations.predicted_nutrition_template_id;
      
      return {
        templateId: templateId,
        targetCalories: nutrition.target_calories || nutrition.daily_calories || Math.round(tdee * 0.8),
        targetProtein: nutrition.target_protein || nutrition.protein_grams || Math.round(userWeight * 2),
        targetCarbs: nutrition.target_carbs || nutrition.carbs_grams || Math.round(userWeight * 3),
        targetFat: nutrition.target_fat || nutrition.fat_grams || Math.round(userWeight * 1),
        calculatedAt: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error calculating nutrition from template:', error);
      return null;
    }
  },

  // Extract workout summary for quick display
  extractWorkoutSummary(recommendations) {
    try {
      const workout = recommendations.workout_recommendation || 
                     recommendations.predictions?.workout_template;
      
      if (!workout) return null;
      
      return {
        templateId: workout.template_id,
        workoutType: workout.workout_type,
        daysPerWeek: workout.days_per_week,
        setsPerExercise: workout.sets_per_exercise,
        exercisesPerSession: workout.exercises_per_session,
        cardioMinutesPerDay: workout.cardio_minutes_per_day || 0,
        schedule: workout.workout_schedule || workout.schedule || workout.weekly_schedule,
        extractedAt: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error extracting workout summary:', error);
      return null;
    }
  },

  // Set all other recommendations as inactive
  async setOtherRecommendationsInactive(userId, currentRecommendationId) {
    try {
      const recommendationsRef = collection(db, 'recommendations');
      const q = query(
        recommendationsRef, 
        where('userId', '==', userId),
        where('isActive', '==', true)
      );
      
      const snapshot = await getDocs(q);
      
      const updatePromises = snapshot.docs
        .filter(doc => doc.id !== currentRecommendationId)
        .map(doc => 
          setDoc(doc.ref, { isActive: false }, { merge: true })
        );
      
      await Promise.all(updatePromises);
    } catch (error) {
      console.error('Error updating recommendation status:', error);
    }
  },

  // Update recommendation with meal plan data
  async updateRecommendationMealPlan(recommendationId, mealPlan) {
    try {
      const recommendationRef = doc(db, 'recommendations', recommendationId);
      await setDoc(recommendationRef, {
        mealPlan: mealPlan,
        mealPlanGeneratedAt: new Date(),
        updatedAt: new Date()
      }, { merge: true });

      console.log('✅ Meal plan updated in recommendation:', recommendationId);
      return true;
    } catch (error) {
      console.error('Error updating recommendation with meal plan:', error);
      throw error;
    }
  },

  // Get current active recommendation
  async getCurrentRecommendation(userId) {
    try {
      const userRef = doc(db, 'users', userId);
      const userDoc = await getDoc(userRef);
      
      if (userDoc.exists() && userDoc.data().currentRecommendationId) {
        const recommendationRef = doc(db, 'recommendations', userDoc.data().currentRecommendationId);
        const recommendationDoc = await getDoc(recommendationRef);
        
        if (recommendationDoc.exists()) {
          return recommendationDoc.data();
        }
      }
      
      return null;
    } catch (error) {
      console.error('Error getting current recommendation:', error);
      return null;
    }
  },

  // Get recommendation history for a user
  async getRecommendationHistory(userId, limitCount = 10) {
    try {
      const recommendationsRef = collection(db, 'recommendations');
      const q = query(
        recommendationsRef,
        where('userId', '==', userId),
        orderBy('createdAt', 'desc'),
        limit(limitCount)
      );
      
      const snapshot = await getDocs(q);
      const recommendations = [];
      
      snapshot.forEach(doc => {
        recommendations.push({
          id: doc.id,
          ...doc.data()
        });
      });
      
      return recommendations;
    } catch (error) {
      console.error('Error getting recommendation history:', error);
      return [];
    }
  },

  // Get recommendations for a specific date
  async getRecommendationsForDate(userId, date) {
    try {
      const recommendationsRef = collection(db, 'recommendations');
      const q = query(
        recommendationsRef,
        where('userId', '==', userId),
        where('date', '==', date),
        orderBy('createdAt', 'desc')
      );
      
      const snapshot = await getDocs(q);
      const recommendations = [];
      
      snapshot.forEach(doc => {
        recommendations.push({
          id: doc.id,
          ...doc.data()
        });
      });
      
      return recommendations;
    } catch (error) {
      console.error('Error getting recommendations for date:', error);
      return [];
    }
  },

  // Get recommendation by ID
  async getRecommendationById(recommendationId) {
    try {
      const recommendationRef = doc(db, 'recommendations', recommendationId);
      const recommendationDoc = await getDoc(recommendationRef);
      
      if (recommendationDoc.exists()) {
        return {
          id: recommendationDoc.id,
          ...recommendationDoc.data()
        };
      }
      
      return null;
    } catch (error) {
      console.error('Error getting recommendation by ID:', error);
      return null;
    }
  },

  // Save recommendation feedback
  async saveRecommendationFeedback(feedbackData) {
    try {
      const timestamp = new Date();
      const feedbackId = `feedback_${feedbackData.recommendationId}_${timestamp.getTime()}`;
      
      const feedbackDoc = {
        id: feedbackId,
        recommendationId: feedbackData.recommendationId,
        feedback: feedbackData.feedback,
        timestamp: timestamp,
        date: timestamp.toISOString().split('T')[0],
        dateTime: timestamp.toISOString()
      };

      const feedbackRef = doc(db, 'recommendationFeedback', feedbackId);
      await setDoc(feedbackRef, feedbackDoc);

      // Update the recommendation with feedback summary
      const recommendationRef = doc(db, 'recommendations', feedbackData.recommendationId);
      await setDoc(recommendationRef, {
        lastFeedback: feedbackData.feedback,
        lastFeedbackDate: timestamp,
        feedbackCount: (await this.getFeedbackCount(feedbackData.recommendationId)) + 1
      }, { merge: true });

      return feedbackDoc;
    } catch (error) {
      console.error('Error saving recommendation feedback:', error);
      throw error;
    }
  },

  // Get feedback count for a recommendation
  async getFeedbackCount(recommendationId) {
    try {
      const feedbackRef = collection(db, 'recommendationFeedback');
      const q = query(feedbackRef, where('recommendationId', '==', recommendationId));
      const snapshot = await getDocs(q);
      return snapshot.size;
    } catch (error) {
      console.error('Error getting feedback count:', error);
      return 0;
    }
  },

  // Get feedback history for a recommendation
  async getFeedbackHistory(recommendationId) {
    try {
      const feedbackRef = collection(db, 'recommendationFeedback');
      const q = query(
        feedbackRef,
        where('recommendationId', '==', recommendationId),
        orderBy('timestamp', 'desc')
      );
      
      const snapshot = await getDocs(q);
      const feedbackHistory = [];
      
      snapshot.forEach(doc => {
        feedbackHistory.push({
          id: doc.id,
          ...doc.data()
        });
      });
      
      return feedbackHistory;
    } catch (error) {
      console.error('Error getting feedback history:', error);
      return [];
    }
  },

  // Helper method to calculate nutrition from template
  calculateNutritionFromTemplate(userData, userProfile, recommendations) {
    try {
      const nutrition = recommendations.predictions?.nutrition_template || 
                       recommendations.nutrition_recommendation;
      
      if (!nutrition || !userProfile || !userData) {
        return null;
      }

      const weight = parseFloat(userData.weight);
      const tdee = userProfile.tdee || 2000;

      return {
        templateId: nutrition.template_id,
        targetCalories: Math.round(tdee * (nutrition.caloric_intake_multiplier || 0.8)),
        targetProtein: Math.round(weight * (nutrition.protein_per_kg || 2)),
        targetCarbs: Math.round(weight * (nutrition.carbs_per_kg || 3)),
        targetFat: Math.round(weight * (nutrition.fat_per_kg || 1)),
        multipliers: {
          caloric: nutrition.caloric_intake_multiplier || 0.8,
          protein: nutrition.protein_per_kg || 2,
          carbs: nutrition.carbs_per_kg || 3,
          fat: nutrition.fat_per_kg || 1
        }
      };
    } catch (error) {
      console.error('Error calculating nutrition from template:', error);
      return null;
    }
  },

  // Helper method to extract workout summary
  extractWorkoutSummary(recommendations) {
    try {
      const workout = recommendations.predictions?.workout_template || 
                     recommendations.workout_recommendation;
      
      if (!workout) {
        return null;
      }

      return {
        templateId: workout.template_id,
        workoutType: workout.workout_type,
        daysPerWeek: workout.days_per_week,
        setsPerExercise: workout.sets_per_exercise,
        exercisesPerSession: workout.exercises_per_session,
        cardioMinutesPerDay: workout.cardio_minutes_per_day || 0,
        cardioSessionsPerDay: workout.cardio_sessions_per_day || 0,
        schedule: workout.workout_schedule || workout.schedule || workout.weekly_schedule
      };
    } catch (error) {
      console.error('Error extracting workout summary:', error);
      return null;
    }
  }
};

// Export individual functions for the RecommendationFeedback component
export const saveRecommendationFeedback = (feedbackData) => {
  return recommendationService.saveRecommendationFeedback(feedbackData);
};

export const getImprovedRecommendation = async (data) => {
  try {
    const response = await fetch('http://localhost:5000/improve-recommendation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error('Failed to get improved recommendation');
    }

    const result = await response.json();
    return result.suggestions;
  } catch (error) {
    console.error('Error getting improved recommendation:', error);
    throw error;
  }
};
