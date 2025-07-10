// frontend/src/services/recommendationService.js
import { db } from './firebaseConfig';
import { doc, setDoc, getDoc, collection, query, orderBy, limit, getDocs, where } from 'firebase/firestore';

export const recommendationService = {
  // Save a new recommendation with timestamp
  async saveRecommendation(userId, userData, recommendations) {
    try {
      const timestamp = new Date();
      const recommendationId = `${userId}_${timestamp.getTime()}`;
      
      const recommendationData = {
        id: recommendationId,
        userId: userId,
        userData: userData,
        recommendations: recommendations,
        createdAt: timestamp,
        date: timestamp.toISOString().split('T')[0], // YYYY-MM-DD format
        dateTime: timestamp.toISOString(),
        isActive: true // Mark as the current active recommendation
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

      return recommendationData;
    } catch (error) {
      console.error('Error saving recommendation:', error);
      throw error;
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
