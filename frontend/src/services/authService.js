// frontend/src/services/authService.js
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  updateProfile,
  sendPasswordResetEmail,
  onAuthStateChanged
} from 'firebase/auth';
import { doc, setDoc, getDoc, updateDoc, collection, query, where, orderBy, getDocs, limit } from 'firebase/firestore';
import { auth, db } from './firebaseConfig'; // Fixed: Import from firebaseConfig instead of self

class AuthService {
  constructor() {
    this.currentUser = null;
    this.authStateListeners = [];

    // Listen for auth state changes
    onAuthStateChanged(auth, (user) => {
      this.currentUser = user;
      this.authStateListeners.forEach(listener => listener(user));
    });
  }

  // Register new user
  async register(email, password, userData) {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Update profile with display name
      await updateProfile(user, {
        displayName: `${userData.firstName} ${userData.lastName}`
      });

      // Create user document in Firestore
      await setDoc(doc(db, 'users', user.uid), {
        uid: user.uid,
        email: user.email,
        firstName: userData.firstName,
        lastName: userData.lastName,
        createdAt: new Date(),
        lastLogin: new Date(),
        profileCompleted: false,
        currentWeek: 1,
        totalWeeks: 0
      });

      return { success: true, user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Login user
  async login(email, password) {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Update last login
      await updateDoc(doc(db, 'users', user.uid), {
        lastLogin: new Date()
      });

      return { success: true, user };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Logout user
  async logout() {
    try {
      await signOut(auth);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Reset password
  async resetPassword(email) {
    try {
      await sendPasswordResetEmail(auth, email);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Save user fitness profile
  async saveUserProfile(profileData) {
    if (!this.currentUser) throw new Error('User not authenticated');

    try {
      const userRef = doc(db, 'users', this.currentUser.uid);

      await updateDoc(userRef, {
        profile: profileData,
        profileCompleted: true,
        profileUpdatedAt: new Date()
      });

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Save weekly recommendation
  async saveWeeklyRecommendation(profileData, recommendations) {
    if (!this.currentUser) throw new Error('User not authenticated');

    try {
      const weekId = `${this.currentUser.uid}_week_${Date.now()}`;

      // Save to weekly_recommendations collection
      await setDoc(doc(db, 'weekly_recommendations', weekId), {
        userId: this.currentUser.uid,
        weekNumber: await this.getCurrentWeekNumber(),
        profileData,
        recommendations,
        createdAt: new Date(),
        status: 'active',
        completedDays: 0,
        adherenceData: {},
        weeklyProgress: {}
      });

      // Update user's current week
      await this.incrementUserWeek();

      return { success: true, weekId };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Get user's current week number
  async getCurrentWeekNumber() {
    if (!this.currentUser) return 1;

    try {
      const userDoc = await getDoc(doc(db, 'users', this.currentUser.uid));
      const userData = userDoc.data();
      return userData?.currentWeek || 1;
    } catch (error) {
      console.error('Error getting current week:', error);
      return 1;
    }
  }

  // Increment user's week counter
  async incrementUserWeek() {
    if (!this.currentUser) return;

    try {
      const userRef = doc(db, 'users', this.currentUser.uid);
      const userDoc = await getDoc(userRef);
      const userData = userDoc.data();

      const newWeekNumber = (userData?.currentWeek || 0) + 1;
      const totalWeeks = (userData?.totalWeeks || 0) + 1;

      await updateDoc(userRef, {
        currentWeek: newWeekNumber,
        totalWeeks: totalWeeks
      });
    } catch (error) {
      console.error('Error incrementing week:', error);
    }
  }

  // Get user's recommendation history
  async getRecommendationHistory(limit = 10) {
    if (!this.currentUser) return [];

    try {
      const q = query(
        collection(db, 'weekly_recommendations'),
        where('userId', '==', this.currentUser.uid),
        orderBy('createdAt', 'desc'),
        limit(limit)
      );

      const querySnapshot = await getDocs(q);
      const recommendations = [];

      querySnapshot.forEach((doc) => {
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
  }

  // Get current active recommendation
  async getCurrentRecommendation() {
    if (!this.currentUser) return null;

    try {
      const q = query(
        collection(db, 'weekly_recommendations'),
        where('userId', '==', this.currentUser.uid),
        where('status', '==', 'active'),
        orderBy('createdAt', 'desc'),
        limit(1)
      );

      const querySnapshot = await getDocs(q);

      if (!querySnapshot.empty) {
        const doc = querySnapshot.docs[0];
        return {
          id: doc.id,
          ...doc.data()
        };
      }

      return null;
    } catch (error) {
      console.error('Error getting current recommendation:', error);
      return null;
    }
  }

  // Update weekly progress
  async updateWeeklyProgress(weekId, progressData) {
    if (!this.currentUser) throw new Error('User not authenticated');

    try {
      const weekRef = doc(db, 'weekly_recommendations', weekId);

      await updateDoc(weekRef, {
        adherenceData: progressData.adherenceData,
        completedDays: progressData.completedDays,
        weeklyProgress: progressData.weeklyProgress,
        lastUpdated: new Date()
      });

      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Check if user can create new recommendation (weekly limit)
  async canCreateNewRecommendation() {
    if (!this.currentUser) return false;

    try {
      const currentRec = await this.getCurrentRecommendation();

      if (!currentRec) return true;

      // Check if current recommendation is older than 7 days
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);

      const createdAt = currentRec.createdAt.toDate();
      return createdAt < weekAgo;
    } catch (error) {
      console.error('Error checking recommendation eligibility:', error);
      return true;
    }
  }

  // Get user profile data
  async getUserProfile() {
    if (!this.currentUser) return null;

    try {
      const userDoc = await getDoc(doc(db, 'users', this.currentUser.uid));
      return userDoc.exists() ? userDoc.data() : null;
    } catch (error) {
      console.error('Error getting user profile:', error);
      return null;
    }
  }

  // Add auth state listener
  onAuthStateChange(callback) {
    this.authStateListeners.push(callback);

    // Return unsubscribe function
    return () => {
      this.authStateListeners = this.authStateListeners.filter(
        listener => listener !== callback
      );
    };
  }

  // Get current user
  getCurrentUser() {
    return this.currentUser;
  }
}

export const authService = new AuthService();