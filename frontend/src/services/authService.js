import { 
  onAuthStateChanged, 
  signOut as firebaseSignOut,
  updateProfile 
} from 'firebase/auth';
import { 
  doc, 
  setDoc, 
  getDoc, 
  updateDoc, 
  collection, 
  addDoc, 
  getDocs, 
  query, 
  where, 
  orderBy 
} from 'firebase/firestore';
import { auth, db } from './firebaseConfig';

class AuthService {
  constructor() {
    this.currentUser = null;
    this.authStateListeners = [];
  }

  // Get current user
  getCurrentUser() {
    return new Promise((resolve) => {
      if (this.currentUser) {
        resolve(this.currentUser);
        return;
      }

      const unsubscribe = onAuthStateChanged(auth, (user) => {
        unsubscribe();
        this.currentUser = user;
        resolve(user);
      });
    });
  }

  // Listen to auth state changes
  onAuthStateChange(callback) {
    return onAuthStateChanged(auth, (user) => {
      this.currentUser = user;
      callback(user);
    });
  }

  // Sign out
  async signOut() {
    try {
      await firebaseSignOut(auth);
      this.currentUser = null;
      return { success: true };
    } catch (error) {
      console.error('Sign out error:', error);
      return { success: false, error: error.message };
    }
  }

  // Update user profile
  async updateUserProfile(updates) {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      // Update Firebase Auth profile
      if (updates.displayName) {
        await updateProfile(this.currentUser, {
          displayName: updates.displayName
        });
      }

      // Update Firestore user document
      const userRef = doc(db, 'users', this.currentUser.uid);
      await updateDoc(userRef, {
        ...updates,
        updatedAt: new Date()
      });

      return { success: true };
    } catch (error) {
      console.error('Update profile error:', error);
      return { success: false, error: error.message };
    }
  }

  // Save user data to Firestore
  async saveUserData(userData) {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const userRef = doc(db, 'users', this.currentUser.uid);
      await setDoc(userRef, {
        ...userData,
        uid: this.currentUser.uid,
        email: this.currentUser.email,
        createdAt: new Date(),
        updatedAt: new Date()
      }, { merge: true });

      return { success: true };
    } catch (error) {
      console.error('Save user data error:', error);
      return { success: false, error: error.message };
    }
  }

  // Get user data from Firestore
  async getUserData() {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const userRef = doc(db, 'users', this.currentUser.uid);
      const userSnap = await getDoc(userRef);

      if (userSnap.exists()) {
        return { success: true, data: userSnap.data() };
      } else {
        return { success: false, error: 'User data not found' };
      }
    } catch (error) {
      console.error('Get user data error:', error);
      return { success: false, error: error.message };
    }
  }

  // Save recommendation
  async saveRecommendation(recommendationData) {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const recommendationsRef = collection(db, 'recommendations');
      const docRef = await addDoc(recommendationsRef, {
        ...recommendationData,
        userId: this.currentUser.uid,
        createdAt: new Date()
      });

      return { success: true, id: docRef.id };
    } catch (error) {
      console.error('Save recommendation error:', error);
      return { success: false, error: error.message };
    }
  }

  // Get user recommendations
  async getUserRecommendations() {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const recommendationsRef = collection(db, 'recommendations');
      const q = query(
        recommendationsRef,
        where('userId', '==', this.currentUser.uid),
        orderBy('createdAt', 'desc')
      );

      const querySnapshot = await getDocs(q);
      const recommendations = [];
      querySnapshot.forEach((doc) => {
        recommendations.push({
          id: doc.id,
          ...doc.data()
        });
      });

      return { success: true, data: recommendations };
    } catch (error) {
      console.error('Get recommendations error:', error);
      return { success: false, error: error.message };
    }
  }

  // Save daily progress
  async saveDailyProgress(progressData) {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const progressRef = collection(db, 'dailyProgress');
      const docRef = await addDoc(progressRef, {
        ...progressData,
        userId: this.currentUser.uid,
        date: new Date().toISOString().split('T')[0], // YYYY-MM-DD format
        createdAt: new Date()
      });

      return { success: true, id: docRef.id };
    } catch (error) {
      console.error('Save progress error:', error);
      return { success: false, error: error.message };
    }
  }

  // Get user daily progress
  async getUserProgress(limit = 30) {
    try {
      if (!this.currentUser) {
        throw new Error('No authenticated user');
      }

      const progressRef = collection(db, 'dailyProgress');
      const q = query(
        progressRef,
        where('userId', '==', this.currentUser.uid),
        orderBy('createdAt', 'desc')
      );

      const querySnapshot = await getDocs(q);
      const progress = [];
      querySnapshot.forEach((doc) => {
        progress.push({
          id: doc.id,
          ...doc.data()
        });
      });

      return { success: true, data: progress.slice(0, limit) };
    } catch (error) {
      console.error('Get progress error:', error);
      return { success: false, error: error.message };
    }
  }
}

export const authService = new AuthService();
