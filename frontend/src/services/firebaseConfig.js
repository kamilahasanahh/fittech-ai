// frontend/src/services/firebaseConfig.js
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyBjabp8BVm6-Kv8LMagvWk2bwByiYajHjQ",
  authDomain: "fittech-ai-thesis.firebaseapp.com",
  projectId: "fittech-ai-thesis",
  storageBucket: "fittech-ai-thesis.appspot.com",
  messagingSenderId: "772230994937",
  appId: "1:772230994937:web:72d9bc88cd943cb114fb28"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export default app;