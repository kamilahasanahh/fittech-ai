// frontend/src/App.js
import React, { useState, useEffect, useCallback } from 'react';
import { apiService } from './services/api';
import { authService } from './services/authService';
import { recommendationService } from './services/recommendationService';
import { auth } from './services/firebaseConfig';
import AuthForm from './components/AuthForm';
import EnhancedUserInputForm from './components/EnhancedUserInputForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import DailyProgress from './components/DailyProgress';
import Dashboard from './components/Dashboard';
import SystemStatus from './components/SystemStatus';
import OfflineNotice from './components/OfflineNotice';
import './styles/App.css';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import AuthPage from './pages/AuthPage';
import DashboardPage from './pages/DashboardPage';
import RecommendationPage from './pages/RecommendationPage';
import ProgressPage from './pages/ProgressPage';
import InputPage from './pages/InputPage';

function App() {
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [currentRecommendation, setCurrentRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemStatus, setSystemStatus] = useState(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    checkSystemHealth();
    checkAuthState();
    
    // Listen for online/offline events
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (user && location.pathname === '/recommendation') {
      loadSavedRecommendations();
    }
  }, [location.pathname, user]);

  const checkAuthState = async () => {
    try {
      const currentUser = await authService.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
        navigate('/dashboard');
        
        // Load saved recommendations if they exist
        try {
          const currentRec = await recommendationService.getCurrentRecommendation(currentUser.uid);
          if (currentRec) {
            console.log('✅ Found existing recommendation on app load:', currentRec);
            setRecommendations(currentRec.recommendations);
            setUserData(currentRec.userData);
            setCurrentRecommendation(currentRec);
          }
        } catch (error) {
          console.error('Error loading existing recommendations:', error);
        }
      }
    } catch (error) {
      console.error('Error checking auth state:', error);
    }
  };

  const checkSystemHealth = async () => {
    try {
      const status = await apiService.healthCheck();
      setSystemStatus(status);
    } catch (err) {
      console.error('Pemeriksaan kesehatan sistem gagal:', err);
      setSystemStatus({ status: 'error', message: 'Backend tidak merespons' });
    }
  };

  const handleAuthSuccess = (userData) => {
    setUser(userData);
    navigate('/dashboard');
  };

  const handleUserSubmit = async (formData) => {
    setLoading(true);
    setError('');
    
    try {
      console.log('🔄 Mengirim data pengguna:', formData);
      console.log('🔄 Current user:', user);
      console.log('🔄 Firebase Auth user:', auth.currentUser);
      
      // Dapatkan rekomendasi ML dari backend Flask
      const recommendations = await apiService.getRecommendations(formData);
      console.log('✅ Rekomendasi diterima:', recommendations);
      
      // Simpan ke Firebase menggunakan service yang baru
      console.log('🔄 Menyimpan data pengguna ke Firebase...');
      const saveResult = await authService.saveUserData(formData);
      console.log('✅ Data pengguna tersimpan:', saveResult);
      
      // Simpan rekomendasi dengan timestamp menggunakan service baru
      console.log('🔄 Menyimpan rekomendasi ke Firebase...');
      await recommendationService.saveRecommendation(user.uid, formData, recommendations);
      console.log('✅ Rekomendasi tersimpan');
      
      setUserData(formData);
      setRecommendations(recommendations);
      navigate('/recommendation');
      
    } catch (err) {
      console.error('❌ Error mendapatkan rekomendasi:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load saved recommendations when navigating to recommendations page
  const loadSavedRecommendations = async () => {
    if (!user) return;
    
    try {
      setLoading(true);
      console.log('🔄 Loading saved recommendations...');
      
      // Get current active recommendation
      const currentRec = await recommendationService.getCurrentRecommendation(user.uid);
      if (currentRec) {
        console.log('✅ Found saved recommendation:', currentRec);
        setRecommendations(currentRec.recommendations);
        setUserData(currentRec.userData);
        setCurrentRecommendation(currentRec);
      } else {
        console.log('ℹ️ No saved recommendation found');
        setRecommendations(null);
        setUserData(null);
      }
    } catch (error) {
      console.error('❌ Error loading saved recommendations:', error);
      setError('Gagal memuat rekomendasi yang tersimpan');
    } finally {
      setLoading(false);
    }
  };

  const handleProgressUpdate = (progressData) => {
    if (currentRecommendation) {
      setCurrentRecommendation(prev => ({
        ...prev,
        ...progressData
      }));
    }
  };

  const handleLogout = async () => {
    try {
      await authService.signOut();
      setUser(null);
      setUserData(null);
      setRecommendations(null);
      setCurrentRecommendation(null);
      navigate('/');
    } catch (error) {
      console.error('Error logout:', error);
    }
  };

  return (
    <div className="App">
      {!isOnline && <OfflineNotice />}
      
      {/* Header */}
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="header-main">
              <h1>🏋️ XGFitness</h1>
              <p>Sistem Rekomendasi Kebugaran Bertenaga AI</p>
            </div>
            <div className="header-user">
              {user && (
                <div className="user-info">
                  <span>Selamat datang, {user.displayName || 'Pengguna'}!</span>
                  <button onClick={handleLogout} className="btn-secondary small">
                    Keluar
                  </button>
                </div>
              )}
            </div>
          </div>
          <SystemStatus status={systemStatus} />
        </div>
      </header>

      {/* Navigation */}
      <nav className="app-navigation">
        <div className="container">
          <div className="nav-items">
            <button onClick={() => navigate('/dashboard')} className={`nav-item${window.location.pathname === '/dashboard' ? ' active' : ''}`}>📊 Dashboard</button>
            <button onClick={() => navigate('/input')} className={`nav-item${window.location.pathname === '/input' ? ' active' : ''}`}>📝 Rencana Baru</button>
            <button onClick={() => navigate('/recommendation')} className={`nav-item${window.location.pathname === '/recommendation' ? ' active' : ''}`}>🎯 Rekomendasi</button>
            <button onClick={() => navigate('/progress')} className={`nav-item${window.location.pathname === '/progress' ? ' active' : ''}`}>📈 Progress</button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="app-main">
        <div className="container">
          {error && (
            <div className="error-message">
              <p>❌ {error}</p>
              <button onClick={() => setError('')} className="btn-secondary small">
                Tutup
              </button>
            </div>
          )}
          <Routes>
            <Route path="/" element={<AuthPage onAuthSuccess={handleAuthSuccess} />} />
            <Route path="/dashboard" element={<DashboardPage user={user} userData={userData} recommendations={recommendations} onNavigate={navigate} />} />
            <Route path="/input" element={<InputPage onSubmit={handleUserSubmit} loading={loading} initialData={userData} />} />
            <Route path="/recommendation" element={<RecommendationPage recommendations={recommendations} userData={userData} onBack={() => navigate('/input')} onNewRecommendation={() => navigate('/input')} loading={loading} error={error} />} />
            <Route path="/progress" element={<ProgressPage user={user} onProgressUpdate={handleProgressUpdate} userProfile={userData} currentRecommendation={currentRecommendation} />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
