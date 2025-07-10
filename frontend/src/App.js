// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { apiService } from './services/api';
import { authService } from './services/authService';
import { recommendationService } from './services/recommendationService';
import { auth } from './services/firebaseConfig';
import AuthForm from './components/AuthForm';
import EnhancedUserInputForm from './components/EnhancedUserInputForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import DailyProgress from './components/DailyProgress';
import SystemStatus from './components/SystemStatus';
import OfflineNotice from './components/OfflineNotice';
import './styles/App.css';

function App() {
  const [currentView, setCurrentView] = useState('auth');
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [currentRecommendation, setCurrentRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemStatus, setSystemStatus] = useState(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

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

  const checkAuthState = async () => {
    try {
      const currentUser = await authService.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
        setCurrentView('dashboard');
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
    setCurrentView('dashboard');
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
      setCurrentView('recommendations');
      
    } catch (err) {
      console.error('❌ Error mendapatkan rekomendasi:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await authService.signOut();
      setUser(null);
      setUserData(null);
      setRecommendations(null);
      setCurrentRecommendation(null);
      setCurrentView('auth');
    } catch (error) {
      console.error('Error logout:', error);
    }
  };

  const handleBackToForm = () => {
    setCurrentView('input');
    setError('');
  };

  const handleNewRecommendation = async () => {
    // For now, always allow creating new recommendations
    // TODO: Implement logic to check if user has active recommendation
    setCurrentView('input');
    setError('');
  };

  const handleProgressUpdate = (progressData) => {
    if (currentRecommendation) {
      setCurrentRecommendation(prev => ({
        ...prev,
        ...progressData
      }));
    }
  };

  if (currentView === 'auth') {
    return <AuthForm onAuthSuccess={handleAuthSuccess} />;
  }

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
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`nav-item ${currentView === 'dashboard' ? 'active' : ''}`}
            >
              📊 Dashboard
            </button>
            <button
              onClick={() => setCurrentView('input')}
              className={`nav-item ${currentView === 'input' ? 'active' : ''}`}
            >
              📝 Rencana Baru
            </button>
            <button
              onClick={() => setCurrentView('recommendations')}
              className={`nav-item ${currentView === 'recommendations' ? 'active' : ''}`}
            >
              🎯 Rekomendasi
            </button>
            <button
              onClick={() => setCurrentView('progress')}
              className={`nav-item ${currentView === 'progress' ? 'active' : ''}`}
            >
              📈 Progress
            </button>
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

          {currentView === 'input' && (
            <EnhancedUserInputForm
              onSubmit={handleUserSubmit}
              loading={loading}
              initialData={userData}
            />
          )}

          {currentView === 'recommendations' && recommendations && (
            <RecommendationDisplay
              recommendations={recommendations}
              userData={userData}
              onBack={handleBackToForm}
              onNewRecommendation={handleNewRecommendation}
            />
          )}

          {currentView === 'progress' && (
            <DailyProgress
              user={user}
              onProgressUpdate={handleProgressUpdate}
            />
          )}

          {currentView === 'dashboard' && (
            <div className="dashboard">
              <div className="welcome-section">
                <h2>Dashboard</h2>
                <p>Selamat datang di XGFitness! Pilih menu di atas untuk memulai.</p>
                
                {!userData && (
                  <div className="cta-section">
                    <h3>Mulai Perjalanan Kebugaran Anda</h3>
                    <p>Dapatkan rekomendasi kebugaran yang dipersonalisasi berdasarkan profil dan tujuan Anda.</p>
                    <button 
                      onClick={() => setCurrentView('input')} 
                      className="btn-primary"
                    >
                      Buat Rencana Kebugaran
                    </button>
                  </div>
                )}

                {userData && recommendations && (
                  <div className="quick-access">
                    <h3>Akses Cepat</h3>
                    <div className="quick-buttons">
                      <button 
                        onClick={() => setCurrentView('recommendations')} 
                        className="btn-primary"
                      >
                        Lihat Rekomendasi
                      </button>
                      <button 
                        onClick={() => setCurrentView('progress')} 
                        className="btn-secondary"
                      >
                        Cek Progress
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
