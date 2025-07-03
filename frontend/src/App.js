// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { apiService } from './services/api';
import { authService } from './services/authService';
import AuthForm from './components/AuthForm';
import EnhancedUserInputForm from './components/EnhancedUserInputForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import WeeklyProgress from './components/WeeklyProgress';
import SystemStatus from './components/SystemStatus';
import './App.css';
import './components/enhanced-styles.css';

function App() {
  const [currentView, setCurrentView] = useState('auth');
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [currentRecommendation, setCurrentRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    checkSystemHealth();
    
    // Listen for auth state changes
    const unsubscribe = authService.onAuthStateChange((user) => {
      setUser(user);
      if (user) {
        loadUserData();
        setCurrentView('dashboard');
      } else {
        setCurrentView('auth');
        setUserData(null);
        setRecommendations(null);
        setCurrentRecommendation(null);
      }
    });

    return unsubscribe;
  }, []);

  const checkSystemHealth = async () => {
    try {
      const status = await apiService.healthCheck();
      setSystemStatus(status);
    } catch (err) {
      console.error('System health check failed:', err);
      setSystemStatus({ status: 'error', message: 'Backend not responding' });
    }
  };

  const loadUserData = async () => {
    try {
      const profile = await authService.getUserProfile();
      const currentRec = await authService.getCurrentRecommendation();
      
      if (profile) {
        setUserData(profile);
      }
      
      if (currentRec) {
        setCurrentRecommendation(currentRec);
        setRecommendations(currentRec.recommendations);
      }
    } catch (error) {
      console.error('Error loading user data:', error);
    }
  };

  const handleAuthSuccess = (user) => {
    setUser(user);
    setCurrentView('dashboard');
  };

  const handleUserSubmit = async (formData) => {
    setLoading(true);
    setError('');
    
    try {
      console.log('Submitting user data:', formData);
      
      // Get ML recommendations
      const recommendations = await apiService.getRecommendations(formData);
      
      // Save user profile and weekly recommendation
      await authService.saveUserProfile(formData);
      const saveResult = await authService.saveWeeklyRecommendation(formData, recommendations);
      
      if (saveResult.success) {
        setUserData(formData);
        setRecommendations(recommendations);
        setCurrentView('recommendations');
        
        // Reload current recommendation
        await loadUserData();
      } else {
        setError('Failed to save recommendation: ' + saveResult.error);
      }
      
    } catch (err) {
      setError(err.message);
      console.error('Error getting recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await authService.logout();
      setUser(null);
      setUserData(null);
      setRecommendations(null);
      setCurrentRecommendation(null);
      setCurrentView('auth');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleBackToForm = () => {
    setCurrentView('input');
    setError('');
  };

  const handleNewRecommendation = async () => {
    const canCreate = await authService.canCreateNewRecommendation();
    if (canCreate) {
      setCurrentView('input');
      setError('');
    } else {
      setError('You have an active weekly plan. Please complete it before creating a new one.');
    }
  };

  const handleProgressUpdate = (progressData) => {
    // Update current recommendation with new progress
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
      {/* Header */}
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="header-main">
              <h1>🏋️ FitTech AI</h1>
              <p>Thesis-Aligned Fitness Recommendation System</p>
            </div>
            <div className="header-user">
              {user && (
                <div className="user-info">
                  <span>Welcome, {user.displayName || 'User'}!</span>
                  <button onClick={handleLogout} className="btn-secondary small">
                    Logout
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
              📝 New Plan
            </button>
            {currentRecommendation && (
              <button
                onClick={() => setCurrentView('progress')}
                className={`nav-item ${currentView === 'progress' ? 'active' : ''}`}
              >
                📈 Progress
              </button>
            )}
            <button
              onClick={() => setCurrentView('recommendations')}
              className={`nav-item ${currentView === 'recommendations' ? 'active' : ''}`}
              disabled={!recommendations}
            >
              🎯 Recommendations
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="app-main">
        <div className="container">
          {error && (
            <div className="error-banner">
              <h3>⚠️ Error</h3>
              <p>{error}</p>
              <button onClick={() => setError('')} className="btn-secondary">
                Dismiss
              </button>
            </div>
          )}

          {currentView === 'dashboard' && (
            <div className="dashboard-section">
              <div className="section-header">
                <h2>Dashboard</h2>
                <p>Your fitness journey overview</p>
              </div>
              
              <div className="dashboard-grid">
                {/* User Summary Card */}
                <div className="dashboard-card">
                  <h3>👤 Profile Summary</h3>
                  {userData ? (
                    <div className="profile-summary">
                      <div className="profile-item">
                        <span>Current Goal:</span>
                        <span>{userData.profile?.fitness_goal || 'Not set'}</span>
                      </div>
                      <div className="profile-item">
                        <span>Activity Level:</span>
                        <span>{userData.profile?.activity_level || 'Not set'}</span>
                      </div>
                      <div className="profile-item">
                        <span>Current Week:</span>
                        <span>Week {userData.currentWeek || 1}</span>
                      </div>
                      <div className="profile-item">
                        <span>Total Weeks:</span>
                        <span>{userData.totalWeeks || 0} completed</span>
                      </div>
                    </div>
                  ) : (
                    <p>Complete your profile to see summary</p>
                  )}
                </div>

                {/* Current Plan Card */}
                <div className="dashboard-card">
                  <h3>📅 Current Plan</h3>
                  {currentRecommendation ? (
                    <div className="current-plan">
                      <div className="plan-item">
                        <span>Week:</span>
                        <span>Week {currentRecommendation.weekNumber}</span>
                      </div>
                      <div className="plan-item">
                        <span>Progress:</span>
                        <span>{currentRecommendation.completedDays || 0}/7 days</span>
                      </div>
                      <div className="plan-item">
                        <span>Created:</span>
                        <span>{new Date(currentRecommendation.createdAt.seconds * 1000).toLocaleDateString()}</span>
                      </div>
                      <button
                        onClick={() => setCurrentView('progress')}
                        className="btn-primary small"
                      >
                        View Progress
                      </button>
                    </div>
                  ) : (
                    <div className="no-plan">
                      <p>No active plan</p>
                      <button
                        onClick={() => setCurrentView('input')}
                        className="btn-primary small"
                      >
                        Create New Plan
                      </button>
                    </div>
                  )}
                </div>

                {/* Quick Actions Card */}
                <div className="dashboard-card">
                  <h3>⚡ Quick Actions</h3>
                  <div className="quick-actions">
                    <button
                      onClick={handleNewRecommendation}
                      className="btn-primary small"
                    >
                      🆕 New Weekly Plan
                    </button>
                    {recommendations && (
                      <button
                        onClick={() => setCurrentView('recommendations')}
                        className="btn-secondary small"
                      >
                        📋 View Current Recommendations
                      </button>
                    )}
                    <button
                      onClick={checkSystemHealth}
                      className="btn-secondary small"
                    >
                      🔄 Refresh System Status
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {currentView === 'input' && (
            <div className="input-section">
              <div className="section-header">
                <h2>Create Weekly Plan</h2>
                <p>Generate your personalized weekly fitness recommendations</p>
              </div>
              
              <EnhancedUserInputForm 
                onSubmit={handleUserSubmit}
                loading={loading}
                initialData={userData?.profile}
              />
            </div>
          )}

          {currentView === 'recommendations' && recommendations && (
            <div className="recommendations-section">
              <div className="section-header">
                <h2>Your Weekly Recommendations</h2>
                <p>AI-generated plan based on your profile and goals</p>
                
                <div className="action-buttons">
                  <button onClick={handleBackToForm} className="btn-secondary">
                    ← Edit Profile
                  </button>
                  <button onClick={handleNewRecommendation} className="btn-primary">
                    🔄 New Week Plan
                  </button>
                </div>
              </div>

              <RecommendationDisplay 
                userData={userData?.profile || userData}
                recommendations={recommendations}
              />
            </div>
          )}

          {currentView === 'progress' && (
            <div className="progress-section">
              <div className="section-header">
                <h2>Weekly Progress Tracking</h2>
                <p>Track your daily adherence and see your improvement</p>
              </div>

              <WeeklyProgress 
                currentRecommendation={currentRecommendation}
                onProgressUpdate={handleProgressUpdate}
              />
            </div>
          )}

          {loading && (
            <div className="loading-overlay">
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Processing your request...</p>
                <p className="loading-detail">Analyzing profile and generating recommendations</p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-section">
              <h4>System Information</h4>
              <ul>
                <li>✅ 75 Total Templates (15 workout + 60 nutrition)</li>
                <li>✅ XGBoost Classification with 99.73% accuracy</li>
                <li>✅ Harris-Benedict BMR Formulas</li>
                <li>✅ Thesis-Aligned TDEE Multipliers</li>
                <li>✅ Weekly Progress Tracking</li>
                <li>✅ AI Adaptive Recommendations</li>
              </ul>
            </div>
            
            <div className="footer-section">
              <h4>Template Structure</h4>
              <ul>
                <li>Workout: 3 goals × 5 activity levels = 15</li>
                <li>Nutrition: 3 goals × 4 BMI × 5 activities = 60</li>
                <li>Weight change limits for safety</li>
                <li>Evidence-based recommendations</li>
                <li>Weekly adaptation based on progress</li>
              </ul>
            </div>
            
            <div className="footer-section">
              <h4>About</h4>
              <p>
                This system implements the exact methodology described in the thesis
                "Implementation of Diet and Physical Exercise Recommendation Systems Using XGBoost"
                by Kamila Hasanah, Telkom University.
              </p>
              <p>
                Features user authentication, weekly progress tracking, and AI-adaptive
                recommendations based on adherence and progress data.
              </p>
            </div>
          </div>
          
          <div className="footer-bottom">
            <p>&copy; 2025 FitTech AI - Thesis Implementation with Enhanced Features</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;