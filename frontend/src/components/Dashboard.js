import React, { useState, useEffect } from 'react';
import { recommendationService } from '../services/recommendationService';
import { authService } from '../services/authService';

const Dashboard = ({ user, userData, recommendations, onNavigate }) => {
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState({
    currentRecommendation: null,
    recommendationHistory: [],
    progressData: [],
    stats: {
      totalRecommendations: 0,
      lastWeightEntry: null
    }
  });

  useEffect(() => {
    loadDashboardData();
  }, [user]);

  const loadDashboardData = async () => {
    if (!user) return;

    try {
      setLoading(true);
      
      // Get current recommendation
      const currentRec = await recommendationService.getCurrentRecommendation(user.uid);
      
      // Get recommendation history
      const history = await recommendationService.getRecommendationHistory(user.uid, 5);
      
      // Get user progress data
      const progress = await authService.getUserProgress(30);
      
      setDashboardData({
        currentRecommendation: currentRec,
        recommendationHistory: history.success ? history.data : [],
        progressData: progress.success ? progress.data : [],
        stats: {
          totalRecommendations: history.success ? history.data.length : 0,
          lastWeightEntry: userData?.weight || null
        }
      });
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };


  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('id-ID', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  };

  const getMetricCards = () => {
    const currentRec = dashboardData.currentRecommendation;
    const userMetrics = currentRec?.userData || userData;
    
    if (!userMetrics) return [];

    const bmi = userMetrics.weight / ((userMetrics.height / 100) ** 2);
    const bmr = userMetrics.gender === 'Male' 
      ? 88.362 + (13.397 * userMetrics.weight) + (4.799 * userMetrics.height) - (5.677 * userMetrics.age)
      : 447.593 + (9.247 * userMetrics.weight) + (3.098 * userMetrics.height) - (4.330 * userMetrics.age);

    return [
      {
        title: 'BMI',
        value: bmi.toFixed(1),
        unit: 'kg/m²',
        icon: '⚖️',
        status: bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese',
        color: bmi < 18.5 ? '#FFA500' : bmi < 25 ? '#4CAF50' : bmi < 30 ? '#FF9800' : '#F44336'
      },
      {
        title: 'Berat Badan',
        value: userMetrics.weight,
        unit: 'kg',
        icon: '🏋️',
        status: `Target: ${userMetrics.fitness_goal}`,
        color: '#2196F3'
      },
      {
        title: 'BMR',
        value: Math.round(bmr),
        unit: 'kcal/hari',
        icon: '🔥',
        status: 'Kalori Basal',
        color: '#FF5722'
      },

    ];
  };

  const getRecentActivity = () => {
    return dashboardData.progressData.slice(0, 5).map(progress => ({
      date: progress.date,
      activities: [
        progress.workout && 'Workout',
        progress.nutrition && 'Nutrition',
        progress.hydration && 'Hydration'
      ].filter(Boolean),
      notes: progress.notes || ''
    }));
  };

  if (loading) {
    return (
      <div className="dashboard loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Memuat data dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard XGFitness</h2>
        <p>Selamat datang kembali, {user.displayName || 'Pengguna'}!</p>
      </div>

      {/* Quick Stats */}
      <div className="metric-cards">
        {getMetricCards().map((metric, index) => (
          <div 
            key={index} 
            className="metric-card" 
            style={{ 
              borderLeft: `4px solid ${metric.color}`,
              '--metric-color': metric.color 
            }}
          >
            <div className="metric-icon">{metric.icon}</div>
            <div className="metric-content">
              <div className="metric-value">
                {metric.value} <span className="metric-unit">{metric.unit}</span>
              </div>
              <div className="metric-title">{metric.title}</div>
              <div className="metric-status" style={{ color: metric.color }}>
                {metric.status}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Current Recommendation Status */}
      {dashboardData.currentRecommendation && (
        <div className="current-recommendation">
          <h3>🎯 Rekomendasi Aktif</h3>
          <div className="recommendation-summary">
            <div className="rec-date">
              <span className="label">Dibuat:</span>
              <span className="value">{formatDate(dashboardData.currentRecommendation.createdAt.toDate())}</span>
            </div>
            <div className="rec-goal">
              <span className="label">Target:</span>
              <span className="value">{dashboardData.currentRecommendation.userData.fitness_goal}</span>
            </div>
            <div className="rec-actions">
              <button 
                className="btn-primary small"
                onClick={() => onNavigate('recommendations')}
              >
                Lihat Detail
              </button>
              <button 
                className="btn-secondary small"
                onClick={() => onNavigate('progress')}
              >
                Update Progress
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="recent-activity">
        <h3>📈 Aktivitas Terbaru</h3>
        {getRecentActivity().length > 0 ? (
          <div className="activity-list">
            {getRecentActivity().map((activity, index) => (
              <div key={index} className="activity-item">
                <div className="activity-date">{formatDate(activity.date)}</div>
                <div className="activity-details">
                  <div className="activity-badges">
                    {activity.activities.map((act, i) => (
                      <span key={i} className="activity-badge">{act}</span>
                    ))}
                  </div>
                  {activity.notes && (
                    <div className="activity-notes">{activity.notes}</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-activity">
            <p>Belum ada aktivitas tercatat. Mulai catat progress Anda!</p>
            <button 
              className="btn-primary"
              onClick={() => onNavigate('progress')}
            >
              Mulai Catat Progress
            </button>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <h3>⚡ Aksi Cepat</h3>
        <div className="action-grid">
          <button 
            className="action-card"
            onClick={() => onNavigate('input')}
          >
            <div className="action-icon">📝</div>
            <div className="action-title">Rencana Baru</div>
            <div className="action-desc">Buat rekomendasi fitness baru</div>
          </button>
          
          <button 
            className="action-card"
            onClick={() => onNavigate('recommendations')}
          >
            <div className="action-icon">🎯</div>
            <div className="action-title">Lihat Rekomendasi</div>
            <div className="action-desc">Cek workout & nutrition plan</div>
          </button>
          
          <button 
            className="action-card"
            onClick={() => onNavigate('progress')}
          >
            <div className="action-icon">📈</div>
            <div className="action-title">Update Progress</div>
            <div className="action-desc">Catat aktivitas harian</div>
          </button>
        </div>
      </div>

      {/* Legacy fallback for when no userData is available */}
      {(!userData && recommendations) && (
        <div className="dashboard-summary">
          <h3>📋 Ringkasan Rekomendasi Terkini</h3>
          <div className="summary-cards">
            <div className="summary-card workout">
              <h4>🏋️ Program Latihan</h4>
              <ul>
                <li>Jenis: {recommendations.workout_recommendation?.workout_type}</li>
                <li>Frekuensi: {recommendations.workout_recommendation?.days_per_week} hari/minggu</li>
                <li>Durasi Cardio: {recommendations.workout_recommendation?.cardio_minutes_per_day} menit/hari</li>
                <li>Set per Latihan: {recommendations.workout_recommendation?.sets_per_exercise}</li>
              </ul>
            </div>
            <div className="summary-card nutrition">
              <h4>🍎 Program Nutrisi</h4>
              <ul>
                <li>Kalori Harian: {Math.round(recommendations.nutrition_recommendation?.target_calories)} kkal</li>
                <li>Protein: {Math.round(recommendations.nutrition_recommendation?.target_protein)} gram</li>
                <li>Karbohidrat: {Math.round(recommendations.nutrition_recommendation?.target_carbs)} gram</li>
                <li>Lemak: {Math.round(recommendations.nutrition_recommendation?.target_fat)} gram</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
