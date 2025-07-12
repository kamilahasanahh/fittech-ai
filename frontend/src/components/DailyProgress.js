import React, { useState, useEffect, useCallback } from 'react';
import { db } from '../services/firebaseConfig';
import { recommendationService } from '../services/recommendationService';
import { doc, setDoc, getDoc } from 'firebase/firestore';
import RecommendationFeedback from './RecommendationFeedback';

const DailyProgress = ({ user, onProgressUpdate, userProfile, currentRecommendation }) => {
  const [progressData, setProgressData] = useState({
    workout: false,
    nutrition: false,
    hydration: false,
    notes: '',
    mood: '',
    workoutRating: 0,
    energyLevel: 0,
    sleepQuality: 0,
    stressLevel: 0,
    recommendationEffectiveness: 0,
    weight: '',
    bodyFat: '',
    measurements: {
      chest: '',
      waist: '',
      arms: '',
      thighs: ''
    }
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [recommendationHistory, setRecommendationHistory] = useState([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState(null);
  const [showRecommendationHistory, setShowRecommendationHistory] = useState(false);

  const today = new Date().toISOString().split('T')[0];

  // Get fitness goal info based on user's actual goal
  const getFitnessGoalInfo = () => {
    const goal = userProfile?.fitness_goal || 'Fat Loss';
    
    switch (goal) {
      case 'Fat Loss':
        return {
          icon: '📉',
          title: 'Menurunkan Berat Badan',
          description: 'Fokus pada pembakaran kalori dan pengurangan lemak tubuh'
        };
      case 'Muscle Gain':
        return {
          icon: '💪',
          title: 'Menambah Massa Otot',
          description: 'Membangun otot dengan latihan beban dan nutrisi yang tepat'
        };
      case 'Maintenance':
        return {
          icon: '⚖️',
          title: 'Mempertahankan Bentuk Tubuh',
          description: 'Menjaga kondisi fisik dan berat badan yang sudah ideal'
        };
      default:
        return {
          icon: '🎯',
          title: 'Target Fitness',
          description: 'Mencapai tujuan kesehatan dan kebugaran optimal'
        };
    }
  };

  // Get activity level info
  const getActivityLevelInfo = () => {
    const level = userProfile?.activity_level || 'Low Activity';
    
    switch (level) {
      case 'Low Activity':
        return {
          icon: '🚶‍♂️',
          title: 'Aktivitas Rendah',
          multiplier: '1.29',
          description: 'Olahraga ringan dengan intensitas rendah'
        };
      case 'Moderate Activity':
        return {
          icon: '🏃‍♂️',
          title: 'Aktivitas Sedang',
          multiplier: '1.55',
          description: 'Olahraga teratur dengan intensitas sedang'
        };
      case 'High Activity':
        return {
          icon: '🏋️‍♂️',
          title: 'Aktivitas Tinggi',
          multiplier: '1.81',
          description: 'Olahraga intensif dengan frekuensi tinggi'
        };
      default:
        return {
          icon: '🎯',
          title: 'Level Aktivitas',
          multiplier: '1.0',
          description: 'Tingkat aktivitas belum ditentukan'
        };
    }
  };

  // Load today's progress data
  const loadTodaysProgress = useCallback(async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      const progressDoc = await getDoc(progressRef);
      
      if (progressDoc.exists()) {
        const data = progressDoc.data();
        setProgressData({
          workout: data.workout || false,
          nutrition: data.nutrition || false,
          hydration: data.hydration || false,
          notes: data.notes || '',
          mood: data.mood || '',
          workoutRating: data.workoutRating || 0,
          energyLevel: data.energyLevel || 0,
          sleepQuality: data.sleepQuality || 0,
          stressLevel: data.stressLevel || 0,
          recommendationEffectiveness: data.recommendationEffectiveness || 0,
          weight: data.weight || '',
          bodyFat: data.bodyFat || '',
          measurements: data.measurements || {
            chest: '',
            waist: '',
            arms: '',
            thighs: ''
          }
        });
      }
      

    } catch (error) {
      console.error('Error loading progress:', error);
    } finally {
      setLoading(false);
    }
  }, [user, today]);

  // Load recommendation history
  const loadRecommendationHistory = useCallback(async () => {
    if (!user) return;
    
    try {
      const history = await recommendationService.getRecommendationHistory(user.uid, 20);
      setRecommendationHistory(history);
    } catch (error) {
      console.error('Error loading recommendation history:', error);
    }
  }, [user]);

  useEffect(() => {
    loadTodaysProgress();
    loadRecommendationHistory();
  }, [loadTodaysProgress, loadRecommendationHistory]);

  const saveProgress = async (newData) => {
    if (!user) return;
    
    setSaving(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      await setDoc(progressRef, {
        ...newData,
        date: today,
        userId: user.uid,
        updatedAt: new Date()
      });



      onProgressUpdate && onProgressUpdate(newData);
    } catch (error) {
      console.error('Error saving progress:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleGoalToggle = (goal) => {
    const newData = {
      ...progressData,
      [goal]: !progressData[goal]
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const handleNotesChange = (e) => {
    const newData = {
      ...progressData,
      notes: e.target.value
    };
    setProgressData(newData);
    // Debounce saving notes
    setTimeout(() => saveProgress(newData), 1000);
  };

  const handleFeedbackChange = (field, value) => {
    const newData = {
      ...progressData,
      [field]: value
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const handleMeasurementChange = (field, value) => {
    const newData = {
      ...progressData,
      measurements: {
        ...progressData.measurements,
        [field]: value
      }
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const getCompletionPercentage = () => {
    const goals = [progressData.workout, progressData.nutrition, progressData.hydration];
    const completed = goals.filter(Boolean).length;
    return Math.round((completed / goals.length) * 100);
  };

  const getMotivationalMessage = () => {
    const percentage = getCompletionPercentage();
    if (percentage === 100) {
      return "🎉 Luar biasa! Anda telah menyelesaikan semua target hari ini!";
    } else if (percentage >= 66) {
      return "💪 Bagus sekali! Tinggal sedikit lagi untuk mencapai target harian!";
    } else if (percentage >= 33) {
      return "👍 Terus semangat! Anda sudah di jalur yang tepat!";
    } else {
      return "🌟 Mulai hari ini dengan semangat! Setiap langkah kecil berarti!";
    }
  };

  // Helper functions for feedback
  const getMoodEmoji = (mood) => {
    const moodEmojis = {
      'excellent': '😄',
      'good': '🙂',
      'neutral': '😐',
      'bad': '😔',
      'terrible': '😢'
    };
    return moodEmojis[mood] || '😐';
  };

  const getRatingStars = (rating) => {
    return '⭐'.repeat(rating) + '☆'.repeat(5 - rating);
  };

  const getLevelLabel = (level) => {
    if (level >= 4) return 'Sangat Tinggi';
    if (level >= 3) return 'Tinggi';
    if (level >= 2) return 'Sedang';
    if (level >= 1) return 'Rendah';
    return 'Sangat Rendah';
  };

  const getEffectivenessLabel = (rating) => {
    if (rating >= 4) return 'Sangat Efektif';
    if (rating >= 3) return 'Efektif';
    if (rating >= 2) return 'Cukup Efektif';
    if (rating >= 1) return 'Kurang Efektif';
    return 'Tidak Efektif';
  };

  if (loading) {
    return (
      <div className="daily-progress loading">
        <div className="spinner"></div>
        <p>Memuat progress hari ini...</p>
      </div>
    );
  }

  return (
    <div className="daily-progress">


      {/* Recommendation History Section */}
      <div className="recommendation-history-section">
        <div className="section-header">
          <h3>📋 Riwayat Rekomendasi</h3>
          <button 
            className="btn-secondary"
            onClick={() => setShowRecommendationHistory(!showRecommendationHistory)}
          >
            {showRecommendationHistory ? 'Sembunyikan' : 'Tampilkan'} Riwayat
          </button>
        </div>
        
        {showRecommendationHistory && (
          <div className="recommendation-history">
            {recommendationHistory.length === 0 ? (
              <div className="no-history">
                <p>Belum ada riwayat rekomendasi</p>
                <p>Buat rekomendasi pertama Anda untuk memulai tracking!</p>
              </div>
            ) : (
              <div className="history-list">
                {recommendationHistory.map((recommendation, index) => (
                  <div key={recommendation.id} className="history-item">
                    <div className="history-header">
                      <div className="history-date">
                        <span className="date">
                          📅 {new Date(recommendation.createdAt.seconds * 1000).toLocaleDateString('id-ID', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </span>
                        <span className="time">
                          🕐 {new Date(recommendation.createdAt.seconds * 1000).toLocaleTimeString('id-ID', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                      </div>
                      <div className="history-status">
                        {recommendation.isActive && (
                          <span className="active-badge">✅ Aktif</span>
                        )}
                        {index === 0 && !recommendation.isActive && (
                          <span className="latest-badge">🆕 Terbaru</span>
                        )}
                      </div>
                    </div>
                    
                    <div className="history-summary">
                      <div className="summary-item">
                        <span className="label">Tujuan:</span>
                        <span className="value">{recommendation.userData.fitness_goal}</span>
                      </div>
                      <div className="summary-item">
                        <span className="label">Aktivitas:</span>
                        <span className="value">{recommendation.userData.activity_level}</span>
                      </div>
                      <div className="summary-item">
                        <span className="label">BMI:</span>
                        <span className="value">
                          {recommendation.recommendations.user_metrics?.bmi?.toFixed(1) || 
                           (recommendation.userData?.weight && recommendation.userData?.height ? 
                             (recommendation.userData.weight / ((recommendation.userData.height / 100) ** 2)).toFixed(1) : 
                             'N/A')}
                        </span>
                      </div>
                    </div>
                    
                    <div className="history-actions">
                      <button 
                        className="btn-outline"
                        onClick={() => setSelectedRecommendation(recommendation)}
                      >
                        👁️ Lihat Detail
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Selected Recommendation Modal */}
      {selectedRecommendation && (
        <div className="recommendation-modal">
          <div className="modal-content">
            <div className="modal-header">
              <h3>📋 Detail Rekomendasi</h3>
              <button 
                className="close-btn"
                onClick={() => setSelectedRecommendation(null)}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <div className="recommendation-display-mini">
                <div className="mini-section">
                  <h4>👤 Profil Pengguna</h4>
                  <div className="profile-mini">
                    <span>Usia: {selectedRecommendation.userData.age} tahun</span>
                    <span>Berat: {selectedRecommendation.userData.weight} kg</span>
                    <span>Tinggi: {selectedRecommendation.userData.height} cm</span>
                  </div>
                </div>
                
                {selectedRecommendation.recommendations.workout_recommendation && (
                  <div className="mini-section">
                    <h4>🏋️ Workout</h4>
                    <div className="workout-mini">
                      <span>Jenis: {selectedRecommendation.recommendations.workout_recommendation.workout_type}</span>
                      <span>Hari/Minggu: {selectedRecommendation.recommendations.workout_recommendation.days_per_week}</span>
                      <span>Kardio: {selectedRecommendation.recommendations.workout_recommendation.cardio_minutes_per_day} menit</span>
                    </div>
                  </div>
                )}
                
                {selectedRecommendation.recommendations.nutrition_recommendation && (
                  <div className="mini-section">
                    <h4>🍎 Nutrisi</h4>
                    <div className="nutrition-mini">
                      <span>Kalori: {selectedRecommendation.recommendations.nutrition_recommendation.target_calories} kkal</span>
                      <span>Protein: {selectedRecommendation.recommendations.nutrition_recommendation.target_protein}g</span>
                      <span>Karbohidrat: {selectedRecommendation.recommendations.nutrition_recommendation.target_carbs}g</span>
                    </div>
                  </div>
                )}
                
                {selectedRecommendation.recommendations.confidence_scores && (
                  <div className="mini-section">
                    <h4>🎯 Tingkat Kepercayaan</h4>
                    <div className="confidence-mini">
                      <span>Keseluruhan: {Math.round(selectedRecommendation.recommendations.confidence_scores.overall_confidence * 100)}%</span>
                      <span>Level: {selectedRecommendation.recommendations.confidence_scores.confidence_level}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recommendation Feedback Section */}
      {currentRecommendation && (
        <RecommendationFeedback
          currentRecommendation={currentRecommendation}
          userProfile={userProfile}
          user={user}
          onRecommendationUpdate={(newRecommendation) => {
            // Handle recommendation update
            console.log('New recommendation applied:', newRecommendation);
            // You can add logic here to update the current recommendation
          }}
        />
      )}

      <div className="progress-tips">
        <h3>💡 Tips Hari Ini</h3>
        <div className="tips-grid">
          <div className="tip-card">
            <h4>🕐 Waktu Terbaik Berolahraga</h4>
            <p>Pagi hari (06:00-08:00) atau sore (16:00-18:00) adalah waktu optimal untuk berolahraga.</p>
          </div>
          <div className="tip-card">
            <h4>🥗 Nutrisi Seimbang</h4>
            <p>Pastikan setiap makanan mengandung protein, karbohidrat, dan sayuran untuk nutrisi optimal.</p>
          </div>
          <div className="tip-card">
            <h4>😴 Istirahat Cukup</h4>
            <p>Tidur 7-9 jam per malam sangat penting untuk pemulihan otot dan metabolisme yang baik.</p>
          </div>
        </div>
      </div>


    </div>
  );
};

export default DailyProgress;
