import React, { useState, useEffect, useCallback } from 'react';
import { db } from '../services/firebaseConfig';
import { recommendationService } from '../services/recommendationService';
import { doc, setDoc, getDoc } from 'firebase/firestore';

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
      <div className="progress-header">
        <h2>📈 Progress Harian</h2>
        <p>Pantau pencapaian target fitness Anda hari ini</p>
        <div className="date-info">
          <span>📅 {new Date().toLocaleDateString('id-ID', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          })}</span>
        </div>
      </div>

      <div className="progress-overview">
        <div className="completion-circle">
          <div className="circle-progress">
            <svg viewBox="0 0 36 36" className="circular-chart">
              <path
                className="circle-bg"
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="circle"
                strokeDasharray={`${getCompletionPercentage()}, 100`}
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <text x="18" y="20.35" className="percentage">
                {getCompletionPercentage()}%
              </text>
            </svg>
          </div>
          <div className="completion-text">
            <h3>Target Harian</h3>
            <p>{getMotivationalMessage()}</p>
          </div>
        </div>


      </div>

      <div className="daily-goals">
        <h3>🎯 Target Hari Ini</h3>
        
        <div className="goal-list">
          <div className={`goal-item ${progressData.workout ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">🏋️‍♂️</div>
              <div className="goal-info">
                <h4>Latihan Fisik</h4>
                <p>Selesaikan sesi latihan sesuai program Anda</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.workout ? 'active' : ''}`}
              onClick={() => handleGoalToggle('workout')}
              disabled={saving}
            >
              {progressData.workout ? '✅' : '⭕'}
            </button>
          </div>

          <div className={`goal-item ${progressData.nutrition ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">🍎</div>
              <div className="goal-info">
                <h4>Target Nutrisi</h4>
                <p>Ikuti panduan makanan dan kalori harian</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.nutrition ? 'active' : ''}`}
              onClick={() => handleGoalToggle('nutrition')}
              disabled={saving}
            >
              {progressData.nutrition ? '✅' : '⭕'}
            </button>
          </div>

          <div className={`goal-item ${progressData.hydration ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">💧</div>
              <div className="goal-info">
                <h4>Hidrasi</h4>
                <p>Minum air putih minimal 8 gelas (2 liter)</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.hydration ? 'active' : ''}`}
              onClick={() => handleGoalToggle('hydration')}
              disabled={saving}
            >
              {progressData.hydration ? '✅' : '⭕'}
            </button>
          </div>
        </div>
      </div>

      <div className="progress-notes">
        <h3>📝 Catatan Harian</h3>
        <textarea
          placeholder="Bagaimana perasaan Anda hari ini? Catat pencapaian, tantangan, atau hal-hal yang ingin Anda ingat..."
          value={progressData.notes}
          onChange={handleNotesChange}
          rows={4}
          disabled={saving}
        />
        {saving && <p className="saving-indicator">💾 Menyimpan...</p>}
      </div>

      {/* Mood & Feedback Section */}
      <div className="feedback-section">
        <h3>😊 Feedback Harian</h3>
        
        <div className="feedback-grid">
          {/* Mood */}
          <div className="feedback-card">
            <h4>Bagaimana mood Anda hari ini?</h4>
            <div className="mood-selector">
              {[
                { value: 'excellent', label: 'Sangat Baik', emoji: '😄' },
                { value: 'good', label: 'Baik', emoji: '🙂' },
                { value: 'neutral', label: 'Biasa', emoji: '😐' },
                { value: 'bad', label: 'Kurang', emoji: '😔' },
                { value: 'terrible', label: 'Buruk', emoji: '😢' }
              ].map(mood => (
                <button
                  key={mood.value}
                  className={`mood-btn ${progressData.mood === mood.value ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('mood', mood.value)}
                  disabled={saving}
                >
                  <span className="mood-emoji">{mood.emoji}</span>
                  <span className="mood-label">{mood.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Workout Rating */}
          <div className="feedback-card">
            <h4>Rate workout Anda hari ini (1-5)</h4>
            <div className="rating-selector">
              {[1, 2, 3, 4, 5].map(rating => (
                <button
                  key={rating}
                  className={`rating-btn ${progressData.workoutRating === rating ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('workoutRating', rating)}
                  disabled={saving}
                >
                  {rating}
                </button>
              ))}
            </div>
            {progressData.workoutRating > 0 && (
              <div className="rating-display">
                <span className="stars">{getRatingStars(progressData.workoutRating)}</span>
                <span className="rating-text">({progressData.workoutRating}/5)</span>
              </div>
            )}
          </div>

          {/* Energy Level */}
          <div className="feedback-card">
            <h4>Level energi Anda hari ini</h4>
            <div className="level-selector">
              {[1, 2, 3, 4, 5].map(level => (
                <button
                  key={level}
                  className={`level-btn ${progressData.energyLevel === level ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('energyLevel', level)}
                  disabled={saving}
                >
                  {level}
                </button>
              ))}
            </div>
            {progressData.energyLevel > 0 && (
              <div className="level-display">
                <span className="level-text">{getLevelLabel(progressData.energyLevel)}</span>
              </div>
            )}
          </div>

          {/* Sleep Quality */}
          <div className="feedback-card">
            <h4>Kualitas tidur malam kemarin</h4>
            <div className="level-selector">
              {[1, 2, 3, 4, 5].map(level => (
                <button
                  key={level}
                  className={`level-btn ${progressData.sleepQuality === level ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('sleepQuality', level)}
                  disabled={saving}
                >
                  {level}
                </button>
              ))}
            </div>
            {progressData.sleepQuality > 0 && (
              <div className="level-display">
                <span className="level-text">{getLevelLabel(progressData.sleepQuality)}</span>
              </div>
            )}
          </div>

          {/* Stress Level */}
          <div className="feedback-card">
            <h4>Level stres Anda hari ini</h4>
            <div className="level-selector">
              {[1, 2, 3, 4, 5].map(level => (
                <button
                  key={level}
                  className={`level-btn ${progressData.stressLevel === level ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('stressLevel', level)}
                  disabled={saving}
                >
                  {level}
                </button>
              ))}
            </div>
            {progressData.stressLevel > 0 && (
              <div className="level-display">
                <span className="level-text">{getLevelLabel(progressData.stressLevel)}</span>
              </div>
            )}
          </div>

          {/* Recommendation Effectiveness */}
          <div className="feedback-card">
            <h4>Seberapa efektif rekomendasi untuk Anda?</h4>
            <div className="level-selector">
              {[1, 2, 3, 4, 5].map(level => (
                <button
                  key={level}
                  className={`level-btn ${progressData.recommendationEffectiveness === level ? 'selected' : ''}`}
                  onClick={() => handleFeedbackChange('recommendationEffectiveness', level)}
                  disabled={saving}
                >
                  {level}
                </button>
              ))}
            </div>
            {progressData.recommendationEffectiveness > 0 && (
              <div className="level-display">
                <span className="level-text">{getEffectivenessLabel(progressData.recommendationEffectiveness)}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Body Measurements Section */}
      <div className="measurements-section">
        <h3>📏 Pengukuran Tubuh</h3>
        <div className="measurements-grid">
          <div className="measurement-input">
            <label>Berat Badan (kg)</label>
            <input
              type="number"
              placeholder="Contoh: 65.5"
              value={progressData.weight}
              onChange={(e) => handleFeedbackChange('weight', e.target.value)}
              disabled={saving}
              step="0.1"
            />
          </div>
          <div className="measurement-input">
            <label>Body Fat %</label>
            <input
              type="number"
              placeholder="Contoh: 15.2"
              value={progressData.bodyFat}
              onChange={(e) => handleFeedbackChange('bodyFat', e.target.value)}
              disabled={saving}
              step="0.1"
            />
          </div>
          <div className="measurement-input">
            <label>Lingkar Dada (cm)</label>
            <input
              type="number"
              placeholder="Contoh: 95"
              value={progressData.measurements.chest}
              onChange={(e) => handleMeasurementChange('chest', e.target.value)}
              disabled={saving}
            />
          </div>
          <div className="measurement-input">
            <label>Lingkar Pinggang (cm)</label>
            <input
              type="number"
              placeholder="Contoh: 80"
              value={progressData.measurements.waist}
              onChange={(e) => handleMeasurementChange('waist', e.target.value)}
              disabled={saving}
            />
          </div>
          <div className="measurement-input">
            <label>Lingkar Lengan (cm)</label>
            <input
              type="number"
              placeholder="Contoh: 32"
              value={progressData.measurements.arms}
              onChange={(e) => handleMeasurementChange('arms', e.target.value)}
              disabled={saving}
            />
          </div>
          <div className="measurement-input">
            <label>Lingkar Paha (cm)</label>
            <input
              type="number"
              placeholder="Contoh: 55"
              value={progressData.measurements.thighs}
              onChange={(e) => handleMeasurementChange('thighs', e.target.value)}
              disabled={saving}
            />
          </div>
        </div>
      </div>

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
                        <span className="value">{recommendation.recommendations.user_metrics?.bmi?.toFixed(1) || 'N/A'}</span>
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
                    <h4>🎯 Confidence</h4>
                    <div className="confidence-mini">
                      <span>Overall: {Math.round(selectedRecommendation.recommendations.confidence_scores.overall_confidence * 100)}%</span>
                      <span>Level: {selectedRecommendation.recommendations.confidence_scores.confidence_level}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
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

      {getCompletionPercentage() === 100 && (
        <div className="celebration">
          <div className="celebration-content">
            <h3>🎉 Selamat!</h3>
            <p>Anda telah menyelesaikan semua target hari ini!</p>
            <p>Terus pertahankan konsistensi Anda!</p>
            <button className="btn-primary">
              Bagikan Pencapaian
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DailyProgress;
