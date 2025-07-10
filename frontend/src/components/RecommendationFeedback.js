import React, { useState, useEffect } from 'react';
import { saveRecommendationFeedback, getImprovedRecommendation, recommendationService } from '../services/recommendationService';
import { useNavigate } from 'react-router-dom';

const RecommendationFeedback = ({ currentRecommendation, userProfile, onRecommendationUpdate, user }) => {
  // Extract the actual recommendation data from the stored structure
  const recommendations = currentRecommendation?.recommendations || currentRecommendation;
  
  // Debug logging
  console.log('Current Recommendation:', currentRecommendation);
  console.log('Extracted Recommendations:', recommendations);
  console.log('Workout Recommendation:', recommendations?.workout_recommendation);
  console.log('Nutrition Recommendation:', recommendations?.nutrition_recommendation);
  const [feedback, setFeedback] = useState({
    workoutDifficulty: 'just_right',
    workoutEnjoyment: 'enjoyed',
    workoutEffectiveness: 'effective',
    nutritionSatisfaction: 'satisfied',
    energyLevel: 'good',
    recovery: 'good',
    overallSatisfaction: 'satisfied'
  });
  
  const [suggestions, setSuggestions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [alert, setAlert] = useState(null);

  const feedbackOptions = {
    workoutDifficulty: {
      too_easy: { label: 'Terlalu Mudah', icon: '😴', color: 'text-blue-500' },
      just_right: { label: 'Tepat', icon: '😊', color: 'text-green-500' },
      too_hard: { label: 'Terlalu Sulit', icon: '😰', color: 'text-red-500' }
    },
    workoutEnjoyment: {
      enjoyed: { label: 'Menyukainya', icon: '😄', color: 'text-green-500' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'text-yellow-500' },
      disliked: { label: 'Tidak Suka', icon: '😞', color: 'text-red-500' }
    },
    workoutEffectiveness: {
      effective: { label: 'Sangat Efektif', icon: '💪', color: 'text-green-500' },
      somewhat: { label: 'Agak Efektif', icon: '👍', color: 'text-yellow-500' },
      not_effective: { label: 'Tidak Efektif', icon: '👎', color: 'text-red-500' }
    },
    nutritionSatisfaction: {
      satisfied: { label: 'Puas', icon: '😋', color: 'text-green-500' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'text-yellow-500' },
      unsatisfied: { label: 'Tidak Puas', icon: '😕', color: 'text-red-500' }
    },
    energyLevel: {
      great: { label: 'Energi Besar', icon: '⚡', color: 'text-green-500' },
      good: { label: 'Energi Baik', icon: '👍', color: 'text-blue-500' },
      low: { label: 'Energi Rendah', icon: '😴', color: 'text-red-500' }
    },
    recovery: {
      great: { label: 'Pemulihan Besar', icon: '🔄', color: 'text-green-500' },
      good: { label: 'Pemulihan Baik', icon: '👍', color: 'text-blue-500' },
      poor: { label: 'Pemulihan Buruk', icon: '😫', color: 'text-red-500' }
    },
    overallSatisfaction: {
      satisfied: { label: 'Puas', icon: '😊', color: 'text-green-500' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'text-yellow-500' },
      unsatisfied: { label: 'Tidak Puas', icon: '😞', color: 'text-red-500' }
    }
  };

  const handleFeedbackChange = (category, value) => {
    setFeedback(prev => ({
      ...prev,
      [category]: value
    }));
  };

  const generateSuggestions = async () => {
    try {
      console.log('Generating suggestions with:', {
        currentRecommendation: recommendations,
        userProfile,
        feedback
      });
      
      const improvedRecommendation = await getImprovedRecommendation({
        currentRecommendation: recommendations,
        userProfile,
        feedback
      });
      
      console.log('Improved recommendation received:', improvedRecommendation);
      return improvedRecommendation;
    } catch (error) {
      console.error('Error generating suggestions:', error);
      return null;
    }
  };

  const saveFeedback = async () => {
    setIsLoading(true);
    try {
      console.log('Saving feedback with data:', feedback);
      console.log('User profile:', userProfile);
      console.log('Current recommendation:', recommendations);
      
      await saveRecommendationFeedback({
        recommendationId: currentRecommendation.id || Date.now(),
        feedback,
        timestamp: new Date().toISOString()
      });
      
      const result = await generateSuggestions();
      console.log('Generated suggestions result:', result);
      
      if (result) {
        setSuggestions(result);
      }
    } catch (error) {
      console.error('Error saving feedback:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const applySuggestion = async (suggestion) => {
    try {
      if (!user || !userProfile) {
        setAlert('Gagal menerapkan rencana baru: data pengguna tidak ditemukan.');
        return;
      }
      // Save the new recommendation to Firestore
      const saved = await recommendationService.saveRecommendation(
        user.uid,
        userProfile,
        suggestion
      );
      setAlert('Rencana baru berhasil diterapkan!');
      // Dispatch a global event so other components can refetch
      window.dispatchEvent(new Event('recommendationUpdated'));
      if (onRecommendationUpdate) {
        onRecommendationUpdate(saved);
      }
      setSuggestions(null);
      setShowFeedback(false);
    } catch (error) {
      setAlert('Gagal menerapkan rencana baru. Silakan coba lagi.');
      console.error('Error applying suggestion:', error);
    }
  };

  const getFeedbackCard = (category, title) => (
    <div className="feedback-card">
      <h4 className="feedback-title">{title}</h4>
      <div className="feedback-options">
        {Object.entries(feedbackOptions[category]).map(([key, option]) => (
          <button
            key={key}
            className={`feedback-option ${feedback[category] === key ? 'selected' : ''}`}
            onClick={() => handleFeedbackChange(category, key)}
          >
            <span className="feedback-icon">{option.icon}</span>
            <span className={`feedback-label ${option.color}`}>{option.label}</span>
          </button>
        ))}
      </div>
    </div>
  );

  const navigate = useNavigate();

  return (
    <div className="recommendation-feedback">
      {alert && (
        <div className="alert-success">
          <span>{alert}</span>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button className="btn-primary" onClick={() => navigate('/recommendation')}>
              Lihat Rekomendasi
            </button>
            <button className="close-alert" onClick={() => setAlert(null)}>
              ✕
            </button>
          </div>
        </div>
      )}
      {!showFeedback ? (
        <div className="feedback-prompt">
          <div className="feedback-prompt-content">
            <h3>Bagaimana latihan Anda hari ini? 💪</h3>
            <p>Bantu kami meningkatkan rekomendasi dengan berbagi pengalaman Anda!</p>
            <button 
              className="feedback-prompt-button"
              onClick={() => setShowFeedback(true)}
            >
              Berikan Feedback
            </button>
          </div>
        </div>
      ) : (
        <div className="feedback-form">
          <div className="feedback-header">
            <h3>Feedback Latihan & Nutrisi</h3>
            <button 
              className="close-feedback"
              onClick={() => setShowFeedback(false)}
            >
              ✕
            </button>
          </div>

          <div className="feedback-sections">
            <div className="feedback-section">
              <h4>Pengalaman Latihan</h4>
              {getFeedbackCard('workoutDifficulty', 'Tingkat Kesulitan Latihan')}
              {getFeedbackCard('workoutEnjoyment', 'Kesenangan Latihan')}
              {getFeedbackCard('workoutEffectiveness', 'Efektivitas Latihan')}
            </div>

            <div className="feedback-section">
              <h4>Nutrisi & Pemulihan</h4>
              {getFeedbackCard('nutritionSatisfaction', 'Kepuasan Nutrisi')}
              {getFeedbackCard('energyLevel', 'Tingkat Energi')}
              {getFeedbackCard('recovery', 'Pemulihan')}
            </div>

            <div className="feedback-section">
              <h4>Keseluruhan</h4>
              {getFeedbackCard('overallSatisfaction', 'Kepuasan Keseluruhan')}
            </div>
          </div>

          <div className="feedback-actions">
            <button 
              className="save-feedback-button"
              onClick={saveFeedback}
              disabled={isLoading}
            >
              {isLoading ? 'Menganalisis...' : 'Simpan Feedback & Dapatkan Saran'}
            </button>
          </div>

          {suggestions && (
            <div className="suggestions-section">
              <h4>💡 Saran yang Dipersonalisasi</h4>
              <div className="suggestions-content">
                <div className="suggestion-card">
                  <h5>Perubahan yang Direkomendasikan</h5>
                  <div className="suggestion-details">
                    {suggestions.workoutChanges && (
                      <div className="suggestion-item">
                        <span className="suggestion-icon">🏋️</span>
                        <span className="suggestion-text">{suggestions.workoutChanges}</span>
                      </div>
                    )}
                    {suggestions.nutritionChanges && (
                      <div className="suggestion-item">
                        <span className="suggestion-icon">🥗</span>
                        <span className="suggestion-text">{suggestions.nutritionChanges}</span>
                      </div>
                    )}
                  </div>
                  
                  {/* Plan Comparison */}
                  <div className="plan-comparison">
                    <h6>📊 Perbandingan Rencana</h6>
                    <div className="comparison-grid">
                      <div className="current-plan">
                        <h6>Rencana Saat Ini</h6>
                        <div className="plan-details">
                          {recommendations.workout_recommendation ? (
                            <div className="plan-section">
                              <h7>🏋️ Latihan</h7>
                              <div className="plan-item">
                                <span>Jenis:</span>
                                <span>{recommendations.workout_recommendation.workout_type || 'N/A'}</span>
                              </div>
                              <div className="plan-item">
                                <span>Hari/Minggu:</span>
                                <span>{recommendations.workout_recommendation.days_per_week || 'N/A'}</span>
                              </div>
                              <div className="plan-item">
                                <span>Kardio:</span>
                                <span>{recommendations.workout_recommendation.cardio_minutes_per_day || 'N/A'} menit</span>
                              </div>
                              <div className="plan-item">
                                <span>Set:</span>
                                <span>{recommendations.workout_recommendation.sets_per_exercise || 'N/A'}</span>
                              </div>
                            </div>
                          ) : (
                            <div className="plan-section">
                              <h7>🏋️ Latihan</h7>
                              <p>Data latihan tidak tersedia</p>
                            </div>
                          )}
                          {recommendations.nutrition_recommendation ? (
                            <div className="plan-section">
                              <h7>🥗 Nutrisi</h7>
                              <div className="plan-item">
                                <span>Kalori:</span>
                                <span>{recommendations.nutrition_recommendation.target_calories || 'N/A'} kkal</span>
                              </div>
                              <div className="plan-item">
                                <span>Protein:</span>
                                <span>{recommendations.nutrition_recommendation.target_protein || 'N/A'}g</span>
                              </div>
                              <div className="plan-item">
                                <span>Karbohidrat:</span>
                                <span>{recommendations.nutrition_recommendation.target_carbs || 'N/A'}g</span>
                              </div>
                              <div className="plan-item">
                                <span>Lemak:</span>
                                <span>{recommendations.nutrition_recommendation.target_fat || 'N/A'}g</span>
                              </div>
                            </div>
                          ) : (
                            <div className="plan-section">
                              <h7>🥗 Nutrisi</h7>
                              <p>Data nutrisi tidak tersedia</p>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="comparison-arrow">➡️</div>
                      
                      <div className="suggested-plan">
                        <h6>Rencana yang Disarankan</h6>
                        <div className="plan-details">
                          {suggestions.newRecommendation.workout_recommendation ? (
                            <div className="plan-section">
                              <h7>🏋️ Latihan</h7>
                              <div className="plan-item">
                                <span>Jenis:</span>
                                <span className={recommendations.workout_recommendation?.workout_type !== suggestions.newRecommendation.workout_recommendation.workout_type ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.workout_recommendation.workout_type || 'N/A'}
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Hari/Minggu:</span>
                                <span className={recommendations.workout_recommendation?.days_per_week !== suggestions.newRecommendation.workout_recommendation.days_per_week ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.workout_recommendation.days_per_week || 'N/A'}
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Kardio:</span>
                                <span className={recommendations.workout_recommendation?.cardio_minutes_per_day !== suggestions.newRecommendation.workout_recommendation.cardio_minutes_per_day ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.workout_recommendation.cardio_minutes_per_day || 'N/A'} menit
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Set:</span>
                                <span className={recommendations.workout_recommendation?.sets_per_exercise !== suggestions.newRecommendation.workout_recommendation.sets_per_exercise ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.workout_recommendation.sets_per_exercise || 'N/A'}
                                </span>
                              </div>
                            </div>
                          ) : (
                            <div className="plan-section">
                              <h7>🏋️ Latihan</h7>
                              <p>Data latihan tidak tersedia</p>
                            </div>
                          )}
                          {suggestions.newRecommendation.nutrition_recommendation ? (
                            <div className="plan-section">
                              <h7>🥗 Nutrisi</h7>
                              <div className="plan-item">
                                <span>Kalori:</span>
                                <span className={recommendations.nutrition_recommendation?.target_calories !== suggestions.newRecommendation.nutrition_recommendation.target_calories ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.nutrition_recommendation.target_calories || 'N/A'} kkal
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Protein:</span>
                                <span className={recommendations.nutrition_recommendation?.target_protein !== suggestions.newRecommendation.nutrition_recommendation.target_protein ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.nutrition_recommendation.target_protein || 'N/A'}g
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Karbohidrat:</span>
                                <span className={recommendations.nutrition_recommendation?.target_carbs !== suggestions.newRecommendation.nutrition_recommendation.target_carbs ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.nutrition_recommendation.target_carbs || 'N/A'}g
                                </span>
                              </div>
                              <div className="plan-item">
                                <span>Lemak:</span>
                                <span className={recommendations.nutrition_recommendation?.target_fat !== suggestions.newRecommendation.nutrition_recommendation.target_fat ? 'highlight-change' : ''}>
                                  {suggestions.newRecommendation.nutrition_recommendation.target_fat || 'N/A'}g
                                </span>
                              </div>
                            </div>
                          ) : (
                            <div className="plan-section">
                              <h7>🥗 Nutrisi</h7>
                              <p>Data nutrisi tidak tersedia</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="suggestion-reasoning">
                    <h6>🤔 Mengapa Perubahan Ini?</h6>
                    <p>{suggestions.reasoning}</p>
                  </div>
                  
                  <div className="suggestion-actions">
                    <button 
                      className="apply-suggestion-button"
                      onClick={() => applySuggestion(suggestions.newRecommendation)}
                    >
                      Terapkan Rekomendasi Baru
                    </button>
                    <button 
                      className="keep-current-button"
                      onClick={() => setSuggestions(null)}
                    >
                      Pertahankan Rencana Saat Ini
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RecommendationFeedback; 