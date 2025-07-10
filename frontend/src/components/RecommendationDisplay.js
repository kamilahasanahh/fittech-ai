import React, { useState, useEffect } from 'react';
import { nutritionService } from '../services/nutritionService';
import { mealPlanService } from '../services/mealPlanService';

const RecommendationDisplay = ({ recommendations, userData, onBack, onNewRecommendation }) => {
  const [nutritionData, setNutritionData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mealPlan, setMealPlan] = useState(null);
  const [mealPlanLoading, setMealPlanLoading] = useState(true);

  // Calculate user's daily macro targets based on API response
  const calculateDailyMacros = () => {
    if (!recommendations || !userData) return null;

    const nutrition = recommendations.nutrition_recommendation || recommendations.nutrition;
    if (!nutrition) return null;

    // Use the actual calculated values from the API response
    return {
      calories: Math.round(nutrition.target_calories || nutrition.caloric_intake || 2000),
      protein: Math.round(nutrition.target_protein || (nutrition.protein_per_kg * parseFloat(userData.weight)) || 100),
      carbs: Math.round(nutrition.target_carbs || (nutrition.carbs_per_kg * parseFloat(userData.weight)) || 200),
      fat: Math.round(nutrition.target_fat || (nutrition.fat_per_kg * parseFloat(userData.weight)) || 60)
    };
  };

  // Load nutrition data from JSON file and generate meal plan
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load nutrition data
        const jsonData = await nutritionService.loadNutritionData();
        setNutritionData(jsonData);
        setLoading(false);

        // Generate meal plan if we have user data
        if (recommendations && userData) {
          setMealPlanLoading(true);
          const dailyMacros = calculateDailyMacros();
          
          if (dailyMacros) {
            const mealPlanResult = await mealPlanService.generateDailyMealPlan(
              dailyMacros.calories,
              dailyMacros.protein,
              dailyMacros.carbs,
              dailyMacros.fat
            );

            if (mealPlanResult.success) {
              const transformedMealPlan = mealPlanService.transformMealPlanToFrontend(mealPlanResult);
              setMealPlan(transformedMealPlan);
            } else {
              console.warn('Failed to load organized meal plan, using fallback');
              setMealPlan(null);
            }
          }
          setMealPlanLoading(false);
        }
      } catch (error) {
        console.error('Error loading data:', error);
        setLoading(false);
        setMealPlanLoading(false);
      }
    };

    loadData();
  }, [recommendations, userData]);

  if (!recommendations) {
    return (
      <div className="recommendation-container">
        <div className="no-recommendations">
          <h2>🤔 Belum Ada Rekomendasi</h2>
          <p>Silakan isi formulir profil terlebih dahulu untuk mendapatkan rekomendasi yang dipersonalisasi.</p>
          <button onClick={onBack} className="btn-primary">
            Kembali ke Formulir
          </button>
        </div>
      </div>
    );
  }

  // Map API response structure to component expected structure
  const workout = recommendations.workout_recommendation || recommendations.workout;
  const nutrition = recommendations.nutrition_recommendation || recommendations.nutrition;

  // Calculate food portions based on template requirements
  const calculateFoodPortions = (targetMacros) => {
    if (!nutritionData.length || !targetMacros) return [];

    const suggestions = [];
    
    // Distribute daily calories across meals
    const mealDistribution = {
      sarapan: { percentage: 0.25, name: '🌅 Sarapan' },
      makan_siang: { percentage: 0.40, name: '☀️ Makan Siang' },
      makan_malam: { percentage: 0.30, name: '🌙 Makan Malam' },
      camilan: { percentage: 0.05, name: '🍪 Camilan' }
    };

    Object.entries(mealDistribution).forEach(([mealType, config]) => {
      const targetCalories = targetMacros.calories * config.percentage;
      const targetProtein = targetMacros.protein * config.percentage;
      
      // Select appropriate foods for this meal using nutrition service
      let selectedFoods = [];
      
      if (mealType === 'sarapan') {
        selectedFoods = nutritionService.getFoodsByCategory('breakfast');
      } else if (mealType === 'makan_siang') {
        selectedFoods = nutritionService.getFoodsByCategory('lunch');
      } else if (mealType === 'makan_malam') {
        selectedFoods = nutritionService.getFoodsByCategory('dinner');
      } else if (mealType === 'camilan') {
        selectedFoods = nutritionService.getFoodsByCategory('snack');
      }

      // If no specific foods found, use all available
      if (selectedFoods.length === 0) {
        if (mealType === 'camilan') {
          // For snacks, use lighter options
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('jagung') ||
            food.name.toLowerCase().includes('telur rebus') ||
            food.calories < 200
          ).slice(0, 2);
        } else {
          selectedFoods = nutritionData.slice(0, 3);
        }
      }

      // Calculate portions to meet targets
      const mealFoods = selectedFoods.slice(0, mealType === 'camilan' ? 1 : 3).map(food => {
        // Calculate how many grams needed to meet portion of target calories
        const divisor = mealType === 'camilan' ? 1 : 3; // Only 1 food for snacks, 3 for meals
        const gramsNeeded = Math.min(
          mealType === 'camilan' ? 100 : 200, 
          Math.max(
            mealType === 'camilan' ? 30 : 50, 
            (targetCalories / divisor) / (food.calories / 100)
          )
        );
        
        return {
          ...food,
          grams: Math.round(gramsNeeded),
          actualCalories: Math.round((food.calories / 100) * gramsNeeded),
          actualProtein: Math.round((food.protein / 100) * gramsNeeded * 10) / 10,
          actualCarbs: Math.round((food.carbs / 100) * gramsNeeded * 10) / 10,
          actualFat: Math.round((food.fat / 100) * gramsNeeded * 10) / 10
        };
      });

      suggestions.push({
        meal: config.name,
        targetCalories: Math.round(targetCalories),
        targetProtein: Math.round(targetProtein),
        foods: mealFoods
      });
    });

    return suggestions;
  };

  const dailyMacros = calculateDailyMacros();
  
  // Use organized meal plan if available, otherwise use fallback calculation
  const foodSuggestions = mealPlan && mealPlan.length > 0 
    ? mealPlan 
    : (dailyMacros ? calculateFoodPortions(dailyMacros) : []);

  return (
    <div className="recommendation-container">
      <div className="recommendation-header">
        <h2>🎯 Rekomendasi XGFitness Anda</h2>
        <p>Rekomendasi yang dipersonalisasi berdasarkan profil dan tujuan Anda</p>
      </div>

      {/* User Profile Summary */}
      <div className="profile-summary">
        <h3>👤 Profil Anda</h3>
        <div className="profile-grid">
          <div className="profile-item">
            <span className="label">Usia:</span>
            <span className="value">{userData?.age} tahun</span>
          </div>
          <div className="profile-item">
            <span className="label">Jenis Kelamin:</span>
            <span className="value">{userData?.gender === 'Male' ? 'Pria' : 'Wanita'}</span>
          </div>
          <div className="profile-item">
            <span className="label">Tinggi:</span>
            <span className="value">{userData?.height} cm</span>
          </div>
          <div className="profile-item">
            <span className="label">Berat:</span>
            <span className="value">{userData?.weight} kg</span>
          </div>
          <div className="profile-item">
            <span className="label">Tujuan:</span>
            <span className="value">{userData?.fitness_goal}</span>
          </div>
          <div className="profile-item">
            <span className="label">Tingkat Aktivitas:</span>
            <span className="value">{userData?.activity_level}</span>
          </div>
        </div>
      </div>

      {/* User Metrics from API */}
      {recommendations.user_metrics && (
        <div className="metrics-summary">
          <h3>📊 Analisis Tubuh Anda</h3>
          <div className="metrics-grid">
            <div className="metric-item">
              <span className="label">BMI:</span>
              <span className="value">{recommendations.user_metrics.bmi?.toFixed(1)}</span>
            </div>
            <div className="metric-item">
              <span className="label">Kategori BMI:</span>
              <span className="value">{recommendations.user_metrics.bmi_category}</span>
            </div>
            <div className="metric-item">
              <span className="label">BMR:</span>
              <span className="value">{Math.round(recommendations.user_metrics.bmr)} kkal</span>
            </div>
            <div className="metric-item">
              <span className="label">TDEE:</span>
              <span className="value">{Math.round(recommendations.user_metrics.tdee)} kkal</span>
            </div>
          </div>
        </div>
      )}

      {/* Confidence Scores */}
      {recommendations.confidence_scores && (
        <div className="confidence-summary">
          <h3>🎯 Tingkat Kepercayaan AI</h3>
          <div className="confidence-grid">
            <div className="confidence-item">
              <span className="label">Kepercayaan Keseluruhan:</span>
              <span className="value">{Math.round(recommendations.confidence_scores.overall_confidence * 100)}%</span>
            </div>
            <div className="confidence-item">
              <span className="label">Kepercayaan Nutrisi:</span>
              <span className="value">{Math.round(recommendations.confidence_scores.nutrition_confidence * 100)}%</span>
            </div>
            <div className="confidence-item">
              <span className="label">Kepercayaan Workout:</span>
              <span className="value">{Math.round(recommendations.confidence_scores.workout_confidence * 100)}%</span>
            </div>
            <div className="confidence-item">
              <span className="label">Level:</span>
              <span className="value">{recommendations.confidence_scores.confidence_level}</span>
            </div>
          </div>
          <p className="confidence-message">{recommendations.confidence_scores.confidence_message}</p>
        </div>
      )}

      {/* Workout Recommendations */}
      {workout && (
        <div className="workout-section">
          <h3>🏋️ Program Latihan Harian</h3>
          <div className="workout-details">
            <div className="workout-card">
              <h4>📅 Jadwal Mingguan</h4>
              <div className="workout-grid">
                <div className="workout-item">
                  <span className="label">Jenis Olahraga:</span>
                  <span className="value">{workout.workout_type || 'Strength Training'}</span>
                </div>
                <div className="workout-item">
                  <span className="label">Hari per Minggu:</span>
                  <span className="value">{workout.days_per_week || 3} hari</span>
                </div>
                <div className="workout-item">
                  <span className="label">Set Harian:</span>
                  <span className="value">{workout.sets_per_exercise || 3} set</span>
                </div>
                <div className="workout-item">
                  <span className="label">Latihan per Sesi:</span>
                  <span className="value">{workout.exercises_per_session || 5} latihan</span>
                </div>
              </div>
            </div>

            {workout.cardio_minutes_per_day && (
              <div className="cardio-card">
                <h4>🏃 Kardio Harian</h4>
                <div className="cardio-details">
                  <div className="cardio-item">
                    <span className="label">Durasi per Hari:</span>
                    <span className="value">{workout.cardio_minutes_per_day} menit</span>
                  </div>
                  <div className="cardio-item">
                    <span className="label">Sesi per Hari:</span>
                    <span className="value">{workout.cardio_sessions_per_day || 1} sesi</span>
                  </div>
                </div>
              </div>
            )}

            {workout.workout_schedule && (
              <div className="schedule-card">
                <h4>📋 Jadwal yang Disarankan</h4>
                <p className="schedule-text">{workout.workout_schedule}</p>
              </div>
            )}

            {/* Workout Type Explanation */}
            <div className="workout-explanation-card">
              <h4>📚 Penjelasan Jenis Latihan</h4>
              <div className="explanation-content">
                <div className="workout-type-section">
                  <h5>🏋️ Full Body Workouts</h5>
                  <ul>
                    <li><strong>Struktur:</strong> Melatih semua kelompok otot utama dalam setiap sesi</li>
                    <li><strong>Cocok untuk:</strong> Pemula, orang dengan waktu terbatas</li>
                    <li><strong>Kelompok Otot:</strong> Kaki, dada, punggung, bahu, lengan, inti</li>
                  </ul>
                </div>

                <div className="workout-type-section">
                  <h5>🔄 Upper/Lower Split</h5>
                  <ul>
                    <li><strong>A (Upper):</strong> Dada, punggung, bahu, lengan</li>
                    <li><strong>B (Lower):</strong> Kaki, bokong, inti</li>
                    <li><strong>Cocok untuk:</strong> Atlet menengah dengan ketersediaan waktu sedang</li>
                  </ul>
                </div>

                <div className="workout-type-section">
                  <h5>💪 Push/Pull/Legs Split</h5>
                  <ul>
                    <li><strong>A (Push):</strong> Dada, bahu, trisep</li>
                    <li><strong>B (Pull):</strong> Punggung, bisep</li>
                    <li><strong>C (Legs):</strong> Paha depan, paha belakang, bokong, betis</li>
                    <li><strong>Cocok untuk:</strong> Atlet lanjutan dengan ketersediaan waktu tinggi</li>
                  </ul>
                </div>

                <div className="schedule-notation">
                  <h5>📅 Notasi Jadwal</h5>
                  <ul>
                    <li><strong>W</strong> = Hari latihan (Workout day)</li>
                    <li><strong>X</strong> = Hari istirahat (Rest day)</li>
                    <li><strong>A/B/C</strong> = Sesi latihan berbeda dalam rutinitas split</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Nutrition Recommendations */}
      {nutrition && dailyMacros && (
        <div className="nutrition-section">
          <h3>🍎 Program Nutrisi Harian</h3>
          
          <div className="nutrition-targets">
            <h4>🎯 Target Harian Berdasarkan Template</h4>
            <div className="macro-grid">
              <div className="macro-card calories">
                <div className="macro-icon">🔥</div>
                <div className="macro-info">
                  <span className="macro-label">Asupan Kalori</span>
                  <span className="macro-value">{dailyMacros.calories} kkal</span>
                </div>
              </div>
              <div className="macro-card protein">
                <div className="macro-icon">🥩</div>
                <div className="macro-info">
                  <span className="macro-label">Asupan Protein</span>
                  <span className="macro-value">{dailyMacros.protein} gram</span>
                </div>
              </div>
              <div className="macro-card carbs">
                <div className="macro-icon">🍞</div>
                <div className="macro-info">
                  <span className="macro-label">Asupan Karbohidrat</span>
                  <span className="macro-value">{dailyMacros.carbs} gram</span>
                </div>
              </div>
              <div className="macro-card fat">
                <div className="macro-icon">🥑</div>
                <div className="macro-info">
                  <span className="macro-label">Asupan Lemak</span>
                  <span className="macro-value">{dailyMacros.fat} gram</span>
                </div>
              </div>
            </div>
          </div>

          {/* Food Suggestions with Calculated Portions */}
          {(!loading && !mealPlanLoading) && foodSuggestions.length > 0 && (
            <div className="food-suggestions">
              <h4>🍽️ {mealPlan ? 'Rencana Makan Berdasarkan Template' : 'Porsi Makanan Indonesia Berdasarkan Template'}</h4>
              <p className="suggestions-subtitle">
                {mealPlan ? 'Kombinasi makanan yang sudah diatur untuk mencapai target nutrisi harian Anda' : 'Porsi yang dihitung untuk mencapai target nutrisi harian Anda'}
              </p>
              
              {foodSuggestions.map((meal, index) => (
                <div key={index} className="meal-section">
                  <h5 className="meal-title">{meal.meal}</h5>
                  {meal.mealName && (
                    <div className="meal-info">
                      <p className="meal-name"><strong>{meal.mealName}</strong></p>
                      {meal.description && <p className="meal-description">{meal.description}</p>}
                    </div>
                  )}
                  <div className="meal-target">
                    <span>Target: {meal.targetCalories} kkal | {meal.targetProtein}g protein</span>
                  </div>
                  
                  <div className="food-list">
                    {meal.foods.map((food, foodIndex) => (
                      <div key={foodIndex} className="food-item-calculated">
                        <div className="food-header">
                          <span className="food-name">{food.name}</span>
                          <span className="food-portion">{food.grams}g</span>
                        </div>
                        <div className="food-macros">
                          <span className="macro">🔥 {food.actualCalories} kkal</span>
                          <span className="macro">🥩 {food.actualProtein}g protein</span>
                          <span className="macro">🍞 {food.actualCarbs}g karbo</span>
                          <span className="macro">🥑 {food.actualFat}g lemak</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Loading state for meal plan */}
          {mealPlanLoading && (
            <div className="meal-plan-loading">
              <h4>🔄 Memuat Rencana Makan...</h4>
              <p>Sedang menyusun kombinasi makanan yang optimal untuk Anda...</p>
            </div>
          )}
        </div>
      )}

      {/* Template Information */}
      <div className="template-info">
        <h4>📊 Informasi Template</h4>
        <div className="template-details">
          <p><strong>Template berdasarkan:</strong></p>
          <ul>
            <li>Tujuan: {userData?.fitness_goal}</li>
            <li>Level aktivitas: {userData?.activity_level}</li>
            <li>BMI kategori: {recommendations.user_metrics?.bmi_category || userData?.bmi_category || 'Normal'}</li>
            <li>Template Nutrisi ID: {nutrition?.template_id}</li>
            <li>Template Workout ID: {workout?.template_id}</li>
            <li>Kalkulasi nutrisi per kg berat badan</li>
          </ul>
        </div>
      </div>


      {/* Action Buttons */}
      <div className="recommendation-actions">
        <button onClick={onBack} className="btn-secondary">
          ← Kembali ke Formulir
        </button>
        <button onClick={onNewRecommendation} className="btn-primary">
          🆕 Buat Rekomendasi Baru
        </button>
      </div>

      {/* Tips Section */}
      <div className="tips-section">
        <h4>💡 Tips Sukses dengan Porsi yang Tepat</h4>
        <div className="tips-grid">
          <div className="tip-card">
            <span className="tip-icon">⚖️</span>
            <p>Gunakan timbangan digital untuk mengukur porsi makanan secara akurat</p>
          </div>
          <div className="tip-card">
            <span className="tip-icon">📱</span>
            <p>Catat asupan makanan harian di fitur progress tracking</p>
          </div>
          <div className="tip-card">
            <span className="tip-icon">🥗</span>
            <p>Variasikan sumber protein dan karbohidrat setiap harinya</p>
          </div>
          <div className="tip-card">
            <span className="tip-icon">💧</span>
            <p>Minum 2-3 liter air putih setiap hari untuk metabolisme optimal</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationDisplay;