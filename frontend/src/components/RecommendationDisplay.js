import React, { useState, useEffect } from 'react';

const RecommendationDisplay = ({ recommendations, userData, onBack, onNewRecommendation }) => {
  const [nutritionData, setNutritionData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Nutrition data from CSV (per 100g)
  useEffect(() => {
    const loadNutritionData = async () => {
      try {
        const csvData = [
          { name: "Mie Telur (Ditambah, Masak)", calories: 138, carbs: 25.16, protein: 4.54, fat: 2.07 },
          { name: "Roti Gandum", calories: 67, carbs: 12.26, protein: 2.37, fat: 1.07 },
          { name: "Ayam Goreng tanpa Pelapis (Kulit Dimakan)", calories: 260, carbs: 0.0, protein: 28.62, fat: 15.35 },
          { name: "Ikan Panggang", calories: 126, carbs: 0.33, protein: 21.94, fat: 3.44 },
          { name: "Nasi Putih", calories: 130, carbs: 28, protein: 2.7, fat: 0.3 },
          { name: "Tempe Goreng", calories: 193, carbs: 9.4, protein: 18.3, fat: 10.8 },
          { name: "Tahu Goreng", calories: 271, carbs: 10.1, protein: 17.2, fat: 20.2 },
          { name: "Sayur Bayam", calories: 23, carbs: 3.6, protein: 2.9, fat: 0.4 }
        ];
        setNutritionData(csvData);
        setLoading(false);
      } catch (error) {
        console.error('Error loading nutrition data:', error);
        setLoading(false);
      }
    };

    loadNutritionData();
  }, []);

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

  const { workout, nutrition } = recommendations;

  // Calculate user's daily macro targets based on template output
  const calculateDailyMacros = () => {
    if (!nutrition || !userData) return null;

    const weight = parseFloat(userData.weight) || 70;
    
    // Based on template structure: caloric_intake, protein_per_kg, carbs_per_kg, fat_per_kg
    return {
      calories: Math.round(nutrition.caloric_intake || 2000),
      protein: Math.round((nutrition.protein_per_kg || 1.6) * weight),
      carbs: Math.round((nutrition.carbs_per_kg || 4) * weight),
      fat: Math.round((nutrition.fat_per_kg || 1) * weight)
    };
  };

  // Calculate food portions based on template requirements
  const calculateFoodPortions = (targetMacros) => {
    if (!nutritionData.length || !targetMacros) return [];

    const suggestions = [];
    
    // Distribute daily calories across meals
    const mealDistribution = {
      sarapan: { percentage: 0.25, name: '🌅 Sarapan' },
      makan_siang: { percentage: 0.35, name: '☀️ Makan Siang' },
      makan_malam: { percentage: 0.30, name: '🌙 Makan Malam' },
      cemilan: { percentage: 0.10, name: '🍎 Cemilan' }
    };

    Object.entries(mealDistribution).forEach(([mealType, config]) => {
      const targetCalories = targetMacros.calories * config.percentage;
      const targetProtein = targetMacros.protein * config.percentage;
      
      // Select appropriate foods for this meal
      let selectedFoods = [];
      
      if (mealType === 'sarapan') {
        selectedFoods = nutritionData.filter(food => 
          food.name.includes('Roti') || food.name.includes('Telur') || food.name.includes('Mie')
        );
      } else if (mealType === 'makan_siang') {
        selectedFoods = nutritionData.filter(food => 
          food.name.includes('Nasi') || food.name.includes('Ayam') || food.name.includes('Ikan')
        );
      } else if (mealType === 'makan_malam') {
        selectedFoods = nutritionData.filter(food => 
          food.name.includes('Ikan') || food.name.includes('Tempe') || food.name.includes('Sayur')
        );
      } else {
        selectedFoods = nutritionData.filter(food => 
          food.name.includes('Tempe') || food.name.includes('Tahu')
        );
      }

      // If no specific foods found, use all available
      if (selectedFoods.length === 0) {
        selectedFoods = nutritionData.slice(0, 3);
      }

      // Calculate portions to meet targets
      const mealFoods = selectedFoods.slice(0, 3).map(food => {
        // Calculate how many grams needed to meet portion of target calories
        const gramsNeeded = Math.min(200, Math.max(50, (targetCalories / 3) / (food.calories / 100)));
        
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
  const foodSuggestions = dailyMacros ? calculateFoodPortions(dailyMacros) : [];

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
          {!loading && foodSuggestions.length > 0 && (
            <div className="food-suggestions">
              <h4>🍽️ Porsi Makanan Indonesia Berdasarkan Template</h4>
              <p className="suggestions-subtitle">Porsi yang dihitung untuk mencapai target nutrisi harian Anda</p>
              
              {foodSuggestions.map((meal, index) => (
                <div key={index} className="meal-section">
                  <h5 className="meal-title">{meal.meal}</h5>
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
            <li>BMI kategori: {userData?.bmi_category || 'Normal'}</li>
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