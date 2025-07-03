// frontend/src/components/RecommendationDisplay.js
import React from 'react';

const RecommendationDisplay = ({ userData, recommendations }) => {
  if (!userData || !recommendations) {
    return <div>No recommendations available</div>;
  }

  const { exercise_recommendation, nutrition_recommendation, calculated_metrics, template_info } = recommendations;

  return (
    <div className="recommendation-display">
      {/* User Summary */}
      <div className="user-summary">
        <h3>Profile Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="label">Age:</span>
            <span className="value">{userData.age} years</span>
          </div>
          <div className="summary-item">
            <span className="label">Gender:</span>
            <span className="value">{userData.gender}</span>
          </div>
          <div className="summary-item">
            <span className="label">Height:</span>
            <span className="value">{userData.height} cm</span>
          </div>
          <div className="summary-item">
            <span className="label">Current Weight:</span>
            <span className="value">{userData.weight} kg</span>
          </div>
          <div className="summary-item">
            <span className="label">Target Weight:</span>
            <span className="value">{userData.target_weight} kg</span>
          </div>
          <div className="summary-item">
            <span className="label">Goal:</span>
            <span className="value">{userData.fitness_goal}</span>
          </div>
          <div className="summary-item">
            <span className="label">Activity Level:</span>
            <span className="value">{userData.activity_level}</span>
          </div>
        </div>
      </div>

      {/* Calculated Metrics */}
      <div className="metrics-section">
        <h3>Your Calculated Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{calculated_metrics.bmr}</div>
            <div className="metric-label">BMR (kcal/day)</div>
            <div className="metric-description">Basal Metabolic Rate</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{calculated_metrics.tdee}</div>
            <div className="metric-label">TDEE (kcal/day)</div>
            <div className="metric-description">Total Daily Energy Expenditure</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{calculated_metrics.bmi}</div>
            <div className="metric-label">BMI</div>
            <div className="metric-description">{calculated_metrics.bmi_category}</div>
          </div>
        </div>
      </div>

      {/* Exercise Recommendations */}
      <div className="exercise-section">
        <div className="section-header">
          <h3>🏋️ Exercise Recommendations</h3>
          <span className="template-id">Template ID: {exercise_recommendation.template_id}</span>
        </div>
        
        <div className="recommendation-grid">
          <div className="recommendation-card exercise">
            <div className="card-icon">💪</div>
            <div className="card-content">
              <h4>Training Volume</h4>
              <div className="main-value">{exercise_recommendation.training_volume}</div>
              <div className="unit">sets per muscle group per week</div>
              <div className="description">
                Optimal volume for {userData.fitness_goal.toLowerCase()} based on your {userData.activity_level.toLowerCase()} lifestyle
              </div>
            </div>
          </div>

          <div className="recommendation-card exercise">
            <div className="card-icon">📅</div>
            <div className="card-content">
              <h4>Training Frequency</h4>
              <div className="main-value">{exercise_recommendation.training_frequency}</div>
              <div className="unit">sessions per week</div>
              <div className="description">
                Balanced approach providing optimal stimulus and recovery time
              </div>
            </div>
          </div>

          <div className="recommendation-card exercise">
            <div className="card-icon">❤️</div>
            <div className="card-content">
              <h4>Cardio Volume</h4>
              <div className="main-value">{exercise_recommendation.cardio_volume}</div>
              <div className="unit">minutes per week</div>
              <div className="description">
                Cardiovascular training to support your {userData.fitness_goal.toLowerCase()} goals
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Nutrition Recommendations */}
      <div className="nutrition-section">
        <div className="section-header">
          <h3>🥗 Nutrition Recommendations</h3>
          <span className="template-id">Template ID: {nutrition_recommendation.template_id}</span>
        </div>

        <div className="recommendation-grid">
          <div className="recommendation-card nutrition calories">
            <div className="card-icon">🔥</div>
            <div className="card-content">
              <h4>Daily Calories</h4>
              <div className="main-value">{nutrition_recommendation.daily_calories}</div>
              <div className="unit">kcal per day</div>
              <div className="description">
                {userData.fitness_goal === 'Muscle Gain' && 'Moderate surplus for muscle growth'}
                {userData.fitness_goal === 'Fat Loss' && 'Controlled deficit for fat loss'}
                {userData.fitness_goal === 'Maintenance' && 'Maintenance calories for stability'}
              </div>
            </div>
          </div>

          <div className="recommendation-card nutrition protein">
            <div className="card-icon">🥩</div>
            <div className="card-content">
              <h4>Protein</h4>
              <div className="main-value">{nutrition_recommendation.daily_protein}g</div>
              <div className="unit">per day</div>
              <div className="description">
                {(nutrition_recommendation.daily_protein / userData.weight).toFixed(1)}g per kg bodyweight
              </div>
            </div>
          </div>

          <div className="recommendation-card nutrition carbs">
            <div className="card-icon">🍞</div>
            <div className="card-content">
              <h4>Carbohydrates</h4>
              <div className="main-value">{nutrition_recommendation.daily_carbs}g</div>
              <div className="unit">per day</div>
              <div className="description">
                {(nutrition_recommendation.daily_carbs / userData.weight).toFixed(1)}g per kg bodyweight
              </div>
            </div>
          </div>

          <div className="recommendation-card nutrition fats">
            <div className="card-icon">🥑</div>
            <div className="card-content">
              <h4>Fats</h4>
              <div className="main-value">{nutrition_recommendation.daily_fat}g</div>
              <div className="unit">per day</div>
              <div className="description">
                Essential fats for hormone production and health
              </div>
            </div>
          </div>
        </div>

        {/* Macronutrient Breakdown */}
        <div className="macro-breakdown">
          <h4>Macronutrient Distribution</h4>
          <div className="macro-bars">
            {(() => {
              const proteinCals = nutrition_recommendation.daily_protein * 4;
              const carbCals = nutrition_recommendation.daily_carbs * 4;
              const fatCals = nutrition_recommendation.daily_fat * 9;
              const totalCals = proteinCals + carbCals + fatCals;
              
              const proteinPct = Math.round((proteinCals / totalCals) * 100);
              const carbPct = Math.round((carbCals / totalCals) * 100);
              const fatPct = Math.round((fatCals / totalCals) * 100);

              return (
                <>
                  <div className="macro-bar">
                    <div className="macro-label">Protein ({proteinPct}%)</div>
                    <div className="bar">
                      <div className="fill protein" style={{width: `${proteinPct}%`}}></div>
                    </div>
                    <div className="macro-value">{nutrition_recommendation.daily_protein}g = {proteinCals} kcal</div>
                  </div>
                  
                  <div className="macro-bar">
                    <div className="macro-label">Carbs ({carbPct}%)</div>
                    <div className="bar">
                      <div className="fill carbs" style={{width: `${carbPct}%`}}></div>
                    </div>
                    <div className="macro-value">{nutrition_recommendation.daily_carbs}g = {carbCals} kcal</div>
                  </div>
                  
                  <div className="macro-bar">
                    <div className="macro-label">Fats ({fatPct}%)</div>
                    <div className="bar">
                      <div className="fill fats" style={{width: `${fatPct}%`}}></div>
                    </div>
                    <div className="macro-value">{nutrition_recommendation.daily_fat}g = {fatCals} kcal</div>
                  </div>
                </>
              );
            })()}
          </div>
        </div>
      </div>

      {/* System Information */}
      {template_info && (
        <div className="system-info">
          <h3>🤖 AI System Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Total Templates:</span>
              <span className="value">{template_info.total_templates}</span>
            </div>
            <div className="info-item">
              <span className="label">Workout Templates:</span>
              <span className="value">{template_info.workout_templates}</span>
            </div>
            <div className="info-item">
              <span className="label">Nutrition Templates:</span>
              <span className="value">{template_info.nutrition_templates}</span>
            </div>
            <div className="info-item">
              <span className="label">Thesis Aligned:</span>
              <span className="value">{template_info.thesis_aligned ? '✅ Yes' : '❌ No'}</span>
            </div>
          </div>
          
          <div className="system-details">
            <p><strong>Methodology:</strong> XGBoost classification using 75 evidence-based templates</p>
            <p><strong>Template Structure:</strong> 15 workout (3 goals × 5 activities) + 60 nutrition (3 goals × 4 BMI × 5 activities)</p>
            <p><strong>Calculations:</strong> Harris-Benedict BMR equations with exact TDEE multipliers from research</p>
          </div>
        </div>
      )}

      {/* Guidelines */}
      <div className="guidelines-section">
        <h3>📋 Guidelines & Tips</h3>
        
        <div className="guidelines-grid">
          <div className="guideline-card">
            <h4>🏋️ Exercise Guidelines</h4>
            <ul>
              <li>Focus on compound movements (squats, deadlifts, bench press)</li>
              <li>Progressive overload: gradually increase weight or reps</li>
              <li>Rest 48-72 hours between training same muscle groups</li>
              <li>Include warm-up and cool-down in every session</li>
            </ul>
          </div>
          
          <div className="guideline-card">
            <h4>🥗 Nutrition Guidelines</h4>
            <ul>
              <li>Eat protein with every meal to support muscle synthesis</li>
              <li>Time carbohydrates around workouts for energy</li>
              <li>Include healthy fats for hormone production</li>
              <li>Stay hydrated: aim for 2-3 liters of water daily</li>
            </ul>
          </div>
          
          <div className="guideline-card">
            <h4>📈 Progress Tracking</h4>
            <ul>
              <li>Weigh yourself at the same time each day</li>
              <li>Take progress photos weekly</li>
              <li>Track your workouts and strength gains</li>
              <li>Monitor energy levels and sleep quality</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationDisplay;