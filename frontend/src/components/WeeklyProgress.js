// frontend/src/components/WeeklyProgress.js
import React, { useState, useEffect } from 'react';
import { authService } from '../services/authService';

const WeeklyProgress = ({ currentRecommendation, onProgressUpdate }) => {
  const [progressData, setProgressData] = useState({
    completedDays: 0,
    adherenceData: {},
    weeklyProgress: {}
  });
  const [loading, setLoading] = useState(false);

  const daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

  useEffect(() => {
    if (currentRecommendation) {
      setProgressData({
        completedDays: currentRecommendation.completedDays || 0,
        adherenceData: currentRecommendation.adherenceData || {},
        weeklyProgress: currentRecommendation.weeklyProgress || {}
      });
    }
  }, [currentRecommendation]);

  const handleDayCompletion = (day, isCompleted) => {
    const newAdherenceData = {
      ...progressData.adherenceData,
      [day]: isCompleted
    };

    const completedDays = Object.values(newAdherenceData).filter(Boolean).length;

    setProgressData(prev => ({
      ...prev,
      adherenceData: newAdherenceData,
      completedDays
    }));
  };

  const handleProgressSave = async () => {
    if (!currentRecommendation) return;

    setLoading(true);
    try {
      const result = await authService.updateWeeklyProgress(
        currentRecommendation.id,
        progressData
      );

      if (result.success) {
        onProgressUpdate && onProgressUpdate(progressData);
      }
    } catch (error) {
      console.error('Error saving progress:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!currentRecommendation) {
    return (
      <div className="weekly-progress-empty">
        <h3>📅 No Active Weekly Plan</h3>
        <p>Create a new recommendation to start tracking your progress!</p>
      </div>
    );
  }

  const adherencePercentage = Math.round((progressData.completedDays / 7) * 100);

  return (
    <div className="weekly-progress">
      <div className="progress-header">
        <h3>📊 Week {currentRecommendation.weekNumber} Progress</h3>
        <div className="progress-stats">
          <div className="stat">
            <span className="stat-value">{progressData.completedDays}/7</span>
            <span className="stat-label">Days Completed</span>
          </div>
          <div className="stat">
            <span className="stat-value">{adherencePercentage}%</span>
            <span className="stat-label">Adherence</span>
          </div>
        </div>
      </div>

      <div className="weekly-checklist">
        <h4>Daily Completion Tracker</h4>
        <div className="days-grid">
          {daysOfWeek.map(day => (
            <div key={day} className="day-item">
              <label className="day-checkbox">
                <input
                  type="checkbox"
                  checked={progressData.adherenceData[day] || false}
                  onChange={(e) => handleDayCompletion(day, e.target.checked)}
                />
                <span className="day-name">{day}</span>
                <span className="checkmark">
                  {progressData.adherenceData[day] ? '✅' : '⬜'}
                </span>
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="progress-summary">
        <div className="adherence-bar">
          <div className="bar-background">
            <div 
              className="bar-fill"
              style={{ width: `${adherencePercentage}%` }}
            ></div>
          </div>
          <span className="bar-text">{adherencePercentage}% Complete</span>
        </div>

        <div className="adherence-feedback">
          {adherencePercentage >= 80 && (
            <div className="feedback excellent">
              🎉 Excellent adherence! You're on track for great results.
            </div>
          )}
          {adherencePercentage >= 60 && adherencePercentage < 80 && (
            <div className="feedback good">
              👍 Good progress! Try to maintain consistency for optimal results.
            </div>
          )}
          {adherencePercentage < 60 && (
            <div className="feedback needs-improvement">
              💪 Keep pushing! Consistency is key to reaching your goals.
            </div>
          )}
        </div>
      </div>

      <div className="progress-actions">
        <button
          onClick={handleProgressSave}
          disabled={loading}
          className="btn-primary"
        >
          {loading ? (
            <>
              <span className="spinner small"></span>
              Saving...
            </>
          ) : (
            'Save Progress'
          )}
        </button>

        {progressData.completedDays >= 7 && (
          <div className="week-complete">
            <h4>🎯 Week Complete!</h4>
            <p>Ready to generate your next week's recommendations with AI adaptation.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default WeeklyProgress;