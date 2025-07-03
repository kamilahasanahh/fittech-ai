// frontend/src/components/EnhancedUserInputForm.js
import React, { useState, useEffect } from 'react';
import { FITNESS_GOALS, ACTIVITY_LEVELS } from '../services/api';
import { validateWeightGoal, WEIGHT_CHANGE_LIMITS } from '../utils/validationRules';
import { authService } from '../services/authService';

const EnhancedUserInputForm = ({ onSubmit, loading, initialData }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    target_weight: '',
    fitness_goal: '',
    activity_level: ''
  });

  const [currentStep, setCurrentStep] = useState(1);
  const [validationErrors, setValidationErrors] = useState({});
  const [weightValidation, setWeightValidation] = useState(null);
  const [userWeekInfo, setUserWeekInfo] = useState(null);
  const [canCreateNewRec, setCanCreateNewRec] = useState(true);

  useEffect(() => {
    if (initialData) {
      setFormData(initialData);
    }
    checkUserEligibility();
  }, [initialData]);

  useEffect(() => {
    // Validate weight goal whenever relevant fields change
    if (formData.weight && formData.target_weight && formData.fitness_goal && formData.age) {
      const validation = validateWeightGoal(
        parseFloat(formData.weight),
        parseFloat(formData.target_weight),
        formData.fitness_goal,
        parseInt(formData.age),
        formData.gender
      );
      setWeightValidation(validation);
    }
  }, [formData.weight, formData.target_weight, formData.fitness_goal, formData.age, formData.gender]);

  const checkUserEligibility = async () => {
    try {
      const canCreate = await authService.canCreateNewRecommendation();
      const currentWeek = await authService.getCurrentWeekNumber();
      
      setCanCreateNewRec(canCreate);
      setUserWeekInfo({
        currentWeek,
        canCreate,
        message: canCreate 
          ? `Ready for Week ${currentWeek} recommendations`
          : 'You have an active weekly plan. Complete it before creating a new one.'
      });
    } catch (error) {
      console.error('Error checking user eligibility:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateStep = (step) => {
    const errors = {};

    switch (step) {
      case 1:
        if (!formData.age || formData.age < 18 || formData.age > 65) {
          errors.age = 'Age must be between 18 and 65';
        }
        if (!formData.gender) {
          errors.gender = 'Please select your gender';
        }
        break;

      case 2:
        if (!formData.height || formData.height < 150 || formData.height > 200) {
          errors.height = 'Height must be between 150 and 200 cm';
        }
        if (!formData.weight || formData.weight < 45 || formData.weight > 150) {
          errors.weight = 'Weight must be between 45 and 150 kg';
        }
        if (!formData.target_weight || formData.target_weight < 45 || formData.target_weight > 150) {
          errors.target_weight = 'Target weight must be between 45 and 150 kg';
        }
        
        // Add weight goal validation
        if (formData.weight && formData.target_weight && formData.fitness_goal && weightValidation && !weightValidation.isValid) {
          errors.target_weight = weightValidation.errors[0];
        }
        break;

      case 3:
        if (!formData.fitness_goal) {
          errors.fitness_goal = 'Please select your fitness goal';
        }
        break;

      case 4:
        if (!formData.activity_level) {
          errors.activity_level = 'Please select your activity level';
        }
        
        // Final validation check
        if (!canCreateNewRec) {
          errors.eligibility = 'You have an active weekly plan. Please complete it before creating a new one.';
        }
        break;

      default:
        break;
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    setCurrentStep(currentStep - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!canCreateNewRec) {
      setValidationErrors({ eligibility: 'You have an active weekly plan. Please complete it before creating a new one.' });
      return;
    }
    
    if (validateStep(4) && weightValidation?.isValid) {
      // Convert string values to appropriate types
      const processedData = {
        ...formData,
        age: parseInt(formData.age),
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        target_weight: parseFloat(formData.target_weight)
      };
      
      onSubmit(processedData);
    }
  };

  const renderWeightLimitInfo = () => {
    if (!formData.fitness_goal) return null;

    const limits = WEIGHT_CHANGE_LIMITS[formData.fitness_goal];
    
    return (
      <div className="weight-limit-info">
        <h4>📊 Healthy {formData.fitness_goal} Guidelines:</h4>
        <ul>
          {formData.fitness_goal === 'Muscle Gain' && (
            <>
              <li>Recommended gain: 0.25-{limits.maxWeightGain}kg per week</li>
              <li>Maximum safe gain: {limits.maxTotalGain}kg total</li>
              <li>Minimum duration: {limits.minWeeksRequired} weeks</li>
            </>
          )}
          {formData.fitness_goal === 'Fat Loss' && (
            <>
              <li>Recommended loss: 0.5-{limits.maxWeightLoss}kg per week</li>
              <li>Maximum safe loss: {limits.maxTotalLoss}kg total</li>
              <li>Minimum duration: {limits.minWeeksRequired} weeks</li>
            </>
          )}
          {formData.fitness_goal === 'Maintenance' && (
            <>
              <li>Target variation: ±{limits.maxVariation}kg</li>
              <li>Focus on body composition over weight</li>
            </>
          )}
        </ul>
      </div>
    );
  };

  const renderWeightValidation = () => {
    if (!weightValidation) return null;

    return (
      <div className={`weight-validation ${weightValidation.isValid ? 'valid' : 'invalid'}`}>
        {!weightValidation.isValid && (
          <div className="validation-errors">
            <h4>⚠️ Weight Goal Issues:</h4>
            {weightValidation.errors.map((error, index) => (
              <p key={index} className="error-text">{error}</p>
            ))}
          </div>
        )}
        
        {weightValidation.isValid && weightValidation.estimatedWeeks > 0 && (
          <div className="validation-success">
            <h4>✅ Goal Timeline:</h4>
            <p>Estimated duration: {weightValidation.estimatedWeeks} weeks</p>
          </div>
        )}
        
        {weightValidation.recommendations.length > 0 && (
          <div className="recommendations">
            <h4>💡 Recommendations:</h4>
            {weightValidation.recommendations.map((rec, index) => (
              <p key={index}>{rec}</p>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderUserWeekInfo = () => {
    if (!userWeekInfo) return null;

    return (
      <div className={`user-week-info ${userWeekInfo.canCreate ? 'ready' : 'blocked'}`}>
        <div className="week-status">
          <span className="week-number">Week {userWeekInfo.currentWeek}</span>
          <span className="status-message">{userWeekInfo.message}</span>
        </div>
        {!userWeekInfo.canCreate && (
          <div className="blocked-info">
            <p>📅 Complete your current weekly plan before generating new recommendations.</p>
            <p>This ensures progressive adaptation and prevents overtraining.</p>
          </div>
        )}
      </div>
    );
  };

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="form-step">
            <h3>Personal Information</h3>
            {renderUserWeekInfo()}
            
            <div className="form-group">
              <label htmlFor="age">Age *</label>
              <input
                type="number"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleInputChange}
                min="18"
                max="65"
                placeholder="Enter your age (18-65)"
                className={validationErrors.age ? 'error' : ''}
              />
              {validationErrors.age && <span className="error-text">{validationErrors.age}</span>}
              <div className="field-help">
                Age affects metabolism and safe training intensity
              </div>
            </div>

            <div className="form-group">
              <label>Gender *</label>
              <div className="radio-group">
                <label className="radio-option">
                  <input
                    type="radio"
                    name="gender"
                    value="Male"
                    checked={formData.gender === 'Male'}
                    onChange={handleInputChange}
                  />
                  <span>Male</span>
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="gender"
                    value="Female"
                    checked={formData.gender === 'Female'}
                    onChange={handleInputChange}
                  />
                  <span>Female</span>
                </label>
              </div>
              {validationErrors.gender && <span className="error-text">{validationErrors.gender}</span>}
              <div className="field-help">
                Gender affects BMR calculations and training recommendations
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="form-step">
            <h3>Body Measurements & Goals</h3>
            
            <div className="form-group">
              <label htmlFor="height">Height (cm) *</label>
              <input
                type="number"
                id="height"
                name="height"
                value={formData.height}
                onChange={handleInputChange}
                min="150"
                max="200"
                step="0.1"
                placeholder="Enter your height in cm"
                className={validationErrors.height ? 'error' : ''}
              />
              {validationErrors.height && <span className="error-text">{validationErrors.height}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="weight">Current Weight (kg) *</label>
              <input
                type="number"
                id="weight"
                name="weight"
                value={formData.weight}
                onChange={handleInputChange}
                min="45"
                max="150"
                step="0.1"
                placeholder="Enter your current weight in kg"
                className={validationErrors.weight ? 'error' : ''}
              />
              {validationErrors.weight && <span className="error-text">{validationErrors.weight}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="target_weight">Target Weight (kg) *</label>
              <input
                type="number"
                id="target_weight"
                name="target_weight"
                value={formData.target_weight}
                onChange={handleInputChange}
                min="45"
                max="150"
                step="0.1"
                placeholder="Enter your target weight in kg"
                className={validationErrors.target_weight ? 'error' : ''}
              />
              {validationErrors.target_weight && <span className="error-text">{validationErrors.target_weight}</span>}
              <div className="field-help">
                Target weight will be validated based on your fitness goal
              </div>
            </div>

            {renderWeightLimitInfo()}
            {renderWeightValidation()}
          </div>
        );

      case 3:
        return (
          <div className="form-step">
            <h3>Fitness Goal</h3>
            
            <div className="form-group">
              <label>What is your primary fitness goal? *</label>
              <div className="option-cards">
                {FITNESS_GOALS.map(goal => (
                  <label key={goal.value} className={`option-card ${formData.fitness_goal === goal.value ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="fitness_goal"
                      value={goal.value}
                      checked={formData.fitness_goal === goal.value}
                      onChange={handleInputChange}
                    />
                    <div className="option-content">
                      <h4>{goal.label}</h4>
                      <p>{goal.description}</p>
                      <div className="goal-specs">
                        {goal.value === 'Muscle Gain' && (
                          <div className="spec-list">
                            <span>• Recommended: 0.25-0.5kg/week</span>
                            <span>• Duration: 12+ weeks</span>
                            <span>• Max safe gain: 12kg</span>
                          </div>
                        )}
                        {goal.value === 'Fat Loss' && (
                          <div className="spec-list">
                            <span>• Recommended: 0.5-1kg/week</span>
                            <span>• Duration: 8+ weeks</span>
                            <span>• Max safe loss: 20kg</span>
                          </div>
                        )}
                        {goal.value === 'Maintenance' && (
                          <div className="spec-list">
                            <span>• Weight variation: ±2kg</span>
                            <span>• Focus: Body composition</span>
                            <span>• Sustainable habits</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </label>
                ))}
              </div>
              {validationErrors.fitness_goal && <span className="error-text">{validationErrors.fitness_goal}</span>}
            </div>
          </div>
        );

      case 4:
        return (
          <div className="form-step">
            <h3>Activity Level</h3>
            
            <div className="form-group">
              <label>What is your current activity level? *</label>
              <div className="option-cards">
                {ACTIVITY_LEVELS.map(activity => (
                  <label key={activity.value} className={`option-card ${formData.activity_level === activity.value ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="activity_level"
                      value={activity.value}
                      checked={formData.activity_level === activity.value}
                      onChange={handleInputChange}
                    />
                    <div className="option-content">
                      <h4>{activity.label}</h4>
                      <p>{activity.description}</p>
                      <span className="multiplier">TDEE Multiplier: {activity.multiplier}</span>
                    </div>
                  </label>
                ))}
              </div>
              {validationErrors.activity_level && <span className="error-text">{validationErrors.activity_level}</span>}
            </div>

            {/* Final Summary */}
            <div className="form-summary">
              <h4>📋 Summary & Weekly Plan</h4>
              <div className="summary-grid">
                <div className="summary-item">
                  <span>Week:</span>
                  <span>Week {userWeekInfo?.currentWeek || 1}</span>
                </div>
                <div className="summary-item">
                  <span>Goal:</span>
                  <span>{formData.fitness_goal}</span>
                </div>
                <div className="summary-item">
                  <span>Target:</span>
                  <span>{formData.weight}kg → {formData.target_weight}kg</span>
                </div>
                {weightValidation?.estimatedWeeks && (
                  <div className="summary-item">
                    <span>Estimated Timeline:</span>
                    <span>{weightValidation.estimatedWeeks} weeks</span>
                  </div>
                )}
              </div>
              
              <div className="weekly-plan-info">
                <h5>📅 About Weekly Plans:</h5>
                <ul>
                  <li>Each recommendation is designed for one week of training</li>
                  <li>Track your progress daily and update at week's end</li>
                  <li>AI adjusts recommendations based on your adherence and progress</li>
                  <li>Sustainable approach prevents overtraining and ensures progress</li>
                </ul>
              </div>

              {!canCreateNewRec && (
                <div className="eligibility-warning">
                  <h5>⚠️ Active Plan Detected</h5>
                  <p>You have an active weekly plan. Complete it before generating new recommendations for optimal progression.</p>
                </div>
              )}
            </div>

            {validationErrors.eligibility && (
              <div className="error-banner">
                {validationErrors.eligibility}
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="enhanced-user-input-form">
      {/* Progress Bar */}
      <div className="progress-bar">
        <div className="progress-steps">
          {[1, 2, 3, 4].map(step => (
            <div
              key={step}
              className={`progress-step ${step <= currentStep ? 'active' : ''} ${step < currentStep ? 'completed' : ''}`}
            >
              <div className="step-number">{step}</div>
              <div className="step-label">
                {step === 1 && 'Personal'}
                {step === 2 && 'Measurements'}
                {step === 3 && 'Goal'}
                {step === 4 && 'Activity'}
              </div>
            </div>
          ))}
        </div>
        <div className="progress-line">
          <div 
            className="progress-fill"
            style={{ width: `${((currentStep - 1) / 3) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Form Content */}
      <form onSubmit={handleSubmit} className="form-content">
        {renderStep()}

        {/* Navigation Buttons */}
        <div className="form-navigation">
          <button
            type="button"
            onClick={handleBack}
            disabled={currentStep === 1}
            className="btn-secondary"
          >
            ← Back
          </button>

          {currentStep < 4 ? (
            <button
              type="button"
              onClick={handleNext}
              className="btn-primary"
              disabled={!canCreateNewRec && currentStep === 4}
            >
              Next →
            </button>
          ) : (
            <button
              type="submit"
              disabled={loading || !canCreateNewRec || (weightValidation && !weightValidation.isValid)}
              className="btn-primary submit-btn"
            >
              {loading ? (
                <>
                  <span className="spinner small"></span>
                  Generating Week {userWeekInfo?.currentWeek || 1} Plan...
                </>
              ) : (
                `Generate Week ${userWeekInfo?.currentWeek || 1} Recommendations 🚀`
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default EnhancedUserInputForm;