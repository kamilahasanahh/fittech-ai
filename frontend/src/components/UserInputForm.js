// frontend/src/components/UserInputForm.js
import React, { useState, useEffect } from 'react';
import { FITNESS_GOALS, ACTIVITY_LEVELS } from '../services/api';

const UserInputForm = ({ onSubmit, loading, initialData }) => {
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

  useEffect(() => {
    if (initialData) {
      setFormData(initialData);
    }
  }, [initialData]);

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

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateStep(4)) {
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

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="form-step">
            <h3>Personal Information</h3>
            
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
            </div>
          </div>
        );

      case 2:
        return (
          <div className="form-step">
            <h3>Body Measurements</h3>
            
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
            </div>
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
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="user-input-form">
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
            >
              Next →
            </button>
          ) : (
            <button
              type="submit"
              disabled={loading}
              className="btn-primary submit-btn"
            >
              {loading ? (
                <>
                  <span className="spinner small"></span>
                  Analyzing...
                </>
              ) : (
                'Get My Recommendations 🚀'
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default UserInputForm;