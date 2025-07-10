import React, { useState } from 'react';
import { auth } from '../services/firebaseConfig';

const EnhancedUserInputForm = ({ onSubmit, onRecommendationsReceived }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    fitness_goal: '',
    activity_level: ''
  });

  const [currentStep, setCurrentStep] = useState(1);
  const [validationErrors, setValidationErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [submissionError, setSubmissionError] = useState('');

  // Indonesian fitness goals and activity levels
  const FITNESS_GOALS = {
    'weight_loss': {
      label: 'Menurunkan Berat Badan',
      description: 'Fokus pada pembakaran kalori dan pengurangan lemak tubuh',
      icon: '📉'
    },
    'muscle_gain': {
      label: 'Menambah Massa Otot',
      description: 'Membangun otot dengan latihan beban dan nutrisi yang tepat',
      icon: '💪'
    },
    'maintenance': {
      label: 'Mempertahankan Bentuk Tubuh',
      description: 'Menjaga kondisi fisik dan berat badan yang sudah ideal',
      icon: '⚖️'
    },
    'endurance': {
      label: 'Meningkatkan Stamina',
      description: 'Fokus pada cardio dan daya tahan tubuh',
      icon: '🏃'
    },
    'strength': {
      label: 'Meningkatkan Kekuatan',
      description: 'Latihan intensif untuk kekuatan dan power',
      icon: '🏋️'
    }
  };

  const ACTIVITY_LEVELS = {
    'sedentary': {
      label: 'Tidak Aktif',
      description: 'Duduk sepanjang hari, tidak berolahraga',
      multiplier: '1.2x',
      icon: '🪑'
    },
    'lightly_active': {
      label: 'Sedikit Aktif',
      description: 'Olahraga ringan 1-3 hari per minggu',
      multiplier: '1.375x',
      icon: '🚶'
    },
    'moderately_active': {
      label: 'Cukup Aktif',
      description: 'Olahraga sedang 3-5 hari per minggu',
      multiplier: '1.55x',
      icon: '🏃'
    },
    'very_active': {
      label: 'Sangat Aktif',
      description: 'Olahraga berat 6-7 hari per minggu',
      multiplier: '1.725x',
      icon: '🏋️'
    },
    'extra_active': {
      label: 'Ekstra Aktif',
      description: 'Olahraga sangat berat, kerja fisik',
      multiplier: '1.9x',
      icon: '💪'
    }
  };

  const steps = [
    { number: 1, label: 'Informasi Dasar' },
    { number: 2, label: 'Tujuan Fitness' },
    { number: 3, label: 'Level Aktivitas' },
    { number: 4, label: 'Konfirmasi' }
  ];

  const validateStep = (step) => {
    const errors = {};
    
    switch (step) {
      case 1:
        if (!formData.age || formData.age < 16 || formData.age > 80) {
          errors.age = 'Usia harus antara 16-80 tahun';
        }
        if (!formData.gender) {
          errors.gender = 'Jenis kelamin harus dipilih';
        }
        if (!formData.height || formData.height < 120 || formData.height > 250) {
          errors.height = 'Tinggi badan harus antara 120-250 cm';
        }
        if (!formData.weight || formData.weight < 30 || formData.weight > 300) {
          errors.weight = 'Berat badan harus antara 30-300 kg';
        }
        break;
      case 2:
        if (!formData.fitness_goal) {
          errors.fitness_goal = 'Tujuan fitness harus dipilih';
        }
        break;
      case 3:
        if (!formData.activity_level) {
          errors.activity_level = 'Level aktivitas harus dipilih';
        }
        break;
      default:
        // No validation needed for other steps
        break;
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
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

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateStep(currentStep)) return;
    
    setLoading(true);
    setSubmissionError('');

    try {
      // Call the backend API
      const response = await fetch('http://localhost:5000/api/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${await auth.currentUser?.getIdToken()}`
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      // Call the parent callbacks
      onSubmit(formData);
      onRecommendationsReceived(data);
      
    } catch (error) {
      console.error('Error submitting form:', error);
      setSubmissionError(
        error.message.includes('fetch') 
          ? 'Tidak dapat terhubung ke server. Periksa koneksi internet Anda.'
          : 'Terjadi kesalahan saat memproses data. Silakan coba lagi.'
      );
    } finally {
      setLoading(false);
    }
  };

  const calculateBMI = () => {
    if (formData.height && formData.weight) {
      const heightInM = formData.height / 100;
      return (formData.weight / (heightInM * heightInM)).toFixed(1);
    }
    return null;
  };

  const getBMICategory = (bmi) => {
    if (bmi < 18.5) return { text: 'Kurus', color: '#3b82f6' };
    if (bmi < 25) return { text: 'Normal', color: '#10b981' };
    if (bmi < 30) return { text: 'Kelebihan Berat', color: '#f59e0b' };
    return { text: 'Obesitas', color: '#ef4444' };
  };

  const progressPercentage = (currentStep / steps.length) * 100;

  return (
    <div className="user-input-form">
      <div className="progress-bar">
        <div className="progress-steps">
          {steps.map((step) => (
            <div 
              key={step.number}
              className={`progress-step ${
                step.number === currentStep ? 'active' : ''
              } ${step.number < currentStep ? 'completed' : ''}`}
            >
              <div className="step-number">{step.number}</div>
              <div className="step-label">{step.label}</div>
            </div>
          ))}
        </div>
        <div className="progress-line">
          <div 
            className="progress-fill" 
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
      </div>

      {submissionError && (
        <div className="error-banner">
          <h3>Terjadi Kesalahan</h3>
          <p>{submissionError}</p>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {currentStep === 1 && (
          <div className="form-step">
            <h3>Informasi Dasar Anda</h3>
            
            <div className="form-group">
              <label htmlFor="age">Usia (tahun)</label>
              <input
                type="number"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleInputChange}
                placeholder="Contoh: 25"
                min="16"
                max="80"
                className={validationErrors.age ? 'error' : ''}
              />
              {validationErrors.age && (
                <span className="error-text">{validationErrors.age}</span>
              )}
            </div>

            <div className="form-group">
              <label>Jenis Kelamin</label>
              <div className="radio-group">
                <div className="radio-option">
                  <input
                    type="radio"
                    id="male"
                    name="gender"
                    value="male"
                    checked={formData.gender === 'male'}
                    onChange={handleInputChange}
                  />
                  <label htmlFor="male">👨 Pria</label>
                </div>
                <div className="radio-option">
                  <input
                    type="radio"
                    id="female"
                    name="gender"
                    value="female"
                    checked={formData.gender === 'female'}
                    onChange={handleInputChange}
                  />
                  <label htmlFor="female">👩 Wanita</label>
                </div>
              </div>
              {validationErrors.gender && (
                <span className="error-text">{validationErrors.gender}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="height">Tinggi Badan (cm)</label>
              <input
                type="number"
                id="height"
                name="height"
                value={formData.height}
                onChange={handleInputChange}
                placeholder="Contoh: 170"
                min="120"
                max="250"
                className={validationErrors.height ? 'error' : ''}
              />
              {validationErrors.height && (
                <span className="error-text">{validationErrors.height}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="weight">Berat Badan (kg)</label>
              <input
                type="number"
                id="weight"
                name="weight"
                value={formData.weight}
                onChange={handleInputChange}
                placeholder="Contoh: 65"
                min="30"
                max="300"
                step="0.1"
                className={validationErrors.weight ? 'error' : ''}
              />
              {validationErrors.weight && (
                <span className="error-text">{validationErrors.weight}</span>
              )}
            </div>

            {formData.height && formData.weight && (
              <div className="bmi-info">
                <h4>📊 Indeks Massa Tubuh (BMI)</h4>
                <div className="bmi-result">
                  <span className="bmi-value">{calculateBMI()}</span>
                  <span 
                    className="bmi-category"
                    style={{ color: getBMICategory(parseFloat(calculateBMI())).color }}
                  >
                    {getBMICategory(parseFloat(calculateBMI())).text}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {currentStep === 2 && (
          <div className="form-step">
            <h3>Apa Tujuan Fitness Anda?</h3>
            <div className="option-cards">
              {Object.entries(FITNESS_GOALS).map(([key, goal]) => (
                <div
                  key={key}
                  className={`option-card ${formData.fitness_goal === key ? 'selected' : ''}`}
                  onClick={() => handleInputChange({ target: { name: 'fitness_goal', value: key } })}
                >
                  <input
                    type="radio"
                    name="fitness_goal"
                    value={key}
                    checked={formData.fitness_goal === key}
                    onChange={handleInputChange}
                  />
                  <div className="option-content">
                    <h4>{goal.icon} {goal.label}</h4>
                    <p>{goal.description}</p>
                  </div>
                </div>
              ))}
            </div>
            {validationErrors.fitness_goal && (
              <span className="error-text">{validationErrors.fitness_goal}</span>
            )}
          </div>
        )}

        {currentStep === 3 && (
          <div className="form-step">
            <h3>Seberapa Aktif Anda Saat Ini?</h3>
            <div className="option-cards">
              {Object.entries(ACTIVITY_LEVELS).map(([key, level]) => (
                <div
                  key={key}
                  className={`option-card ${formData.activity_level === key ? 'selected' : ''}`}
                  onClick={() => handleInputChange({ target: { name: 'activity_level', value: key } })}
                >
                  <input
                    type="radio"
                    name="activity_level"
                    value={key}
                    checked={formData.activity_level === key}
                    onChange={handleInputChange}
                  />
                  <div className="option-content">
                    <h4>{level.icon} {level.label}</h4>
                    <p>{level.description}</p>
                    <span className="multiplier">Faktor: {level.multiplier}</span>
                  </div>
                </div>
              ))}
            </div>
            {validationErrors.activity_level && (
              <span className="error-text">{validationErrors.activity_level}</span>
            )}
          </div>
        )}

        {currentStep === 4 && (
          <div className="form-step">
            <h3>Konfirmasi Data Anda</h3>
            <div className="confirmation-summary">
              <div className="summary-card">
                <h4>📋 Ringkasan Profil</h4>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="label">Usia:</span>
                    <span className="value">{formData.age} tahun</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Jenis Kelamin:</span>
                    <span className="value">{formData.gender === 'male' ? 'Pria' : 'Wanita'}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Tinggi Badan:</span>
                    <span className="value">{formData.height} cm</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Berat Badan:</span>
                    <span className="value">{formData.weight} kg</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">BMI:</span>
                    <span className="value">{calculateBMI()} ({getBMICategory(parseFloat(calculateBMI())).text})</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Tujuan:</span>
                    <span className="value">{FITNESS_GOALS[formData.fitness_goal]?.label}</span>
                  </div>
                  <div className="summary-item">
                    <span className="label">Level Aktivitas:</span>
                    <span className="value">{ACTIVITY_LEVELS[formData.activity_level]?.label}</span>
                  </div>
                </div>
              </div>
              
              <div className="confirmation-note">
                <p>
                  <strong>📝 Catatan:</strong> Data ini akan digunakan untuk membuat 
                  rekomendasi fitness yang personal untuk Anda. Pastikan semua informasi sudah benar.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="form-navigation">
          {currentStep > 1 && (
            <button
              type="button"
              onClick={handlePrevious}
              className="btn-secondary"
              disabled={loading}
            >
              ← Sebelumnya
            </button>
          )}
          
          <div style={{ flex: 1 }}></div>
          
          {currentStep < steps.length ? (
            <button
              type="button"
              onClick={handleNext}
              className="btn-primary"
              disabled={loading}
            >
              Selanjutnya →
            </button>
          ) : (
            <button
              type="submit"
              className="btn-primary submit-btn"
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner small"></div>
                  Memproses...
                </>
              ) : (
                '🎯 Buat Rekomendasi'
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default EnhancedUserInputForm;