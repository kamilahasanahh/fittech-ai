import React, { useState, useEffect } from 'react';
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
  const [bmi, setBmi] = useState(null);
  const [bmiCategory, setBmiCategory] = useState(null);

  // Indonesian fitness goals and activity levels (only 3 goals as requested)
  const FITNESS_GOALS = {
    'Fat Loss': {
      label: 'Menurunkan Berat Badan',
      description: 'Fokus pada pembakaran kalori dan pengurangan lemak tubuh',
      icon: '📉'
    },
    'Muscle Gain': {
      label: 'Menambah Massa Otot',
      description: 'Membangun otot dengan latihan beban dan nutrisi yang tepat',
      icon: '💪'
    },
    'Maintenance': {
      label: 'Mempertahankan Bentuk Tubuh',
      description: 'Menjaga kondisi fisik dan berat badan yang sudah ideal',
      icon: '⚖️'
    }
  };

  const ACTIVITY_LEVELS = {
    'Low Activity': {
      label: 'Aktivitas Rendah',
      description: 'Kurang dari 150 menit aktivitas fisik sedang ATAU kurang dari 75 menit aktivitas fisik berat per minggu',
      multiplier: '1.29',
      icon: '🚶‍♂️'
    },
    'Moderate Activity': {
      label: 'Aktivitas Sedang',
      description: '150-300 menit aktivitas fisik sedang ATAU 75-150 menit aktivitas fisik berat per minggu',
      multiplier: '1.55',
      icon: '🏃‍♂️'
    },
    'High Activity': {
      label: 'Aktivitas Tinggi',
      description: 'Lebih dari 300 menit aktivitas fisik sedang ATAU lebih dari 150 menit aktivitas fisik berat per minggu',
      multiplier: '1.81',
      icon: '🏋️‍♂️'
    }
  };

  const steps = [
    { number: 1, label: 'Informasi Dasar' },
    { number: 2, label: 'Tujuan Fitness' },
    { number: 3, label: 'Level Aktivitas' },
    { number: 4, label: 'Konfirmasi' }
  ];

  // Calculate BMI and update category whenever height or weight changes
  useEffect(() => {
    if (formData.height && formData.weight) {
      const heightInM = formData.height / 100;
      const calculatedBmi = (formData.weight / (heightInM * heightInM));
      setBmi(calculatedBmi);
      
      let category;
      if (calculatedBmi < 18.5) {
        category = { text: 'Kurus', color: '#3b82f6', restriction: 'underweight' };
      } else if (calculatedBmi < 25) {
        category = { text: 'Normal', color: '#10b981', restriction: 'normal' };
      } else if (calculatedBmi < 30) {
        category = { text: 'Kelebihan Berat', color: '#f59e0b', restriction: 'overweight' };
      } else {
        category = { text: 'Obesitas', color: '#ef4444', restriction: 'obese' };
      }
      setBmiCategory(category);
    } else {
      setBmi(null);
      setBmiCategory(null);
    }
  }, [formData.height, formData.weight]);

  // Check if a fitness goal is allowed based on BMI category
  const isFitnessGoalAllowed = (goalKey) => {
    if (!bmiCategory) return true; // Allow all goals if BMI not calculated yet
    
    const restrictions = {
      'underweight': ['Muscle Gain', 'Maintenance'], // Only muscle gain and maintenance
      'normal': ['Fat Loss', 'Muscle Gain', 'Maintenance'], // All goals allowed
      'overweight': ['Fat Loss', 'Maintenance'], // Only fat loss and maintenance
      'obese': ['Fat Loss'] // Only fat loss
    };
    
    return restrictions[bmiCategory.restriction].includes(goalKey);
  };

  const validateStep = (step) => {
    const errors = {};
    
    switch (step) {
      case 1:
        if (!formData.age || formData.age < 18 || formData.age > 65) {
          errors.age = 'Usia harus antara 18-65 tahun';
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
        } else if (!isFitnessGoalAllowed(formData.fitness_goal)) {
          errors.fitness_goal = 'Tujuan fitness ini tidak direkomendasikan untuk kategori BMI Anda';
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

  // Handle age input restrictions
  const handleAgeKeyDown = (e) => {
    // Allow: backspace, delete, tab, escape, enter, arrows
    const allowedKeys = ['Backspace', 'Delete', 'Tab', 'Escape', 'Enter', 'ArrowLeft', 'ArrowRight'];
    
    if (allowedKeys.includes(e.key)) {
      return;
    }
    
    // Allow numbers 0-9
    if (e.key >= '0' && e.key <= '9') {
      const currentValue = e.target.value;
      const newValue = currentValue + e.key;
      
      // Prevent typing if it would result in invalid age
      if (newValue.length > 2 || parseInt(newValue) > 65) {
        e.preventDefault();
      }
      return;
    }
    
    // Block all other keys
    e.preventDefault();
  };

  const handleAgeBlur = (e) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value < 18) {
      // If user typed a number less than 18, clear it
      setFormData(prev => ({
        ...prev,
        age: ''
      }));
      setValidationErrors(prev => ({
        ...prev,
        age: 'Usia harus antara 18-65 tahun'
      }));
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Special handling for age input to enforce 18-65 range
    if (name === 'age') {
      const ageValue = parseInt(value);
      // Allow empty input for editing, but restrict invalid values
      if (value !== '' && (ageValue < 18 || ageValue > 65 || isNaN(ageValue))) {
        return; // Don't update state with invalid age values
      }
    }
    
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

    // If fitness goal is being changed and it's not allowed, clear it
    if (name === 'fitness_goal' && !isFitnessGoalAllowed(value)) {
      return; // Don't allow selection of restricted goals
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
                onKeyDown={handleAgeKeyDown}
                onBlur={handleAgeBlur}
                placeholder="Contoh: 25"
                min="18"
                max="65"
                className={validationErrors.age ? 'error' : ''}
              />
              {validationErrors.age && (
                <span className="error-text">{validationErrors.age}</span>
              )}
              <small style={{ color: '#6b7280', fontSize: '0.8em' }}>
                Usia harus antara 18-65 tahun
              </small>
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

            {bmi && bmiCategory && (
              <div className="bmi-info">
                <h4>📊 Indeks Massa Tubuh (BMI)</h4>
                <div className="bmi-result">
                  <span className="bmi-value">{bmi.toFixed(1)}</span>
                  <span 
                    className="bmi-category"
                    style={{ color: bmiCategory.color }}
                  >
                    {bmiCategory.text}
                  </span>
                </div>
                {bmiCategory.restriction !== 'normal' && (
                  <div className="bmi-restriction-note">
                    <p style={{ fontSize: '0.9em', marginTop: '10px', padding: '8px', backgroundColor: '#f0f9ff', borderRadius: '4px' }}>
                      💡 <strong>Catatan:</strong> Berdasarkan BMI Anda, beberapa tujuan fitness mungkin tidak tersedia pada langkah selanjutnya untuk hasil yang optimal.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {currentStep === 2 && (
          <div className="form-step">
            <h3>Apa Tujuan Fitness Anda?</h3>
            {bmiCategory && bmiCategory.restriction !== 'normal' && (
              <div className="bmi-restriction-info" style={{ 
                padding: '12px', 
                backgroundColor: '#f0f9ff', 
                borderRadius: '8px', 
                marginBottom: '20px',
                border: '1px solid #bfdbfe'
              }}>
                <p style={{ margin: 0, fontSize: '0.9em' }}>
                  <strong>📋 Rekomendasi berdasarkan BMI Anda ({bmiCategory.text}):</strong><br/>
                  Pilihan yang diarsir tidak direkomendasikan untuk kategori BMI Anda saat ini.
                </p>
              </div>
            )}
            <div className="option-cards">
              {Object.entries(FITNESS_GOALS).map(([key, goal]) => {
                const isAllowed = isFitnessGoalAllowed(key);
                return (
                  <div
                    key={key}
                    className={`option-card ${formData.fitness_goal === key ? 'selected' : ''} ${!isAllowed ? 'disabled' : ''}`}
                    onClick={() => {
                      if (isAllowed) {
                        handleInputChange({ target: { name: 'fitness_goal', value: key } });
                      }
                    }}
                    style={{
                      opacity: isAllowed ? 1 : 0.5,
                      cursor: isAllowed ? 'pointer' : 'not-allowed',
                      backgroundColor: !isAllowed ? '#f9fafb' : ''
                    }}
                  >
                    <input
                      type="radio"
                      name="fitness_goal"
                      value={key}
                      checked={formData.fitness_goal === key}
                      onChange={handleInputChange}
                      disabled={!isAllowed}
                    />
                    <div className="option-content">
                      <h4>{goal.icon} {goal.label}</h4>
                      <p>{goal.description}</p>
                      {!isAllowed && (
                        <small style={{ color: '#6b7280', fontStyle: 'italic' }}>
                          Tidak direkomendasikan untuk BMI Anda
                        </small>
                      )}
                    </div>
                  </div>
                );
              })}
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
            
            {/* Informational card */}
            <div className="info-card" style={{
              backgroundColor: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '12px',
              padding: '16px',
              marginTop: '20px',
              textAlign: 'center'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                marginBottom: '8px'
              }}>
                <span style={{ fontSize: '20px' }}>💡</span>
                <strong style={{ color: '#495057' }}>Informasi</strong>
              </div>
              <p style={{
                margin: 0,
                color: '#6c757d',
                fontSize: '14px',
                lineHeight: '1.4'
              }}>
                Aktivitas fisik = semua gerakan tubuh (olahraga + pekerjaan + aktivitas harian) yang membuat Anda bergerak aktif dan berkeringat
              </p>
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
                    <span className="value">{bmi ? bmi.toFixed(1) : '-'} ({bmiCategory ? bmiCategory.text : '-'})</span>
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
