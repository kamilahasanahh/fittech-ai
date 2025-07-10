import React, { useState, useEffect } from 'react';

const EnhancedUserInputForm = ({ onSubmit, loading: parentLoading, initialData }) => {
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
  // Use loading state from parent component
  const [submissionError, setSubmissionError] = useState('');
  const [bmi, setBmi] = useState(null);
  const [bmiCategory, setBmiCategory] = useState(null);
  const [storedDataLoaded, setStoredDataLoaded] = useState(false);

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
      description: 'Olahraga ringan dengan intensitas rendah',
      multiplier: '1.29',
      icon: '🚶‍♂️'
    },
    'Moderate Activity': {
      label: 'Aktivitas Sedang',
      description: 'Olahraga teratur dengan intensitas sedang',
      multiplier: '1.55',
      icon: '🏃‍♂️'
    },
    'High Activity': {
      label: 'Aktivitas Tinggi',
      description: 'Olahraga intensif dengan frekuensi tinggi',
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

  // Load stored user data on component mount
  useEffect(() => {
    const loadStoredData = () => {
      try {
        const storedData = localStorage.getItem('fittech_user_data');
        if (storedData) {
          const parsedData = JSON.parse(storedData);
          
          // Only load age and height if they exist and are valid
          if (parsedData.age && parsedData.height) {
            setFormData(prev => ({
              ...prev,
              age: parsedData.age.toString(),
              height: parsedData.height.toString(),
              gender: parsedData.gender || ''
            }));
            setStoredDataLoaded(true);
          }
        }
      } catch (error) {
        console.error('Error loading stored user data:', error);
      }
    };

    loadStoredData();
  }, []);

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
    // Allow: backspace, delete, tab, escape, enter, arrows, and numbers
    const allowedKeys = ['Backspace', 'Delete', 'Tab', 'Escape', 'Enter', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'];
    
    if (allowedKeys.includes(e.key)) {
      return;
    }
    
    // Allow numbers 0-9
    if (e.key >= '0' && e.key <= '9') {
      const currentValue = e.target.value;
      const newValue = currentValue + e.key;
      
      // Only prevent if the new value would be more than 2 digits or greater than 99
      // We'll handle the 18-65 range validation in onChange and onBlur
      if (newValue.length > 2 || parseInt(newValue) > 99) {
        e.preventDefault();
      }
      return;
    }
    
    // Block all other keys (like letters, symbols, etc.)
    e.preventDefault();
  };

  const handleAgeBlur = (e) => {
    const value = e.target.value;
    
    // If empty, just return
    if (value === '') {
      return;
    }
    
    const ageValue = parseInt(value);
    
    // Check if it's a valid number and within range
    if (isNaN(ageValue) || ageValue < 18 || ageValue > 65) {
      setValidationErrors(prev => ({
        ...prev,
        age: 'Usia harus antara 18-65 tahun'
      }));
      
      // Don't clear the input, let user fix it
      return;
    }
    
    // Clear any previous validation errors if age is valid
    if (validationErrors.age) {
      setValidationErrors(prev => ({
        ...prev,
        age: ''
      }));
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Special handling for age input - allow typing but clear validation errors
    if (name === 'age') {
      // Clear validation errors when user starts typing
      if (validationErrors.age) {
        setValidationErrors(prev => ({
          ...prev,
          age: ''
        }));
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
    
    setSubmissionError('');

    try {
      // Prepare data for the parent component
      const submissionData = {
        ...formData,
        // Ensure numeric values are properly converted
        age: parseInt(formData.age),
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        gender: formData.gender === 'male' ? 'Male' : 'Female'
      };

      console.log('🔄 Submitting form data to parent:', submissionData);
      
      // Save age, height, and gender to localStorage for future use
      try {
        const dataToStore = {
          age: submissionData.age,
          height: submissionData.height,
          gender: submissionData.gender
        };
        localStorage.setItem('fittech_user_data', JSON.stringify(dataToStore));
        console.log('💾 Saved user data to localStorage:', dataToStore);
      } catch (error) {
        console.error('Error saving user data to localStorage:', error);
      }
      
      // Call the parent's submit handler (which handles API calls and state updates)
      await onSubmit(submissionData);
      
    } catch (error) {
      console.error('❌ Error submitting form:', error);
      setSubmissionError(
        error.message.includes('fetch') 
          ? 'Tidak dapat terhubung ke server. Periksa koneksi internet Anda.'
          : `Terjadi kesalahan saat memproses data: ${error.message}`
      );
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

      {storedDataLoaded && (
        <div className="info-banner" style={{
          backgroundColor: '#dbeafe',
          border: '1px solid #3b82f6',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '20px',
          color: '#1e40af'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>💾</span>
              <div>
                <strong>Data Tersimpan Ditemukan!</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '0.9em' }}>
                  Usia, tinggi badan, dan jenis kelamin Anda telah dimuat dari penyimpanan lokal. 
                  Anda hanya perlu memasukkan berat badan saat ini.
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                localStorage.removeItem('fittech_user_data');
                setFormData({
                  age: '',
                  gender: '',
                  height: '',
                  weight: '',
                  fitness_goal: '',
                  activity_level: ''
                });
                setStoredDataLoaded(false);
              }}
              style={{
                background: 'none',
                border: '1px solid #3b82f6',
                borderRadius: '4px',
                padding: '4px 8px',
                fontSize: '0.8em',
                color: '#3b82f6',
                cursor: 'pointer',
                whiteSpace: 'nowrap'
              }}
            >
              Mulai Baru
            </button>
          </div>
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
              {storedDataLoaded && (
                <small style={{ color: '#059669', fontSize: '0.8em', fontWeight: '500' }}>
                  ⚡ Hanya berat badan yang perlu diperbarui - data lainnya sudah tersimpan!
                </small>
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
              disabled={parentLoading}
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
              disabled={parentLoading}
            >
              Selanjutnya →
            </button>
          ) : (
            <button
              type="submit"
              className="btn-primary submit-btn"
              disabled={parentLoading}
            >
              {parentLoading ? (
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
