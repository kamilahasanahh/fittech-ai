import React, { useState, useEffect, useCallback } from 'react';
import { db } from '../services/firebaseConfig';
import { doc, setDoc, getDoc } from 'firebase/firestore';

const DailyProgress = ({ user, onProgressUpdate }) => {
  const [progressData, setProgressData] = useState({
    workout: false,
    nutrition: false,
    hydration: false,
    notes: ''
  });
  const [streak, setStreak] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const today = new Date().toISOString().split('T')[0];

  useEffect(() => {
    loadTodaysProgress();
  }, [user, loadTodaysProgress]);

  const loadTodaysProgress = useCallback(async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      const progressDoc = await getDoc(progressRef);
      
      if (progressDoc.exists()) {
        const data = progressDoc.data();
        setProgressData({
          workout: data.workout || false,
          nutrition: data.nutrition || false,
          hydration: data.hydration || false,
          notes: data.notes || ''
        });
      }
      
      // Load streak
      const userRef = doc(db, 'users', user.uid);
      const userDoc = await getDoc(userRef);
      if (userDoc.exists()) {
        setStreak(userDoc.data().currentStreak || 0);
      }
    } catch (error) {
      console.error('Error loading progress:', error);
    } finally {
      setLoading(false);
    }
  }, [user, today]);

  const saveProgress = async (newData) => {
    if (!user) return;
    
    setSaving(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      await setDoc(progressRef, {
        ...newData,
        date: today,
        userId: user.uid,
        updatedAt: new Date()
      });

      // Update streak if all goals completed
      if (newData.workout && newData.nutrition && newData.hydration) {
        const userRef = doc(db, 'users', user.uid);
        const userDoc = await getDoc(userRef);
        const currentStreak = userDoc.exists() ? (userDoc.data().currentStreak || 0) : 0;
        
        await setDoc(userRef, {
          currentStreak: currentStreak + 1,
          lastActiveDate: today
        }, { merge: true });
        
        setStreak(currentStreak + 1);
      }

      onProgressUpdate && onProgressUpdate(newData);
    } catch (error) {
      console.error('Error saving progress:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleGoalToggle = (goal) => {
    const newData = {
      ...progressData,
      [goal]: !progressData[goal]
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const handleNotesChange = (e) => {
    const newData = {
      ...progressData,
      notes: e.target.value
    };
    setProgressData(newData);
    // Debounce saving notes
    setTimeout(() => saveProgress(newData), 1000);
  };

  const getCompletionPercentage = () => {
    const goals = [progressData.workout, progressData.nutrition, progressData.hydration];
    const completed = goals.filter(Boolean).length;
    return Math.round((completed / goals.length) * 100);
  };

  const getMotivationalMessage = () => {
    const percentage = getCompletionPercentage();
    if (percentage === 100) {
      return "🎉 Luar biasa! Anda telah menyelesaikan semua target hari ini!";
    } else if (percentage >= 66) {
      return "💪 Bagus sekali! Tinggal sedikit lagi untuk mencapai target harian!";
    } else if (percentage >= 33) {
      return "👍 Terus semangat! Anda sudah di jalur yang tepat!";
    } else {
      return "🌟 Mulai hari ini dengan semangat! Setiap langkah kecil berarti!";
    }
  };

  if (loading) {
    return (
      <div className="daily-progress loading">
        <div className="spinner"></div>
        <p>Memuat progress hari ini...</p>
      </div>
    );
  }

  return (
    <div className="daily-progress">
      <div className="progress-header">
        <h2>📈 Progress Harian</h2>
        <p>Pantau pencapaian target fitness Anda hari ini</p>
        <div className="date-info">
          <span>📅 {new Date().toLocaleDateString('id-ID', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          })}</span>
        </div>
      </div>

      <div className="progress-overview">
        <div className="completion-circle">
          <div className="circle-progress">
            <svg viewBox="0 0 36 36" className="circular-chart">
              <path
                className="circle-bg"
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="circle"
                strokeDasharray={`${getCompletionPercentage()}, 100`}
                d="M18 2.0845
                  a 15.9155 15.9155 0 0 1 0 31.831
                  a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <text x="18" y="20.35" className="percentage">
                {getCompletionPercentage()}%
              </text>
            </svg>
          </div>
          <div className="completion-text">
            <h3>Target Harian</h3>
            <p>{getMotivationalMessage()}</p>
          </div>
        </div>

        <div className="streak-info">
          <div className="streak-number">{streak}</div>
          <div className="streak-label">Hari Berturut-turut</div>
          <div className="streak-icon">🔥</div>
        </div>
      </div>

      <div className="daily-goals">
        <h3>🎯 Target Hari Ini</h3>
        
        <div className="goal-list">
          <div className={`goal-item ${progressData.workout ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">🏋️‍♂️</div>
              <div className="goal-info">
                <h4>Latihan Fisik</h4>
                <p>Selesaikan sesi latihan sesuai program Anda</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.workout ? 'active' : ''}`}
              onClick={() => handleGoalToggle('workout')}
              disabled={saving}
            >
              {progressData.workout ? '✅' : '⭕'}
            </button>
          </div>

          <div className={`goal-item ${progressData.nutrition ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">🍎</div>
              <div className="goal-info">
                <h4>Target Nutrisi</h4>
                <p>Ikuti panduan makanan dan kalori harian</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.nutrition ? 'active' : ''}`}
              onClick={() => handleGoalToggle('nutrition')}
              disabled={saving}
            >
              {progressData.nutrition ? '✅' : '⭕'}
            </button>
          </div>

          <div className={`goal-item ${progressData.hydration ? 'completed' : ''}`}>
            <div className="goal-content">
              <div className="goal-icon">💧</div>
              <div className="goal-info">
                <h4>Hidrasi</h4>
                <p>Minum air putih minimal 8 gelas (2 liter)</p>
              </div>
            </div>
            <button
              className={`goal-toggle ${progressData.hydration ? 'active' : ''}`}
              onClick={() => handleGoalToggle('hydration')}
              disabled={saving}
            >
              {progressData.hydration ? '✅' : '⭕'}
            </button>
          </div>
        </div>
      </div>

      <div className="progress-notes">
        <h3>📝 Catatan Harian</h3>
        <textarea
          placeholder="Bagaimana perasaan Anda hari ini? Catat pencapaian, tantangan, atau hal-hal yang ingin Anda ingat..."
          value={progressData.notes}
          onChange={handleNotesChange}
          rows={4}
          disabled={saving}
        />
        {saving && <p className="saving-indicator">💾 Menyimpan...</p>}
      </div>

      <div className="progress-tips">
        <h3>💡 Tips Hari Ini</h3>
        <div className="tips-grid">
          <div className="tip-card">
            <h4>🕐 Waktu Terbaik Berolahraga</h4>
            <p>Pagi hari (06:00-08:00) atau sore (16:00-18:00) adalah waktu optimal untuk berolahraga.</p>
          </div>
          <div className="tip-card">
            <h4>🥗 Nutrisi Seimbang</h4>
            <p>Pastikan setiap makanan mengandung protein, karbohidrat, dan sayuran untuk nutrisi optimal.</p>
          </div>
          <div className="tip-card">
            <h4>😴 Istirahat Cukup</h4>
            <p>Tidur 7-9 jam per malam sangat penting untuk pemulihan otot dan metabolisme yang baik.</p>
          </div>
        </div>
      </div>

      {getCompletionPercentage() === 100 && (
        <div className="celebration">
          <div className="celebration-content">
            <h3>🎉 Selamat!</h3>
            <p>Anda telah menyelesaikan semua target hari ini!</p>
            <p>Streak Anda: <strong>{streak} hari</strong> 🔥</p>
            <button className="btn-primary">
              Bagikan Pencapaian
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DailyProgress;
