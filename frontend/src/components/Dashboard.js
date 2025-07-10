import React from 'react';

const Dashboard = ({ user, recommendations, onNavigate }) => {
  const formatDate = (date) => {
    return new Intl.DateTimeFormat('id-ID', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }).format(date);
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Dashboard XGFitness</h2>
        <p>Selamat datang kembali! Hari ini adalah {formatDate(new Date())}</p>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-card">
          <h3>👤 Profil Saya</h3>
          <div className="profile-summary">
            <div className="profile-item">
              <span className="label">Nama:</span>
              <span className="value">{user.displayName || 'Belum diatur'}</span>
            </div>
            <div className="profile-item">
              <span className="label">Email:</span>
              <span className="value">{user.email}</span>
            </div>
            <div className="profile-item">
              <span className="label">Bergabung:</span>
              <span className="value">
                {new Date(user.metadata.creationTime).toLocaleDateString('id-ID')}
              </span>
            </div>
            <div className="profile-item">
              <span className="label">Status:</span>
              <span className="value">Aktif</span>
            </div>
          </div>
        </div>

        <div className="dashboard-card">
          <h3>🎯 Rencana Fitness Saya</h3>
          {recommendations ? (
            <div className="current-plan">
              <div className="plan-item">
                <span className="label">Jenis Program:</span>
                <span className="value">
                  {recommendations.workout_recommendations?.workout_type || 'Program Kustom'}
                </span>
              </div>
              <div className="plan-item">
                <span className="label">Target Kalori:</span>
                <span className="value">
                  {recommendations.nutrition_recommendations?.caloric_intake 
                    ? `${Math.round(recommendations.nutrition_recommendations.caloric_intake)} kkal/hari`
                    : 'Belum dihitung'
                  }
                </span>
              </div>
              <div className="plan-item">
                <span className="label">Frekuensi Latihan:</span>
                <span className="value">
                  {recommendations.workout_recommendations?.days_per_week 
                    ? `${recommendations.workout_recommendations.days_per_week} hari/minggu`
                    : 'Belum diatur'
                  }
                </span>
              </div>
              <div className="plan-item">
                <span className="label">Status:</span>
                <span className="value">Aktif</span>
              </div>
            </div>
          ) : (
            <div className="no-plan">
              <p>Anda belum memiliki rencana fitness.</p>
              <p>Buat rencana baru untuk memulai perjalanan fitness Anda!</p>
            </div>
          )}
        </div>

        <div className="dashboard-card">
          <h3>⚡ Aksi Cepat</h3>
          <div className="quick-actions">
            <button 
              onClick={() => onNavigate('input')}
              className="btn-primary small"
            >
              📝 Buat Rencana Baru
            </button>
            {recommendations && (
              <button 
                onClick={() => onNavigate('recommendations')}
                className="btn-secondary small"
              >
                👀 Lihat Rekomendasi
              </button>
            )}
            <button className="btn-secondary small">
              📊 Lihat Progress
            </button>
            <button className="btn-secondary small">
              💡 Tips Hari Ini
            </button>
          </div>
        </div>

        <div className="dashboard-card">
          <h3>📈 Statistik Minggu Ini</h3>
          <div className="stats-summary">
            <div className="stat-item">
              <div className="stat-number">0</div>
              <div className="stat-label">Hari Latihan</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">0</div>
              <div className="stat-label">Kalori Terbakar</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">0</div>
              <div className="stat-label">Target Tercapai</div>
            </div>
          </div>
          <p className="stats-note">
            <small>*Statistik akan muncul setelah Anda mulai menjalankan program</small>
          </p>
        </div>

        <div className="dashboard-card">
          <h3>💡 Tips Fitness Hari Ini</h3>
          <div className="daily-tip">
            <h4>Hidrasi yang Cukup</h4>
            <p>
              Pastikan Anda minum air putih minimal 8 gelas per hari. Hidrasi yang baik 
              membantu meningkatkan performa latihan dan mempercepat pemulihan otot.
            </p>
            <div className="tip-actions">
              <button className="btn-secondary small">
                💧 Set Pengingat Minum
              </button>
            </div>
          </div>
        </div>

        <div className="dashboard-card">
          <h3>🏆 Pencapaian</h3>
          <div className="achievements">
            <div className="achievement-item">
              <span className="achievement-icon">🎖️</span>
              <div className="achievement-info">
                <h4>Pengguna Baru</h4>
                <p>Selamat bergabung dengan XGFitness!</p>
              </div>
            </div>
            <div className="achievement-placeholder">
              <p>Pencapaian lainnya akan muncul saat Anda aktif menggunakan aplikasi.</p>
            </div>
          </div>
        </div>
      </div>

      {recommendations && (
        <div className="dashboard-summary">
          <h3>📋 Ringkasan Rekomendasi Terkini</h3>
          <div className="summary-cards">
            <div className="summary-card workout">
              <h4>🏋️ Program Latihan</h4>
              <ul>
                <li>Jenis: {recommendations.workout_recommendations?.workout_type}</li>
                <li>Frekuensi: {recommendations.workout_recommendations?.days_per_week} hari/minggu</li>
                <li>Durasi Cardio: {recommendations.workout_recommendations?.cardio_minutes_per_day} menit/hari</li>
                <li>Set per Latihan: {recommendations.workout_recommendations?.sets_per_exercise}</li>
              </ul>
            </div>
            <div className="summary-card nutrition">
              <h4>🍎 Program Nutrisi</h4>
              <ul>
                <li>Kalori Harian: {Math.round(recommendations.nutrition_recommendations?.caloric_intake)} kkal</li>
                <li>Protein: {Math.round(recommendations.nutrition_recommendations?.protein_per_kg * 70)} gram</li>
                <li>Karbohidrat: {Math.round(recommendations.nutrition_recommendations?.carbs_per_kg * 70)} gram</li>
                <li>Lemak: {Math.round(recommendations.nutrition_recommendations?.fat_per_kg * 70)} gram</li>
              </ul>
              <small>*Perhitungan berdasarkan berat badan 70kg sebagai contoh</small>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
