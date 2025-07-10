import React from 'react';

const SystemStatus = ({ status }) => {
  if (!status) {
    return (
      <div className="system-status loading">
        <span>⏳ Memeriksa status sistem...</span>
      </div>
    );
  }

  const isHealthy = status.status === 'healthy';
  const hasMLSystem = status.model_loaded || status.ml_system_loaded;

  return (
    <div className={`system-status ${isHealthy ? 'healthy' : 'error'}`}>
      <span>{isHealthy ? '✅' : '❌'} {isHealthy ? 'Sistem Online' : 'Sistem Offline'}</span>
      
      {hasMLSystem && (
        <div className="status-details">
          <span>🤖 Model XGBoost Siap</span>
          {status.total_templates && <span>📊 {status.total_templates} Template</span>}
          {status.thesis_aligned && <span>🎓 Sesuai Penelitian</span>}
          {status.single_model && <span>🔗 Model Tunggal</span>}
        </div>
      )}
      
      {!isHealthy && (
        <div className="status-error">
          <p>⚠️ Server backend tidak merespons. Jalankan server Flask di port 5000</p>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;