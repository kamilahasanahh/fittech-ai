import React from 'react';
import RecommendationDisplay from '../components/RecommendationDisplay';

const RecommendationPage = ({ recommendations, userData, onBack, onNewRecommendation, loading, error }) => {
  if (loading) {
    return (
      <div className="loading-message">
        <div className="spinner"></div>
        <p>Memuat rekomendasi...</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="error-message">
        <p>❌ {error}</p>
      </div>
    );
  }
  if (!recommendations) {
    return (
      <div className="no-recommendations">
        <h2>🤔 Belum Ada Rekomendasi</h2>
        <p>Anda belum memiliki rekomendasi yang tersimpan. Silakan buat rencana baru untuk mendapatkan rekomendasi yang dipersonalisasi.</p>
        <button onClick={onBack} className="btn-primary">
          Buat Rencana Baru
        </button>
      </div>
    );
  }
  return (
    <RecommendationDisplay
      recommendations={recommendations}
      userData={userData}
      onBack={onBack}
      onNewRecommendation={onNewRecommendation}
    />
  );
};

export default RecommendationPage; 