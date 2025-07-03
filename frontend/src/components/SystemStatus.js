// frontend/src/components/SystemStatus.js
import React from 'react';

const SystemStatus = ({ status }) => {
  if (!status) {
    return (
      <div className="system-status loading">
        <span className="status-indicator">⏳</span>
        <span>Checking system status...</span>
      </div>
    );
  }

  const getStatusDisplay = () => {
    if (status.status === 'healthy') {
      return {
        indicator: '✅',
        text: 'System Online',
        className: 'healthy'
      };
    } else if (status.status === 'error') {
      return {
        indicator: '❌',
        text: 'System Offline',
        className: 'error'
      };
    } else {
      return {
        indicator: '⚠️',
        text: 'System Issues',
        className: 'warning'
      };
    }
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div className={`system-status ${statusDisplay.className}`}>
      <span className="status-indicator">{statusDisplay.indicator}</span>
      <span className="status-text">{statusDisplay.text}</span>
      
      {status.ml_system_loaded && (
        <div className="status-details">
          <span>🤖 ML System Ready</span>
          {status.total_templates && (
            <span>📊 {status.total_templates} Templates</span>
          )}
          {status.thesis_aligned && (
            <span>🎓 Thesis Aligned</span>
          )}
        </div>
      )}
      
      {status.status === 'error' && (
        <div className="status-error">
          <p>⚠️ Backend server is not responding</p>
          <p>Make sure the Flask server is running on port 5000</p>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;