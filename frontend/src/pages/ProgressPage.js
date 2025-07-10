import React from 'react';
import DailyProgress from '../components/DailyProgress';

const ProgressPage = ({ user, onProgressUpdate, userProfile, currentRecommendation }) => {
  return (
    <DailyProgress
      user={user}
      onProgressUpdate={onProgressUpdate}
      userProfile={userProfile}
      currentRecommendation={currentRecommendation}
    />
  );
};

export default ProgressPage; 