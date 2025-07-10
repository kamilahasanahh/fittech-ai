import React from 'react';
import Dashboard from '../components/Dashboard';

const DashboardPage = ({ user, userData, recommendations, onNavigate }) => {
  return (
    <Dashboard
      user={user}
      userData={userData}
      recommendations={recommendations}
      onNavigate={onNavigate}
    />
  );
};

export default DashboardPage; 