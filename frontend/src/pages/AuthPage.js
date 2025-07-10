import React from 'react';
import AuthForm from '../components/AuthForm';

const AuthPage = ({ onAuthSuccess }) => {
  return <AuthForm onAuthSuccess={onAuthSuccess} />;
};

export default AuthPage; 