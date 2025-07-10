import React from 'react';
import EnhancedUserInputForm from '../components/EnhancedUserInputForm';

const InputPage = ({ onSubmit, loading, initialData }) => {
  return (
    <EnhancedUserInputForm
      onSubmit={onSubmit}
      loading={loading}
      initialData={initialData}
    />
  );
};

export default InputPage; 