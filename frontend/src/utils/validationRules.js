// frontend/src/utils/validationRules.js
export const VALIDATION_RULES = {
  age: { min: 18, max: 65 },
  height: { min: 150, max: 200 },
  weight: { min: 45, max: 150 }
};

// Validation for fitness goals (no target weight needed)
export const validateFitnessGoal = (fitnessGoal) => {
  const validGoals = ['Fat Loss', 'Muscle Gain', 'Maintenance'];
  
  return {
    isValid: validGoals.includes(fitnessGoal),
    recommendations: getFitnessGoalRecommendations(fitnessGoal)
  };
};

const getFitnessGoalRecommendations = (fitnessGoal) => {
  const recommendations = [];
  
  switch (fitnessGoal) {
    case 'Muscle Gain':
      recommendations.push("Focus on strength training and adequate protein intake");
      recommendations.push("Aim for progressive overload in your workouts");
      recommendations.push("Maintain a slight caloric surplus for muscle growth");
      break;
      
    case 'Fat Loss':
      recommendations.push("Combine cardio and strength training for best results");
      recommendations.push("Maintain adequate protein to preserve muscle mass");
      recommendations.push("Create a moderate caloric deficit");
      break;
      
    case 'Maintenance':
      recommendations.push("Focus on body composition rather than weight change");
      recommendations.push("Maintain consistent training and nutrition habits");
      recommendations.push("Adjust calories based on activity level and goals");
      break;
      
    default:
      recommendations.push("Please select a valid fitness goal");
      break;
  }
  
  return recommendations;
};