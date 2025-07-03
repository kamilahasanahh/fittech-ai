// frontend/src/utils/validationRules.js
export const VALIDATION_RULES = {
  age: { min: 18, max: 65 },
  height: { min: 150, max: 200 },
  weight: { min: 45, max: 150 },
  target_weight: { min: 45, max: 150 }
};

// Safe weight change limits based on fitness research
export const WEIGHT_CHANGE_LIMITS = {
  'Muscle Gain': {
    minWeeksRequired: 12, // Minimum 12 weeks for healthy muscle gain
    maxWeightGain: 0.5,   // Max 0.5kg per week
    maxTotalGain: 12      // Max 12kg total gain
  },
  'Fat Loss': {
    minWeeksRequired: 8,  // Minimum 8 weeks for sustainable fat loss
    maxWeightLoss: 1.0,   // Max 1kg per week
    maxTotalLoss: 20      // Max 20kg total loss
  },
  'Maintenance': {
    maxVariation: 2       // Max ±2kg variation
  }
};

export const validateWeightGoal = (currentWeight, targetWeight, fitnessGoal, userAge, userGender) => {
  const weightDifference = targetWeight - currentWeight;
  const absoluteDifference = Math.abs(weightDifference);
  
  // Calculate healthy BMI range for user
  const minHealthyWeight = 18.5 * Math.pow(currentWeight / Math.sqrt(currentWeight), 2);
  const maxHealthyWeight = 24.9 * Math.pow(currentWeight / Math.sqrt(currentWeight), 2);
  
  const errors = [];
  
  // Check if target weight is within healthy BMI range
  if (targetWeight < 45) {
    errors.push("Target weight cannot be below 45kg for health reasons");
  }
  
  if (targetWeight > 150) {
    errors.push("Target weight cannot exceed 150kg");
  }
  
  // Validate based on fitness goal
  switch (fitnessGoal) {
    case 'Muscle Gain':
      if (weightDifference <= 0) {
        errors.push("For muscle gain, target weight must be higher than current weight");
      }
      
      if (weightDifference > WEIGHT_CHANGE_LIMITS['Muscle Gain'].maxTotalGain) {
        errors.push(`Maximum healthy muscle gain is ${WEIGHT_CHANGE_LIMITS['Muscle Gain'].maxTotalGain}kg. Consider a smaller initial goal.`);
      }
      
      const muscleGainWeeks = Math.ceil(weightDifference / WEIGHT_CHANGE_LIMITS['Muscle Gain'].maxWeightGain);
      if (muscleGainWeeks > 52) {
        errors.push(`Your goal would take ${muscleGainWeeks} weeks. Consider a smaller target (recommended: max 6kg gain).`);
      }
      break;
      
    case 'Fat Loss':
      if (weightDifference >= 0) {
        errors.push("For fat loss, target weight must be lower than current weight");
      }
      
      if (absoluteDifference > WEIGHT_CHANGE_LIMITS['Fat Loss'].maxTotalLoss) {
        errors.push(`Maximum recommended fat loss is ${WEIGHT_CHANGE_LIMITS['Fat Loss'].maxTotalLoss}kg. Consider a smaller initial goal.`);
      }
      
      // Don't allow target weight below healthy minimum
      if (targetWeight < 50) {
        errors.push("Target weight should not go below 50kg for health and safety reasons");
      }
      
      const fatLossWeeks = Math.ceil(absoluteDifference / WEIGHT_CHANGE_LIMITS['Fat Loss'].maxWeightLoss);
      if (fatLossWeeks > 26) {
        errors.push(`Your goal would take ${fatLossWeeks} weeks. Consider a smaller target (recommended: max 15kg loss).`);
      }
      break;
      
    case 'Maintenance':
      if (absoluteDifference > WEIGHT_CHANGE_LIMITS['Maintenance'].maxVariation) {
        errors.push(`For maintenance, target weight should be within ±${WEIGHT_CHANGE_LIMITS['Maintenance'].maxVariation}kg of current weight`);
      }
      break;
  }
  
  // Age-based adjustments
  if (userAge > 50 && absoluteDifference > 10) {
    errors.push("For users over 50, we recommend smaller weight changes (max 10kg) for safety");
  }
  
  // Calculate estimated timeline
  let estimatedWeeks = 0;
  if (fitnessGoal === 'Muscle Gain' && weightDifference > 0) {
    estimatedWeeks = Math.ceil(weightDifference / WEIGHT_CHANGE_LIMITS['Muscle Gain'].maxWeightGain);
  } else if (fitnessGoal === 'Fat Loss' && weightDifference < 0) {
    estimatedWeeks = Math.ceil(absoluteDifference / WEIGHT_CHANGE_LIMITS['Fat Loss'].maxWeightLoss);
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    estimatedWeeks,
    recommendations: generateWeightGoalRecommendations(currentWeight, targetWeight, fitnessGoal)
  };
};

const generateWeightGoalRecommendations = (currentWeight, targetWeight, fitnessGoal) => {
  const weightDifference = targetWeight - currentWeight;
  const recommendations = [];
  
  if (fitnessGoal === 'Muscle Gain' && weightDifference > 6) {
    recommendations.push(`Consider an initial goal of ${currentWeight + 6}kg, then reassess`);
    recommendations.push("Focus on strength training and adequate protein intake");
    recommendations.push("Aim for 0.25-0.5kg gain per week for lean muscle growth");
  }
  
  if (fitnessGoal === 'Fat Loss' && Math.abs(weightDifference) > 10) {
    recommendations.push(`Consider an initial goal of ${currentWeight - 10}kg, then reassess`);
    recommendations.push("Sustainable fat loss is 0.5-1kg per week");
    recommendations.push("Focus on caloric deficit with adequate protein to preserve muscle");
  }
  
  if (fitnessGoal === 'Maintenance') {
    recommendations.push("Focus on body composition rather than weight change");
    recommendations.push("Maintain consistent training and nutrition habits");
  }
  
  return recommendations;
};