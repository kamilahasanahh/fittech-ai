import React, { useState, useEffect } from 'react';
import {
  Box,
  Stack,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  SimpleGrid,
  Badge,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useColorModeValue,
  Divider,
  Card,
  CardBody,
  CardHeader,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  List,
  ListItem,
  ListIcon,
  Icon,
  Flex,
  Spacer,
  Skeleton,
  SkeletonText,
  SkeletonCircle
} from '@chakra-ui/react';
import { CheckCircleIcon, InfoIcon, WarningIcon } from '@chakra-ui/icons';
import { nutritionService } from '../services/nutritionService';
import { mealPlanService } from '../services/mealPlanService';
import { apiService } from '../services/api';

const RecommendationDisplay = ({ recommendations, userData, onBack, onNewRecommendation, onMealPlanGenerated }) => {
  const [nutritionData, setNutritionData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mealPlan, setMealPlan] = useState(null);
  const [mealPlanLoading, setMealPlanLoading] = useState(true);
  const [backendMealPlan, setBackendMealPlan] = useState(null);
  const [backendMealPlanLoading, setBackendMealPlanLoading] = useState(true);

  const gradientBg = useColorModeValue(
    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)'
  );

  // Calculate user's daily macro targets based on API response
  const calculateDailyMacros = () => {
    if (!recommendations || !userData) return null;

    const nutrition = recommendations?.predictions?.nutrition_template;
    if (!nutrition) return null;

    // Calculate based on TDEE from user_profile and standard ratios
    const tdee = recommendations?.user_profile?.tdee || 2000;
    const weight = parseFloat(userData.weight);
    
    // Standard macro calculations for different goals
    let calories, protein, carbs, fat;
    
    if (userData.fitness_goal === 'Fat Loss') {
      calories = Math.round(tdee * 0.8); // 20% deficit
      protein = Math.round(weight * 2.2); // High protein for fat loss
      fat = Math.round(weight * 0.8);
      carbs = Math.round((calories - (protein * 4) - (fat * 9)) / 4);
    } else if (userData.fitness_goal === 'Muscle Gain') {
      calories = Math.round(tdee * 1.1); // 10% surplus
      protein = Math.round(weight * 2.0);
      fat = Math.round(weight * 1.0);
      carbs = Math.round((calories - (protein * 4) - (fat * 9)) / 4);
    } else { // Maintenance
      calories = Math.round(tdee);
      protein = Math.round(weight * 1.8);
      fat = Math.round(weight * 0.9);
      carbs = Math.round((calories - (protein * 4) - (fat * 9)) / 4);
    }

    return { calories, protein, carbs, fat };
  };

  // Load nutrition data from JSON file and generate meal plan
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load nutrition data
        const jsonData = await nutritionService.loadNutritionData();
        setNutritionData(jsonData);
        setLoading(false);

        // Generate meal plan if we have user data
        if (recommendations && userData) {
          setMealPlanLoading(true);
          setBackendMealPlanLoading(true);
          
          const dailyMacros = calculateDailyMacros();
          
          if (dailyMacros) {
            // Generate meal plan using the local service
            const mealPlanResult = await mealPlanService.generateDailyMealPlan(
              dailyMacros.calories,
              dailyMacros.protein,
              dailyMacros.carbs,
              dailyMacros.fat
            );

            if (mealPlanResult.success) {
              const transformedMealPlan = mealPlanService.transformMealPlanToFrontend(mealPlanResult);
              setMealPlan(transformedMealPlan);
            } else {
              console.warn('Failed to load organized meal plan, using fallback');
              setMealPlan(null);
            }
            setMealPlanLoading(false);

            // Also fetch meal plan from the backend API for comparison
            try {
              console.log('🔄 Fetching meal plan from backend with macros:', dailyMacros);
              const backendPlan = await apiService.getMealPlan(
                dailyMacros.calories,
                dailyMacros.protein,
                dailyMacros.carbs,
                dailyMacros.fat,
                { dietary_restrictions: [] } // Add preferences if needed
              );
              console.log('✅ Backend meal plan received:', backendPlan);
              setBackendMealPlan(backendPlan);
              
              // Store the detailed meal plan in the recommendation for future reference
              if (backendPlan && backendPlan.meal_plan && onMealPlanGenerated) {
                console.log('🔄 Storing meal plan for recommendation history...');
                onMealPlanGenerated(backendPlan.meal_plan);
              }
            } catch (backendError) {
              console.error('❌ Failed to fetch backend meal plan:', backendError);
              setBackendMealPlan(null);
            }
            setBackendMealPlanLoading(false);
          }
        }
      } catch (error) {
        console.error('Error loading data:', error);
        setLoading(false);
        setMealPlanLoading(false);
        setBackendMealPlanLoading(false);
      }
    };

    loadData();
  }, [recommendations, userData]);

  if (!recommendations) {
    return (
      <Box textAlign="center" py={10}>
        <Heading size="lg" mb={4}>🤔 Belum Ada Rekomendasi</Heading>
        <Text color="gray.600" mb={6}>
          Silakan isi formulir profil terlebih dahulu untuk mendapatkan rekomendasi yang dipersonalisasi.
        </Text>
        <Button colorScheme="brand" onClick={onBack}>
          Kembali ke Formulir
        </Button>
      </Box>
    );
  }

  // Map API response structure to component expected structure
  // Extract workout and nutrition data from the API response
  const workout = recommendations?.predictions?.workout_template;
  const nutrition = recommendations?.predictions?.nutrition_template;

  // Calculate food portions based on template requirements
  const calculateFoodPortions = (targetMacros) => {
    if (!nutritionData.length || !targetMacros) return [];

    const suggestions = [];
    
    // Distribute daily calories across meals
    const mealDistribution = {
      sarapan: { percentage: 0.25, name: '🌅 Sarapan' },
      makan_siang: { percentage: 0.40, name: '☀️ Makan Siang' },
      makan_malam: { percentage: 0.30, name: '🌙 Makan Malam' },
      camilan: { percentage: 0.05, name: '🍪 Camilan' }
    };

    Object.entries(mealDistribution).forEach(([mealType, config]) => {
      const targetCalories = targetMacros.calories * config.percentage;
      const targetProtein = targetMacros.protein * config.percentage;
      
      // Select appropriate foods for this meal using nutrition service
      let selectedFoods = [];
      
      if (mealType === 'sarapan') {
        selectedFoods = nutritionService.getFoodsByCategory('breakfast');
      } else if (mealType === 'makan_siang') {
        selectedFoods = nutritionService.getFoodsByCategory('lunch');
      } else if (mealType === 'makan_malam') {
        selectedFoods = nutritionService.getFoodsByCategory('dinner');
      } else if (mealType === 'camilan') {
        selectedFoods = nutritionService.getFoodsByCategory('snack');
      }

      // If no specific foods found, use all available
      if (selectedFoods.length === 0) {
        if (mealType === 'camilan') {
          // For snacks, use lighter options
          selectedFoods = nutritionData.filter(food => 
            food.name.toLowerCase().includes('jagung') ||
            food.name.toLowerCase().includes('telur rebus') ||
            food.calories < 200
          ).slice(0, 2);
        } else {
          selectedFoods = nutritionData.slice(0, 3);
        }
      }

      // Calculate portions to meet targets
      const mealFoods = selectedFoods.slice(0, mealType === 'camilan' ? 1 : 3).map(food => {
        // Calculate how many grams needed to meet portion of target calories
        const divisor = mealType === 'camilan' ? 1 : 3; // Only 1 food for snacks, 3 for meals
        const gramsNeeded = Math.min(
          mealType === 'camilan' ? 100 : 200, 
          Math.max(
            mealType === 'camilan' ? 30 : 50, 
            (targetCalories / divisor) / (food.calories / 100)
          )
        );
        
        return {
          ...food,
          grams: Math.round(gramsNeeded),
          actualCalories: Math.round((food.calories / 100) * gramsNeeded),
          actualProtein: Math.round((food.protein / 100) * gramsNeeded * 10) / 10,
          actualCarbs: Math.round((food.carbs / 100) * gramsNeeded * 10) / 10,
          actualFat: Math.round((food.fat / 100) * gramsNeeded * 10) / 10
        };
      });

      suggestions.push({
        meal: config.name,
        targetCalories: Math.round(targetCalories),
        targetProtein: Math.round(targetProtein),
        foods: mealFoods
      });
    });

    return suggestions;
  };

  const dailyMacros = calculateDailyMacros();
  
  // Use organized meal plan if available, otherwise use fallback calculation
  const foodSuggestions = mealPlan && mealPlan.length > 0 
    ? mealPlan 
    : (dailyMacros ? calculateFoodPortions(dailyMacros) : []);

  // Loading skeleton for the entire component
  if (loading) {
    return (
      <Box maxW={{ base: "full", lg: "6xl" }} mx="auto" p={{ base: 4, md: 8 }}>
        <VStack spacing={{ base: 6, md: 8 }} align="stretch">
          {/* Header Skeleton */}
          <Box textAlign="center" py={{ base: 4, md: 6 }}>
            <Skeleton height="40px" mb={2} />
            <Skeleton height="20px" width="60%" mx="auto" />
          </Box>

          {/* Profile Summary Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="200px" />
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
                {[...Array(6)].map((_, i) => (
                  <Box key={i}>
                    <Skeleton height="16px" width="80px" mb={1} />
                    <Skeleton height="24px" width="120px" />
                  </Box>
                ))}
              </SimpleGrid>
            </CardBody>
          </Card>

          {/* Metrics Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="250px" />
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 4, md: 6 }}>
                {[...Array(4)].map((_, i) => (
                  <Box key={i}>
                    <Skeleton height="16px" width="100px" mb={1} />
                    <Skeleton height="24px" width="80px" />
                  </Box>
                ))}
              </SimpleGrid>
            </CardBody>
          </Card>

          {/* Workout Section Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="300px" />
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <Skeleton height="100px" width="full" />
                <Skeleton height="80px" width="full" />
              </VStack>
            </CardBody>
          </Card>

          {/* Nutrition Section Skeleton */}
          <Card>
            <CardHeader>
              <Skeleton height="24px" width="280px" />
            </CardHeader>
            <CardBody>
              <VStack spacing={4}>
                <Skeleton height="120px" width="full" />
                <Skeleton height="200px" width="full" />
              </VStack>
            </CardBody>
          </Card>
        </VStack>
      </Box>
    );
  }

  return (
    <Box maxW={{ base: "full", lg: "6xl" }} mx="auto" p={{ base: 4, md: 8 }}>
      <VStack spacing={{ base: 6, md: 8 }} align="stretch">
        {/* Header */}
        <Box textAlign="center" py={{ base: 4, md: 6 }}>
          <Heading size={{ base: "lg", md: "xl" }} bgGradient={gradientBg} bgClip="text" mb={2}>
            🎯 Rekomendasi XGFitness Anda
          </Heading>
          <Text color="gray.600" fontSize={{ base: "md", md: "lg" }}>
            Rekomendasi yang dipersonalisasi berdasarkan profil dan tujuan Anda
          </Text>
        </Box>

        {/* User Profile Summary */}
        <Card>
          <CardHeader>
            <Heading size={{ base: "sm", md: "md" }}>👤 Profil Anda</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
              <Stat>
                <StatLabel>Usia</StatLabel>
                <StatNumber>{userData?.age} tahun</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Jenis Kelamin</StatLabel>
                <StatNumber>{userData?.gender === 'Male' ? 'Pria' : 'Wanita'}</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tinggi</StatLabel>
                <StatNumber>{userData?.height} cm</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Berat</StatLabel>
                <StatNumber>{userData?.weight} kg</StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tujuan</StatLabel>
                <StatNumber>
                  {userData?.fitness_goal === 'Fat Loss' ? 'Membakar Lemak' :
                   userData?.fitness_goal === 'Muscle Gain' ? 'Menambah Massa Otot' :
                   userData?.fitness_goal === 'Maintenance' ? 'Mempertahankan Berat' : 
                   userData?.fitness_goal}
                </StatNumber>
              </Stat>
              <Stat>
                <StatLabel>Tingkat Aktivitas</StatLabel>
                <StatNumber>
                  {userData?.activity_level === 'Low Activity' ? 'Aktivitas Rendah' :
                   userData?.activity_level === 'Moderate Activity' ? 'Aktivitas Sedang' :
                   userData?.activity_level === 'High Activity' ? 'Aktivitas Tinggi' :
                   userData?.activity_level}
                </StatNumber>
              </Stat>
            </SimpleGrid>
          </CardBody>
        </Card>

        {/* User Metrics from API */}
        {recommendations?.user_profile && (
          <Card>
            <CardHeader>
              <Heading size={{ base: "sm", md: "md" }}>📊 Analisis Tubuh Anda</Heading>
            </CardHeader>
            <CardBody>
              <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 4, md: 6 }}>
                <Stat>
                  <StatLabel>BMI</StatLabel>
                  <StatNumber>{recommendations.user_profile.bmi?.toFixed(1)}</StatNumber>
                  <StatHelpText>
                    <Badge colorScheme={recommendations.user_profile.bmi_category === 'Normal' ? 'green' : 'orange'}>
                      {recommendations.user_profile.bmi_category === 'Underweight' ? 'Kurus' :
                       recommendations.user_profile.bmi_category === 'Normal' ? 'Normal' :
                       recommendations.user_profile.bmi_category === 'Overweight' ? 'Kelebihan Berat' :
                       recommendations.user_profile.bmi_category === 'Obese' ? 'Obesitas' :
                       recommendations.user_profile.bmi_category}
                    </Badge>
                  </StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>BMR</StatLabel>
                  <StatNumber>{Math.round(recommendations.user_profile.bmr)} kkal</StatNumber>
                  <StatHelpText>Kalori Basal</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>TDEE</StatLabel>
                  <StatNumber>{Math.round(recommendations.user_profile.tdee)} kkal</StatNumber>
                  <StatHelpText>Total Pengeluaran Energi</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel>Kategori BMI</StatLabel>
                  <StatNumber>
                    {recommendations.user_profile.bmi_category === 'Underweight' ? 'Kurus' :
                     recommendations.user_profile.bmi_category === 'Normal' ? 'Normal' :
                     recommendations.user_profile.bmi_category === 'Overweight' ? 'Kelebihan Berat' :
                     recommendations.user_profile.bmi_category === 'Obese' ? 'Obesitas' :
                     recommendations.user_profile.bmi_category}
                  </StatNumber>
                </Stat>
              </SimpleGrid>
            </CardBody>
          </Card>
        )}

        {/* Confidence Scores */}
        {recommendations.model_confidence && (
          <Card>
            <CardHeader>
              <Heading size={{ base: "sm", md: "md" }}>🎯 Tingkat Kepercayaan AI</Heading>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 4, md: 6 }}>
                  <Stat>
                    <StatLabel>Kepercayaan Keseluruhan</StatLabel>
                    <StatNumber color="green.500">
                      {Math.round(((recommendations.model_confidence.nutrition_confidence + recommendations.model_confidence.workout_confidence) / 2) * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <Badge colorScheme="green">Tinggi</Badge>
                    </StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Kepercayaan Nutrisi</StatLabel>
                    <StatNumber color="blue.500">
                      {Math.round(recommendations.model_confidence.nutrition_confidence * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <StatArrow type={recommendations.model_confidence.nutrition_confidence > 0.5 ? 'increase' : 'decrease'} />
                    </StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Kepercayaan Workout</StatLabel>
                    <StatNumber color="purple.500">
                      {Math.round(recommendations.model_confidence.workout_confidence * 100)}%
                    </StatNumber>
                    <StatHelpText>
                      <StatArrow type={recommendations.model_confidence.workout_confidence > 0.5 ? 'increase' : 'decrease'} />
                    </StatHelpText>
                  </Stat>
                </SimpleGrid>

                <Alert status="info" borderRadius="md">
                  <AlertIcon />
                  <Box>
                    <AlertTitle>Level Kepercayaan</AlertTitle>
                    <AlertDescription>
                      {(() => {
                        const explanation = recommendations.enhanced_confidence?.explanation;
                        if (explanation === 'Based on high activity and fat loss goal') return 'Berdasarkan aktivitas tinggi dan tujuan membakar lemak';
                        if (explanation === 'Based on moderate activity and muscle gain goal') return 'Berdasarkan aktivitas sedang dan tujuan menambah massa otot';
                        if (explanation === 'Based on low activity and maintenance goal') return 'Berdasarkan aktivitas rendah dan tujuan mempertahankan berat';
                        if (explanation === 'Based on moderate activity and maintenance goal') return 'Berdasarkan aktivitas sedang dan tujuan mempertahankan berat';
                        if (explanation === 'Based on high activity and muscle gain goal') return 'Berdasarkan aktivitas tinggi dan tujuan menambah massa otot';
                        if (explanation === 'Based on low activity and fat loss goal') return 'Berdasarkan aktivitas rendah dan tujuan membakar lemak';
                        return explanation || 'Rekomendasi dibuat berdasarkan data yang Anda berikan.';
                      })()}
                    </AlertDescription>
                  </Box>
                </Alert>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Workout Recommendations */}
        {workout && (
          <Card>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size={{ base: "sm", md: "md" }}>🏋️ Program Latihan Harian</Heading>
                {workout.template_id && (
                  <Badge colorScheme="blue" fontSize="0.8em">
                    Template ID: {workout.template_id}
                  </Badge>
                )}
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={{ base: 4, md: 6 }}>
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={3}>📅 Jadwal Mingguan</Heading>
                    <SimpleGrid columns={2} spacing={3}>
                      <Stat>
                        <StatLabel>Jenis Olahraga</StatLabel>
                        <StatNumber>
                          {workout.workout_type === 'Full Body' ? 'Seluruh Tubuh' :
                           workout.workout_type === 'Upper/Lower Split' ? 'Split Atas/Bawah' :
                           workout.workout_type === 'Push/Pull/Legs' ? 'Dorong/Tarik/Kaki' :
                           workout.workout_type === 'Strength Training' ? 'Latihan Kekuatan' :
                           workout.workout_type || 'Latihan Kekuatan'}
                        </StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Hari per Minggu</StatLabel>
                        <StatNumber>{workout.days_per_week || 3} hari</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Set Harian</StatLabel>
                        <StatNumber>{workout.sets_per_exercise || 3} set</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Latihan per Sesi</StatLabel>
                        <StatNumber>{workout.exercises_per_session || 5} latihan</StatNumber>
                      </Stat>
                    </SimpleGrid>
                  </Box>

                  {workout.cardio_minutes_per_day && (
                    <Box>
                      <Heading size={{ base: "xs", md: "sm" }} mb={3}>🏃 Kardio Harian</Heading>
                      <SimpleGrid columns={2} spacing={3}>
                        <Stat>
                          <StatLabel>Durasi per Hari</StatLabel>
                          <StatNumber>{workout.cardio_minutes_per_day} menit</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Sesi per Hari</StatLabel>
                          <StatNumber>{workout.cardio_sessions_per_day || 1} sesi</StatNumber>
                        </Stat>
                      </SimpleGrid>
                    </Box>
                  )}
                </SimpleGrid>

                {(workout.workout_schedule || workout.schedule || workout.weekly_schedule) && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={3}>📋 Jadwal yang Disarankan</Heading>
                    <Box p={4} bg="gray.50" borderRadius="md">
                      <Text fontFamily="mono" fontSize={{ base: "md", md: "lg" }}>
                        {workout.workout_schedule || workout.schedule || workout.weekly_schedule}
                      </Text>
                      <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mt={2}>
                        W = Hari latihan, X = Hari istirahat
                      </Text>
                    </Box>
                  </Box>
                )}

                <Box>
                  <Heading size={{ base: "xs", md: "sm" }} mb={3}>📚 Penjelasan Jenis Latihan</Heading>
                  <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>🏋️ Full Body Workouts</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Melatih semua kelompok otot utama dalam setiap sesi. Cocok untuk pemula dan orang dengan waktu terbatas.</Text>
                    </Box>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>🔄 Upper/Lower Split</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Split latihan atas dan bawah tubuh. Cocok untuk atlet menengah dengan ketersediaan waktu sedang.</Text>
                    </Box>
                    <Box p={4} borderWidth={1} borderRadius="md">
                      <Text fontWeight="bold" mb={2} fontSize={{ base: "sm", md: "md" }}>💪 Push/Pull/Legs</Text>
                      <Text fontSize={{ base: "xs", md: "sm" }}>Split berdasarkan gerakan dorong, tarik, dan kaki. Cocok untuk atlet lanjutan.</Text>
                    </Box>
                  </SimpleGrid>
                </Box>
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Nutrition Recommendations */}
        {nutrition && dailyMacros && (
          <Card>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size={{ base: "sm", md: "md" }}>🍎 Program Nutrisi Harian</Heading>
                {nutrition.template_id && (
                  <Badge colorScheme="green" fontSize="0.8em">
                    Template ID: {nutrition.template_id}
                  </Badge>
                )}
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={{ base: 4, md: 6 }} align="stretch">
                <Box>
                  <Heading size={{ base: "xs", md: "sm" }} mb={4}>🎯 Target Harian Berdasarkan Template</Heading>
                  <SimpleGrid columns={{ base: 1, sm: 2, md: 4 }} spacing={{ base: 3, md: 4 }}>
                    <Stat>
                      <StatLabel>🔥 Kalori</StatLabel>
                      <StatNumber>{dailyMacros.calories} kkal</StatNumber>
                      <StatHelpText>
                        <Badge colorScheme="red">Defisit: {Math.round((1 - (dailyMacros.calories / (recommendations?.user_profile?.tdee || 2000))) * 100)}%</Badge>
                      </StatHelpText>
                    </Stat>
                    <Stat>
                      <StatLabel>🥩 Protein</StatLabel>
                      <StatNumber>{dailyMacros.protein}g</StatNumber>
                      <StatHelpText>{Math.round(dailyMacros.protein / parseFloat(userData.weight))}g/kg</StatHelpText>
                    </Stat>
                    <Stat>
                      <StatLabel>🍞 Karbohidrat</StatLabel>
                      <StatNumber>{dailyMacros.carbs}g</StatNumber>
                      <StatHelpText>{Math.round(dailyMacros.carbs / parseFloat(userData.weight))}g/kg</StatHelpText>
                    </Stat>
                    <Stat>
                      <StatLabel>🥑 Lemak</StatLabel>
                      <StatNumber>{dailyMacros.fat}g</StatNumber>
                      <StatHelpText>{Math.round(dailyMacros.fat / parseFloat(userData.weight))}g/kg</StatHelpText>
                    </Stat>
                  </SimpleGrid>
                </Box>

                {/* Food Suggestions with Calculated Portions */}
                {(!mealPlanLoading) && foodSuggestions.length > 0 && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={4}>
                      🍽️ {mealPlan ? 'Rencana Makan Berdasarkan Template' : 'Porsi Makanan Indonesia Berdasarkan Template'}
                    </Heading>
                    <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={4}>
                      {mealPlan ? 'Kombinasi makanan yang sudah diatur untuk mencapai target nutrisi harian Anda' : 'Porsi yang dihitung untuk mencapai target nutrisi harian Anda'}
                    </Text>
                    
                    <VStack spacing={4} align="stretch">
                      {foodSuggestions.map((meal, index) => (
                        <Box key={index} p={4} borderWidth={1} borderRadius="md" bg="gray.50">
                          <Heading size={{ base: "xs", md: "sm" }} mb={2}>{meal.meal}</Heading>
                          {meal.mealName && (
                            <Box mb={2}>
                              <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{meal.mealName}</Text>
                              {meal.description && <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600">{meal.description}</Text>}
                            </Box>
                          )}
                          <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={3}>
                            Target: {meal.targetCalories} kkal | {meal.targetProtein}g protein
                          </Text>
                          
                          <VStack spacing={2} align="stretch">
                            {meal.foods.map((food, foodIndex) => (
                              <Box key={foodIndex} p={3} bg="white" borderRadius="md" borderWidth={1}>
                                <Flex justify="space-between" align="center" mb={2}>
                                  <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{food.name}</Text>
                                  <Badge colorScheme="blue" fontSize={{ base: "2xs", md: "xs" }}>{food.grams}g</Badge>
                                </Flex>
                                <SimpleGrid columns={4} spacing={2}>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {food.actualCalories} kkal</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {food.actualProtein}g</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {food.actualCarbs}g</Text>
                                  <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {food.actualFat}g</Text>
                                </SimpleGrid>
                              </Box>
                            ))}
                          </VStack>
                        </Box>
                      ))}
                    </VStack>
                  </Box>
                )}
                
                {/* Backend AI-Generated Meal Plan */}
                {!backendMealPlanLoading && backendMealPlan && (
                  <Box>
                    <Heading size={{ base: "xs", md: "sm" }} mb={4}>🤖 AI-Generated Meal Plan</Heading>
                    <Text fontSize={{ base: "xs", md: "sm" }} color="gray.600" mb={4}>
                      Rencana makan yang dihasilkan oleh AI berdasarkan template nutrisi dan kebutuhan kalori Anda
                    </Text>
                    
                    {/* Daily Summary */}
                    {backendMealPlan.daily_summary && (
                      <Box mb={4} p={4} bg="blue.50" borderRadius="md">
                        <Heading size={{ base: "xs", md: "sm" }} mb={3}>📊 Ringkasan Harian</Heading>
                        <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3}>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Total Kalori</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_calories)} kkal</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Protein</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_protein)}g</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Karbohidrat</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_carbs)}g</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize={{ base: "2xs", md: "xs" }}>Lemak</StatLabel>
                            <StatNumber fontSize={{ base: "sm", md: "md" }}>{Math.round(backendMealPlan.daily_summary.total_fat)}g</StatNumber>
                          </Stat>
                        </SimpleGrid>
                      </Box>
                    )}

                    {/* Detailed Meals */}
                    {backendMealPlan.meals && (
                      <VStack spacing={4} align="stretch">
                        {Object.entries(backendMealPlan.meals).map(([mealType, mealData]) => (
                          <Box key={mealType} p={4} borderWidth={1} borderRadius="md" bg="green.50">
                            <Heading size={{ base: "xs", md: "sm" }} mb={3}>
                              {mealType === 'breakfast' && '🌅 Sarapan'}
                              {mealType === 'morning_snack' && '🍎 Snack Pagi'}
                              {mealType === 'lunch' && '🌞 Makan Siang'}
                              {mealType === 'afternoon_snack' && '🥜 Snack Sore'}
                              {mealType === 'dinner' && '🌙 Makan Malam'}
                              {mealType === 'evening_snack' && '🍪 Snack Malam'}
                            </Heading>
                            
                            <SimpleGrid columns={4} spacing={2} mb={3}>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {Math.round(mealData.calories)} kkal</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {Math.round(mealData.protein)}g</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {Math.round(mealData.carbs)}g</Text>
                              <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {Math.round(mealData.fat)}g</Text>
                            </SimpleGrid>

                            <VStack spacing={2} align="stretch">
                              {mealData.foods && mealData.foods.map((food, foodIndex) => (
                                <Box key={foodIndex} p={3} bg="white" borderRadius="md" borderWidth={1}>
                                  <Flex justify="space-between" align="center" mb={2}>
                                    <Text fontWeight="bold" fontSize={{ base: "xs", md: "sm" }}>{food.name}</Text>
                                    <Badge colorScheme="green" fontSize={{ base: "2xs", md: "xs" }}>{food.portion}g</Badge>
                                  </Flex>
                                  <SimpleGrid columns={4} spacing={2}>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🔥 {Math.round(food.calories)} kkal</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🥩 {Math.round(food.protein)}g</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🍞 {Math.round(food.carbs)}g</Text>
                                    <Text fontSize={{ base: "2xs", md: "xs" }}>🥑 {Math.round(food.fat)}g</Text>
                                  </SimpleGrid>
                                </Box>
                              ))}
                            </VStack>
                          </Box>
                        ))}
                      </VStack>
                    )}

                    {/* Shopping List */}
                    {backendMealPlan.shopping_list && backendMealPlan.shopping_list.length > 0 && (
                      <Box mt={4} p={4} bg="orange.50" borderRadius="md">
                        <Heading size={{ base: "xs", md: "sm" }} mb={3}>🛒 Daftar Belanja</Heading>
                        <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={2}>
                          {backendMealPlan.shopping_list.map((item, index) => (
                            <Flex key={index} justify="space-between" p={2} bg="white" borderRadius="md">
                              <Text fontSize={{ base: "xs", md: "sm" }}>{item.name}</Text>
                              <Text fontSize={{ base: "xs", md: "sm" }} fontWeight="bold">{item.total_amount}g</Text>
                            </Flex>
                          ))}
                        </SimpleGrid>
                      </Box>
                    )}
                  </Box>
                )}
                
                {/* Loading state for backend meal plan */}
                {backendMealPlanLoading && (
                  <Box p={6} textAlign="center" bg="blue.50" borderRadius="md">
                    <Heading size={{ base: "sm", md: "md" }} mb={2}>🤖 Memuat AI Meal Plan...</Heading>
                    <Text fontSize={{ base: "sm", md: "md" }}>Sedang menghasilkan rencana makan yang dipersonalisasi dengan AI...</Text>
                  </Box>
                )}
                
                {/* Loading state for meal plan */}
                {mealPlanLoading && (
                  <Box p={6} textAlign="center" bg="green.50" borderRadius="md">
                    <Heading size={{ base: "sm", md: "md" }} mb={2}>🔄 Memuat Rencana Makan...</Heading>
                    <Text fontSize={{ base: "sm", md: "md" }}>Sedang menyusun kombinasi makanan yang optimal untuk Anda...</Text>
                  </Box>
                )}
              </VStack>
            </CardBody>
          </Card>
        )}

        {/* Action Buttons */}
        <VStack spacing={4}>
          <HStack justify="center" spacing={{ base: 2, md: 4 }} w="full">
            <Button 
              variant="outline" 
              colorScheme="brand" 
              onClick={onBack}
              size={{ base: "sm", md: "md" }}
              w={{ base: "full", md: "auto" }}
            >
              ← Kembali ke Formulir
            </Button>
            <Button 
              colorScheme="brand" 
              onClick={onNewRecommendation}
              size={{ base: "sm", md: "md" }}
              w={{ base: "full", md: "auto" }}
            >
              🆕 Buat Rekomendasi Baru
            </Button>
          </HStack>
        </VStack>

        {/* Tips Section */}
        <Card>
          <CardHeader>
            <Heading size={{ base: "sm", md: "md" }}>💡 Tips Sukses dengan Porsi yang Tepat</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, sm: 2, lg: 4 }} spacing={{ base: 3, md: 4 }}>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>⚖️</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Gunakan timbangan digital untuk mengukur porsi makanan secara akurat</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>📱</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Catat asupan makanan harian di fitur progress tracking</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>🥗</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Variasikan sumber protein dan karbohidrat setiap harinya</Text>
              </Box>
              <Box textAlign="center" p={{ base: 3, md: 4 }}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mb={2}>💧</Text>
                <Text fontSize={{ base: "xs", md: "sm" }}>Minum 2-3 liter air putih setiap hari untuk metabolisme optimal</Text>
              </Box>
            </SimpleGrid>
          </CardBody>
        </Card>
      </VStack>
    </Box>
  );
};

export default RecommendationDisplay;