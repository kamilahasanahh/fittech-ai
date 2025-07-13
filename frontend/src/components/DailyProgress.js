import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Progress,
  Badge,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  Input,
  Textarea,
  Select,
  IconButton,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  useToast,
  useColorModeValue,
  Divider,
  Flex,
  Spacer,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Skeleton,
  SkeletonText,
  Tooltip,
  Wrap,
  WrapItem,
  Avatar,
  AvatarBadge,
  List,
  ListItem,
  ListIcon
} from '@chakra-ui/react';
import { 
  CheckCircleIcon, 
  TimeIcon, 
  StarIcon, 
  InfoIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ViewIcon,
  CloseIcon,
  EditIcon,
  CheckIcon
} from '@chakra-ui/icons';
import { db } from '../services/firebaseConfig';
import { recommendationService } from '../services/recommendationService';
import { apiService } from '../services/api';
import { doc, setDoc, getDoc } from 'firebase/firestore';


const DailyProgress = ({ user, onProgressUpdate, userProfile, currentRecommendation }) => {
  const [progressData, setProgressData] = useState({
    workout: false,
    nutrition: false,
    hydration: false,
    notes: '',
    mood: '',
    workoutRating: 0,
    energyLevel: 0,
    sleepQuality: 0,
    stressLevel: 0,
    recommendationEffectiveness: 0,
    weight: '',
    bodyFat: '',
    measurements: {
      chest: '',
      waist: '',
      arms: '',
      thighs: ''
    }
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [recommendationHistory, setRecommendationHistory] = useState([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState(null);
  const [showRecommendationHistory, setShowRecommendationHistory] = useState(false);
  const [mealPlanLoading, setMealPlanLoading] = useState(false);
  const [nutritionTemplates, setNutritionTemplates] = useState([]);
  const [workoutTemplates, setWorkoutTemplates] = useState([]);
  const [templatesLoading, setTemplatesLoading] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const today = new Date().toISOString().split('T')[0];

  // Color mode values
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const cardBg = useColorModeValue('white', 'gray.700');
  const textColor = useColorModeValue('gray.800', 'white');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');

  // Load nutrition templates
  const loadNutritionTemplates = useCallback(async () => {
    try {
      console.log('🔄 Loading nutrition templates...');
      const templates = await apiService.getNutritionTemplates();
      console.log('✅ Nutrition templates loaded:', templates);
      setNutritionTemplates(templates);
      return templates;
    } catch (error) {
      console.error('❌ Error loading nutrition templates:', error);
      // Don't show toast for template loading errors - use fallback data instead
      console.log('⚠️ Will use template data from recommendations as fallback');
      return [];
    }
  }, []);

  // Load workout templates
  const loadWorkoutTemplates = useCallback(async () => {
    try {
      console.log('🔄 Loading workout templates...');
      const templates = await apiService.getWorkoutTemplates();
      console.log('✅ Workout templates loaded:', templates);
      setWorkoutTemplates(templates);
      return templates;
    } catch (error) {
      console.error('❌ Error loading workout templates:', error);
      // Don't show toast for template loading errors - use fallback data instead
      console.log('⚠️ Will use template data from recommendations as fallback');
      return [];
    }
  }, []);

  // Load all templates
  const loadAllTemplates = useCallback(async () => {
    setTemplatesLoading(true);
    try {
      const results = await Promise.allSettled([
        loadNutritionTemplates(),
        loadWorkoutTemplates()
      ]);
      
      // Log results but don't throw errors - we'll use fallback data
      results.forEach((result, index) => {
        const templateType = index === 0 ? 'nutrition' : 'workout';
        if (result.status === 'fulfilled') {
          console.log(`✅ ${templateType} templates loaded successfully`);
        } else {
          console.log(`⚠️ ${templateType} templates failed to load, will use fallback data`);
        }
      });
    } catch (error) {
      console.error('❌ Error in loadAllTemplates:', error);
    } finally {
      setTemplatesLoading(false);
    }
  }, [loadNutritionTemplates, loadWorkoutTemplates]);

  // Find nutrition template by ID - with fallback to recommendation data
  const findNutritionTemplate = (templateId, recommendationData = null) => {
    // First try to find in loaded templates
    if (nutritionTemplates.length > 0) {
      console.log('🔍 Looking for nutrition template ID:', templateId, 'in loaded templates:', nutritionTemplates.length);
      const template = nutritionTemplates.find(template => template.template_id === parseInt(templateId));
      if (template) {
        console.log('✅ Found nutrition template in loaded templates:', template);
        return template;
      }
    }
    
    // Second: try to get from stored template data in recommendation
    if (recommendationData?.templateData?.nutritionTemplate) {
      const storedTemplate = recommendationData.templateData.nutritionTemplate;
      if (storedTemplate.template_id === parseInt(templateId)) {
        console.log('✅ Found nutrition template in stored template data:', storedTemplate);
        return storedTemplate;
      }
    }
    
    // Third: try to extract template data from recommendation
    if (recommendationData?.recommendations) {
      const nutritionRec = recommendationData.recommendations.nutrition_recommendation || 
                          recommendationData.recommendations.predictions?.nutrition_template;
      
      if (nutritionRec && nutritionRec.template_id === parseInt(templateId)) {
        console.log('✅ Found nutrition template data in recommendation:', nutritionRec);
        return nutritionRec;
      }
    }
    
    console.warn(`❌ Nutrition template ${templateId} not found. Available templates:`, nutritionTemplates.map(t => t?.template_id));
    return null;
  };

  // Find workout template by ID - with fallback to recommendation data
  const findWorkoutTemplate = (templateId, recommendationData = null) => {
    // First try to find in loaded templates
    if (workoutTemplates.length > 0) {
      console.log('🔍 Looking for workout template ID:', templateId, 'in loaded templates:', workoutTemplates.length);
      const template = workoutTemplates.find(template => template.template_id === parseInt(templateId));
      if (template) {
        console.log('✅ Found workout template in loaded templates:', template);
        return template;
      }
    }
    
    // Second: try to get from stored template data in recommendation
    if (recommendationData?.templateData?.workoutTemplate) {
      const storedTemplate = recommendationData.templateData.workoutTemplate;
      if (storedTemplate.template_id === parseInt(templateId)) {
        console.log('✅ Found workout template in stored template data:', storedTemplate);
        return storedTemplate;
      }
    }
    
    // Third: try to extract template data from recommendation
    if (recommendationData?.recommendations) {
      const workoutRec = recommendationData.recommendations.workout_recommendation || 
                        recommendationData.recommendations.predictions?.workout_template;
      
      if (workoutRec && workoutRec.template_id === parseInt(templateId)) {
        console.log('✅ Found workout template data in recommendation:', workoutRec);
        return workoutRec;
      }
    }
    
    console.warn(`❌ Workout template ${templateId} not found. Available templates:`, workoutTemplates.map(t => t?.template_id));
    return null;
  };

  // Get fitness goal info based on user's actual goal
  const getFitnessGoalInfo = () => {
    const goal = userProfile?.fitness_goal || 'Fat Loss';
    
    switch (goal) {
      case 'Fat Loss':
        return {
          icon: '📉',
          title: 'Menurunkan Berat Badan',
          description: 'Fokus pada pembakaran kalori dan pengurangan lemak tubuh',
          color: 'red'
        };
      case 'Muscle Gain':
        return {
          icon: '💪',
          title: 'Menambah Massa Otot',
          description: 'Membangun otot dengan latihan beban dan nutrisi yang tepat',
          color: 'blue'
        };
      case 'Maintenance':
        return {
          icon: '⚖️',
          title: 'Mempertahankan Bentuk Tubuh',
          description: 'Menjaga kondisi fisik dan berat badan yang sudah ideal',
          color: 'green'
        };
      default:
        return {
          icon: '🎯',
          title: 'Target Fitness',
          description: 'Mencapai tujuan kesehatan dan kebugaran optimal',
          color: 'purple'
        };
    }
  };

  // Get activity level info
  const getActivityLevelInfo = () => {
    const level = userProfile?.activity_level || 'Low Activity';
    
    switch (level) {
      case 'Low Activity':
        return {
          icon: '🚶‍♂️',
          title: 'Aktivitas Rendah',
          multiplier: '1.29',
          description: 'Olahraga ringan dengan intensitas rendah',
          color: 'orange'
        };
      case 'Moderate Activity':
        return {
          icon: '🏃‍♂️',
          title: 'Aktivitas Sedang',
          multiplier: '1.55',
          description: 'Olahraga teratur dengan intensitas sedang',
          color: 'yellow'
        };
      case 'High Activity':
        return {
          icon: '🏋️‍♂️',
          title: 'Aktivitas Tinggi',
          multiplier: '1.81',
          description: 'Olahraga intensif dengan frekuensi tinggi',
          color: 'green'
        };
      default:
        return {
          icon: '🎯',
          title: 'Level Aktivitas',
          multiplier: '1.0',
          description: 'Tingkat aktivitas belum ditentukan',
          color: 'gray'
        };
    }
  };

  // Load today's progress data
  const loadTodaysProgress = useCallback(async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      const progressDoc = await getDoc(progressRef);
      
      if (progressDoc.exists()) {
        const data = progressDoc.data();
        setProgressData({
          workout: data.workout || false,
          nutrition: data.nutrition || false,
          hydration: data.hydration || false,
          notes: data.notes || '',
          mood: data.mood || '',
          workoutRating: data.workoutRating || 0,
          energyLevel: data.energyLevel || 0,
          sleepQuality: data.sleepQuality || 0,
          stressLevel: data.stressLevel || 0,
          recommendationEffectiveness: data.recommendationEffectiveness || 0,
          weight: data.weight || '',
          bodyFat: data.bodyFat || '',
          measurements: data.measurements || {
            chest: '',
            waist: '',
            arms: '',
            thighs: ''
          }
        });
      }
    } catch (error) {
      console.error('Error loading progress:', error);
      toast({
        title: 'Error',
        description: 'Gagal memuat data progress hari ini',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setLoading(false);
    }
  }, [user, today, toast]);

  // Helper function to generate meal plan for historical recommendations
  const generateMealPlanForRecommendation = async (recommendation) => {
    // First check if the recommendation already has a meal plan stored directly
    if (recommendation.mealPlan) {
      console.log('✅ Meal plan already exists in recommendation object:', recommendation.id);
      return recommendation;
    }
    
    // Second, try to load the meal plan from Firebase (might have been saved after initial creation)
    try {
      const savedRecommendation = await recommendationService.getRecommendationById(recommendation.id);
      if (savedRecommendation && savedRecommendation.mealPlan) {
        console.log('✅ Found saved meal plan in Firebase for recommendation:', recommendation.id);
        return {
          ...recommendation,
          mealPlan: savedRecommendation.mealPlan
        };
      }
    } catch (error) {
      console.warn('❌ Could not retrieve recommendation from Firebase:', error);
    }
    
    // Third, check if meal plan exists in the original recommendations data
    const originalMealPlan = recommendation.recommendations?.meal_plan ||
                            recommendation.recommendations?.daily_meal_plan ||
                            recommendation.recommendations?.predictions?.meal_plan;
    
    if (originalMealPlan) {
      console.log('✅ Found original meal plan in recommendation data:', originalMealPlan);
      return {
        ...recommendation,
        mealPlan: originalMealPlan
      };
    }
    
    console.log('🔍 No existing meal plan found, will NOT generate new one to preserve original data');
    console.log('� Recommendation structure:', {
      hasMealPlan: !!recommendation.mealPlan,
      hasRecommendationsMealPlan: !!recommendation.recommendations?.meal_plan,
      hasRecommendationsDailyMealPlan: !!recommendation.recommendations?.daily_meal_plan,
      hasPredictionsMealPlan: !!recommendation.recommendations?.predictions?.meal_plan,
      recommendationKeys: Object.keys(recommendation.recommendations || {}),
      predictionsKeys: Object.keys(recommendation.recommendations?.predictions || {})
    });
    
    // Instead of generating a new meal plan, return with a message that original meal plan was not found
    return {
      ...recommendation,
      mealPlan: null,
      mealPlanStatus: 'original_not_found'
    };
  };

  // Load recommendation history
  const loadRecommendationHistory = useCallback(async () => {
    if (!user) return;
    
    try {
      const history = await recommendationService.getRecommendationHistory(user.uid, 20);
      setRecommendationHistory(history);
    } catch (error) {
      console.error('Error loading recommendation history:', error);
    }
  }, [user]);

  useEffect(() => {
    loadTodaysProgress();
    loadRecommendationHistory();
    loadAllTemplates();
  }, [loadTodaysProgress, loadRecommendationHistory, loadAllTemplates]);

  const saveProgress = async (newData) => {
    if (!user) return;
    
    setSaving(true);
    try {
      const progressRef = doc(db, 'userProgress', `${user.uid}_${today}`);
      await setDoc(progressRef, {
        ...newData,
        date: today,
        userId: user.uid,
        updatedAt: new Date()
      });

      onProgressUpdate && onProgressUpdate(newData);
      
      toast({
        title: 'Berhasil',
        description: 'Progress berhasil disimpan',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error saving progress:', error);
      toast({
        title: 'Error',
        description: 'Gagal menyimpan progress',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleGoalToggle = (goal) => {
    const newData = {
      ...progressData,
      [goal]: !progressData[goal]
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const handleNotesChange = (e) => {
    const newData = {
      ...progressData,
      notes: e.target.value
    };
    setProgressData(newData);
    // Debounce saving notes
    setTimeout(() => saveProgress(newData), 1000);
  };

  const handleFeedbackChange = (field, value) => {
    const newData = {
      ...progressData,
      [field]: value
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const handleMeasurementChange = (field, value) => {
    const newData = {
      ...progressData,
      measurements: {
        ...progressData.measurements,
        [field]: value
      }
    };
    setProgressData(newData);
    saveProgress(newData);
  };

  const getCompletionPercentage = () => {
    const goals = [progressData.workout, progressData.nutrition, progressData.hydration];
    const completed = goals.filter(Boolean).length;
    return Math.round((completed / goals.length) * 100);
  };

  const getMotivationalMessage = () => {
    const percentage = getCompletionPercentage();
    if (percentage === 100) {
      return "🎉 Luar biasa! Anda telah menyelesaikan semua target hari ini!";
    } else if (percentage >= 66) {
      return "💪 Bagus sekali! Tinggal sedikit lagi untuk mencapai target harian!";
    } else if (percentage >= 33) {
      return "👍 Terus semangat! Anda sudah di jalur yang tepat!";
    } else {
      return "🌟 Mulai hari ini dengan semangat! Setiap langkah kecil berarti!";
    }
  };

  // Helper functions for feedback
  const getMoodEmoji = (mood) => {
    const moodEmojis = {
      'excellent': '😄',
      'good': '🙂',
      'neutral': '😐',
      'bad': '😔',
      'terrible': '😢'
    };
    return moodEmojis[mood] || '😐';
  };

  const getRatingStars = (rating) => {
    return '⭐'.repeat(rating) + '☆'.repeat(5 - rating);
  };

  const getLevelLabel = (level) => {
    if (level >= 4) return 'Sangat Tinggi';
    if (level >= 3) return 'Tinggi';
    if (level >= 2) return 'Sedang';
    if (level >= 1) return 'Rendah';
    return 'Sangat Rendah';
  };

  const getEffectivenessLabel = (rating) => {
    if (rating >= 4) return 'Sangat Efektif';
    if (rating >= 3) return 'Efektif';
    if (rating >= 2) return 'Cukup Efektif';
    if (rating >= 1) return 'Kurang Efektif';
    return 'Tidak Efektif';
  };

  const fitnessGoalInfo = getFitnessGoalInfo();
  const activityLevelInfo = getActivityLevelInfo();
  const completionPercentage = getCompletionPercentage();

  if (loading) {
    return (
      <Box maxW="container.xl" mx="auto" p={{ base: 4, md: 6 }}>
        <VStack spacing={6} align="stretch">
          <Skeleton height="60px" />
          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
            <Skeleton height="200px" />
            <Skeleton height="200px" />
            <Skeleton height="200px" />
          </SimpleGrid>
          <Skeleton height="300px" />
        </VStack>
      </Box>
    );
  }

  return (
    <Box maxW="container.xl" mx="auto" p={{ base: 4, md: 6 }}>
      <VStack spacing={8} align="stretch">
        {/* Header Section */}
        <Box>
          <Heading size={{ base: "lg", md: "xl" }} mb={2}>
            📊 Progress Harian
          </Heading>
          <Text color={mutedTextColor} fontSize={{ base: "sm", md: "md" }}>
            {new Date().toLocaleDateString('id-ID', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </Text>
        </Box>

        {/* Recommendation History */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardHeader>
            <Heading size={{ base: "md", md: "lg" }}>📋 Riwayat Rekomendasi</Heading>
          </CardHeader>
          
          <CardBody>
            {recommendationHistory.length === 0 ? (
              <Alert status="info" borderRadius="md">
                <AlertIcon />
                <Box>
                  <AlertTitle>Belum ada riwayat</AlertTitle>
                  <AlertDescription>
                    Buat rekomendasi pertama Anda untuk memulai tracking!
                  </AlertDescription>
                </Box>
              </Alert>
            ) : (
              <VStack spacing={4} align="stretch">
                {recommendationHistory.slice(0, 5).map((recommendation, index) => (
                  <Card key={recommendation.id} variant="outline">
                    <CardBody>
                      <VStack align="stretch" spacing={4}>
                        {/* Header with date and status */}
                        <HStack justify="space-between">
                          <VStack align="start" spacing={1}>
                            <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>
                              📅 {new Date(recommendation.createdAt.seconds * 1000).toLocaleDateString('id-ID', {
                                weekday: 'long',
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                              })}
                            </Text>
                            <Text fontSize="xs" color={mutedTextColor}>
                              🕐 {new Date(recommendation.createdAt.seconds * 1000).toLocaleTimeString('id-ID', {
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </Text>
                          </VStack>
                          <HStack spacing={2}>
                            {recommendation.isActive && (
                              <Badge colorScheme="green" variant="subtle">
                                ✅ Aktif
                              </Badge>
                            )}
                            {index === 0 && !recommendation.isActive && (
                              <Badge colorScheme="blue" variant="subtle">
                                🆕 Terbaru
                              </Badge>
                            )}
                          </HStack>
                        </HStack>
                        
                        {/* User profile summary */}
                        <SimpleGrid columns={{ base: 2, sm: 4 }} spacing={3}>
                          <Stat>
                            <StatLabel fontSize="xs">Tujuan</StatLabel>
                            <StatNumber fontSize="sm">{recommendation.userData.fitness_goal}</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize="xs">Aktivitas</StatLabel>
                            <StatNumber fontSize="sm">{recommendation.userData.activity_level}</StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize="xs">BMI</StatLabel>
                            <StatNumber fontSize="sm">
                              {recommendation.recommendations.user_metrics?.bmi?.toFixed(1) || 
                               (recommendation.userData?.weight && recommendation.userData?.height ? 
                                 (recommendation.userData.weight / ((recommendation.userData.height / 100) ** 2)).toFixed(1) : 
                                 'N/A')}
                            </StatNumber>
                          </Stat>
                          <Stat>
                            <StatLabel fontSize="xs">TDEE</StatLabel>
                            <StatNumber fontSize="sm">
                              {Math.round(
                                recommendation.userProfile?.tdee || 
                                recommendation.recommendations?.user_profile?.tdee ||
                                recommendation.recommendations?.user_metrics?.tdee || 
                                recommendation.recommendations?.user_metrics?.total_daily_energy_expenditure ||
                                0
                              )} kkal
                            </StatNumber>
                          </Stat>
                        </SimpleGrid>
                        
                        {/* View Detail Button */}
                        <Button
                          size="sm"
                          variant="outline"
                          leftIcon={<ViewIcon />}
                          onClick={async () => {
                            console.log('🔍 Opening recommendation details:', recommendation);
                            
                            // Check if meal plan needs to be generated
                            const updatedRecommendation = await generateMealPlanForRecommendation(recommendation);
                            setSelectedRecommendation(updatedRecommendation);
                            onOpen();
                          }}
                        >
                          Lihat Detail Lengkap
                        </Button>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
                
                {/* Show more button if there are more than 5 recommendations */}
                {recommendationHistory.length > 5 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowRecommendationHistory(!showRecommendationHistory)}
                  >
                    {showRecommendationHistory ? 'Tampilkan Lebih Sedikit' : `Tampilkan ${recommendationHistory.length - 5} Rekomendasi Lainnya`}
                  </Button>
                )}
                
                {showRecommendationHistory && recommendationHistory.length > 5 && (
                  <VStack spacing={4} align="stretch">
                    {recommendationHistory.slice(5).map((recommendation, index) => (
                      <Card key={recommendation.id} variant="outline">
                        <CardBody>
                          <VStack align="stretch" spacing={3}>
                            <HStack justify="space-between">
                              <VStack align="start" spacing={1}>
                                <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>
                                  📅 {new Date(recommendation.createdAt.seconds * 1000).toLocaleDateString('id-ID', {
                                    weekday: 'long',
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                  })}
                                </Text>
                                <Text fontSize="xs" color={mutedTextColor}>
                                  🕐 {new Date(recommendation.createdAt.seconds * 1000).toLocaleTimeString('id-ID', {
                                    hour: '2-digit',
                                    minute: '2-digit'
                                  })}
                                </Text>
                              </VStack>
                            </HStack>
                            
                            <SimpleGrid columns={{ base: 1, sm: 3 }} spacing={3}>
                              <Stat>
                                <StatLabel fontSize="xs">Tujuan</StatLabel>
                                <StatNumber fontSize="sm">{recommendation.userData.fitness_goal}</StatNumber>
                              </Stat>
                              <Stat>
                                <StatLabel fontSize="xs">Aktivitas</StatLabel>
                                <StatNumber fontSize="sm">{recommendation.userData.activity_level}</StatNumber>
                              </Stat>
                              <Stat>
                                <StatLabel fontSize="xs">BMI</StatLabel>
                                <StatNumber fontSize="sm">
                                  {recommendation.recommendations.user_metrics?.bmi?.toFixed(1) || 
                                   (recommendation.userData?.weight && recommendation.userData?.height ? 
                                     (recommendation.userData.weight / ((recommendation.userData.height / 100) ** 2)).toFixed(1) : 
                                     'N/A')}
                                </StatNumber>
                              </Stat>
                            </SimpleGrid>
                            
                            <Button
                              size="sm"
                              variant="outline"
                              leftIcon={<ViewIcon />}
                              onClick={async () => {
                                console.log('🔍 Opening recommendation details:', recommendation);
                                
                                const updatedRecommendation = await generateMealPlanForRecommendation(recommendation);
                                setSelectedRecommendation(updatedRecommendation);
                                onOpen();
                              }}
                            >
                              Lihat Detail
                            </Button>
                          </VStack>
                        </CardBody>
                      </Card>
                    ))}
                  </VStack>
                )}
              </VStack>
            )}
          </CardBody>
        </Card>

        {/* Tips Section */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardHeader>
            <Heading size={{ base: "md", md: "lg" }}>💡 Tips Hari Ini</Heading>
          </CardHeader>
          <CardBody>
            <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
              <VStack align="start" spacing={3} p={4} bg="blue.50" borderRadius="md">
                <Text fontSize="2xl">🕐</Text>
                <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>
                  Waktu Terbaik Berolahraga
                </Text>
                <Text fontSize="sm" color={mutedTextColor}>
                  Pagi hari (06:00-08:00) atau sore (16:00-18:00) adalah waktu optimal untuk berolahraga.
                </Text>
              </VStack>
              
              <VStack align="start" spacing={3} p={4} bg="green.50" borderRadius="md">
                <Text fontSize="2xl">🥗</Text>
                <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>
                  Nutrisi Seimbang
                </Text>
                <Text fontSize="sm" color={mutedTextColor}>
                  Pastikan setiap makanan mengandung protein, karbohidrat, dan sayuran untuk nutrisi optimal.
                </Text>
              </VStack>
              
              <VStack align="start" spacing={3} p={4} bg="purple.50" borderRadius="md">
                <Text fontSize="2xl">😴</Text>
                <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>
                  Istirahat Cukup
                </Text>
                <Text fontSize="sm" color={mutedTextColor}>
                  Tidur 7-9 jam per malam sangat penting untuk pemulihan otot dan metabolisme yang baik.
                </Text>
              </VStack>
            </SimpleGrid>
          </CardBody>
        </Card>
      </VStack>

      {/* Recommendation Detail Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size={{ base: "full", md: "xl" }}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>📋 Detail Rekomendasi</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {selectedRecommendation && (
              <VStack spacing={6} align="stretch">
                {/* Complete User Profile */}
                <Card variant="outline">
                  <CardHeader>
                    <Heading size="md">👤 Profil Anda</Heading>
                  </CardHeader>
                  <CardBody>
                    <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                      <Stat>
                        <StatLabel>Usia</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.age} tahun</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Jenis Kelamin</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.gender === 'Male' ? 'Pria' : 'Wanita'}</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Tinggi</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.height} cm</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Berat</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.weight} kg</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Tujuan</StatLabel>
                        <StatNumber>
                          {selectedRecommendation.userData.fitness_goal === 'Fat Loss' ? 'Membakar Lemak' :
                           selectedRecommendation.userData.fitness_goal === 'Muscle Gain' ? 'Menambah Otot' :
                           selectedRecommendation.userData.fitness_goal === 'Maintenance' ? 'Mempertahankan' :
                           selectedRecommendation.userData.fitness_goal}
                        </StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Tingkat Aktivitas</StatLabel>
                        <StatNumber>
                          {selectedRecommendation.userData.activity_level === 'High Activity' ? 'Aktivitas Tinggi' :
                           selectedRecommendation.userData.activity_level === 'Moderate Activity' ? 'Aktivitas Sedang' :
                           selectedRecommendation.userData.activity_level === 'Low Activity' ? 'Aktivitas Rendah' :
                           selectedRecommendation.userData.activity_level}
                        </StatNumber>
                      </Stat>
                    </SimpleGrid>
                  </CardBody>
                </Card>

                {/* Body Analysis */}
                {(selectedRecommendation.userProfile || selectedRecommendation.recommendations?.user_profile || selectedRecommendation.recommendations?.user_metrics) && (
                  <Card variant="outline">
                    <CardHeader>
                      <Heading size="md">📊 Analisis Tubuh Anda</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>BMI</StatLabel>
                          <StatNumber>
                            {(selectedRecommendation.userProfile?.bmi || 
                              selectedRecommendation.recommendations?.user_metrics?.bmi ||
                              (selectedRecommendation.userData?.weight && selectedRecommendation.userData?.height ? 
                                (selectedRecommendation.userData.weight / ((selectedRecommendation.userData.height / 100) ** 2)).toFixed(1) : 
                                'N/A'))}
                          </StatNumber>
                          <StatHelpText>
                            {selectedRecommendation.userProfile?.bmi_category || 
                             selectedRecommendation.recommendations?.user_metrics?.bmi_category ||
                             'Kategori BMI'}
                          </StatHelpText>
                        </Stat>
                        <Stat>
                          <StatLabel>BMR</StatLabel>
                          <StatNumber>
                            {Math.round(
                              selectedRecommendation.userProfile?.bmr || 
                              selectedRecommendation.recommendations?.user_profile?.bmr ||
                              selectedRecommendation.recommendations?.user_metrics?.bmr || 
                              selectedRecommendation.recommendations?.user_metrics?.basal_metabolic_rate ||
                              0
                            )} kkal
                          </StatNumber>
                          <StatHelpText>Kalori Basal</StatHelpText>
                        </Stat>
                        <Stat>
                          <StatLabel>TDEE</StatLabel>
                          <StatNumber>
                            {Math.round(
                              selectedRecommendation.userProfile?.tdee || 
                              selectedRecommendation.recommendations?.user_profile?.tdee ||
                              selectedRecommendation.recommendations?.user_metrics?.tdee || 
                              selectedRecommendation.recommendations?.user_metrics?.total_daily_energy_expenditure ||
                              0
                            )} kkal
                          </StatNumber>
                          <StatHelpText>Total Pengeluaran Energi</StatHelpText>
                        </Stat>
                        <Stat>
                          <StatLabel>Kategori BMI</StatLabel>
                          <StatNumber>
                            {(() => {
                              const category = selectedRecommendation.userProfile?.bmi_category || 
                                              selectedRecommendation.recommendations?.user_metrics?.bmi_category ||
                                              'Unknown';
                              
                              return category === 'Normal' ? 'Normal' :
                                     category === 'Overweight' ? 'Kelebihan Berat' :
                                     category === 'Obese' ? 'Obesitas' :
                                     category === 'Underweight' ? 'Kekurangan Berat' :
                                     category;
                            })()}
                          </StatNumber>
                        </Stat>
                      </SimpleGrid>
                    </CardBody>
                  </Card>
                )}

                {/* AI Confidence */}
                {selectedRecommendation.recommendations.enhanced_confidence && (
                  <Card variant="outline">
                    <CardHeader>
                      <Heading size="md">🎯 Tingkat Kepercayaan AI</Heading>
                    </CardHeader>
                    <CardBody>
                      <VStack spacing={4} align="stretch">
                        <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                          <Stat>
                            <StatLabel>Kepercayaan Keseluruhan</StatLabel>
                            <StatNumber>{Math.round(selectedRecommendation.recommendations.enhanced_confidence.confidence_score * 100)}%</StatNumber>
                            <StatHelpText>{selectedRecommendation.recommendations.enhanced_confidence.confidence_level}</StatHelpText>
                          </Stat>
                          {selectedRecommendation.recommendations.model_confidence && (
                            <>
                              <Stat>
                                <StatLabel>Kepercayaan Nutrisi</StatLabel>
                                <StatNumber>{Math.round(selectedRecommendation.recommendations.model_confidence.nutrition_confidence * 100)}%</StatNumber>
                              </Stat>
                              <Stat>
                                <StatLabel>Kepercayaan Workout</StatLabel>
                                <StatNumber>{Math.round(selectedRecommendation.recommendations.model_confidence.workout_confidence * 100)}%</StatNumber>
                              </Stat>
                            </>
                          )}
                        </SimpleGrid>
                        <Text fontSize="sm" color="gray.600">
                          Level Kepercayaan: {selectedRecommendation.recommendations.enhanced_confidence.explanation}
                        </Text>
                      </VStack>
                    </CardBody>
                  </Card>
                )}

                {/* Workout Recommendation Section */}
                {(selectedRecommendation.recommendations.workout_recommendation || 
                  selectedRecommendation.recommendations.predictions?.workout_template) && (
                  <Card variant="outline">
                    <CardHeader>
                      <HStack justify="space-between">
                        <Heading size="md">🏋️ Program Latihan Harian</Heading>
                        {(selectedRecommendation.recommendations.workout_recommendation?.template_id || 
                          selectedRecommendation.recommendations.predictions?.workout_template?.template_id) && (
                          <Badge colorScheme="blue" fontSize="0.8em">
                            Template ID: {selectedRecommendation.recommendations.workout_recommendation?.template_id || 
                                         selectedRecommendation.recommendations.predictions?.workout_template?.template_id}
                          </Badge>
                        )}
                      </HStack>
                    </CardHeader>
                    <CardBody>
                      <VStack spacing={4} align="stretch">
                        <Text fontWeight="bold" fontSize="md">📅 Jadwal Mingguan</Text>
                        
                        {(() => {
                          const workout = selectedRecommendation.recommendations.workout_recommendation || 
                                         selectedRecommendation.recommendations.predictions?.workout_template;
                          
                          if (!workout) return <Text>No workout data available</Text>;
                          
                          return (
                            <>
                              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                <Stat>
                                  <StatLabel>Jenis Olahraga</StatLabel>
                                  <StatNumber fontSize="sm">
                                    {workout.workout_type === 'Upper/Lower Split' ? 'Split Atas/Bawah' :
                                     workout.workout_type === 'Push/Pull/Legs' ? 'Push/Pull/Legs' :
                                     workout.workout_type === 'Full Body' ? 'Full Body' :
                                     workout.workout_type || 'Tidak tersedia'}
                                  </StatNumber>
                                </Stat>
                                <Stat>
                                  <StatLabel>Hari per Minggu</StatLabel>
                                  <StatNumber fontSize="sm">{workout.days_per_week || 'Tidak tersedia'} hari</StatNumber>
                                </Stat>
                                <Stat>
                                  <StatLabel>Set Harian</StatLabel>
                                  <StatNumber fontSize="sm">{workout.sets_per_exercise || 'Tidak tersedia'} set</StatNumber>
                                </Stat>
                                <Stat>
                                  <StatLabel>Latihan per Sesi</StatLabel>
                                  <StatNumber fontSize="sm">{workout.exercises_per_session || 'Tidak tersedia'} latihan</StatNumber>
                                </Stat>
                              </SimpleGrid>

                              {/* Cardio Section */}
                              {workout.cardio_minutes_per_day > 0 && (
                                <>
                                  <Text fontWeight="bold" fontSize="md" mt={4}>🏃 Kardio Harian</Text>
                                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Durasi per Hari</StatLabel>
                                      <StatNumber fontSize="sm">{workout.cardio_minutes_per_day} menit</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Sesi per Hari</StatLabel>
                                      <StatNumber fontSize="sm">{workout.cardio_sessions_per_day || 1} sesi</StatNumber>
                                    </Stat>
                                  </SimpleGrid>
                                </>
                              )}

                              {/* Workout schedule */}
                              {(workout.workout_schedule || workout.schedule || workout.weekly_schedule) && (
                                <Box mt={4}>
                                  <Text fontWeight="bold" mb={2}>📋 Jadwal yang Disarankan</Text>
                                  <Box p={3} bg="blue.50" borderRadius="md" border="1px" borderColor="blue.200">
                                    <Text fontFamily="mono" fontSize="lg" textAlign="center" fontWeight="bold">
                                      {workout.workout_schedule || workout.schedule || workout.weekly_schedule}
                                    </Text>
                                    <Text fontSize="xs" color="gray.600" mt={2} textAlign="center">
                                      W = Hari latihan, X = Hari istirahat, A/B/C = Jenis latihan
                                    </Text>
                                  </Box>
                                </Box>
                              )}

                              {/* Workout Type Explanations */}
                              <Box mt={4}>
                                <Text fontWeight="bold" mb={3}>📚 Penjelasan Jenis Latihan</Text>
                                <VStack spacing={3} align="stretch">
                                  <Box p={3} bg="gray.50" borderRadius="md">
                                    <Text fontWeight="bold" fontSize="sm">🏋️ Full Body Workouts</Text>
                                    <Text fontSize="xs" color="gray.600" mt={1}>
                                      Melatih semua kelompok otot utama dalam setiap sesi. Cocok untuk pemula dan orang dengan waktu terbatas.
                                    </Text>
                                  </Box>
                                  <Box p={3} bg="gray.50" borderRadius="md">
                                    <Text fontWeight="bold" fontSize="sm">🔄 Upper/Lower Split</Text>
                                    <Text fontSize="xs" color="gray.600" mt={1}>
                                      Split latihan atas dan bawah tubuh. Cocok untuk atlet menengah dengan ketersediaan waktu sedang.
                                    </Text>
                                  </Box>
                                  <Box p={3} bg="gray.50" borderRadius="md">
                                    <Text fontWeight="bold" fontSize="sm">💪 Push/Pull/Legs</Text>
                                    <Text fontSize="xs" color="gray.600" mt={1}>
                                      Split berdasarkan gerakan dorong, tarik, dan kaki. Cocok untuk atlet lanjutan.
                                    </Text>
                                  </Box>
                                </VStack>
                              </Box>
                            </>
                          );
                        })()}
                      </VStack>
                    </CardBody>
                  </Card>
                )}

                {/* Nutrition Recommendation Section - Always show if we have recommendations */}
                {selectedRecommendation.recommendations && (
                  <Card variant="outline">
                    <CardHeader>
                      <HStack justify="space-between">
                        <Heading size="md">🍎 Program Nutrisi Harian</Heading>
                        {(() => {
                          const templateId = selectedRecommendation.recommendations.nutrition_recommendation?.template_id || 
                                           selectedRecommendation.recommendations.predictions?.nutrition_template?.template_id ||
                                           selectedRecommendation.recommendations.predicted_nutrition_template_id;
                          return templateId && (
                            <Badge colorScheme="green" fontSize="0.8em">
                              Template ID: {templateId}
                            </Badge>
                          );
                        })()}
                      </HStack>
                    </CardHeader>
                    <CardBody>
                      <VStack spacing={4} align="stretch">
                        <Text fontWeight="bold" fontSize="md">🎯 Target Harian Berdasarkan Template</Text>
                        
                        {(() => {
                          // Debug log the recommendations structure
                          console.log('🔍 Recommendations structure:', selectedRecommendation.recommendations);
                          
                          const nutrition = selectedRecommendation.recommendations.nutrition_recommendation || 
                                           selectedRecommendation.recommendations.predictions?.nutrition_template ||
                                           selectedRecommendation.recommendations.nutrition_template;
                          
                          // Get template ID and find the actual template
                          const templateId = nutrition?.template_id || 
                                           selectedRecommendation.recommendations.predicted_nutrition_template_id;
                          
                          const nutritionTemplate = findNutritionTemplate(templateId, selectedRecommendation);
                          
                          // Extract user metrics for calculations with better fallback
                          const userWeight = selectedRecommendation.userData?.weight || 70;
                          const userProfile = selectedRecommendation.userProfile || 
                                             selectedRecommendation.recommendations?.user_profile ||
                                             selectedRecommendation.recommendations?.user_metrics;
                          const tdee = userProfile?.tdee || 
                                      userProfile?.total_daily_energy_expenditure || 
                                      2000;
                          
                          // Calculate target values using template data if available
                          let targetCalories, targetProtein, targetCarbs, targetFat, deficitInfo;
                          
                          if (nutritionTemplate) {
                            // Use template multipliers
                            targetCalories = Math.round(tdee * nutritionTemplate.caloric_intake_multiplier);
                            targetProtein = Math.round(userWeight * nutritionTemplate.protein_per_kg);
                            targetCarbs = Math.round(userWeight * nutritionTemplate.carbs_per_kg);
                            targetFat = Math.round(userWeight * nutritionTemplate.fat_per_kg);
                            
                            // Calculate deficit/surplus percentage
                            const calorieChange = Math.round((nutritionTemplate.caloric_intake_multiplier - 1) * 100);
                            if (calorieChange < 0) {
                              deficitInfo = `Defisit: ${Math.abs(calorieChange)}%`;
                            } else if (calorieChange > 0) {
                              deficitInfo = `Surplus: ${calorieChange}%`;
                            } else {
                              deficitInfo = 'Maintenance';
                            }
                          } else if (nutrition) {
                            // Fallback: use direct values from nutrition data and try to calculate from recommendation
                            targetCalories = nutrition.target_calories || nutrition.daily_calories || 
                                           nutrition.scaled_calories || Math.round(tdee * 0.8); // Default 20% deficit
                            targetProtein = nutrition.target_protein || nutrition.protein_grams || 
                                          nutrition.scaled_protein || Math.round(userWeight * 2); // Default 2g/kg
                            targetCarbs = nutrition.target_carbs || nutrition.carbs_grams || 
                                        nutrition.scaled_carbs || Math.round(userWeight * 3); // Default 3g/kg
                            targetFat = nutrition.target_fat || nutrition.fat_grams || 
                                      nutrition.scaled_fat || Math.round(userWeight * 1); // Default 1g/kg
                            
                            // Try to determine deficit/surplus from calorie comparison
                            const estimatedDeficit = Math.round(((targetCalories - tdee) / tdee) * 100);
                            if (estimatedDeficit < -5) {
                              deficitInfo = `Defisit: ${Math.abs(estimatedDeficit)}%`;
                            } else if (estimatedDeficit > 5) {
                              deficitInfo = `Surplus: ${estimatedDeficit}%`;
                            } else {
                              deficitInfo = 'Maintenance';
                            }
                          } else {
                            // Last resort: use basic calculations
                            targetCalories = Math.round(tdee * 0.8); // Default 20% deficit
                            targetProtein = Math.round(userWeight * 2);
                            targetCarbs = Math.round(userWeight * 3);
                            targetFat = Math.round(userWeight * 1);
                            deficitInfo = 'Estimasi (Template tidak tersedia)';
                          }
                          
                          return (
                            <>
                              <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4}>
                                <Stat>
                                  <StatLabel>🔥 Kalori</StatLabel>
                                  <StatNumber fontSize="sm">
                                    {Math.round(targetCalories || 0)} kkal
                                  </StatNumber>
                                  <StatHelpText fontSize="xs">
                                    {deficitInfo}
                                  </StatHelpText>
                                </Stat>
                                <Stat>
                                  <StatLabel>🥩 Protein</StatLabel>
                                  <StatNumber fontSize="sm">
                                    {Math.round(targetProtein || 0)}g
                                  </StatNumber>
                                  <StatHelpText fontSize="xs">
                                    {nutritionTemplate ? `${nutritionTemplate.protein_per_kg}g/kg` : 'N/A'}
                                  </StatHelpText>
                                </Stat>
                                <Stat>
                                  <StatLabel>🍞 Karbohidrat</StatLabel>
                                  <StatNumber fontSize="sm">
                                    {Math.round(targetCarbs || 0)}g
                                  </StatNumber>
                                  <StatHelpText fontSize="xs">
                                    {nutritionTemplate ? `${nutritionTemplate.carbs_per_kg}g/kg` : 'N/A'}
                                  </StatHelpText>
                                </Stat>
                                <Stat>
                                  <StatLabel>🥑 Lemak</StatLabel>
                                  <StatNumber fontSize="sm">
                                    {Math.round(targetFat || 0)}g
                                  </StatNumber>
                                  <StatHelpText fontSize="xs">
                                    {nutritionTemplate ? `${nutritionTemplate.fat_per_kg}g/kg` : 'N/A'}
                                  </StatHelpText>
                                </Stat>
                              </SimpleGrid>

                              {/* Template Information */}
                              {nutritionTemplate && (
                                <Box mt={4} p={3} bg="blue.50" borderRadius="md" border="1px" borderColor="blue.200">
                                  <Text fontWeight="bold" fontSize="sm" mb={2}>📋 Informasi Template</Text>
                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={3}>
                                    <Text fontSize="xs">
                                      <Text as="span" fontWeight="semibold">Tujuan:</Text> {nutritionTemplate.goal}
                                    </Text>
                                    <Text fontSize="xs">
                                      <Text as="span" fontWeight="semibold">Kategori BMI:</Text> {nutritionTemplate.bmi_category}
                                    </Text>
                                    <Text fontSize="xs">
                                      <Text as="span" fontWeight="semibold">Multiplier Kalori:</Text> {nutritionTemplate.caloric_intake_multiplier}x
                                    </Text>
                                  </SimpleGrid>
                                </Box>
                              )}

                              {/* Rest of the meal plan section remains the same */}

                              {/* Meal Plan Section - Enhanced to show detailed meal plan */}
                              <Box mt={4}>
                                <Text fontWeight="bold" mb={3}>🍽️ Rencana Makan Berdasarkan Template</Text>
                                <Text fontSize="sm" color="gray.600" mb={4}>
                                  Kombinasi makanan yang sudah diatur untuk mencapai target nutrisi harian Anda
                                </Text>
                                
                                {/* Check if meal plan exists */}
                                {(selectedRecommendation.mealPlan || mealPlanLoading) ? (
                                  mealPlanLoading ? (
                                    <Alert status="info" borderRadius="md">
                                      <AlertIcon />
                                      <Box>
                                        <AlertTitle fontSize="sm">Sedang Menyiapkan Rencana Makan...</AlertTitle>
                                        <AlertDescription fontSize="xs">
                                          Mohon tunggu, sistem sedang membuat rencana makan yang disesuaikan dengan kebutuhan nutrisi Anda.
                                        </AlertDescription>
                                      </Box>
                                    </Alert>
                                  ) : (
                                    <VStack spacing={4} align="stretch">
                                      {/* Handle meal plan data structure */}
                                      {(() => {
                                        const mealPlan = selectedRecommendation.mealPlan;
                                        
                                        if (!mealPlan) return null;
                                        
                                        // Check different possible structures
                                        const dailyPlan = mealPlan.daily_meal_plan || mealPlan.meals || mealPlan;
                                        
                                        console.log('🍽️ Displaying meal plan:', dailyPlan);
                                        
                                        // Helper function to render meal
                                        const renderMeal = (mealKey, mealData, bgColor, borderColor, icon, title) => {
                                          if (!mealData || !mealData.foods || mealData.foods.length === 0) return null;
                                          
                                          const mealName = mealData.meal_name || mealData.name || title;
                                          const description = mealData.description || '';
                                          const calories = mealData.scaled_calories || mealData.total_calories || 0;
                                          const protein = mealData.scaled_protein || mealData.total_protein || 0;
                                          const foods = mealData.foods || [];
                                          
                                          return (
                                            <Box key={mealKey} p={4} bg={bgColor} borderRadius="md" border="1px" borderColor={borderColor}>
                                              <Text fontWeight="bold" fontSize="sm" mb={2}>{icon} {title}</Text>
                                              <Text fontSize="sm" fontWeight="semibold">{mealName}</Text>
                                              {description && <Text fontSize="xs" color="gray.600" mb={2}>{description}</Text>}
                                              <Text fontSize="xs" color="blue.600" mb={3}>
                                                Target: {Math.round(calories)} kkal | {Math.round(protein)}g protein
                                              </Text>
                                              
                                              {/* Food items */}
                                              <VStack spacing={2} align="stretch">
                                                {foods.map((food, index) => (
                                                  <HStack key={index} justify="space-between" p={2} bg="white" borderRadius="sm" fontSize="xs">
                                                    <VStack align="start" spacing={0}>
                                                      <Text fontWeight="semibold">{food.nama || food.name || 'Unknown Food'}</Text>
                                                      <Text color="gray.600">{Math.round(food.amount || food.quantity || 0)}g</Text>
                                                    </VStack>
                                                    <HStack spacing={3}>
                                                      <Text>🔥 {Math.round(food.calories || 0)} kkal</Text>
                                                      <Text>🥩 {Math.round(food.protein || 0)}g</Text>
                                                      <Text>🍞 {Math.round(food.carbs || food.carbohydrates || 0)}g</Text>
                                                      <Text>🥑 {Math.round(food.fat || food.fats || 0)}g</Text>
                                                    </HStack>
                                                  </HStack>
                                                ))}
                                              </VStack>
                                            </Box>
                                          );
                                        };
                                        
                                        return (
                                          <>
                                            {/* Sarapan */}
                                            {renderMeal('sarapan', dailyPlan.sarapan || dailyPlan.breakfast, 'orange.50', 'orange.200', '🌅', 'Sarapan')}
                                            
                                            {/* Makan Siang */}
                                            {renderMeal('makan_siang', dailyPlan.makan_siang || dailyPlan.lunch, 'yellow.50', 'yellow.200', '☀️', 'Makan Siang')}
                                            
                                            {/* Makan Malam */}
                                            {renderMeal('makan_malam', dailyPlan.makan_malam || dailyPlan.dinner, 'purple.50', 'purple.200', '🌙', 'Makan Malam')}
                                            
                                            {/* Snack */}
                                            {renderMeal('snack', dailyPlan.snack, 'green.50', 'green.200', '🍪', 'Camilan')}
                                          </>
                                        );
                                      })()}
                                    </VStack>
                                  )
                                ) : (
                                  /* Show that meal plan will be generated or was not found */
                                  selectedRecommendation.mealPlanStatus === 'original_not_found' ? (
                                    <Alert status="warning" borderRadius="md">
                                      <AlertIcon />
                                      <Box>
                                        <AlertTitle fontSize="sm">Rencana Makan Asli Tidak Ditemukan</AlertTitle>
                                        <AlertDescription fontSize="xs">
                                          Rencana makan asli dari rekomendasi ini tidak tersimpan. 
                                          Data nutrisi dan template masih tersedia untuk referensi.
                                        </AlertDescription>
                                      </Box>
                                    </Alert>
                                  ) : (
                                    <Alert status="info" borderRadius="md">
                                      <AlertIcon />
                                      <Box>
                                        <AlertTitle fontSize="sm">Rencana Makan Akan Dibuat</AlertTitle>
                                        <AlertDescription fontSize="xs">
                                          Rencana makan detail akan dibuat otomatis berdasarkan template nutrisi dan target kalori Anda. 
                                          Sistem akan mengatur kombinasi makanan yang seimbang untuk mencapai target harian.
                                        </AlertDescription>
                                      </Box>
                                    </Alert>
                                  )
                                )}
                              </Box>

                              {/* Additional nutrition details */}
                              {nutrition.meal_plan && (
                                <Box mt={4}>
                                  <Text fontWeight="bold" mb={2}>🍽️ Rencana Makan Detail:</Text>
                                  <Box p={3} bg="gray.50" borderRadius="md">
                                    <Text fontSize="sm">{nutrition.meal_plan}</Text>
                                  </Box>
                                </Box>
                              )}

                              {nutrition.dietary_notes && (
                                <Box>
                                  <Text fontWeight="bold" mb={2}>📝 Catatan Diet:</Text>
                                  <Box p={3} bg="gray.50" borderRadius="md">
                                    <Text fontSize="sm">{nutrition.dietary_notes}</Text>
                                  </Box>
                                </Box>
                              )}
                            </>
                          );
                        })()}
                      </VStack>
                    </CardBody>
                  </Card>
                )}
              </VStack>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default DailyProgress;
