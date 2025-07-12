import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Card,
  CardBody,
  CardHeader,
  SimpleGrid,
  Select,
  useToast,
  useColorModeValue,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Divider,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Progress,
  IconButton,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Skeleton,
  SkeletonText,
  Wrap,
  WrapItem,
  Tooltip
} from '@chakra-ui/react';
import { 
  CheckIcon, 
  CloseIcon, 
  InfoIcon, 
  StarIcon,
  ArrowForwardIcon,
  ViewIcon
} from '@chakra-ui/icons';
import { saveRecommendationFeedback, getImprovedRecommendation, recommendationService } from '../services/recommendationService';
import { useNavigate } from 'react-router-dom';

const RecommendationFeedback = ({ currentRecommendation, userProfile, onRecommendationUpdate, user }) => {
  // Extract the actual recommendation data from the stored structure
  const recommendations = currentRecommendation?.recommendations || currentRecommendation;
  
  // Get workout and nutrition recommendations from the correct nested structure
  const workoutRecommendation = recommendations?.predictions?.workout_template || recommendations?.workout_recommendation;
  const nutritionRecommendation = recommendations?.predictions?.nutrition_template || recommendations?.nutrition_recommendation;
  
  // Debug logging
  console.log('Current Recommendation:', currentRecommendation);
  console.log('Extracted Recommendations:', recommendations);
  console.log('Workout Recommendation:', workoutRecommendation);
  console.log('Nutrition Recommendation:', nutritionRecommendation);
  
  // Additional debugging for template data
  console.log('Template IDs in recommendation:', {
    workout_template_id: workoutRecommendation?.template_id || recommendations?.workout_template_id,
    nutrition_template_id: nutritionRecommendation?.template_id || recommendations?.nutrition_template_id
  });
  
  const [feedback, setFeedback] = useState({
    workoutDifficulty: 'just_right',
    workoutEnjoyment: 'enjoyed',
    workoutEffectiveness: 'effective',
    nutritionSatisfaction: 'satisfied',
    energyLevel: 'good',
    recovery: 'good',
    overallSatisfaction: 'satisfied'
  });
  
  const [suggestions, setSuggestions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [alert, setAlert] = useState(null);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  const navigate = useNavigate();

  // Color mode values
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const cardBg = useColorModeValue('white', 'gray.700');
  const textColor = useColorModeValue('gray.800', 'white');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');

  const feedbackOptions = {
    workoutDifficulty: {
      too_easy: { label: 'Terlalu Mudah', icon: '😴', color: 'blue' },
      just_right: { label: 'Tepat', icon: '😊', color: 'green' },
      too_hard: { label: 'Terlalu Sulit', icon: '😰', color: 'red' }
    },
    workoutEnjoyment: {
      enjoyed: { label: 'Menyukainya', icon: '😄', color: 'green' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'yellow' },
      disliked: { label: 'Tidak Suka', icon: '😞', color: 'red' }
    },
    workoutEffectiveness: {
      effective: { label: 'Sangat Efektif', icon: '💪', color: 'green' },
      somewhat: { label: 'Agak Efektif', icon: '👍', color: 'yellow' },
      not_effective: { label: 'Tidak Efektif', icon: '👎', color: 'red' }
    },
    nutritionSatisfaction: {
      satisfied: { label: 'Puas', icon: '😋', color: 'green' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'yellow' },
      unsatisfied: { label: 'Tidak Puas', icon: '😕', color: 'red' }
    },
    energyLevel: {
      great: { label: 'Energi Besar', icon: '⚡', color: 'green' },
      good: { label: 'Energi Baik', icon: '👍', color: 'blue' },
      low: { label: 'Energi Rendah', icon: '😴', color: 'red' }
    },
    recovery: {
      great: { label: 'Pemulihan Besar', icon: '🔄', color: 'green' },
      good: { label: 'Pemulihan Baik', icon: '👍', color: 'blue' },
      poor: { label: 'Pemulihan Buruk', icon: '😫', color: 'red' }
    },
    overallSatisfaction: {
      satisfied: { label: 'Puas', icon: '😊', color: 'green' },
      neutral: { label: 'Biasa Saja', icon: '😐', color: 'yellow' },
      unsatisfied: { label: 'Tidak Puas', icon: '😞', color: 'red' }
    }
  };

  const handleFeedbackChange = (category, value) => {
    setFeedback(prev => ({
      ...prev,
      [category]: value
    }));
  };

  const generateSuggestions = async () => {
    try {
      console.log('Generating suggestions with:', {
        currentRecommendation: recommendations,
        userProfile,
        feedback
      });
      
      const improvedRecommendation = await getImprovedRecommendation({
        currentRecommendation: recommendations,
        userProfile,
        feedback
      });
      
      console.log('Improved recommendation received:', improvedRecommendation);
      return improvedRecommendation;
    } catch (error) {
      console.error('Error generating suggestions:', error);
      return null;
    }
  };

  const saveFeedback = async () => {
    setIsLoading(true);
    try {
      console.log('Saving feedback with data:', feedback);
      console.log('User profile:', userProfile);
      console.log('Current recommendation:', recommendations);
      
      await saveRecommendationFeedback({
        recommendationId: currentRecommendation.id || Date.now(),
        feedback,
        timestamp: new Date().toISOString()
      });
      
      const result = await generateSuggestions();
      console.log('Generated suggestions result:', result);
      
      if (result) {
        setSuggestions(result);
        toast({
          title: 'Feedback berhasil disimpan',
          description: 'Saran yang dipersonalisasi telah dibuat untuk Anda',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      console.error('Error saving feedback:', error);
      toast({
        title: 'Error',
        description: 'Gagal menyimpan feedback',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const applySuggestion = async (suggestion) => {
    try {
      if (!user || !userProfile) {
        setAlert('Gagal menerapkan rencana baru: data pengguna tidak ditemukan.');
        return;
      }
      // Save the new recommendation to Firestore
      const saved = await recommendationService.saveRecommendation(
        user.uid,
        userProfile,
        suggestion
      );
      setAlert('Rencana baru berhasil diterapkan!');
      // Dispatch a global event so other components can refetch
      window.dispatchEvent(new Event('recommendationUpdated'));
      if (onRecommendationUpdate) {
        onRecommendationUpdate(saved);
      }
      setSuggestions(null);
      setShowFeedback(false);
    } catch (error) {
      setAlert('Gagal menerapkan rencana baru. Silakan coba lagi.');
      console.error('Error applying suggestion:', error);
    }
  };

  const getFeedbackCard = (category, title) => (
    <Card variant="outline" borderWidth="1px" borderColor={borderColor}>
      <CardHeader pb={3}>
        <Heading size="sm">{title}</Heading>
      </CardHeader>
      <CardBody pt={0}>
        <VStack spacing={3} align="stretch">
          {Object.entries(feedbackOptions[category]).map(([key, option]) => (
            <Button
              key={key}
              variant={feedback[category] === key ? 'solid' : 'outline'}
              colorScheme={option.color}
              size={{ base: "sm", md: "md" }}
              onClick={() => handleFeedbackChange(category, key)}
              justifyContent="flex-start"
              h="auto"
              py={3}
              px={4}
            >
              <HStack spacing={3} w="full">
                <Text fontSize="lg">{option.icon}</Text>
                <Text fontSize={{ base: "sm", md: "md" }}>{option.label}</Text>
                {feedback[category] === key && <CheckIcon ml="auto" />}
              </HStack>
            </Button>
          ))}
        </VStack>
      </CardBody>
    </Card>
  );

  return (
    <Box>
      {alert && (
        <Alert status="success" mb={4} borderRadius="md">
          <AlertIcon />
          <Box flex={1}>
            <AlertTitle>Berhasil!</AlertTitle>
            <AlertDescription>{alert}</AlertDescription>
          </Box>
          <HStack spacing={2}>
            <Button
              size="sm"
              colorScheme="blue"
              onClick={() => navigate('/recommendation')}
            >
              Lihat Rekomendasi
            </Button>
            <IconButton
              size="sm"
              icon={<CloseIcon />}
              onClick={() => setAlert(null)}
              aria-label="Close alert"
            />
          </HStack>
        </Alert>
      )}
      
      {!showFeedback ? (
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardBody textAlign="center" py={8}>
            <VStack spacing={4}>
              <Text fontSize="4xl">💪</Text>
              <Heading size={{ base: "md", md: "lg" }}>
                Bagaimana latihan Anda hari ini?
              </Heading>
              <Text color={mutedTextColor} fontSize={{ base: "sm", md: "md" }}>
                Bantu kami meningkatkan rekomendasi dengan berbagi pengalaman Anda!
              </Text>
              <Button
                colorScheme="blue"
                size={{ base: "md", md: "lg" }}
                onClick={() => setShowFeedback(true)}
                leftIcon={<InfoIcon />}
              >
                Berikan Feedback
              </Button>
            </VStack>
          </CardBody>
        </Card>
      ) : (
        <VStack spacing={6} align="stretch">
          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
            <CardHeader>
              <HStack justify="space-between">
                <Heading size={{ base: "md", md: "lg" }}>Feedback Latihan & Nutrisi</Heading>
                <IconButton
                  icon={<CloseIcon />}
                  onClick={() => setShowFeedback(false)}
                  aria-label="Close feedback"
                  variant="ghost"
                />
              </HStack>
            </CardHeader>
            <CardBody>
              <VStack spacing={6} align="stretch">
                {/* Workout Experience */}
                <Box>
                  <Heading size="md" mb={4}>🏋️ Pengalaman Latihan</Heading>
                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                    {getFeedbackCard('workoutDifficulty', 'Tingkat Kesulitan Latihan')}
                    {getFeedbackCard('workoutEnjoyment', 'Kesenangan Latihan')}
                    {getFeedbackCard('workoutEffectiveness', 'Efektivitas Latihan')}
                  </SimpleGrid>
                </Box>

                <Divider />

                {/* Nutrition & Recovery */}
                <Box>
                  <Heading size="md" mb={4}>🥗 Nutrisi & Pemulihan</Heading>
                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                    {getFeedbackCard('nutritionSatisfaction', 'Kepuasan Nutrisi')}
                    {getFeedbackCard('energyLevel', 'Tingkat Energi')}
                    {getFeedbackCard('recovery', 'Pemulihan')}
                  </SimpleGrid>
                </Box>

                <Divider />

                {/* Overall */}
                <Box>
                  <Heading size="md" mb={4}>📊 Keseluruhan</Heading>
                  {getFeedbackCard('overallSatisfaction', 'Kepuasan Keseluruhan')}
                </Box>

                <Button
                  colorScheme="blue"
                  size="lg"
                  onClick={saveFeedback}
                  isLoading={isLoading}
                  loadingText="Menganalisis..."
                  leftIcon={<ArrowForwardIcon />}
                >
                  Simpan Feedback & Dapatkan Saran
                </Button>
              </VStack>
            </CardBody>
          </Card>

          {suggestions && (
            <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
              <CardHeader>
                <Heading size={{ base: "md", md: "lg" }}>💡 Saran yang Dipersonalisasi</Heading>
              </CardHeader>
              <CardBody>
                <VStack spacing={6} align="stretch">
                  {/* Recommended Changes */}
                  <Box>
                    <Heading size="md" mb={4}>Perubahan yang Direkomendasikan</Heading>
                    <VStack spacing={3} align="stretch">
                      {suggestions.workoutChanges && (
                        <HStack p={4} bg="blue.50" borderRadius="md" spacing={3}>
                          <Text fontSize="xl">🏋️</Text>
                          <Text fontSize={{ base: "sm", md: "md" }}>{suggestions.workoutChanges}</Text>
                        </HStack>
                      )}
                      {suggestions.nutritionChanges && (
                        <HStack p={4} bg="green.50" borderRadius="md" spacing={3}>
                          <Text fontSize="xl">🥗</Text>
                          <Text fontSize={{ base: "sm", md: "md" }}>{suggestions.nutritionChanges}</Text>
                        </HStack>
                      )}
                    </VStack>
                  </Box>

                  {/* Plan Comparison */}
                  <Box>
                    <Heading size="md" mb={4}>📊 Perbandingan Rencana</Heading>
                    <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
                      {/* Current Plan */}
                      <Card variant="outline">
                        <CardHeader>
                          <Heading size="sm">Rencana Saat Ini</Heading>
                        </CardHeader>
                        <CardBody>
                          <VStack spacing={4} align="stretch">
                            {workoutRecommendation && (
                              <Box>
                                <Text fontWeight="bold" mb={2}>🏋️ Latihan</Text>
                                <SimpleGrid columns={2} spacing={2}>
                                  <Stat>
                                    <StatLabel fontSize="xs">Jenis</StatLabel>
                                    <StatNumber fontSize="sm">{workoutRecommendation.workout_type || 'Tidak ditentukan'}</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Hari/Minggu</StatLabel>
                                    <StatNumber fontSize="sm">{workoutRecommendation.days_per_week || 'Tidak ditentukan'}</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Template ID</StatLabel>
                                    <StatNumber fontSize="sm">{workoutRecommendation.template_id || 'Tidak tersedia'}</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Goal</StatLabel>
                                    <StatNumber fontSize="sm">{workoutRecommendation.goal || 'Tidak ditentukan'}</StatNumber>
                                  </Stat>
                                </SimpleGrid>
                              </Box>
                            )}
                            {nutritionRecommendation && (
                              <Box>
                                <Text fontWeight="bold" mb={2}>🥗 Nutrisi</Text>
                                <SimpleGrid columns={2} spacing={2}>
                                  <Stat>
                                    <StatLabel fontSize="xs">Protein/kg</StatLabel>
                                    <StatNumber fontSize="sm">{nutritionRecommendation.protein_per_kg || 'Tidak ditentukan'}g</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Karbohidrat/kg</StatLabel>
                                    <StatNumber fontSize="sm">{nutritionRecommendation.carbs_per_kg || 'Tidak ditentukan'}g</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Lemak/kg</StatLabel>
                                    <StatNumber fontSize="sm">{nutritionRecommendation.fat_per_kg || 'Tidak ditentukan'}g</StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Template ID</StatLabel>
                                    <StatNumber fontSize="sm">{nutritionRecommendation.template_id || 'Tidak tersedia'}</StatNumber>
                                  </Stat>
                                </SimpleGrid>
                                {recommendations?.user_profile && (
                                  <Box mt={3} p={3} bg="gray.50" borderRadius="md">
                                    <Text fontWeight="bold" fontSize="sm" mb={2}>Perhitungan Kalori:</Text>
                                    <SimpleGrid columns={2} spacing={2}>
                                      <Stat>
                                        <StatLabel fontSize="xs">BMR</StatLabel>
                                        <StatNumber fontSize="sm">{Math.round(recommendations.user_profile.bmr)} kkal</StatNumber>
                                      </Stat>
                                      <Stat>
                                        <StatLabel fontSize="xs">TDEE</StatLabel>
                                        <StatNumber fontSize="sm">{Math.round(recommendations.user_profile.tdee)} kkal</StatNumber>
                                      </Stat>
                                    </SimpleGrid>
                                  </Box>
                                )}
                              </Box>
                            )}
                          </VStack>
                        </CardBody>
                      </Card>

                      {/* Suggested Plan */}
                      <Card variant="outline" borderColor="blue.200">
                        <CardHeader>
                          <Heading size="sm">Rencana yang Disarankan</Heading>
                        </CardHeader>
                        <CardBody>
                          <VStack spacing={4} align="stretch">
                            {suggestions.newRecommendation.workout_recommendation && (
                              <Box>
                                <Text fontWeight="bold" mb={2}>🏋️ Latihan</Text>
                                <SimpleGrid columns={2} spacing={2}>
                                  <Stat>
                                    <StatLabel fontSize="xs">Jenis</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={workoutRecommendation?.workout_type !== suggestions.newRecommendation.workout_recommendation.workout_type ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {suggestions.newRecommendation.workout_recommendation.workout_type || 'Tidak ditentukan'}
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Hari/Minggu</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={workoutRecommendation?.days_per_week !== suggestions.newRecommendation.workout_recommendation.days_per_week ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {suggestions.newRecommendation.workout_recommendation.days_per_week || 'Tidak ditentukan'}
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Kardio</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={workoutRecommendation?.cardio_minutes_per_day !== suggestions.newRecommendation.workout_recommendation.cardio_minutes_per_day ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {suggestions.newRecommendation.workout_recommendation.cardio_minutes_per_day || 'Tidak ditentukan'} menit
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Set</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={workoutRecommendation?.sets_per_exercise !== suggestions.newRecommendation.workout_recommendation.sets_per_exercise ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {suggestions.newRecommendation.workout_recommendation.sets_per_exercise || 'Tidak ditentukan'}
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                </SimpleGrid>
                              </Box>
                            )}
                            {suggestions.newRecommendation.nutrition_recommendation && (
                              <Box>
                                <Text fontWeight="bold" mb={2}>🥗 Nutrisi</Text>
                                <SimpleGrid columns={2} spacing={2}>
                                  <Stat>
                                    <StatLabel fontSize="xs">Kalori</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={Math.round(recommendations?.user_profile?.tdee) !== Math.round(suggestions.newRecommendation.nutrition_recommendation.target_calories) ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {Math.round(suggestions.newRecommendation.nutrition_recommendation.target_calories) || 'Tidak ditentukan'} kkal
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Protein</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={Math.round(nutritionRecommendation?.protein_per_kg * (userProfile?.weight || 70)) !== Math.round(suggestions.newRecommendation.nutrition_recommendation.target_protein) ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {Math.round(suggestions.newRecommendation.nutrition_recommendation.target_protein) || 'Tidak ditentukan'}g
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Karbohidrat</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={Math.round(nutritionRecommendation?.carbs_per_kg * (userProfile?.weight || 70)) !== Math.round(suggestions.newRecommendation.nutrition_recommendation.target_carbs) ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {Math.round(suggestions.newRecommendation.nutrition_recommendation.target_carbs) || 'Tidak ditentukan'}g
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                  <Stat>
                                    <StatLabel fontSize="xs">Lemak</StatLabel>
                                    <StatNumber fontSize="sm">
                                      <Badge 
                                        colorScheme={Math.round(nutritionRecommendation?.fat_per_kg * (userProfile?.weight || 70)) !== Math.round(suggestions.newRecommendation.nutrition_recommendation.target_fat) ? 'blue' : 'gray'}
                                        variant="subtle"
                                      >
                                        {Math.round(suggestions.newRecommendation.nutrition_recommendation.target_fat) || 'Tidak ditentukan'}g
                                      </Badge>
                                    </StatNumber>
                                  </Stat>
                                </SimpleGrid>
                              </Box>
                            )}
                          </VStack>
                        </CardBody>
                      </Card>
                    </SimpleGrid>
                  </Box>

                  {/* Reasoning */}
                  <Box p={4} bg="purple.50" borderRadius="md">
                    <Heading size="sm" mb={2}>🤔 Mengapa Perubahan Ini?</Heading>
                    <Text fontSize={{ base: "sm", md: "md" }}>{suggestions.reasoning}</Text>
                  </Box>

                  {/* Actions */}
                  <HStack spacing={4} justify="center">
                    <Button
                      colorScheme="blue"
                      size={{ base: "md", md: "lg" }}
                      onClick={() => applySuggestion(suggestions.newRecommendation)}
                      leftIcon={<CheckIcon />}
                    >
                      Terapkan Rekomendasi Baru
                    </Button>
                    <Button
                      variant="outline"
                      size={{ base: "md", md: "lg" }}
                      onClick={() => setSuggestions(null)}
                    >
                      Pertahankan Rencana Saat Ini
                    </Button>
                  </HStack>
                </VStack>
              </CardBody>
            </Card>
          )}
        </VStack>
      )}
    </Box>
  );
};

export default RecommendationFeedback; 