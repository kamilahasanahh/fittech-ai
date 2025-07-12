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
import { doc, setDoc, getDoc } from 'firebase/firestore';
import RecommendationFeedback from './RecommendationFeedback';

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

  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();

  const today = new Date().toISOString().split('T')[0];

  // Color mode values
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const cardBg = useColorModeValue('white', 'gray.700');
  const textColor = useColorModeValue('gray.800', 'white');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');

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
  }, [loadTodaysProgress, loadRecommendationHistory]);

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

        {/* User Profile Summary */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardBody>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
              <VStack align="start" spacing={3}>
                <HStack>
                  <Text fontSize="2xl">{fitnessGoalInfo.icon}</Text>
                  <Box>
                    <Text fontWeight="bold" fontSize={{ base: "md", md: "lg" }}>
                      {fitnessGoalInfo.title}
                    </Text>
                    <Text color={mutedTextColor} fontSize="sm">
                      {fitnessGoalInfo.description}
                    </Text>
                  </Box>
                </HStack>
                <Badge colorScheme={fitnessGoalInfo.color} variant="subtle" px={3} py={1}>
                  Tujuan Utama
                </Badge>
              </VStack>
              
              <VStack align="start" spacing={3}>
                <HStack>
                  <Text fontSize="2xl">{activityLevelInfo.icon}</Text>
                  <Box>
                    <Text fontWeight="bold" fontSize={{ base: "md", md: "lg" }}>
                      {activityLevelInfo.title}
                    </Text>
                    <Text color={mutedTextColor} fontSize="sm">
                      Faktor: {activityLevelInfo.multiplier}
                    </Text>
                  </Box>
                </HStack>
                <Badge colorScheme={activityLevelInfo.color} variant="subtle" px={3} py={1}>
                  Level Aktivitas
                </Badge>
              </VStack>
            </SimpleGrid>
          </CardBody>
        </Card>

        {/* Recommendation History */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardHeader>
            <HStack justify="space-between">
              <Heading size={{ base: "md", md: "lg" }}>📋 Riwayat Rekomendasi</Heading>
              <Button
                size={{ base: "sm", md: "md" }}
                variant="outline"
                rightIcon={showRecommendationHistory ? <ChevronUpIcon /> : <ChevronDownIcon />}
                onClick={() => setShowRecommendationHistory(!showRecommendationHistory)}
              >
                {showRecommendationHistory ? 'Sembunyikan' : 'Tampilkan'} Riwayat
              </Button>
            </HStack>
          </CardHeader>
          
          {showRecommendationHistory && (
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
                  {recommendationHistory.map((recommendation, index) => (
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
                            onClick={() => {
                              setSelectedRecommendation(recommendation);
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
            </CardBody>
          )}
        </Card>

        {/* Recommendation Feedback */}
        {currentRecommendation && (
          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
            <CardHeader>
              <Heading size={{ base: "md", md: "lg" }}>💬 Feedback Rekomendasi</Heading>
            </CardHeader>
            <CardBody>
              <RecommendationFeedback
                currentRecommendation={currentRecommendation}
                userProfile={userProfile}
                user={user}
                onRecommendationUpdate={(newRecommendation) => {
                  console.log('New recommendation applied:', newRecommendation);
                }}
              />
            </CardBody>
          </Card>
        )}

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
                <Card variant="outline">
                  <CardHeader>
                    <Heading size="md">👤 Profil Pengguna</Heading>
                  </CardHeader>
                  <CardBody>
                    <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                      <Stat>
                        <StatLabel>Usia</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.age} tahun</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Berat</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.weight} kg</StatNumber>
                      </Stat>
                      <Stat>
                        <StatLabel>Tinggi</StatLabel>
                        <StatNumber>{selectedRecommendation.userData.height} cm</StatNumber>
                      </Stat>
                    </SimpleGrid>
                  </CardBody>
                </Card>

                {selectedRecommendation.recommendations.workout_recommendation && (
                  <Card variant="outline">
                    <CardHeader>
                      <Heading size="md">🏋️ Workout</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>Jenis</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.workout_recommendation.workout_type}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Hari/Minggu</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.workout_recommendation.days_per_week}</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Kardio</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.workout_recommendation.cardio_minutes_per_day} menit</StatNumber>
                        </Stat>
                      </SimpleGrid>
                    </CardBody>
                  </Card>
                )}

                {selectedRecommendation.recommendations.nutrition_recommendation && (
                  <Card variant="outline">
                    <CardHeader>
                      <Heading size="md">🍎 Nutrisi</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                        <Stat>
                          <StatLabel>Kalori</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.nutrition_recommendation.target_calories} kkal</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Protein</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.nutrition_recommendation.target_protein}g</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Karbohidrat</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.nutrition_recommendation.target_carbs}g</StatNumber>
                        </Stat>
                      </SimpleGrid>
                    </CardBody>
                  </Card>
                )}

                {selectedRecommendation.recommendations.confidence_scores && (
                  <Card variant="outline">
                    <CardHeader>
                      <Heading size="md">🎯 Tingkat Kepercayaan</Heading>
                    </CardHeader>
                    <CardBody>
                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                        <Stat>
                          <StatLabel>Keseluruhan</StatLabel>
                          <StatNumber>{Math.round(selectedRecommendation.recommendations.confidence_scores.overall_confidence * 100)}%</StatNumber>
                        </Stat>
                        <Stat>
                          <StatLabel>Level</StatLabel>
                          <StatNumber>{selectedRecommendation.recommendations.confidence_scores.confidence_level}</StatNumber>
                        </Stat>
                      </SimpleGrid>
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
