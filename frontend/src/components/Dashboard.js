import React, { useState, useEffect } from 'react';
import { recommendationService } from '../services/recommendationService';
import { authService } from '../services/authService';
import {
  Box,
  Flex,
  Stack,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  SimpleGrid,
  Badge,
  useColorModeValue,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Skeleton,
  Divider
} from '@chakra-ui/react';

const Dashboard = ({ user, userData, recommendations, onNavigate }) => {
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState({
    currentRecommendation: null,
    recommendationHistory: [],
    progressData: [],
    stats: {
      totalRecommendations: 0,
      lastWeightEntry: null
    }
  });

  useEffect(() => {
    loadDashboardData();
  }, [user]);

  const loadDashboardData = async () => {
    if (!user) return;

    try {
      setLoading(true);
      const currentRec = await recommendationService.getCurrentRecommendation(user.uid);
      const history = await recommendationService.getRecommendationHistory(user.uid, 5);
      const progress = await authService.getUserProgress(30);
      setDashboardData({
        currentRecommendation: currentRec,
        recommendationHistory: history.success ? history.data : [],
        progressData: progress.success ? progress.data : [],
        stats: {
          totalRecommendations: history.success ? history.data.length : 0,
          lastWeightEntry: userData?.weight || null
        }
      });
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('id-ID', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  };

  const getMetricCards = () => {
    const currentRec = dashboardData.currentRecommendation;
    const userMetrics = currentRec?.userData || userData;
    if (!userMetrics) return [];
    const bmi = userMetrics.weight / ((userMetrics.height / 100) ** 2);
    const bmr = userMetrics.gender === 'Male' 
      ? 88.362 + (13.397 * userMetrics.weight) + (4.799 * userMetrics.height) - (5.677 * userMetrics.age)
      : 447.593 + (9.247 * userMetrics.weight) + (3.098 * userMetrics.height) - (4.330 * userMetrics.age);
    return [
      {
        title: 'BMI',
        value: bmi.toFixed(1),
        unit: 'kg/m²',
        icon: '⚖️',
        status: bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese',
        color: bmi < 18.5 ? 'orange.400' : bmi < 25 ? 'green.400' : bmi < 30 ? 'orange.500' : 'red.500'
      },
      {
        title: 'Berat Badan',
        value: userMetrics.weight,
        unit: 'kg',
        icon: '🏋️',
        status: `Target: ${userMetrics.fitness_goal}`,
        color: 'blue.400'
      },
      {
        title: 'BMR',
        value: Math.round(bmr),
        unit: 'kcal/hari',
        icon: '🔥',
        status: 'Kalori Basal',
        color: 'purple.400'
      },
    ];
  };

  const gradientBg = useColorModeValue(
    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)'
  );

  if (loading) {
    return (
      <Box p={8}>
        <Skeleton height="40px" mb={4} />
        <Skeleton height="120px" mb={4} />
        <Skeleton height="80px" mb={4} />
      </Box>
    );
  }

  return (
    <Box px={{ base: 4, md: 8 }} py={{ base: 4, md: 8 }}>
      <VStack align="start" spacing={{ base: 4, md: 6 }} w="full">
        {/* Header */}
        <Box w="full">
          <Heading size={{ base: "md", md: "lg" }} bgGradient={gradientBg} bgClip="text">
            Dashboard XGFitness
          </Heading>
          <Text color="gray.600" fontSize={{ base: "sm", md: "md" }}>
            Selamat datang kembali, {user.displayName || 'Pengguna'}!
          </Text>
        </Box>

        {/* Quick Stats */}
        <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 4, md: 6 }} w="full">
          {getMetricCards().map((metric, idx) => (
            <Box
              key={idx}
              bg="white"
              borderRadius="xl"
              boxShadow="md"
              p={{ base: 4, md: 5 }}
              borderLeftWidth={6}
              borderLeftColor={metric.color}
              transition="box-shadow 0.2s"
              _hover={{ boxShadow: 'xl' }}
            >
              <Flex align="center" mb={2}>
                <Text fontSize={{ base: "xl", md: "2xl" }} mr={2}>{metric.icon}</Text>
                <Text fontWeight="bold" fontSize={{ base: "lg", md: "xl" }}>{metric.title}</Text>
              </Flex>
              <Text fontSize={{ base: "xl", md: "2xl" }} fontWeight="bold">
                {metric.value} <Text as="span" fontSize={{ base: "sm", md: "md" }} color="gray.500">{metric.unit}</Text>
              </Text>
              <Text fontSize={{ base: "xs", md: "sm" }} color={metric.color} fontWeight="semibold">
                {metric.status}
              </Text>
            </Box>
          ))}
        </SimpleGrid>

        {/* Current Recommendation Status */}
        {dashboardData.currentRecommendation && (
          <Box w="full" bg="white" borderRadius="xl" boxShadow="md" p={{ base: 4, md: 6 }}>
            <Heading size={{ base: "sm", md: "md" }} mb={2}>🎯 Rekomendasi Aktif</Heading>
            <VStack align="start" spacing={4} w="full">
              <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={4} w="full">
                <Box>
                  <Text fontSize="sm" color="gray.500">Dibuat:</Text>
                  <Text fontWeight="bold" fontSize={{ base: "sm", md: "md" }}>{formatDate(dashboardData.currentRecommendation.createdAt.toDate())}</Text>
                </Box>
                <Box>
                  <Text fontSize="sm" color="gray.500">Target:</Text>
                  <Badge colorScheme="purple" fontSize={{ base: "xs", md: "md" }} px={2} py={1} borderRadius="md">
                    {dashboardData.currentRecommendation.userData.fitness_goal}
                  </Badge>
                </Box>
              </SimpleGrid>
              <HStack spacing={{ base: 2, md: 4 }} w="full" justify={{ base: "center", md: "flex-end" }}>
                <Button colorScheme="brand" size={{ base: "sm", md: "md" }} onClick={() => onNavigate('recommendation')}>Lihat Detail</Button>
                <Button variant="outline" colorScheme="brand" size={{ base: "sm", md: "md" }} onClick={() => onNavigate('progress')}>Update Progress</Button>
              </HStack>
            </VStack>
          </Box>
        )}

        {/* Quick Actions */}
        <Box w="full">
          <Heading size={{ base: "sm", md: "md" }} mb={3}>⚡ Aksi Cepat</Heading>
          <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
            <Button
              h={{ base: "100px", md: "120px" }}
              borderRadius="xl"
              bgGradient={gradientBg}
              color="white"
              fontWeight="bold"
              fontSize={{ base: "md", md: "lg" }}
              boxShadow="md"
              _hover={{ opacity: 0.9 }}
              onClick={() => onNavigate('input')}
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              p={{ base: 2, md: 4 }}
            >
              <Text fontSize={{ base: "2xl", md: "3xl" }}>📝</Text>
              <Text fontSize={{ base: "sm", md: "md" }}>Rencana Baru</Text>
              <Text fontSize={{ base: "xs", md: "sm" }} opacity={0.8}>Buat rekomendasi fitness baru</Text>
            </Button>
            <Button
              h={{ base: "100px", md: "120px" }}
              borderRadius="xl"
              bgGradient={gradientBg}
              color="white"
              fontWeight="bold"
              fontSize={{ base: "md", md: "lg" }}
              boxShadow="md"
              _hover={{ opacity: 0.9 }}
              onClick={() => onNavigate('recommendations')}
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              p={{ base: 2, md: 4 }}
            >
              <Text fontSize={{ base: "2xl", md: "3xl" }}>🎯</Text>
              <Text fontSize={{ base: "sm", md: "md" }}>Lihat Rekomendasi</Text>
              <Text fontSize={{ base: "xs", md: "sm" }} opacity={0.8}>Cek workout & nutrition plan</Text>
            </Button>
            <Button
              h={{ base: "100px", md: "120px" }}
              borderRadius="xl"
              bgGradient={gradientBg}
              color="white"
              fontWeight="bold"
              fontSize={{ base: "md", md: "lg" }}
              boxShadow="md"
              _hover={{ opacity: 0.9 }}
              onClick={() => onNavigate('progress')}
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              p={{ base: 2, md: 4 }}
            >
              <Text fontSize={{ base: "2xl", md: "3xl" }}>📈</Text>
              <Text fontSize={{ base: "sm", md: "md" }}>Update Progress</Text>
              <Text fontSize={{ base: "xs", md: "sm" }} opacity={0.8}>Catat aktivitas harian</Text>
            </Button>
          </SimpleGrid>
        </Box>

        {/* Legacy fallback for when no userData is available */}
        {(!userData && recommendations) && (
          <Box w="full" bg="white" borderRadius="xl" boxShadow="md" p={6}>
            <Heading size="md" mb={3}>📋 Ringkasan Rekomendasi Terkini</Heading>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
              <Box borderWidth={1} borderRadius="md" p={4}>
                <Heading size="sm" mb={2}>🏋️ Program Latihan</Heading>
                <VStack align="start" spacing={1}>
                  <Text>Jenis: {recommendations.workout_recommendation?.workout_type}</Text>
                  <Text>Frekuensi: {recommendations.workout_recommendation?.days_per_week} hari/minggu</Text>
                  <Text>Durasi Cardio: {recommendations.workout_recommendation?.cardio_minutes_per_day} menit/hari</Text>
                  <Text>Set per Latihan: {recommendations.workout_recommendation?.sets_per_exercise}</Text>
                </VStack>
              </Box>
              <Box borderWidth={1} borderRadius="md" p={4}>
                <Heading size="sm" mb={2}>🍎 Program Nutrisi</Heading>
                <VStack align="start" spacing={1}>
                  <Text>Kalori Harian: {Math.round(recommendations.nutrition_recommendation?.target_calories)} kkal</Text>
                  <Text>Protein: {Math.round(recommendations.nutrition_recommendation?.target_protein)} gram</Text>
                  <Text>Karbohidrat: {Math.round(recommendations.nutrition_recommendation?.target_carbs)} gram</Text>
                  <Text>Lemak: {Math.round(recommendations.nutrition_recommendation?.target_fat)} gram</Text>
                </VStack>
              </Box>
            </SimpleGrid>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default Dashboard;
