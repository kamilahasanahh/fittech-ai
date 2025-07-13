// frontend/src/App.js
import React, { useState, useEffect, useCallback } from 'react';
import { 
  ChakraProvider, 
  Box, 
  Flex, 
  Text, 
  Button, 
  Container, 
  VStack, 
  HStack,
  Alert,
  AlertIcon,
  AlertDescription,
  CloseButton,
  useToast
} from '@chakra-ui/react';
import theme from './theme';
import { apiService } from './services/api';
import { authService } from './services/authService';
import { recommendationService } from './services/recommendationService';
import { auth } from './services/firebaseConfig';
import AuthForm from './components/AuthForm';
import EnhancedUserInputForm from './components/EnhancedUserInputForm';
import RecommendationDisplay from './components/RecommendationDisplay';
import DailyProgress from './components/DailyProgress';
import Dashboard from './components/Dashboard';
import SystemStatus from './components/SystemStatus';
import OfflineNotice from './components/OfflineNotice';
import './styles/App.css';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import AuthPage from './pages/AuthPage';
import DashboardPage from './pages/DashboardPage';
import RecommendationPage from './pages/RecommendationPage';
import ProgressPage from './pages/ProgressPage';
import InputPage from './pages/InputPage';



function App() {
  const [user, setUser] = useState(null);
  const [userData, setUserData] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [currentRecommendation, setCurrentRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemStatus, setSystemStatus] = useState(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    checkSystemHealth();
    checkAuthState();
    
    // Listen for online/offline events
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (user && location.pathname === '/recommendation') {
      loadSavedRecommendations();
    }
  }, [location.pathname, user]);

  const handleMealPlanGenerated = async (mealPlan) => {
    if (!currentRecommendation || !user) return;
    
    try {
      console.log('🔄 Updating recommendation with meal plan:', mealPlan);
      
      // Update the current recommendation with meal plan data
      const updatedRecommendation = {
        ...currentRecommendation,
        mealPlan: mealPlan
      };
      
      // Save to Firebase
      await recommendationService.updateRecommendationMealPlan(
        currentRecommendation.id, 
        mealPlan
      );
      
      // Update local state
      setCurrentRecommendation(updatedRecommendation);
      console.log('✅ Meal plan saved to recommendation history');
      
    } catch (error) {
      console.error('❌ Error saving meal plan to recommendation:', error);
    }
  };

  const checkAuthState = async () => {
    try {
      const currentUser = await authService.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
        navigate('/dashboard');
        
        // Load saved recommendations if they exist
        try {
          const currentRec = await recommendationService.getCurrentRecommendation(currentUser.uid);
          if (currentRec) {
            console.log('✅ Found existing recommendation on app load:', currentRec);
            setRecommendations(currentRec.recommendations);
            setUserData(currentRec.userData);
            setCurrentRecommendation(currentRec);
          }
        } catch (error) {
          console.error('Error loading existing recommendations:', error);
        }
      }
    } catch (error) {
      console.error('Error checking auth state:', error);
    }
  };

  const checkSystemHealth = async () => {
    try {
      const status = await apiService.healthCheck();
      setSystemStatus(status);
    } catch (err) {
      console.error('Pemeriksaan kesehatan sistem gagal:', err);
      setSystemStatus({ status: 'error', message: 'Backend tidak merespons' });
    }
  };

  const handleAuthSuccess = (userData) => {
    setUser(userData);
    navigate('/dashboard');
  };

  const handleUserSubmit = async (formData) => {
    setLoading(true);
    setError('');
    
    try {
      console.log('🔄 Mengirim data pengguna:', formData);
      console.log('🔄 Current user:', user);
      console.log('🔄 Firebase Auth user:', auth.currentUser);
      
      // Dapatkan rekomendasi ML dari backend Flask
      const recommendations = await apiService.getRecommendations(formData);
      console.log('✅ Rekomendasi diterima:', recommendations);
      
      // Simpan ke Firebase menggunakan service yang baru
      console.log('🔄 Menyimpan data pengguna ke Firebase...');
      const saveResult = await authService.saveUserData(formData);
      console.log('✅ Data pengguna tersimpan:', saveResult);
      
      // Extract userProfile data from recommendations for comprehensive storage
      const userProfile = recommendations.user_metrics || recommendations.user_profile || null;
      
      // Simpan rekomendasi dengan timestamp dan data lengkap menggunakan service baru
      console.log('🔄 Menyimpan rekomendasi ke Firebase...');
      const savedRecommendation = await recommendationService.saveRecommendation(
        user.uid, 
        formData, 
        recommendations, 
        userProfile, 
        null // displayData can be added later if needed
      );
      console.log('✅ Rekomendasi tersimpan:', savedRecommendation);
      
      setUserData(formData);
      setRecommendations(recommendations);
      
      // Update current recommendation with the saved data including userProfile
      setCurrentRecommendation({
        id: savedRecommendation.id || Date.now(),
        userData: formData,
        recommendations: recommendations,
        userProfile: userProfile,
        createdAt: { seconds: Date.now() / 1000 },
        ...savedRecommendation
      });
      
      navigate('/recommendation');
      
    } catch (err) {
      console.error('❌ Error mendapatkan rekomendasi:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load saved recommendations when navigating to recommendations page
  const loadSavedRecommendations = async () => {
    if (!user) return;
    
    try {
      setLoading(true);
      console.log('🔄 Loading saved recommendations...');
      
      // Get current active recommendation
      const currentRec = await recommendationService.getCurrentRecommendation(user.uid);
      if (currentRec) {
        console.log('✅ Found saved recommendation:', currentRec);
        setRecommendations(currentRec.recommendations);
        setUserData(currentRec.userData);
        setCurrentRecommendation(currentRec);
      } else {
        console.log('ℹ️ No saved recommendation found');
        setRecommendations(null);
        setUserData(null);
      }
    } catch (error) {
      console.error('❌ Error loading saved recommendations:', error);
      setError('Gagal memuat rekomendasi yang tersimpan');
    } finally {
      setLoading(false);
    }
  };

  const handleProgressUpdate = (progressData) => {
    if (currentRecommendation) {
      setCurrentRecommendation(prev => ({
        ...prev,
        ...progressData
      }));
    }
  };

  const handleLogout = async () => {
    try {
      await authService.signOut();
      setUser(null);
      setUserData(null);
      setRecommendations(null);
      setCurrentRecommendation(null);
      navigate('/');
    } catch (error) {
      console.error('Error logout:', error);
    }
  };

  return (
    <ChakraProvider theme={theme}>
      <Box minH="100vh" bg="gray.50">
        {!isOnline && <OfflineNotice />}
        
        {/* Header */}
        <Box 
          bg="white" 
          boxShadow="sm" 
          borderBottom="1px" 
          borderColor="gray.200"
          position="sticky"
          top={0}
          zIndex={10}
        >
          <Container maxW="container.xl" py={{ base: 2, md: 4 }} px={{ base: 4, md: 6 }}>
            <Flex 
              direction={{ base: "column", md: "row" }} 
              justify="space-between" 
              align={{ base: "start", md: "center" }}
              gap={{ base: 3, md: 0 }}
            >
              <VStack align="start" spacing={1}>
                <Text 
                  fontSize={{ base: "xl", md: "2xl" }} 
                  fontWeight="bold" 
                  bgGradient="linear(to-r, brand.500, blue.500)"
                  bgClip="text"
                >
                  🏋️ XGFitness
                </Text>
                <Text color="gray.600" fontSize={{ base: "xs", md: "sm" }}>
                  Sistem Rekomendasi Kebugaran Bertenaga AI
                </Text>
              </VStack>
              
              {user && (
                <HStack spacing={{ base: 2, md: 4 }} w={{ base: "full", md: "auto" }} justify={{ base: "space-between", md: "flex-end" }}>
                  <Text color="gray.700" fontSize={{ base: "sm", md: "md" }}>
                    Selamat datang, {user.displayName || 'Pengguna'}!
                  </Text>
                  <Button 
                    size={{ base: "xs", md: "sm" }}
                    variant="outline" 
                    colorScheme="brand"
                    onClick={handleLogout}
                  >
                    Keluar
                  </Button>
                </HStack>
              )}
            </Flex>
            <SystemStatus status={systemStatus} />
          </Container>
        </Box>

        {/* Navigation */}
        <Box bg="white" borderBottom="1px" borderColor="gray.200">
          <Container maxW="container.xl" px={{ base: 4, md: 6 }}>
            <Box 
              overflowX="auto" 
              css={{
                '&::-webkit-scrollbar': { display: 'none' },
                '-ms-overflow-style': 'none',
                'scrollbarWidth': 'none'
              }}
            >
              <HStack spacing={0} py={2} minW="max-content">
                {[
                  { path: '/dashboard', label: '📊 Dashboard', icon: '📊' },
                  { path: '/input', label: '📝 Rencana Baru', icon: '📝' },
                  { path: '/recommendation', label: '🎯 Rekomendasi', icon: '🎯' },
                  { path: '/progress', label: '📈 Progress', icon: '📈' }
                ].map((item) => (
                  <Button
                    key={item.path}
                    variant={location.pathname === item.path ? "solid" : "ghost"}
                    colorScheme={location.pathname === item.path ? "brand" : "gray"}
                    onClick={() => navigate(item.path)}
                    borderRadius="md"
                    mx={1}
                    size={{ base: "xs", md: "sm" }}
                    fontSize={{ base: "xs", md: "sm" }}
                    px={{ base: 2, md: 3 }}
                    _hover={{
                      bg: location.pathname === item.path ? "brand.600" : "gray.100"
                    }}
                  >
                    {item.label}
                  </Button>
                ))}
              </HStack>
            </Box>
          </Container>
        </Box>

        {/* Main Content */}
        <Box as="main" py={{ base: 4, md: 8 }}>
          <Container maxW="container.xl" px={{ base: 4, md: 6 }}>
            {error && (
              <Alert status="error" mb={6} borderRadius="md">
                <AlertIcon />
                <AlertDescription flex={1}>
                  {error}
                </AlertDescription>
                <CloseButton
                  alignSelf="flex-start"
                  position="relative"
                  right={-1}
                  top={-1}
                  onClick={() => setError('')}
                />
              </Alert>
            )}
            <Routes>
              <Route path="/" element={<AuthPage onAuthSuccess={handleAuthSuccess} />} />
              <Route path="/dashboard" element={<DashboardPage user={user} userData={userData} recommendations={recommendations} onNavigate={navigate} />} />
              <Route path="/input" element={<InputPage onSubmit={handleUserSubmit} loading={loading} initialData={userData} />} />
              <Route path="/recommendation" element={<RecommendationPage recommendations={recommendations} userData={userData} onBack={() => navigate('/input')} onNewRecommendation={() => navigate('/input')} onMealPlanGenerated={handleMealPlanGenerated} loading={loading} error={error} />} />
              <Route path="/progress" element={<ProgressPage user={user} onProgressUpdate={handleProgressUpdate} userProfile={userData} currentRecommendation={currentRecommendation} />} />
            </Routes>
          </Container>
        </Box>
      </Box>
    </ChakraProvider>
  );
}

export default App;
