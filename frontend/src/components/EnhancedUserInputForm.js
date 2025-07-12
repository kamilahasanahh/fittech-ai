import React, { useState, useEffect } from 'react';
import {
  Box,
  Stack,
  VStack,
  HStack,
  Heading,
  Text,
  FormControl,
  FormLabel,
  Input,
  Radio,
  RadioGroup,
  Button,
  Progress,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Badge,
  useColorModeValue,
  useToast,
  Divider,
  SimpleGrid,
  Skeleton
} from '@chakra-ui/react';

const EnhancedUserInputForm = ({ onSubmit, loading: parentLoading, initialData }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    height: '',
    weight: '',
    fitness_goal: '',
    activity_level: ''
  });

  const [currentStep, setCurrentStep] = useState(1);
  const [validationErrors, setValidationErrors] = useState({});
  // Use loading state from parent component
  const [submissionError, setSubmissionError] = useState('');
  const [bmi, setBmi] = useState(null);
  const [bmiCategory, setBmiCategory] = useState(null);
  const [storedDataLoaded, setStoredDataLoaded] = useState(false);

  // Indonesian fitness goals and activity levels (only 3 goals as requested)
  const FITNESS_GOALS = {
    'Fat Loss': {
      label: 'Menurunkan Berat Badan',
      description: 'Fokus pada pembakaran kalori dan pengurangan lemak tubuh',
      icon: '📉'
    },
    'Muscle Gain': {
      label: 'Menambah Massa Otot',
      description: 'Membangun otot dengan latihan beban dan nutrisi yang tepat',
      icon: '💪'
    },
    'Maintenance': {
      label: 'Mempertahankan Bentuk Tubuh',
      description: 'Menjaga kondisi fisik dan berat badan yang sudah ideal',
      icon: '⚖️'
    }
  };

  const ACTIVITY_LEVELS = {
    'Low Activity': {
      label: 'Aktivitas Rendah',
      description: 'Kurang dari 150 menit aktivitas fisik sedang ATAU kurang dari 75 menit aktivitas fisik berat per minggu',
      multiplier: '1.29',
      icon: '🚶‍♂️'
    },
    'Moderate Activity': {
      label: 'Aktivitas Sedang',
      description: '150-300 menit aktivitas fisik sedang ATAU 75-150 menit aktivitas fisik berat per minggu',
      multiplier: '1.55',
      icon: '🏃‍♂️'
    },
    'High Activity': {
      label: 'Aktivitas Tinggi',
      description: 'Lebih dari 300 menit aktivitas fisik sedang ATAU lebih dari 150 menit aktivitas fisik berat per minggu',
      multiplier: '1.81',
      icon: '🏋️‍♂️'
    }
  };

  const steps = [
    { number: 1, label: 'Informasi Dasar' },
    { number: 2, label: 'Tujuan Fitness' },
    { number: 3, label: 'Level Aktivitas' },
    { number: 4, label: 'Konfirmasi' }
  ];

  // Load stored user data on component mount
  useEffect(() => {
    const loadStoredData = () => {
      try {
        const storedData = localStorage.getItem('fittech_user_data');
        if (storedData) {
          const parsedData = JSON.parse(storedData);
          
          // Load gender regardless of other fields
          const updatedFormData = { ...formData };
          
          if (parsedData.gender) {
            // Normalize gender to lowercase to match form values
            updatedFormData.gender = parsedData.gender.toLowerCase();
          }
          
          // Load age and height if they exist and are valid
          if (parsedData.age && parsedData.height) {
            updatedFormData.age = parsedData.age.toString();
            updatedFormData.height = parsedData.height.toString();
            setStoredDataLoaded(true);
          }
          
          setFormData(updatedFormData);
        }
      } catch (error) {
        console.error('Error loading stored user data:', error);
      }
    };

    loadStoredData();
  }, []);

  // Calculate BMI and update category whenever height or weight changes
  useEffect(() => {
    if (formData.height && formData.weight) {
      const heightInM = formData.height / 100;
      const calculatedBmi = (formData.weight / (heightInM * heightInM));
      setBmi(calculatedBmi);
      
      let category;
      if (calculatedBmi < 18.5) {
        category = { text: 'Kurus', color: '#3b82f6', restriction: 'underweight' };
      } else if (calculatedBmi < 25) {
        category = { text: 'Normal', color: '#10b981', restriction: 'normal' };
      } else if (calculatedBmi < 30) {
        category = { text: 'Kelebihan Berat', color: '#f59e0b', restriction: 'overweight' };
      } else {
        category = { text: 'Obesitas', color: '#ef4444', restriction: 'obese' };
      }
      setBmiCategory(category);
    } else {
      setBmi(null);
      setBmiCategory(null);
    }
  }, [formData.height, formData.weight]);

  // Check if a fitness goal is allowed based on BMI category
  const isFitnessGoalAllowed = (goalKey) => {
    if (!bmiCategory) return true; // Allow all goals if BMI not calculated yet
    
    const restrictions = {
      'underweight': ['Muscle Gain', 'Maintenance'], // Only muscle gain and maintenance
      'normal': ['Fat Loss', 'Muscle Gain', 'Maintenance'], // All goals allowed
      'overweight': ['Fat Loss', 'Maintenance'], // Only fat loss and maintenance
      'obese': ['Fat Loss'] // Only fat loss
    };
    
    return restrictions[bmiCategory.restriction].includes(goalKey);
  };

  const validateStep = (step) => {
    const errors = {};
    
    switch (step) {
      case 1:
        if (!formData.age || formData.age < 18 || formData.age > 65) {
          errors.age = 'Usia harus antara 18-65 tahun';
        }
        if (!formData.gender) {
          errors.gender = 'Jenis kelamin harus dipilih';
        }
        if (!formData.height || formData.height < 120 || formData.height > 250) {
          errors.height = 'Tinggi badan harus antara 120-250 cm';
        }
        if (!formData.weight || formData.weight < 30 || formData.weight > 300) {
          errors.weight = 'Berat badan harus antara 30-300 kg';
        }
        break;
      case 2:
        if (!formData.fitness_goal) {
          errors.fitness_goal = 'Tujuan fitness harus dipilih';
        } else if (!isFitnessGoalAllowed(formData.fitness_goal)) {
          errors.fitness_goal = 'Tujuan fitness ini tidak direkomendasikan untuk kategori BMI Anda';
        }
        break;
      case 3:
        if (!formData.activity_level) {
          errors.activity_level = 'Level aktivitas harus dipilih';
        }
        break;
      default:
        // No validation needed for other steps
        break;
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Handle age input restrictions
  const handleAgeKeyDown = (e) => {
    // Allow: backspace, delete, tab, escape, enter, arrows, and numbers
    const allowedKeys = ['Backspace', 'Delete', 'Tab', 'Escape', 'Enter', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'];
    
    if (allowedKeys.includes(e.key)) {
      return;
    }
    
    // Allow numbers 0-9
    if (e.key >= '0' && e.key <= '9') {
      const currentValue = e.target.value;
      const newValue = currentValue + e.key;
      
      // Only prevent if the new value would be more than 2 digits or greater than 99
      // We'll handle the 18-65 range validation in onChange and onBlur
      if (newValue.length > 2 || parseInt(newValue) > 99) {
        e.preventDefault();
      }
      return;
    }
    
    // Block all other keys (like letters, symbols, etc.)
    e.preventDefault();
  };

  const handleAgeBlur = (e) => {
    const value = e.target.value;
    
    // If empty, just return
    if (value === '') {
      return;
    }
    
    const ageValue = parseInt(value);
    
    // Check if it's a valid number and within range
    if (isNaN(ageValue) || ageValue < 18 || ageValue > 65) {
      setValidationErrors(prev => ({
        ...prev,
        age: 'Usia harus antara 18-65 tahun'
      }));
      
      // Don't clear the input, let user fix it
      return;
    }
    
    // Clear any previous validation errors if age is valid
    if (validationErrors.age) {
      setValidationErrors(prev => ({
        ...prev,
        age: ''
      }));
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    // Special handling for age input - allow typing but clear validation errors
    if (name === 'age') {
      // Clear validation errors when user starts typing
      if (validationErrors.age) {
        setValidationErrors(prev => ({
          ...prev,
          age: ''
        }));
      }
    }
    
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear validation error for this field
    if (validationErrors[name]) {
      setValidationErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }

    // If fitness goal is being changed and it's not allowed, clear it
    if (name === 'fitness_goal' && !isFitnessGoalAllowed(value)) {
      return; // Don't allow selection of restricted goals
    }
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateStep(currentStep)) return;
    
    setSubmissionError('');

    try {
      // Prepare data for the parent component
      const submissionData = {
        ...formData,
        // Ensure numeric values are properly converted
        age: parseInt(formData.age),
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        gender: formData.gender === 'male' ? 'Male' : 'Female'
      };

      console.log('🔄 Submitting form data to parent:', submissionData);
      
      // Save age, height, and gender to localStorage for future use
      try {
        const dataToStore = {
          age: submissionData.age,
          height: submissionData.height,
          gender: submissionData.gender
        };
        localStorage.setItem('fittech_user_data', JSON.stringify(dataToStore));
        console.log('💾 Saved user data to localStorage:', dataToStore);
      } catch (error) {
        console.error('Error saving user data to localStorage:', error);
      }
      
      // Call the parent's submit handler (which handles API calls and state updates)
      await onSubmit(submissionData);
      
    } catch (error) {
      console.error('❌ Error submitting form:', error);
      setSubmissionError(
        error.message.includes('fetch') 
          ? 'Tidak dapat terhubung ke server. Periksa koneksi internet Anda.'
          : `Terjadi kesalahan saat memproses data: ${error.message}`
      );
    }
  };

  const progressPercentage = (currentStep / steps.length) * 100;

  const gradientBg = useColorModeValue(
    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)'
  );
  const toast = useToast();

  return (
    <Box maxW={{ base: "full", md: "lg" }} mx="auto" mt={{ base: 4, md: 8 }} p={{ base: 4, md: 8 }} bg="white" borderRadius="xl" boxShadow="lg">
      {/* Progress Bar */}
      <Box mb={{ base: 6, md: 8 }}>
        <Box 
          overflowX="auto" 
          css={{
            '&::-webkit-scrollbar': { display: 'none' },
            '-ms-overflow-style': 'none',
            'scrollbarWidth': 'none'
          }}
        >
          <HStack justify="space-between" mb={2} minW="max-content" spacing={{ base: 1, md: 2 }}>
            {steps.map((step) => (
              <VStack key={step.number} spacing={0} flex={1} minW={{ base: "60px", md: "80px" }}>
                <Box
                  w={{ base: 6, md: 8 }}
                  h={{ base: 6, md: 8 }}
                  borderRadius="full"
                  bg={step.number === currentStep ? 'brand.500' : step.number < currentStep ? 'green.400' : 'gray.200'}
                  color={step.number === currentStep || step.number < currentStep ? 'white' : 'gray.500'}
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  fontWeight="bold"
                  fontSize={{ base: "sm", md: "md" }}
                  mb={1}
                  border={step.number === currentStep ? '2px solid' : '1px solid'}
                  borderColor={step.number === currentStep ? 'brand.600' : 'gray.300'}
                  transition="all 0.2s"
                >
                  {step.number}
                </Box>
                <Text fontSize={{ base: "2xs", md: "xs" }} color={step.number === currentStep ? 'brand.600' : 'gray.500'} textAlign="center">
                  {step.label}
                </Text>
              </VStack>
            ))}
          </HStack>
        </Box>
        <Progress value={progressPercentage} size="sm" colorScheme="brand" borderRadius="md" />
      </Box>

      {/* Error Banner */}
      {submissionError && (
        <Alert status="error" mb={4} borderRadius="md">
          <AlertIcon />
          <Box flex={1}>
            <AlertTitle>Terjadi Kesalahan</AlertTitle>
            <AlertDescription>{submissionError}</AlertDescription>
          </Box>
        </Alert>
      )}

      {/* Info Banner for stored data */}
      {storedDataLoaded && (
        <Alert status="info" mb={4} borderRadius="md" bg="blue.50" color="blue.800" borderColor="blue.300">
          <HStack justify="space-between" w="full">
            <HStack>
              <Text fontSize="xl">💾</Text>
              <Box>
                <Text fontWeight="bold">Data Tersimpan Ditemukan!</Text>
                <Text fontSize="sm">
                  Usia, tinggi badan, dan jenis kelamin Anda telah dimuat dari penyimpanan lokal. Anda hanya perlu memasukkan berat badan saat ini.
                </Text>
              </Box>
            </HStack>
            <Button
              size="xs"
              variant="outline"
              colorScheme="blue"
              onClick={() => {
                localStorage.removeItem('fittech_user_data');
                setFormData({
                  age: '',
                  gender: '',
                  height: '',
                  weight: '',
                  fitness_goal: '',
                  activity_level: ''
                });
                setStoredDataLoaded(false);
              }}
            >
              Mulai Baru
            </Button>
          </HStack>
        </Alert>
      )}

      <form onSubmit={handleSubmit}>
        {/* Step 1: Basic Info */}
        {currentStep === 1 && (
          <VStack spacing={6} align="stretch">
            <Heading size="md">Informasi Dasar Anda</Heading>
            <FormControl isInvalid={!!validationErrors.age}>
              <FormLabel>Usia (tahun)</FormLabel>
              <Input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleInputChange}
                onKeyDown={handleAgeKeyDown}
                onBlur={handleAgeBlur}
                placeholder="Contoh: 25"
                min={18}
                max={65}
                disabled={parentLoading}
              />
              {validationErrors.age && <Text color="red.500" fontSize="sm">{validationErrors.age}</Text>}
              <Text color="gray.500" fontSize="xs">Usia harus antara 18-65 tahun</Text>
            </FormControl>
            <FormControl isInvalid={!!validationErrors.gender}>
              <FormLabel>Jenis Kelamin</FormLabel>
              <RadioGroup
                name="gender"
                value={formData.gender}
                onChange={(val) => handleInputChange({ target: { name: 'gender', value: val } })}
                isDisabled={parentLoading}
              >
                <HStack spacing={6}>
                  <Radio value="male">👨 Pria</Radio>
                  <Radio value="female">👩 Wanita</Radio>
                </HStack>
              </RadioGroup>
              {validationErrors.gender && <Text color="red.500" fontSize="sm">{validationErrors.gender}</Text>}
            </FormControl>
            <FormControl isInvalid={!!validationErrors.height}>
              <FormLabel>Tinggi Badan (cm)</FormLabel>
              <Input
                type="number"
                name="height"
                value={formData.height}
                onChange={handleInputChange}
                placeholder="Contoh: 170"
                min={120}
                max={250}
                disabled={parentLoading}
              />
              {validationErrors.height && <Text color="red.500" fontSize="sm">{validationErrors.height}</Text>}
            </FormControl>
            <FormControl isInvalid={!!validationErrors.weight}>
              <FormLabel>Berat Badan (kg)</FormLabel>
              <Input
                type="number"
                name="weight"
                value={formData.weight}
                onChange={handleInputChange}
                placeholder="Contoh: 65"
                min={30}
                max={300}
                step={0.1}
                disabled={parentLoading}
              />
              {validationErrors.weight && <Text color="red.500" fontSize="sm">{validationErrors.weight}</Text>}
              {storedDataLoaded && (
                <Text color="green.600" fontSize="xs" fontWeight="medium">
                  ⚡ Hanya berat badan yang perlu diperbarui - data lainnya sudah tersimpan!
                </Text>
              )}
            </FormControl>
            {bmi && bmiCategory && (
              <Box bg="gray.50" borderRadius="md" p={4} mt={2}>
                <Text fontWeight="bold" mb={1}>📊 Indeks Massa Tubuh (BMI)</Text>
                <HStack>
                  <Text fontSize="2xl" fontWeight="bold">{bmi.toFixed(1)}</Text>
                  <Badge colorScheme={bmiCategory.restriction === 'underweight' ? 'blue' : bmiCategory.restriction === 'normal' ? 'green' : bmiCategory.restriction === 'overweight' ? 'orange' : 'red'} fontSize="md">
                    {bmiCategory.text}
                  </Badge>
                </HStack>
                {bmiCategory.restriction !== 'normal' && (
                  <Box mt={2} bg="blue.50" borderRadius="md" p={2}>
                    <Text fontSize="sm">
                      💡 <b>Catatan:</b> Berdasarkan BMI Anda, beberapa tujuan fitness mungkin tidak tersedia pada langkah selanjutnya untuk hasil yang optimal.
                    </Text>
                  </Box>
                )}
              </Box>
            )}
          </VStack>
        )}

        {/* Step 2: Fitness Goal */}
        {currentStep === 2 && (
          <VStack spacing={6} align="stretch">
            <Heading size={{ base: "sm", md: "md" }}>Apa Tujuan Fitness Anda?</Heading>
            {bmiCategory && bmiCategory.restriction !== 'normal' && (
              <Alert status="info" mb={2} borderRadius="md" bg="blue.50" color="blue.800" borderColor="blue.300">
                <AlertIcon />
                <Box>
                  <Text fontSize={{ base: "xs", md: "sm" }}>
                    <b>📋 Rekomendasi berdasarkan BMI Anda ({bmiCategory.text}):</b><br />
                    Pilihan yang diarsir tidak direkomendasikan untuk kategori BMI Anda saat ini.
                  </Text>
                </Box>
              </Alert>
            )}
            <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
              {Object.entries(FITNESS_GOALS).map(([key, goal]) => {
                const isAllowed = isFitnessGoalAllowed(key);
                return (
                  <Box
                    key={key}
                    borderWidth={2}
                    borderColor={formData.fitness_goal === key ? 'brand.500' : 'gray.200'}
                    borderRadius="lg"
                    p={4}
                    bg={formData.fitness_goal === key ? 'brand.50' : 'white'}
                    opacity={isAllowed ? 1 : 0.5}
                    cursor={isAllowed ? 'pointer' : 'not-allowed'}
                    onClick={() => {
                      if (isAllowed) {
                        handleInputChange({ target: { name: 'fitness_goal', value: key } });
                      }
                    }}
                    transition="all 0.2s"
                  >
                    <Radio
                      name="fitness_goal"
                      value={key}
                      isChecked={formData.fitness_goal === key}
                      onChange={handleInputChange}
                      isDisabled={!isAllowed}
                      colorScheme="brand"
                      mb={2}
                    >
                      <Text fontWeight="bold" fontSize="lg">{goal.icon} {goal.label}</Text>
                    </Radio>
                    <Text fontSize="sm" color="gray.600">{goal.description}</Text>
                    {!isAllowed && (
                      <Text color="gray.400" fontSize="xs" fontStyle="italic" mt={2}>
                        Tidak direkomendasikan untuk BMI Anda
                      </Text>
                    )}
                  </Box>
                );
              })}
            </SimpleGrid>
            {validationErrors.fitness_goal && <Text color="red.500" fontSize="sm">{validationErrors.fitness_goal}</Text>}
          </VStack>
        )}

        {/* Step 3: Activity Level */}
        {currentStep === 3 && (
          <VStack spacing={6} align="stretch">
            <Heading size={{ base: "sm", md: "md" }}>Seberapa Aktif Anda Saat Ini?</Heading>
            <SimpleGrid columns={{ base: 1, sm: 2, md: 3 }} spacing={{ base: 3, md: 4 }}>
              {Object.entries(ACTIVITY_LEVELS).map(([key, level]) => (
                <Box
                  key={key}
                  borderWidth={2}
                  borderColor={formData.activity_level === key ? 'brand.500' : 'gray.200'}
                  borderRadius="lg"
                  p={4}
                  bg={formData.activity_level === key ? 'brand.50' : 'white'}
                  cursor="pointer"
                  onClick={() => handleInputChange({ target: { name: 'activity_level', value: key } })}
                  transition="all 0.2s"
                >
                  <Radio
                    name="activity_level"
                    value={key}
                    isChecked={formData.activity_level === key}
                    onChange={handleInputChange}
                    colorScheme="brand"
                    mb={2}
                  >
                    <Text fontWeight="bold" fontSize="lg">{level.icon} {level.label}</Text>
                  </Radio>
                  <Text fontSize="sm" color="gray.600">{level.description}</Text>
                  <Text fontSize="xs" color="blue.500" mt={1}>Faktor: {level.multiplier}</Text>
                </Box>
              ))}
            </SimpleGrid>
            <Box bg="gray.50" borderRadius="md" p={3} mt={2} textAlign="center">
              <Text fontSize="sm" color="gray.600">
                💡 <b>Informasi:</b> Aktivitas fisik = semua gerakan tubuh (olahraga + pekerjaan + aktivitas harian) yang membuat Anda bergerak aktif dan berkeringat
              </Text>
            </Box>
            {validationErrors.activity_level && <Text color="red.500" fontSize="sm">{validationErrors.activity_level}</Text>}
          </VStack>
        )}

        {/* Step 4: Confirmation */}
        {currentStep === 4 && (
          <VStack spacing={6} align="stretch">
            <Heading size="md">Konfirmasi Data Anda</Heading>
            <Box bg="gray.50" borderRadius="md" p={4}>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                <VStack align="start" spacing={1}>
                  <Text><b>Usia:</b> {formData.age} tahun</Text>
                  <Text><b>Jenis Kelamin:</b> {formData.gender === 'male' ? 'Pria' : 'Wanita'}</Text>
                  <Text><b>Tinggi Badan:</b> {formData.height} cm</Text>
                  <Text><b>Berat Badan:</b> {formData.weight} kg</Text>
                  <Text><b>BMI:</b> {bmi ? bmi.toFixed(1) : '-'} ({bmiCategory ? bmiCategory.text : '-'})</Text>
                </VStack>
                <VStack align="start" spacing={1}>
                  <Text><b>Tujuan:</b> {FITNESS_GOALS[formData.fitness_goal]?.label}</Text>
                  <Text><b>Level Aktivitas:</b> {ACTIVITY_LEVELS[formData.activity_level]?.label}</Text>
                </VStack>
              </SimpleGrid>
              <Divider my={3} />
              <Text fontSize="sm" color="gray.600">
                📝 <b>Catatan:</b> Data ini akan digunakan untuk membuat rekomendasi fitness yang personal untuk Anda. Pastikan semua informasi sudah benar.
              </Text>
            </Box>
          </VStack>
        )}

        {/* Navigation Buttons */}
        <VStack mt={8} spacing={4}>
          <HStack w="full" justify="space-between">
            {currentStep > 1 && (
              <Button
                onClick={handlePrevious}
                variant="outline"
                colorScheme="brand"
                isDisabled={parentLoading}
                size={{ base: "sm", md: "md" }}
              >
                ← Sebelumnya
              </Button>
            )}
            <Box flex={1} />
            {currentStep < steps.length ? (
              <Button
                onClick={handleNext}
                colorScheme="brand"
                isDisabled={parentLoading}
                size={{ base: "sm", md: "md" }}
              >
                Selanjutnya →
              </Button>
            ) : (
              <Button
                type="submit"
                colorScheme="brand"
                isLoading={parentLoading}
                loadingText="Memproses..."
                fontWeight="bold"
                px={{ base: 6, md: 8 }}
                size={{ base: "sm", md: "md" }}
                w={{ base: "full", md: "auto" }}
              >
                🎯 Buat Rekomendasi
              </Button>
            )}
          </HStack>
        </VStack>
      </form>
    </Box>
  );
};

export default EnhancedUserInputForm;
