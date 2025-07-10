// Nutrition Service for fetching nutrition data from JSON files
class NutritionService {
  constructor() {
    this.nutritionData = null;
    this.loading = false;
  }

  async loadNutritionData() {
    if (this.nutritionData) {
      return this.nutritionData;
    }

    if (this.loading) {
      // Wait for existing request to complete
      while (this.loading) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return this.nutritionData;
    }

    this.loading = true;

    try {
      // Fetch the JSON file from the public directory
      const response = await fetch('/data/nutrition_macro_summary.json');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch nutrition data: ${response.status}`);
      }

      const jsonData = await response.json();
      const nutritionData = this.formatNutritionData(jsonData.nutrition_database);
      
      this.nutritionData = nutritionData;
      return nutritionData;
    } catch (error) {
      console.error('Error loading nutrition data:', error);
      
      // Fallback to hardcoded data if JSON fetch fails
      const fallbackData = [
        { name: "Mie Telur (Ditambah, Masak)", calories: 138, carbs: 25.16, protein: 4.54, fat: 2.07 },
        { name: "Ayam Goreng tanpa Pelapis (Kulit Dimakan)", calories: 260, carbs: 0.0, protein: 28.62, fat: 15.35 },
        { name: "Ikan Panggang", calories: 126, carbs: 0.33, protein: 21.94, fat: 3.44 },
        { name: "Nasi Putih (Butir-Sedang, Dimasak)", calories: 130, carbs: 28.59, protein: 2.38, fat: 0.21 },
        { name: "Tempe Goreng", calories: 225, carbs: 9.0, protein: 18.0, fat: 11.0 },
        { name: "Tahu Goreng", calories: 271, carbs: 10.49, protein: 17.19, fat: 20.18 }
      ];
      
      this.nutritionData = fallbackData;
      return fallbackData;
    } finally {
      this.loading = false;
    }
  }

  formatNutritionData(nutritionDatabase) {
    return nutritionDatabase.map(food => ({
      name: food.name,
      calories: food.calories,
      carbs: food.carbohydrates,
      protein: food.protein,
      fat: food.fat
    }));
  }

  // Get foods by category for meal planning
  getFoodsByCategory(category) {
    if (!this.nutritionData) return [];
    
    const categoryFilters = {
      breakfast: ['Roti', 'Telur', 'Mie', 'Nasi Uduk'],
      lunch: ['Nasi', 'Ayam', 'Ikan', 'Kwetiau', 'Mie Tek Tek'],
      dinner: ['Ikan', 'Tempe', 'Tahu', 'Sayur', 'Sate'],
      snack: ['Jagung', 'Telur Rebus', 'Kentang']
    };
    
    const filters = categoryFilters[category] || [];
    
    return this.nutritionData.filter(food => 
      filters.some(filter => food.name.toLowerCase().includes(filter.toLowerCase()))
    );
  }

  // Get all available foods
  getAllFoods() {
    return this.nutritionData || [];
  }

  // Search foods by name
  searchFoods(query) {
    if (!this.nutritionData) return [];
    
    const searchTerm = query.toLowerCase();
    return this.nutritionData.filter(food => 
      food.name.toLowerCase().includes(searchTerm)
    );
  }
}

// Create singleton instance
const nutritionService = new NutritionService();

export { nutritionService }; 