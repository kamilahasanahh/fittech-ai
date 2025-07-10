// Nutrition Service for fetching nutrition data from CSV files
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
      // Fetch the CSV file from the public directory
      const response = await fetch('/data/nutrition_macro_summary.csv');
      
      if (!response.ok) {
        throw new Error(`Failed to fetch nutrition data: ${response.status}`);
      }

      const csvText = await response.text();
      const nutritionData = this.parseCSV(csvText);
      
      this.nutritionData = nutritionData;
      return nutritionData;
    } catch (error) {
      console.error('Error loading nutrition data:', error);
      
      // Fallback to hardcoded data if CSV fetch fails
      const fallbackData = [
        { name: "Mie Telur (Ditambah, Masak)", calories: 138, carbs: 25.16, protein: 4.54, fat: 2.07 },
        { name: "Roti Gandum", calories: 67, carbs: 12.26, protein: 2.37, fat: 1.07 },
        { name: "Ayam Goreng tanpa Pelapis (Kulit Dimakan)", calories: 260, carbs: 0.0, protein: 28.62, fat: 15.35 },
        { name: "Ikan Panggang", calories: 126, carbs: 0.33, protein: 21.94, fat: 3.44 },
        { name: "Nasi Putih (Butir-Sedang, Dimasak)", calories: 130, carbs: 28.59, protein: 2.38, fat: 0.21 },
        { name: "Tempe Goreng", calories: 34, carbs: 1.79, protein: 2.0, fat: 2.28 },
        { name: "Tahu Goreng", calories: 35, carbs: 1.36, protein: 2.23, fat: 2.62 }
      ];
      
      this.nutritionData = fallbackData;
      return fallbackData;
    } finally {
      this.loading = false;
    }
  }

  parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(header => header.trim().replace(/"/g, ''));
    
    const nutritionData = [];
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      const values = this.parseCSVLine(line);
      
      if (values.length >= headers.length) {
        const food = {
          name: values[0].replace(/"/g, '').trim(),
          calories: parseFloat(values[1]) || 0,
          carbs: parseFloat(values[2]) || 0,
          protein: parseFloat(values[3]) || 0,
          fat: parseFloat(values[4]) || 0
        };
        
        nutritionData.push(food);
      }
    }
    
    return nutritionData;
  }

  parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        values.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    values.push(current);
    return values;
  }

  // Get foods by category for meal planning
  getFoodsByCategory(category) {
    if (!this.nutritionData) return [];
    
    const categoryFilters = {
      breakfast: ['Roti', 'Telur', 'Mie'],
      lunch: ['Nasi', 'Ayam', 'Ikan'],
      dinner: ['Ikan', 'Tempe', 'Tahu', 'Sayur'],
      snack: ['Tempe', 'Tahu']
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