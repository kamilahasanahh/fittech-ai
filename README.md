# XGFitness AI - Advanced Fitness Recommendation System

## 🎯 Overview
XGFitness AI is a sophisticated machine learning system that provides personalized fitness and nutrition recommendations using XGBoost algorithms with advanced feature engineering and BMI-based goal restrictions.

## 🤖 Machine Learning Features

### Advanced Feature Engineering (22 Features)
- **Interaction Features**: BMI×Goal, Age×Activity, BMI×Activity Level, Age×Goal, Gender×Goal interactions
- **Metabolic Ratios**: BMR/weight, TDEE/BMR ratio, calorie needs per kg
- **Health Deviation Scores**: Distance from ideal BMI (22.5), weight/height ratio
- **Boolean Flags**: High metabolism, very active, young adult indicators
- **Core Features**: Age, gender, height, weight, BMI, BMR, TDEE, activity level, fitness goal, BMI category

### Model Architecture
- **XGBoost Models**: Separate optimized models for workout and nutrition recommendations
- **Hyperparameter Tuning**: RandomizedSearchCV with separate parameter grids to prevent overfitting
- **Feature Scaling**: StandardScaler for optimal model performance
- **Anti-Overfitting**: Noise injection and stronger regularization for nutrition model
- **Label Encoding**: Consistent class mapping across models

### Data Pipeline
- **70/15/15 Split**: 70% real training, 15% real validation, 15% logical test data
- **Real Data Priority**: 3,659 real samples (ages 18-65) from household dataset
- **Age Filtering**: Restricted to adults aged 18-65 for safety and relevance

## 🎯 System Features

### Smart BMI-Based Restrictions
- **Underweight**: Only Muscle Gain and Maintenance goals available
- **Normal BMI**: All goals (Fat Loss, Muscle Gain, Maintenance) available
- **Overweight**: Only Fat Loss and Maintenance goals available  
- **Obese**: Only Fat Loss goal available
- **UI Feedback**: Restricted options are greyed out with explanatory text

### Template System
- **9 Workout Templates**: 3 fitness goals × 3 activity levels
- **8 Nutrition Templates**: Evidence-based goal + BMI category combinations
- **Daily Outputs**: All recommendations in practical daily format
- **Activity Levels**: Low (1.29), Moderate (1.55), High (1.81) multipliers

### User Experience
- **Age Restriction**: 18-65 years only for safety
- **Indonesian UI**: Localized interface for Indonesian food database compatibility
- **Firebase Authentication**: Secure user management and data storage
- **Progressive Form**: Step-by-step input with real-time BMI calculation and goal restrictions

## 📁 Clean Project Structure

```
xgfitness/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Consolidated dependencies
│
├── 🗂️ backend/                     # Flask API server
│   ├── 📄 app.py                   # Main Flask application
│   ├── 📄 setup.py                 # Package setup
│   ├── 📁 src/                     # Core source code
│   │   ├── 📄 thesis_model.py      # Main AI model
│   │   ├── 📄 calculations.py      # BMI/BMR/TDEE calculations
│   │   ├── 📄 templates.py         # Template management
│   │   ├── 📄 validation.py        # Input validation
│   │   ├── 📄 meal_plan_calculator.py  # Meal planning
│   │   └── 📄 config.py            # Configuration settings
│   ├── 📁 models/                  # Trained model files
│   └── 📁 logs/                    # Application logs
│
├── 🗂️ frontend/                    # React web application
│   ├── 📄 package.json             # Node.js dependencies
│   ├── 📁 src/                     # React source code
│   ├── 📁 public/                  # Static assets
│   └── 📁 build/                   # Production build
│
├── 🗂️ data/                        # Datasets and templates
│   ├── 📄 nutrition_database.json  # Food database
│   ├── 📄 nutrition_templates.json # Nutrition templates
│   ├── 📄 workout_templates.json   # Workout templates
│   ├── 📁 meals/                   # Meal plan data
│   ├── 📁 nutrition/               # Nutrition data
│   ├── 📁 templates/               # Template definitions
│   └── 📁 backups/                 # Data backups
│
├── 🗂️ tests/                       # Comprehensive test suite
│   ├── 📄 test_comprehensive.py    # Main test suite
│   ├── 📄 test_meal_plans.py       # Meal plan tests
│   └── 📄 test_suite.py            # Additional tests
│
└── 🗂️ visualizations/              # Model analysis & charts
    ├── 📄 generate_clean_pngs.py   # Visualization generator
    ├── 📄 web_visualization_viewer.py  # Web viewer
    ├── 📄 launch_viewer.py         # Viewer launcher
    └── 📈 *.png                    # Generated charts
```

## 🚀 Quick Start

### Initial Setup (First Time Only)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate      # Windows PowerShell
# OR
source .venv/bin/activate   # Linux/Mac
```

### Backend Setup
```bash
# Activate virtual environment
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Install dependencies and train models
cd backend
python train_model.py     
python app.py             
```

### Frontend Setup
```bash
cd frontend
npm install
npm run build
npm start                
```

### Testing
```bash
cd backend
python test_suite.py       # Run comprehensive ML and API tests
```

##  Model Performance
- **Workout Model**: 81.5% accuracy, F1: 0.74 (realistic performance)
- **Nutrition Model**: 92.2% accuracy, F1: 0.91 (reduced from overfitting)
- **Real Data Usage**: 85% authentic data utilization (3,107 real samples)
- **Template Coverage**: All 9 workout and 8 nutrition templates actively used

## 🛠️ API Endpoints
- `POST /api/recommendations`: Get personalized fitness and nutrition recommendations
- `GET /health`: System health check and model status
- `GET /templates`: View available workout and nutrition templates

## 🔄 Key Technical Achievements
- ✅ Advanced 22-feature engineering with interaction terms
- ✅ BMI-based fitness goal restrictions with UI feedback
- ✅ Age restriction (18-65) with form validation
- ✅ Anti-overfitting measures preventing unrealistic model performance
- ✅ Daily-focused recommendations (not weekly)
- ✅ Indonesian localization for food database compatibility
- ✅ Firebase authentication and data persistence
- ✅ Progressive form with real-time BMI calculation

## 📄 License
Private project - All rights reserved.

---
*XGFitness AI - Evidence-based fitness recommendations powered by advanced machine learning.*
