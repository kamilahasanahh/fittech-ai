# FitTech AI - Advanced Fitness Recommendation System

## 🎯 Overview
FitTech AI is a comprehensive machine learning system that provides personalized fitness and nutrition recommendations using dual XGBoost and Random Forest algorithms with advanced 29-feature engineering and BMI-based goal restrictions. The system combines real household data with intelligent synthetic augmentation for robust, evidence-based recommendations.

## 🤖 Machine Learning Features

### Advanced Feature Engineering (29 Features)
- **Interaction Features**: BMI×Goal, Age×Activity, BMI×Activity Level, Age×Goal, Gender×Goal interactions
- **Metabolic Ratios**: BMR/weight, TDEE/BMR ratio, calorie needs per kg
- **Health Deviation Scores**: Distance from ideal BMI (22.5), weight/height ratio
- **Boolean Flags**: High metabolism, very active, young adult indicators
- **Activity Features**: Activity multiplier, intensity scores, moderate/vigorous activity hours
- **Core Features**: Age, gender, height, weight, BMI, BMR, TDEE, activity level, fitness goal, BMI category

### Model Architecture
- **XGBoost Models**: Separate optimized models for workout and nutrition recommendations
- **Random Forest Models**: Dual baseline models for academic comparison and validation
- **Hyperparameter Tuning**: RandomizedSearchCV with separate parameter grids to prevent overfitting
- **Feature Scaling**: StandardScaler for optimal model performance
- **Anti-Overfitting**: Noise injection and stronger regularization for nutrition model
- **Label Encoding**: Consistent class mapping across models

### Data Pipeline
- **70/15/15 Split**: 70% real training, 15% real validation, 15% real test data
- **Real Data Priority**: 3,659 real samples (ages 18-65) from household dataset
- **Synthetic Augmentation**: 2,203 synthetic samples for balanced goal distribution
- **Age Filtering**: Restricted to adults aged 18-65 for safety and relevance
- **Data Persistence**: Training data saved for visualization and analysis

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
fittech-ai/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Consolidated dependencies
│
├── 🗂️ backend/                     # Flask API server
│   ├── 📄 app.py                   # Main Flask application
│   ├── 📄 train_model.py           # Model training pipeline
│   ├── 📄 run_visualizations.py    # Visualization generator
│   ├── 📄 analyze_models.py        # Model analysis utility
│   ├── 📄 training_data.csv        # Generated training dataset
│   ├── 📁 src/                     # Core source code
│   │   ├── 📄 thesis_model.py      # Main AI model (29 features)
│   │   ├── 📄 calculations.py      # BMI/BMR/TDEE calculations
│   │   ├── 📄 templates.py         # Template management
│   │   ├── 📄 validation.py        # Input validation
│   │   ├── 📄 meal_plan_calculator.py  # Meal planning
│   │   └── 📄 config.py            # Configuration settings
│   ├── 📁 models/                  # Trained model files
│   │   ├── 📄 xgfitness_ai_model.pkl  # Production model (XGBoost only)
│   │   ├── 📄 xgfitness_ai_model_production.pkl  # Backup model
│   │   └── 📄 research_model_comparison_production.pkl  # Research model (XGB+RF)
│   ├── 📁 visualizations/          # Generated analysis charts
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
│   └── 📁 nutrition/               # Nutrition data
│
└── ️ tests/                       # Comprehensive test suite
    ├── 📄 test_comprehensive.py    # Main test suite
    └── 📄 test_suite.py            # Additional tests
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

# Train models and generate visualizations
cd backend
python train_model.py        # Train XGBoost + Random Forest models
python run_visualizations.py # Generate comprehensive analysis charts
python app.py               # Start Flask API server
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
python analyze_models.py   # Analyze model differences and performance
```

## 📊 Model Performance
- **XGBoost Workout Model**: 73.2% accuracy, F1: 0.73 (Production model)
- **XGBoost Nutrition Model**: 74.0% accuracy, F1: 0.67 (Production model)
- **Random Forest Workout Model**: 74.0% accuracy, F1: 0.74 (Research comparison)
- **Random Forest Nutrition Model**: 73.0% accuracy, F1: 0.68 (Research comparison)
- **Training Data**: 5,862 total samples (62% real, 38% synthetic)
- **Template Coverage**: All 9 workout and 8 nutrition templates actively used

## 🎨 Visualization & Analysis
The system generates comprehensive research-quality visualizations:
- **Dataset Composition Analysis**: Data source distribution and splits
- **Data Quality & Authenticity**: Real vs synthetic data validation
- **Demographic Analysis**: Age, gender, BMI, physiological distributions
- **Template Assignment Patterns**: Workout/nutrition template usage
- **Model Performance Comparison**: XGBoost vs Random Forest metrics
- **Confusion Matrices**: Prediction accuracy across all models
- **ROC Analysis**: Model performance curves and AUC scores
- **Research Summary Dashboard**: Executive overview for thesis

## 🛠️ API Endpoints
- `POST /api/recommendations`: Get personalized fitness and nutrition recommendations
- `GET /health`: System health check and model status
- `GET /templates`: View available workout and nutrition templates

## � Model Management
The system includes three distinct model files:
- **`xgfitness_ai_model.pkl`**: Production model (2.3 MB) - Used by web app
- **`xgfitness_ai_model_production.pkl`**: Backup production model (2.6 MB)
- **`research_model_comparison_production.pkl`**: Complete research model (24.1 MB) - Used for analysis

## �🔄 Key Technical Achievements
- ✅ Advanced 29-feature engineering with interaction terms and metabolic ratios
- ✅ Dual algorithm implementation (XGBoost + Random Forest) for comparison
- ✅ BMI-based fitness goal restrictions with intelligent UI feedback
- ✅ Age restriction (18-65) with comprehensive form validation
- ✅ Authentic data methodology: 70/15/15 split with 100% real validation/test data
- ✅ Synthetic data augmentation for balanced goal distribution
- ✅ Comprehensive visualization suite for thesis analysis
- ✅ Data persistence for reproducible research
- ✅ Indonesian localization for food database compatibility
- ✅ Firebase authentication and secure data storage

## 📄 License
Private project - All rights reserved.

---
*FitTech AI - Evidence-based fitness recommendations powered by advanced machine learning and comprehensive data analysis.*
